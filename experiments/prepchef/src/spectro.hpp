// Model-free trace spectroscopy.
//
// No learner, no context, no prefetcher: these are properties of the reference
// stream alone.  Their purpose is to tell "the predictor behaves oddly at
// horizon h" apart from "the workload is structured at lag h".
//
// For each lag h, over the scored data-reference stream L[i] with a miss
// indicator m[i] from the same counterfactual demand-only L1 the engine uses:
//
//   recur(h)          P(L[i+h] == L[i])                  self-similarity
//   miss_lift(h)      P(m[i+h] | m[i]) / P(m)            does missing recur at lag h
//   top1_mass(h)      max_d P(L[i+h] - L[i] == d)        how concentrated the h-step
//                                                        delta distribution is; a
//                                                        model-free ceiling for any
//                                                        single-delta predictor
//   top1nz_mass(h)    the same, excluding d = 0.  A delta-0 "preparation"
//                     prepares nothing and can never land on a miss, so the
//                     unrestricted top-1 is dominated by it and says little.
//   top1_miss(h)      P(L[i+h]-L[i] == d* AND m[i+h])/P(m)
//                                                        the part of that ceiling that
//                                                        lands on a real miss -- the
//                                                        model-free analogue of strict
//                                                        coverage
//   missdelta_top1(g) max_d P(M[j+g] - M[j] == d)        the same concentration measured
//                                                        in miss-index space, over the
//                                                        subsequence of missing refs
#pragma once

#include "common.hpp"

#include <unordered_map>

namespace pc {

struct SpectroRow {
    int h = 0;
    double recur = 0, miss_lift = 0, top1_mass = 0, top1_miss = 0, missdelta_top1 = 0;
    double top1nz_mass = 0, top1nz_miss = 0;
    int64_t top1_delta = 0, top1nz_delta = 0, missdelta_delta = 0;
};

struct SpectroResult {
    uint64_t n = 0, n_miss = 0;
    double miss_rate = 0;
    std::vector<SpectroRow> rows;
};

inline SpectroResult spectroscopy(const Trace& tr, int max_lag, double warmup_frac,
                                  size_t l1_sets, int l1_ways, uint64_t sample_stride)
{
    // Pass 1: the scored data-line stream and its miss indicator.
    uint64_t n_data = 0;
    for (uint64_t i = 0; i < tr.n; ++i) if (tr.type(i) != kIFetch) ++n_data;
    const uint64_t score_from = uint64_t(double(n_data) * warmup_frac);

    std::vector<uint64_t> line;
    std::vector<uint8_t> miss;
    line.reserve(n_data - score_from);
    miss.reserve(n_data - score_from);

    LruCache l1;
    l1.configure(l1_sets, l1_ways);
    uint64_t di = 0;
    for (uint64_t i = 0; i < tr.n; ++i) {
        if (tr.type(i) == kIFetch) continue;
        const uint64_t ln = tr.addr(i) >> kLineShift;
        const bool m = !l1.access(ln);          // cache stays warm through warm-up
        if (di >= score_from) { line.push_back(ln); miss.push_back(m ? 1 : 0); }
        ++di;
    }

    SpectroResult out;
    out.n = line.size();
    for (uint8_t m : miss) out.n_miss += m;
    out.miss_rate = out.n ? double(out.n_miss) / double(out.n) : 0.0;

    // The miss subsequence, for the miss-index-space measurement.
    std::vector<uint64_t> mline;
    mline.reserve(out.n_miss);
    for (size_t i = 0; i < line.size(); ++i) if (miss[i]) mline.push_back(line[i]);

    std::unordered_map<int64_t, uint64_t> hist, mhist;
    for (int h = 1; h <= max_lag; ++h) {
        SpectroRow r;
        r.h = h;
        if (out.n <= uint64_t(h)) { out.rows.push_back(r); continue; }

        uint64_t samples = 0, same_line = 0, miss_after_miss = 0, miss_anchor = 0;
        hist.clear();
        hist.reserve(1 << 14);
        for (size_t i = 0; i + size_t(h) < line.size(); i += sample_stride) {
            ++samples;
            if (line[i + h] == line[i]) ++same_line;
            if (miss[i]) { ++miss_anchor; if (miss[i + h]) ++miss_after_miss; }
            ++hist[int64_t(line[i + h]) - int64_t(line[i])];
        }
        if (!samples) { out.rows.push_back(r); continue; }

        int64_t best_d = 0, best_nz_d = 0;
        uint64_t best_c = 0, best_nz_c = 0;
        for (const auto& kv : hist) {
            if (kv.second > best_c) { best_c = kv.second; best_d = kv.first; }
            if (kv.first != 0 && kv.second > best_nz_c) { best_nz_c = kv.second; best_nz_d = kv.first; }
        }

        r.recur = double(same_line) / double(samples);
        r.top1_mass = double(best_c) / double(samples);
        r.top1_delta = best_d;
        r.top1nz_mass = double(best_nz_c) / double(samples);
        r.top1nz_delta = best_nz_d;
        if (miss_anchor && out.miss_rate > 0)
            r.miss_lift = (double(miss_after_miss) / double(miss_anchor)) / out.miss_rate;

        // How much of that top-1 mass lands on a reference that really missed,
        // expressed as a fraction of all misses (so it is comparable with the
        // strict coverage the engine reports).
        uint64_t hit_and_miss = 0, nz_hit_and_miss = 0, scanned = 0;
        for (size_t i = 0; i + size_t(h) < line.size(); i += sample_stride) {
            ++scanned;
            const int64_t d = int64_t(line[i + h]) - int64_t(line[i]);
            if (miss[i + h]) {
                if (d == best_d) ++hit_and_miss;
                if (d == best_nz_d) ++nz_hit_and_miss;
            }
        }
        if (scanned && out.n_miss) {
            r.top1_miss = double(hit_and_miss) * double(sample_stride) / double(out.n_miss);
            r.top1nz_miss = double(nz_hit_and_miss) * double(sample_stride) / double(out.n_miss);
        }

        // Same concentration measure, but walking only the misses.
        if (mline.size() > size_t(h)) {
            mhist.clear();
            mhist.reserve(1 << 14);
            uint64_t ms = 0;
            for (size_t j = 0; j + size_t(h) < mline.size(); ++j) {
                ++ms;
                ++mhist[int64_t(mline[j + h]) - int64_t(mline[j])];
            }
            uint64_t bc = 0;
            int64_t bd = 0;
            for (const auto& kv : mhist) if (kv.second > bc) { bc = kv.second; bd = kv.first; }
            if (ms) { r.missdelta_top1 = double(bc) / double(ms); r.missdelta_delta = bd; }
        }
        out.rows.push_back(r);
    }
    return out;
}

}  // namespace pc
