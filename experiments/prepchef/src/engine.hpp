// The evaluation engine: one place where prefetches are issued, deduplicated,
// expired and credited, shared by PrepChef and every baseline so that no policy
// can be advantaged by its accounting.
//
// Ordering at scored data reference i (this order is normative; audit.cpp
// pins it with a hand-checked synthetic trace):
//   1. expire   every outstanding prefetch with (i - issue_index) > window
//   2. observe  counterfactual demand-only cache (demand traffic only)
//   3. demand   if line(i) is outstanding, credit it exactly once and retire it
//   4. learn    predictor updates from references it has already seen
//   5. issue    at most one new prefetch, suppressed if already outstanding
#pragma once

#include "common.hpp"
#include "predictor.hpp"

#include <chrono>
#include <deque>
#include <unordered_map>

namespace pc {

struct RunCfg {
    uint64_t window = 32;           // usefulness horizon, in data references
    double warmup_frac = 0.20;      // leading fraction of data refs used online, unscored
    double eval_begin = 0.20;       // scored region (fractions of data refs)
    double eval_end = 1.00;
    float value = 1.0f;
    float waste = 0.25f;
    bool allow_self_prefetch = true;   // PC-BASE allows a delta-0 action
    uint64_t max_refs = 0;             // 0 = whole trace
    // Counterfactual L1 used for miss-filtered metrics (demand traffic only).
    size_t l1_sets = 64;
    int l1_ways = 8;
    // Which reward the *learner* is told about.  Window is the handoff's crude
    // protocol; MissFiltered pays only for prefetches that saved a real miss,
    // and charges the waste price for redundant ones.
    enum class Reward { Window, MissFiltered } reward_mode = Reward::Window;
    bool log_events = false;
    // Audit-only: from this total-reference position onward, corrupt every
    // address.  Anything computed from the past alone must be unaffected.
    uint64_t corrupt_after_pos = 0;
};

struct Metrics {
    uint64_t total_refs = 0, instr_refs = 0, data_refs = 0;
    uint64_t scored_refs = 0, scored_misses = 0;
    uint64_t issued = 0, useful = 0, wasted = 0, unresolved = 0;
    uint64_t useful_miss = 0;             // credited *and* would have missed
    uint64_t self_issued = 0, self_useful = 0;   // action == current line
    uint64_t dedup_suppressed = 0;
    double lead_data_sum = 0, lead_total_sum = 0;
    uint64_t distinct_data_lines = 0;
    uint64_t evictions = 0, inserts = 0;
    // Per-arm reporting only.  Nothing in the engine's decisions reads these;
    // they exist so the *learned* action distribution over horizons can be
    // compared against the independently measured model-free spectrum.
    std::vector<uint64_t> arm_issued, arm_useful, arm_useful_miss;
    size_t state_bytes = 0, table_bytes = 0;
    uint64_t update_ops = 0;
    double ns_per_ref = 0;

    double accuracy() const { return issued ? double(useful) / double(issued) : 0.0; }
    double coverage() const { return scored_refs ? double(useful) / double(scored_refs) : 0.0; }
    double strict_accuracy() const { return issued ? double(useful_miss) / double(issued) : 0.0; }
    double strict_coverage() const { return scored_misses ? double(useful_miss) / double(scored_misses) : 0.0; }
    double net_per_1k(float value, float waste) const
    {
        if (!scored_refs) return 0.0;
        double net = double(useful) * value - double(issued - useful) * waste;
        return net / double(scored_refs) * 1000.0;
    }
    double strict_net_per_1k(float value, float waste) const
    {
        if (!scored_refs) return 0.0;
        double net = double(useful_miss) * value - double(issued - useful_miss) * waste;
        return net / double(scored_refs) * 1000.0;
    }
    double mean_lead_data() const { return useful ? lead_data_sum / double(useful) : 0.0; }
    double mean_lead_total() const { return useful ? lead_total_sum / double(useful) : 0.0; }
    double action_rate() const { return scored_refs ? double(issued) / double(scored_refs) : 0.0; }
};

// Event log consumed by the *independent* scorer in scorer.hpp.  It records
// only what an outside observer could see: when a line was demanded and when a
// prefetch for a line was issued.
struct EventLog {
    std::vector<uint64_t> demand_line;     // indexed by scored data index
    std::vector<uint64_t> demand_total;    // total-ref index of that demand
    std::vector<uint64_t> issue_index;     // scored data index of each issue
    std::vector<uint64_t> issue_line;
    uint64_t window = 32;
};

class Engine {
public:
    static void bump_arm(std::vector<uint64_t>& v, uint32_t arm)
    {
        if (v.size() <= arm) v.resize(size_t(arm) + 1, 0);
        ++v[arm];
    }

    Metrics run(const Trace& tr, Predictor& p, const RunCfg& cfg, EventLog* log = nullptr)
    {
        Metrics m;
        // Pass 1: how many data references are there?  (Needed to place the
        // warm-up/eval boundary chronologically, and nothing else.)
        uint64_t limit = cfg.max_refs ? std::min(cfg.max_refs, tr.n) : tr.n;
        uint64_t n_data = 0;
        for (uint64_t i = 0; i < limit; ++i) if (tr.type(i) != kIFetch) ++n_data;

        const uint64_t warm_end   = uint64_t(double(n_data) * cfg.warmup_frac);
        const uint64_t score_from = std::max(warm_end, uint64_t(double(n_data) * cfg.eval_begin));
        const uint64_t score_to   = uint64_t(double(n_data) * cfg.eval_end);

        LruCache l1;
        l1.configure(cfg.l1_sets, cfg.l1_ways);
        p.reset();

        struct Rec { uint64_t issue_index, issue_total; uint64_t ctx, ctx2; int64_t action;
                     bool self_pf; uint32_t arm; };
        std::unordered_map<uint64_t, Rec> outstanding;
        outstanding.reserve(1u << 12);
        std::deque<std::pair<uint64_t, uint64_t>> fifo;   // (issue_index, line)

        if (log) { log->window = cfg.window; log->demand_line.reserve(n_data); }

        auto t0 = std::chrono::steady_clock::now();
        uint64_t di = 0;     // data-reference index (never advanced by ifetch)

        for (uint64_t i = 0; i < limit; ++i) {
            const uint8_t ty = tr.type(i);
            uint64_t ad = tr.addr(i);
            if (cfg.corrupt_after_pos && i >= cfg.corrupt_after_pos)
                ad = (ad ^ mix64(i)) & ((1ull << 48) - 1);
            if (ty == kIFetch) { p.observeInstr(ad); ++m.instr_refs; continue; }

            const uint64_t line = ad >> kLineShift;
            const bool scored = (di >= score_from && di < score_to);

            // 1. expire
            while (!fifo.empty() && di - fifo.front().first > cfg.window) {
                auto [ix, ln] = fifo.front();
                fifo.pop_front();
                auto it = outstanding.find(ln);
                if (it != outstanding.end() && it->second.issue_index == ix) {
                    ++m.wasted;
                    p.reward(it->second.ctx, it->second.ctx2, it->second.action, -cfg.waste);
                    outstanding.erase(it);
                }
            }

            // 2. counterfactual demand-only cache, consulted before crediting so
            //    the reward handed to the learner can reflect real prices
            const bool would_miss = !l1.access(line);

            // 3. demand credit (exactly once; the entry is erased on credit)
            bool credited = false;
            if (scored) {
                auto it = outstanding.find(line);
                if (it != outstanding.end()) {
                    ++m.useful;
                    m.lead_data_sum += double(di - it->second.issue_index);
                    m.lead_total_sum += double(i - it->second.issue_total);
                    if (it->second.self_pf) ++m.self_useful;
                    bump_arm(m.arm_useful, it->second.arm);
                    if (would_miss) bump_arm(m.arm_useful_miss, it->second.arm);
                    const float r = (cfg.reward_mode == RunCfg::Reward::Window)
                                        ? cfg.value
                                        : (would_miss ? cfg.value : -cfg.waste);
                    p.reward(it->second.ctx, it->second.ctx2, it->second.action, r);
                    outstanding.erase(it);
                    credited = true;
                }
            }

            if (scored) {
                ++m.scored_refs;
                if (would_miss) ++m.scored_misses;
                if (credited && would_miss) ++m.useful_miss;
                if (log) { log->demand_line.push_back(line); log->demand_total.push_back(i); }
                // (the scored-region index of this demand is
                //  log->demand_line.size() - 1, which is what the issue log
                //  below is expressed in too)
            }

            // 4 + 5. learn, then speculate
            Proposal pr = p.onData(line, ty, di);
            ++m.data_refs;

            if (scored && pr.act) {
                uint64_t target = pr.target;
                bool self_pf = (target == line);
                if (!cfg.allow_self_prefetch && self_pf) {
                    // registered variant: a "prefetch" of the line just touched
                    // is not a preparation at all, so it is not issued
                } else if (outstanding.count(target)) {
                    ++m.dedup_suppressed;
                } else {
                    outstanding.emplace(target,
                                        Rec{di, i, pr.ctx, pr.ctx2, pr.action, self_pf, pr.arm});
                    fifo.emplace_back(di, target);
                    ++m.issued;
                    bump_arm(m.arm_issued, pr.arm);
                    if (self_pf) ++m.self_issued;
                    if (log) {
                        // Issues are logged in *scored-region* data indices so
                        // the independent scorer never has to know where the
                        // warm-up boundary was.
                        log->issue_index.push_back(di - score_from);
                        log->issue_line.push_back(target);
                    }
                }
            }
            ++di;
        }

        // Anything still outstanding never paid off.
        m.unresolved = uint64_t(outstanding.size());
        m.wasted += m.unresolved;

        auto t1 = std::chrono::steady_clock::now();
        m.total_refs = limit;
        m.ns_per_ref = std::chrono::duration<double, std::nano>(t1 - t0).count() / double(limit);
        m.state_bytes = p.stateBytes();
        m.table_bytes = p.tableBytes();
        m.update_ops = p.updateOps();
        m.evictions = p.evictions();
        m.inserts = p.inserts();
        return m;
    }
};

}  // namespace pc
