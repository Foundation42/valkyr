// Context representations under test.
//
// Every representation answers the same question: "what situation am I in?",
// as a single 64-bit id.  The decision rule downstream never changes, so any
// difference in the results is a difference in the context alone.
#pragma once

#include "common.hpp"

#include <algorithm>
#include <cmath>

namespace pc {

struct CtxCfg {
    enum class Kind {
        TypeDelta,      // access type + bucketed data delta only
        PcOnly,         // most recent instruction line
        ExplicitHist,   // last N *distinct* instruction lines
        Fading,         // multi-time-scale leaky state (Bitty)
        RandomMatched,  // frozen random id, matched cardinality (control)
    };
    enum class Quant { Sign, Ternary, Q2, Q3, Int8, Float };

    Kind kind = Kind::Fading;
    int hist_len = 4;
    std::vector<int> taus{2, 8, 32, 128};
    int heads = 8;
    Quant quant = Quant::Ternary;
    float thresh = 0.20f;

    bool use_type = true;
    bool use_delta = true;
    bool feed_instr = true;         // fading state driven by instruction lines
    bool feed_data = false;         // fading state driven by data lines
    bool shuffle_instr = false;     // control: same marginal, no temporal order
    bool integer_update = false;    // shift-friendly fixed-point leaky state
    bool distinct_instr_only = false;  // integrate only when the instruction
                                       // line changes (most fetches repeat a
                                       // line, and the update is the hot path)
    int instr_stride = 1;              // or integrate every Nth instruction

    uint64_t random_cardinality = 1 << 20;

    std::string label = "bitty4";
};

class ContextEngine {
public:
    void configure(const CtxCfg& c)
    {
        cfg_ = c;
        banks_ = int(cfg_.taus.size());
        if (cfg_.integer_update) {
            for (int t : cfg_.taus)
                if (t & (t - 1)) { std::fprintf(stderr, "integer update needs power-of-two tau\n"); std::exit(2); }
            shift_.clear();
            for (int t : cfg_.taus) { int s = 0; while ((1 << s) < t) ++s; shift_.push_back(s); }
        }
        reset();
    }

    void reset()
    {
        state_.assign(size_t(banks_) * cfg_.heads, 0.0f);
        istate_.assign(size_t(banks_) * cfg_.heads, 0);
        hist_.assign(cfg_.hist_len, 0);
        hist_n_ = 0;
        last_pc_line_ = 0;
        last_pc_addr_ = 0;
        last_integrated_ = ~0ull;
        instr_seen_ = 0;
        last_data_line_ = 0;
        have_last_data_ = false;
        cur_type_ = 0;
        cur_bucket_ = 0;
        data_index_ = 0;
        update_ops_ = 0;
        shuffle_buf_.assign(4096, 0);
        shuffle_fill_ = 0;
        rng_ = 0x243F6A8885A308D3ull;
    }

    // ---- observation --------------------------------------------------
    // Instruction references only ever touch the fading state / history.
    // They never advance the data index; that separation is asserted by the
    // audit suite (see audit.cpp, "index mixing").
    void observeInstr(uint64_t addr)
    {
        uint64_t line = addr >> kLineShift;
        last_pc_addr_ = addr;
        last_pc_line_ = line;

        if (cfg_.kind == CtxCfg::Kind::ExplicitHist) {
            if (hist_n_ == 0 || hist_[(hist_n_ - 1) % cfg_.hist_len] != line) {
                hist_[hist_n_ % cfg_.hist_len] = line;
                ++hist_n_;
            }
            return;
        }
        if (cfg_.kind != CtxCfg::Kind::Fading || !cfg_.feed_instr) return;
        if (cfg_.distinct_instr_only) {
            if (line == last_integrated_) return;
            last_integrated_ = line;
        }
        if (cfg_.instr_stride > 1 && (instr_seen_++ % uint64_t(cfg_.instr_stride)) != 0) return;

        if (cfg_.shuffle_instr) {
            uint64_t slot = next_rand() % shuffle_buf_.size();
            uint64_t fed = shuffle_fill_ ? shuffle_buf_[slot] : line;
            shuffle_buf_[slot] = line;
            if (shuffle_fill_ < shuffle_buf_.size()) ++shuffle_fill_;
            line = fed;
        }
        integrate(line, kInstrSalt);
    }

    // Called once per data reference, *before* id() is read, so the context
    // includes this reference's own type and delta but nothing after it.
    void observeData(uint64_t line, int type)
    {
        cur_type_ = type;
        int64_t d = have_last_data_ ? int64_t(line) - int64_t(last_data_line_) : 0;
        cur_bucket_ = have_last_data_ ? delta_bucket(d) : 0;
        last_data_line_ = line;
        have_last_data_ = true;
        ++data_index_;

        if (cfg_.kind == CtxCfg::Kind::Fading && cfg_.feed_data)
            integrate(line, kDataSalt);
    }

    uint64_t id() const
    {
        uint64_t h = 0xCBF29CE484222325ull;
        switch (cfg_.kind) {
        case CtxCfg::Kind::TypeDelta:
            break;
        case CtxCfg::Kind::PcOnly:
            h = hash_combine(h, last_pc_line_);
            break;
        case CtxCfg::Kind::ExplicitHist: {
            int n = std::min<int>(cfg_.hist_len, int(hist_n_));
            for (int k = 0; k < n; ++k) {
                uint64_t idx = hist_n_ - 1 - uint64_t(k);
                h = hash_combine(h, hist_[idx % cfg_.hist_len]);
            }
            h = hash_combine(h, uint64_t(n));
            break;
        }
        case CtxCfg::Kind::Fading:
            h = quantised_hash(h);
            break;
        case CtxCfg::Kind::RandomMatched:
            return 1 + (mix64(data_index_ * 0x9E3779B97F4A7C15ull) % cfg_.random_cardinality);
        }
        if (cfg_.use_type)  h = hash_combine(h, 0x100u + uint64_t(cur_type_));
        if (cfg_.use_delta) h = hash_combine(h, 0x200u + uint64_t(cur_bucket_ + 32));
        return h | 1;                      // 0 is reserved as "empty slot"
    }

    // Bytes of live temporal state (what an implementation would have to keep
    // per hardware context).  Explicit histories pay 8 bytes per retained line.
    size_t stateBytes() const
    {
        switch (cfg_.kind) {
        case CtxCfg::Kind::TypeDelta:     return 8 + 1;                 // last data line + type
        case CtxCfg::Kind::PcOnly:        return 8 + 8 + 1;
        case CtxCfg::Kind::ExplicitHist:  return size_t(cfg_.hist_len) * 8 + 8 + 1;
        case CtxCfg::Kind::RandomMatched: return 8;
        case CtxCfg::Kind::Fading: {
            size_t scalars = size_t(banks_) * cfg_.heads;
            size_t per = cfg_.integer_update ? 2 : 4;                   // Q8.8 vs float
            return scalars * per + 8 + 1;
        }
        }
        return 0;
    }

    // Scalar leaky-state updates performed on the hot path.
    uint64_t updateOps() const { return update_ops_; }
    uint64_t lastPcAddr() const { return last_pc_addr_; }
    uint64_t lastPcLine() const { return last_pc_line_; }
    const CtxCfg& cfg() const { return cfg_; }

private:
    void integrate(uint64_t line, uint64_t salt)
    {
        const int H = cfg_.heads;
        if (cfg_.integer_update) {
            for (int j = 0; j < H; ++j) {
                int32_t x = signed_feature(line, j, salt) > 0 ? 256 : -256;   // Q8.8
                for (int t = 0; t < banks_; ++t) {
                    int32_t& h = istate_[size_t(t) * H + j];
                    h += (x - h) >> shift_[t];
                }
            }
        } else {
            for (int j = 0; j < H; ++j) {
                float x = signed_feature(line, j, salt);
                for (int t = 0; t < banks_; ++t) {
                    float& h = state_[size_t(t) * H + j];
                    h += (x - h) / float(cfg_.taus[t]);
                }
            }
        }
        update_ops_ += uint64_t(H) * banks_;
    }

    uint64_t quantised_hash(uint64_t h) const
    {
        const int H = cfg_.heads;
        const size_t n = size_t(banks_) * H;
        // Pack quantised levels a few per hash step; the exact packing only has
        // to be deterministic and collision-resistant.
        uint64_t acc = 0;
        int packed = 0;
        for (size_t i = 0; i < n; ++i) {
            float v = cfg_.integer_update ? float(istate_[i]) / 256.0f : state_[i];
            uint64_t q = quantise(v);
            acc = acc * 257u + q;
            if (++packed == 7) { h = hash_combine(h, acc); acc = 0; packed = 0; }
        }
        if (packed) h = hash_combine(h, acc);
        return h;
    }

    uint64_t quantise(float v) const
    {
        switch (cfg_.quant) {
        case CtxCfg::Quant::Sign:    return v >= 0.0f ? 1 : 0;
        case CtxCfg::Quant::Ternary: return v > cfg_.thresh ? 2 : (v < -cfg_.thresh ? 0 : 1);
        case CtxCfg::Quant::Q2:      return level(v, 4);
        case CtxCfg::Quant::Q3:      return level(v, 8);
        case CtxCfg::Quant::Int8:    return level(v, 256);
        case CtxCfg::Quant::Float: {
            float f = v;
            uint32_t bits;
            std::memcpy(&bits, &f, 4);
            return bits;
        }
        }
        return 0;
    }

    static uint64_t level(float v, int L)
    {
        float t = (v + 1.0f) * 0.5f * float(L);
        int q = int(t);
        if (q < 0) q = 0;
        if (q >= L) q = L - 1;
        return uint64_t(q);
    }

    uint64_t next_rand() { rng_ = mix64(rng_); return rng_; }

    CtxCfg cfg_;
    int banks_ = 4;
    std::vector<int> shift_;
    std::vector<float> state_;
    std::vector<int32_t> istate_;
    std::vector<uint64_t> hist_;
    uint64_t hist_n_ = 0;
    uint64_t last_pc_line_ = 0, last_pc_addr_ = 0, last_data_line_ = 0;
    bool have_last_data_ = false;
    int cur_type_ = 0, cur_bucket_ = 0;
    uint64_t data_index_ = 0, update_ops_ = 0;
    uint64_t last_integrated_ = ~0ull, instr_seen_ = 0;
    std::vector<uint64_t> shuffle_buf_;
    size_t shuffle_fill_ = 0;
    uint64_t rng_ = 0;
};

}  // namespace pc
