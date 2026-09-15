// Predictors: PrepChef itself, plus the registered controls and the stronger
// prefetcher baselines.  All of them see exactly the same reference stream and
// are scored by the same engine at the same action budget (degree 1).
#pragma once

#include "common.hpp"
#include "context.hpp"
#include "learner.hpp"

#include <string>

namespace pc {

struct Proposal {
    bool act = false;
    uint64_t target = 0;
    uint64_t ctx = 0;          // the key the reward must be routed back to
    int64_t action = 0;
    uint32_t arm = 0;          // which horizon arm proposed it (reporting only)
};

struct Predictor {
    virtual ~Predictor() = default;
    virtual void reset() = 0;
    virtual void observeInstr(uint64_t addr) { last_pc_ = addr; }
    // Learn from this reference, then decide whether to prepare anything.
    virtual Proposal onData(uint64_t line, int type, uint64_t data_index) = 0;
    virtual void reward(uint64_t, int64_t, float) {}
    virtual size_t stateBytes() const { return 0; }
    virtual size_t tableBytes() const { return 0; }
    virtual uint64_t updateOps() const { return 0; }
    virtual uint64_t evictions() const { return 0; }
    virtual uint64_t inserts() const { return 0; }
    virtual std::string label() const = 0;
    uint64_t last_pc_ = 0;
};

// ---------------------------------------------------------------- PrepChef --

class PrepChef : public Predictor {
public:
    PrepChef(const CtxCfg& c, const LearnCfg& l) : ccfg_(c), lcfg_(l)
    {
        ctx_.configure(ccfg_);
        learn_.configure(lcfg_);
        learn_.ensureBandit();
        if (lcfg_.horizon < 1) lcfg_.horizon = 1;
        ring_.assign(size_t(lcfg_.horizon), {0, 0});
    }

    void reset() override
    {
        ctx_.reset();
        learn_.reset();
        std::fill(ring_.begin(), ring_.end(), std::pair<uint64_t, uint64_t>{0, 0});
        seen_ = 0;
    }

    void observeInstr(uint64_t addr) override { last_pc_ = addr; ctx_.observeInstr(addr); }

    Proposal onData(uint64_t line, int type, uint64_t) override
    {
        // (a) the delayed label: the context recorded `horizon` data references
        //     ago is now known to have been followed by `line`.  The slot read
        //     here is the same one written below, so it must be read first.
        const size_t slot = size_t(seen_ % uint64_t(lcfg_.horizon));
        if (seen_ >= uint64_t(lcfg_.horizon)) {
            const auto& past = ring_[slot];
            int64_t outcome = lcfg_.outcome_delta ? int64_t(line) - int64_t(past.second)
                                                  : int64_t(line);
            learn_.learn(past.first, outcome);
        }

        // (b) advance the context with this reference, then read it
        ctx_.observeData(line, type);
        const uint64_t id = ctx_.id();

        ring_[slot] = {id, line};
        ++seen_;

        // (c) speculate
        Learner::Proposal pr = learn_.propose(id);
        Proposal out;
        out.ctx = id;
        out.action = pr.action;
        if (pr.act) {
            int64_t t = lcfg_.outcome_delta ? int64_t(line) + pr.action : pr.action;
            if (t >= 0) { out.act = true; out.target = uint64_t(t); }
        }
        return out;
    }

    void reward(uint64_t ctx, int64_t action, float r) override { learn_.reward(ctx, action, r); }

    size_t stateBytes() const override { return ctx_.stateBytes(); }
    size_t tableBytes() const override { return learn_.tableBytes(); }
    uint64_t updateOps() const override { return ctx_.updateOps(); }
    uint64_t evictions() const override { return learn_.evictions(); }
    uint64_t inserts() const override { return learn_.inserts(); }
    std::string label() const override { return ccfg_.label + "/" + lcfg_.label; }

    const ContextEngine& context() const { return ctx_; }

private:
    CtxCfg ccfg_;
    LearnCfg lcfg_;
    ContextEngine ctx_;
    Learner learn_;
    std::vector<std::pair<uint64_t, uint64_t>> ring_;   // (context, line) delay line
    uint64_t seen_ = 0;
};

// ------------------------------------------------- G65: multi-horizon PrepChef

// Exactly one thing changes from PrepChef: the action gains a dimension.
// Instead of a = delta it is a = (h, delta), and the *existing* realised-reward
// gate chooses among the arms, with the null action still competing.
//
// Three implementation choices keep "one thing" literally true:
//
//   * One association table, keyed by hash(context, h), at the same capacity
//     the single-horizon learner uses.  The arms compete for the same slots,
//     so multi-horizon gets no memory advantage by construction (evictions are
//     reported).
//   * One delay ring of length max(H), read back h steps, rather than |H|
//     rings.  |H| rings would be a strawman cost.
//   * The context and its hash are computed once per data reference and shared
//     by every arm, so the expensive part -- the fading-state update, which
//     runs per *instruction* -- is not multiplied.  Only the per-data-reference
//     table lookups are.
class MultiPrepChef : public Predictor {
public:
    enum class Select {
        BestEV,          // argmax over arms of realised-reward estimate (the system under test)
        UniformRandom,   // commit to a random arm, then let that arm's gate decide (floor)
        Pooled,          // one estimator fed by every arm's evidence, no ability to tell h apart
        BestEVTested,    // as BestEV, but an untested arm cannot win the argmax:
                         // only one designated explorer may fire on its prior
    };

    MultiPrepChef(const CtxCfg& c, const LearnCfg& l, std::vector<int> horizons,
                  Select sel, std::string label)
        : ccfg_(c), lcfg_(l), horizons_(std::move(horizons)), sel_(sel), label_(std::move(label))
    {
        ctx_.configure(ccfg_);
        learn_.configure(lcfg_);
        learn_.ensureBandit();
        max_h_ = 1;
        for (int h : horizons_) if (h > max_h_) max_h_ = h;
        ring_.assign(size_t(max_h_), {0, 0});
    }

    void reset() override
    {
        ctx_.reset();
        learn_.reset();
        std::fill(ring_.begin(), ring_.end(), std::pair<uint64_t, uint64_t>{0, 0});
        seen_ = 0;
        rng_ = 0x9E3779B97F4A7C15ull;
    }

    void observeInstr(uint64_t addr) override { last_pc_ = addr; ctx_.observeInstr(addr); }

    Proposal onData(uint64_t line, int type, uint64_t) override
    {
        // (a) one delayed label per arm.  All reads happen before the write
        //     below, so no arm can see its own future.
        for (size_t k = 0; k < horizons_.size(); ++k) {
            const uint64_t h = uint64_t(horizons_[k]);
            if (seen_ < h) continue;
            const auto& past = ring_[size_t((seen_ - h) % uint64_t(max_h_))];
            const int64_t outcome = lcfg_.outcome_delta ? int64_t(line) - int64_t(past.second)
                                                        : int64_t(line);
            learn_.learn(key(past.first, sel_ == Select::Pooled ? kPoolId : horizons_[k]), outcome);
        }

        // (b) advance the shared context, then read it once
        ctx_.observeData(line, type);
        const uint64_t id = ctx_.id();
        ring_[size_t(seen_ % uint64_t(max_h_))] = {id, line};
        ++seen_;

        // (c) choose among { null, (h, delta) ... }
        Proposal out;
        switch (sel_) {
        case Select::BestEV:
        case Select::BestEVTested: {
            // With one arm an optimistic prior costs a few wasted prefetches and
            // is then corrected.  Taking the argmax over |H| arms makes it a
            // maximisation bias: some untested arm is almost always sitting at
            // the prior, so the null action stops competing.  BestEVTested
            // therefore ranks only arms that have seen a realised reward, and
            // lets a single designated explorer (the shortest horizon) fire on
            // its prior, so exploration is one arm per context rather than |H|.
            const bool tested_only = (sel_ == Select::BestEVTested);
            float best_u = 0.0f;
            bool have = false;
            size_t explorer = horizons_.size();
            for (size_t k = 0; k < horizons_.size(); ++k) {
                const uint64_t ck = key(id, horizons_[k]);
                Learner::Proposal pr = learn_.propose(ck);
                if (!pr.act) continue;
                if (tested_only && pr.n_reward == 0) {
                    if (explorer == horizons_.size()) {
                        explorer = k;
                        expl_ctx_ = ck; expl_action_ = pr.action;
                    }
                    continue;
                }
                if (!have || pr.utility > best_u) {
                    have = true; best_u = pr.utility;
                    out.ctx = ck; out.action = pr.action; out.arm = uint32_t(k);
                }
            }
            if (!have) {
                if (!tested_only || explorer == horizons_.size()) return Proposal{};
                out.ctx = expl_ctx_; out.action = expl_action_; out.arm = uint32_t(explorer);
            }
            break;
        }
        case Select::UniformRandom: {
            const size_t k = size_t(next_rand() % horizons_.size());
            const uint64_t ck = key(id, horizons_[k]);
            Learner::Proposal pr = learn_.propose(ck);
            if (!pr.act) return Proposal{};
            out.ctx = ck; out.action = pr.action; out.arm = uint32_t(k);
            break;
        }
        case Select::Pooled: {
            const uint64_t ck = key(id, kPoolId);
            Learner::Proposal pr = learn_.propose(ck);
            if (!pr.act) return Proposal{};
            // The pooled estimator cannot tell the horizons apart, so it has no
            // basis for preferring one; it acts at a uniformly random horizon.
            out.ctx = ck; out.action = pr.action;
            out.arm = uint32_t(next_rand() % horizons_.size());
            break;
        }
        }

        const int64_t t = lcfg_.outcome_delta ? int64_t(line) + out.action : out.action;
        if (t < 0) return Proposal{};
        out.act = true;
        out.target = uint64_t(t);
        return out;
    }

    void reward(uint64_t ctx, int64_t action, float r) override { learn_.reward(ctx, action, r); }

    size_t stateBytes() const override { return ctx_.stateBytes() + ring_.size() * 16; }
    size_t tableBytes() const override { return learn_.tableBytes(); }
    uint64_t updateOps() const override { return ctx_.updateOps(); }
    uint64_t evictions() const override { return learn_.evictions(); }
    uint64_t inserts() const override { return learn_.inserts(); }
    std::string label() const override { return label_; }
    const std::vector<int>& horizons() const { return horizons_; }

private:
    static constexpr int kPoolId = 0x7FFF;
    static uint64_t key(uint64_t ctx, int h)
    {
        return hash_combine(ctx, 0xA0000ull + uint64_t(h)) | 1;
    }
    uint64_t next_rand() { rng_ = mix64(rng_); return rng_; }

    CtxCfg ccfg_;
    LearnCfg lcfg_;
    std::vector<int> horizons_;
    Select sel_;
    std::string label_;
    ContextEngine ctx_;
    Learner learn_;
    std::vector<std::pair<uint64_t, uint64_t>> ring_;
    int max_h_ = 1;
    uint64_t seen_ = 0, rng_ = 0;
    uint64_t expl_ctx_ = 0;
    int64_t expl_action_ = 0;
};

// --------------------------------------------------------------- baselines --

class NextLine : public Predictor {
public:
    void reset() override {}
    Proposal onData(uint64_t line, int, uint64_t) override
    {
        Proposal p; p.act = true; p.target = line + 1; p.action = 1; return p;
    }
    std::string label() const override { return "next-line"; }
    size_t stateBytes() const override { return 0; }
};

class LastStride : public Predictor {
public:
    void reset() override { have_ = false; last_ = 0; delta_ = 0; }
    Proposal onData(uint64_t line, int, uint64_t) override
    {
        Proposal p;
        if (have_) {
            int64_t d = int64_t(line) - int64_t(last_);
            if (delta_ != 0) {
                int64_t t = int64_t(line) + delta_;
                if (t >= 0) { p.act = true; p.target = uint64_t(t); p.action = delta_; }
            }
            delta_ = d;
        }
        last_ = line; have_ = true;
        return p;
    }
    std::string label() const override { return "last-stride"; }
    size_t stateBytes() const override { return 16; }
private:
    bool have_ = false;
    uint64_t last_ = 0;
    int64_t delta_ = 0;
};

// Classic PC-indexed stride with a 2-bit confidence counter.
class PcStride : public Predictor {
public:
    explicit PcStride(int log2_entries = 12) : mask_((1u << log2_entries) - 1)
    {
        tab_.assign(size_t(mask_) + 1, Entry{});
    }
    void reset() override { std::fill(tab_.begin(), tab_.end(), Entry{}); }
    Proposal onData(uint64_t line, int, uint64_t) override
    {
        Entry& e = tab_[size_t(mix64(last_pc_) & mask_)];
        Proposal p;
        if (e.valid) {
            int64_t d = int64_t(line) - int64_t(e.last_line);
            if (d == e.stride && d != 0) { if (e.conf < 3) ++e.conf; }
            else { e.stride = d; e.conf = 0; }
        } else { e.valid = 1; }
        e.last_line = line;
        if (e.conf >= 2 && e.stride != 0) {
            int64_t t = int64_t(line) + e.stride;
            if (t >= 0) { p.act = true; p.target = uint64_t(t); p.action = e.stride; }
        }
        return p;
    }
    std::string label() const override { return "pc-stride"; }
    size_t tableBytes() const override { return tab_.size() * sizeof(Entry); }
private:
    struct Entry { uint64_t last_line = 0; int64_t stride = 0; uint8_t conf = 0, valid = 0; };
    uint32_t mask_;
    std::vector<Entry> tab_;
};

// Delta-correlation: last delta -> most likely next delta (Markov, order 1).
class DeltaMarkov : public Predictor {
public:
    explicit DeltaMarkov(int log2_entries = 14, int min_conf = 2)
        : mask_((1u << log2_entries) - 1), min_conf_(min_conf)
    {
        tab_.assign(size_t(mask_) + 1, Entry{});
    }
    void reset() override { std::fill(tab_.begin(), tab_.end(), Entry{}); have_ = false; last_ = 0; prev_d_ = 0; }
    Proposal onData(uint64_t line, int, uint64_t) override
    {
        Proposal p;
        if (have_) {
            int64_t d = int64_t(line) - int64_t(last_);
            Entry& e = tab_[size_t(mix64(uint64_t(prev_d_)) & mask_)];
            if (e.next == d) { if (e.conf < 3) ++e.conf; }
            else if (e.conf == 0) { e.next = d; e.conf = 1; }
            else --e.conf;
            prev_d_ = d;
        }
        last_ = line; have_ = true;

        Entry& q = tab_[size_t(mix64(uint64_t(prev_d_)) & mask_)];
        if (q.conf >= min_conf_) {
            int64_t t = int64_t(line) + q.next;
            if (t >= 0) { p.act = true; p.target = uint64_t(t); p.action = q.next; }
        }
        return p;
    }
    std::string label() const override { return "delta-markov"; }
    size_t tableBytes() const override { return tab_.size() * sizeof(Entry); }
private:
    struct Entry { int64_t next = 0; uint8_t conf = 0; };
    uint32_t mask_;
    int min_conf_;
    std::vector<Entry> tab_;
    bool have_ = false;
    uint64_t last_ = 0;
    int64_t prev_d_ = 0;
};

// Global History Buffer, G/DC flavour: index by last delta into a circular
// buffer of deltas and replay whatever followed the previous occurrence.
class Ghb : public Predictor {
public:
    Ghb(int log2_index = 12, int log2_ghb = 14)
        : imask_((1u << log2_index) - 1), gsize_(size_t(1) << log2_ghb)
    {
        index_.assign(size_t(imask_) + 1, kNil);
        ghb_delta_.assign(gsize_, 0);
        ghb_prev_.assign(gsize_, kNil);
    }
    void reset() override
    {
        std::fill(index_.begin(), index_.end(), kNil);
        std::fill(ghb_prev_.begin(), ghb_prev_.end(), kNil);
        head_ = 0; have_ = false; last_ = 0;
    }
    Proposal onData(uint64_t line, int, uint64_t) override
    {
        Proposal p;
        if (!have_) { last_ = line; have_ = true; return p; }
        int64_t d = int64_t(line) - int64_t(last_);
        last_ = line;

        size_t slot = size_t(mix64(uint64_t(d)) & imask_);
        uint64_t prev = index_[slot];

        uint64_t me = head_;
        ghb_delta_[me % gsize_] = d;
        ghb_prev_[me % gsize_] = prev;
        index_[slot] = me;
        ++head_;

        // The delta that followed the previous occurrence of this delta.
        if (prev != kNil && head_ - prev < gsize_) {
            uint64_t nxt = prev + 1;
            if (nxt < head_ && head_ - nxt < gsize_) {
                int64_t pd = ghb_delta_[nxt % gsize_];
                int64_t t = int64_t(line) + pd;
                if (t >= 0 && pd != 0) { p.act = true; p.target = uint64_t(t); p.action = pd; }
            }
        }
        return p;
    }
    std::string label() const override { return "ghb-gdc"; }
    size_t tableBytes() const override { return index_.size() * 8 + gsize_ * 16; }
private:
    static constexpr uint64_t kNil = ~0ull;
    uint32_t imask_;
    size_t gsize_;
    std::vector<uint64_t> index_;
    std::vector<int64_t> ghb_delta_;
    std::vector<uint64_t> ghb_prev_;
    uint64_t head_ = 0;
    bool have_ = false;
    uint64_t last_ = 0;
};

// Signature/path based, in the spirit of SPP: a rolling hash of recent delta
// buckets indexes a small confidence-weighted delta table.
class SppLite : public Predictor {
public:
    explicit SppLite(int log2_sig = 14, float thresh = 0.35f)
        : mask_((1u << log2_sig) - 1), thresh_(thresh)
    {
        tab_.assign(size_t(mask_) + 1, Entry{});
    }
    void reset() override { std::fill(tab_.begin(), tab_.end(), Entry{}); sig_ = 0; have_ = false; last_ = 0; prev_sig_ = 0; }
    Proposal onData(uint64_t line, int, uint64_t) override
    {
        Proposal p;
        if (have_) {
            int64_t d = int64_t(line) - int64_t(last_);
            Entry& e = tab_[size_t(prev_sig_ & mask_)];
            if (e.delta == d) e.c_hit += 1.0f;
            else if (e.c_hit <= 1.0f) { e.delta = d; e.c_hit = 1.0f; }
            e.c_tot += 1.0f;
            if (e.c_tot > 64.0f) { e.c_tot *= 0.5f; e.c_hit *= 0.5f; }
            sig_ = ((sig_ << 3) ^ uint64_t(delta_bucket(d) + 16)) & mask_;
        }
        last_ = line; have_ = true;
        prev_sig_ = sig_;

        Entry& q = tab_[size_t(sig_ & mask_)];
        if (q.c_tot >= 4.0f && q.c_hit / q.c_tot > thresh_ && q.delta != 0) {
            int64_t t = int64_t(line) + q.delta;
            if (t >= 0) { p.act = true; p.target = uint64_t(t); p.action = q.delta; }
        }
        return p;
    }
    std::string label() const override { return "spp-lite"; }
    size_t tableBytes() const override { return tab_.size() * sizeof(Entry); }
private:
    struct Entry { int64_t delta = 0; float c_hit = 0, c_tot = 0; };
    uint32_t mask_;
    float thresh_;
    std::vector<Entry> tab_;
    uint64_t sig_ = 0, prev_sig_ = 0;
    bool have_ = false;
    uint64_t last_ = 0;
};

// Does nothing; used to measure the cost of the evaluation loop itself.
class NullPredictor : public Predictor {
public:
    void reset() override {}
    Proposal onData(uint64_t, int, uint64_t) override { return Proposal{}; }
    std::string label() const override { return "null"; }
};

}  // namespace pc
