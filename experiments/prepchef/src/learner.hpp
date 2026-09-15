// Association learner + economic null gate.
//
// The estimator is deliberately not the primitive: every variant here answers
// "given this context, what preparation (if any) pays for itself?" and they
// are swapped without touching the context or the scorer.
#pragma once

#include "common.hpp"

#include <cmath>

namespace pc {

struct LearnCfg {
    enum class Kind {
        Counts,          // cumulative counts (PC-BASE)
        DecayCounts,     // exponentially decayed counts
        LeakyEvidence,   // leaky per-action evidence, no global total
        TopK,            // fixed-capacity top-k outcomes
        Bandit,          // tiny contextual value estimate, updated by realised reward
        RealizedEV,      // counts pick the action; a running mean of *realised*
                         // reward decides whether acting here pays at all
    };
    enum class Gate {
        ExpectedUtility, // p*value - (1-p)*waste > 0     (PC-BASE)
        FixedConf,       // p > threshold
        AlwaysBest,      // no gate at all (control)
    };

    Kind kind = Kind::Counts;
    Gate gate = Gate::ExpectedUtility;
    int min_evidence = 6;
    float decay = 0.995f;
    float conf_threshold = 0.5f;
    float value = 1.0f;
    float waste = 0.25f;
    float lr = 0.05f;               // Bandit / RealizedEV
    float ev_init = 0.05f;          // optimistic start, so every context is tried
    int max_actions = 8;            // per-context outcome slots
    int log2_slots = 22;            // context table capacity
    int probe = 8;                  // linear probe depth before LFU replacement
    bool outcome_delta = true;      // action = line delta (vs absolute line)
    int horizon = 1;                // learn context -> the line demanded this
                                    // many data references later (Phase H)
    std::string label = "counts";
};

// Flat, fixed-capacity, open-addressed context table.  Nothing is allocated on
// the hot path; a full probe window evicts the least-supported entry and the
// eviction is counted so a contaminated run is visible rather than silent.
class Learner {
public:
    void configure(const LearnCfg& c)
    {
        cfg_ = c;
        slots_ = size_t(1) << cfg_.log2_slots;
        mask_ = slots_ - 1;
        key_.assign(slots_, 0);
        total_.assign(slots_, 0.0f);
        nact_.assign(slots_, 0);
        act_.assign(slots_ * size_t(cfg_.max_actions), 0);
        cnt_.assign(slots_ * size_t(cfg_.max_actions), 0.0f);
        ev_.assign(slots_, cfg_.ev_init);
        reset();
    }

    void reset()
    {
        std::fill(key_.begin(), key_.end(), 0ull);
        std::fill(total_.begin(), total_.end(), 0.0f);
        std::fill(nact_.begin(), nact_.end(), uint8_t(0));
        std::fill(ev_.begin(), ev_.end(), cfg_.ev_init);
        evictions_ = 0; inserts_ = 0;
    }

    // Record that `action` was the right preparation for `ctx`.
    void learn(uint64_t ctx, int64_t action)
    {
        size_t s = find(ctx, /*create=*/true);
        if (s == kNone) return;
        if (cfg_.kind == LearnCfg::Kind::DecayCounts || cfg_.kind == LearnCfg::Kind::LeakyEvidence) {
            total_[s] *= cfg_.decay;
            float* c = &cnt_[s * cfg_.max_actions];
            for (int i = 0; i < nact_[s]; ++i) c[i] *= cfg_.decay;
        }
        if (cfg_.kind == LearnCfg::Kind::Bandit) {
            // Bandit arms are created on observation but valued by realised
            // reward only (see reward()); counts merely track support.
        }
        bump(s, action, 1.0f);
        total_[s] += 1.0f;
    }

    // Realised outcome of a completed speculation (Bandit only).
    void reward(uint64_t ctx, int64_t action, float r)
    {
        if (cfg_.kind == LearnCfg::Kind::RealizedEV) {
            size_t s = find(ctx, /*create=*/false);
            if (s != kNone) ev_[s] += cfg_.lr * (r - ev_[s]);
            return;
        }
        if (cfg_.kind != LearnCfg::Kind::Bandit) return;
        size_t s = find(ctx, /*create=*/false);
        if (s == kNone) return;
        int64_t* a = &act_[s * cfg_.max_actions];
        float* q = &qval_[s * cfg_.max_actions];
        for (int i = 0; i < nact_[s]; ++i)
            if (a[i] == action) { q[i] += cfg_.lr * (r - q[i]); return; }
    }

    struct Proposal { bool act = false; int64_t action = 0; float p = 0.0f; float utility = 0.0f; };

    Proposal propose(uint64_t ctx) const
    {
        Proposal out;
        size_t s = find_const(ctx);
        if (s == kNone) return out;
        if (total_[s] < float(cfg_.min_evidence)) return out;

        const int64_t* a = &act_[s * cfg_.max_actions];
        const float* c = &cnt_[s * cfg_.max_actions];
        int best = -1;
        float best_c = -1.0f;
        for (int i = 0; i < nact_[s]; ++i)
            if (c[i] > best_c) { best_c = c[i]; best = i; }
        if (best < 0) return out;

        float p = (best_c + 1.0f) / (total_[s] + 2.0f);       // Laplace
        out.p = p;
        out.action = a[best];

        if (cfg_.kind == LearnCfg::Kind::Bandit) {
            const float* q = &qval_[s * cfg_.max_actions];
            int bq = -1; float bqv = 0.0f;                     // null arm has value 0
            for (int i = 0; i < nact_[s]; ++i)
                if (q[i] > bqv) { bqv = q[i]; bq = i; }
            if (bq < 0) {                                      // no arm beats doing nothing yet
                // optimistic start: untried arms use the count estimate once
                float u = p * cfg_.value - (1.0f - p) * cfg_.waste;
                if (u > 0.0f && q[best] == 0.0f) { out.act = true; out.utility = u; }
                return out;
            }
            out.action = a[bq];
            out.utility = bqv;
            out.act = true;
            return out;
        }

        if (cfg_.kind == LearnCfg::Kind::RealizedEV) {
            out.utility = ev_[s];
            out.act = ev_[s] > 0.0f;
            return out;
        }

        float u = p * cfg_.value - (1.0f - p) * cfg_.waste;
        out.utility = u;
        switch (cfg_.gate) {
        case LearnCfg::Gate::ExpectedUtility: out.act = u > 0.0f; break;
        case LearnCfg::Gate::FixedConf:       out.act = p > cfg_.conf_threshold; break;
        case LearnCfg::Gate::AlwaysBest:      out.act = true; break;
        }
        return out;
    }

    size_t tableBytes() const
    {
        return slots_ * (8 + 4 + 1 + size_t(cfg_.max_actions) * (8 + 4)
                         + (cfg_.kind == LearnCfg::Kind::Bandit ? size_t(cfg_.max_actions) * 4 : 0)
                         + (cfg_.kind == LearnCfg::Kind::RealizedEV ? 4 : 0));
    }
    uint64_t evictions() const { return evictions_; }
    uint64_t inserts() const { return inserts_; }
    const LearnCfg& cfg() const { return cfg_; }

    void ensureBandit()
    {
        if (cfg_.kind == LearnCfg::Kind::Bandit && qval_.size() != act_.size())
            qval_.assign(act_.size(), 0.0f);
    }

private:
    static constexpr size_t kNone = ~size_t(0);

    size_t find(uint64_t ctx, bool create)
    {
        size_t h = size_t(mix64(ctx)) & mask_;
        size_t weakest = kNone;
        float weakest_total = 1e30f;
        for (int i = 0; i < cfg_.probe; ++i) {
            size_t s = (h + size_t(i)) & mask_;
            if (key_[s] == ctx) return s;
            if (key_[s] == 0) {
                if (!create) return kNone;
                key_[s] = ctx; total_[s] = 0.0f; nact_[s] = 0; ev_[s] = cfg_.ev_init;
                ++inserts_;
                return s;
            }
            if (total_[s] < weakest_total) { weakest_total = total_[s]; weakest = s; }
        }
        if (!create) return kNone;
        key_[weakest] = ctx; total_[weakest] = 0.0f; nact_[weakest] = 0; ev_[weakest] = cfg_.ev_init;
        ++evictions_;
        return weakest;
    }

    size_t find_const(uint64_t ctx) const
    {
        size_t h = size_t(mix64(ctx)) & mask_;
        for (int i = 0; i < cfg_.probe; ++i) {
            size_t s = (h + size_t(i)) & mask_;
            if (key_[s] == ctx) return s;
            if (key_[s] == 0) return kNone;
        }
        return kNone;
    }

    void bump(size_t s, int64_t action, float w)
    {
        int64_t* a = &act_[s * cfg_.max_actions];
        float* c = &cnt_[s * cfg_.max_actions];
        for (int i = 0; i < nact_[s]; ++i)
            if (a[i] == action) { c[i] += w; return; }
        if (nact_[s] < cfg_.max_actions) {
            int i = nact_[s]++;
            a[i] = action; c[i] = w;
            if (!qval_.empty()) qval_[s * cfg_.max_actions + i] = 0.0f;
            return;
        }
        // Slots full: replace the weakest outcome (top-k behaviour).
        int worst = 0;
        for (int i = 1; i < nact_[s]; ++i) if (c[i] < c[worst]) worst = i;
        if (c[worst] <= w) {
            a[worst] = action; c[worst] = w;
            if (!qval_.empty()) qval_[s * cfg_.max_actions + worst] = 0.0f;
        }
    }

    LearnCfg cfg_;
    size_t slots_ = 0, mask_ = 0;
    std::vector<uint64_t> key_;
    std::vector<float> total_;
    std::vector<uint8_t> nact_;
    std::vector<int64_t> act_;
    std::vector<float> cnt_;
    std::vector<float> qval_;
    std::vector<float> ev_;
    uint64_t evictions_ = 0, inserts_ = 0;
};

}  // namespace pc
