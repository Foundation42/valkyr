// PrepChef experimental driver.
//
//   prepchef audit  <trace...>            Phase A: scorer audits + Gate A
//   prepchef base   <trace...>            PC-BASE headline numbers + price sweep
//   prepchef ctx    <trace...>            Phase B: context ablation
//   prepchef rep    <trace...>            Phase C: Bitty representation sweep
//   prepchef learn  <trace...>            Phase D: learner ablation
//   prepchef base2  <trace...>            Phase E: strong baselines
//   prepchef splits <trace...>            chronological splits + rolling windows
//   prepchef drift  <trace_a> <trace_b>   concept-drift / phase-change probe
//
// Every run appends one row to results/runs.csv.  Nothing is printed that is
// not also written there.

#include "common.hpp"
#include "context.hpp"
#include "learner.hpp"
#include "predictor.hpp"
#include "engine.hpp"
#include "scorer.hpp"

#include <cmath>
#include <cstdarg>
#include <functional>
#include <memory>

using namespace pc;

// ------------------------------------------------------------------- csv ----

static std::FILE* g_csv = nullptr;

static void csv_open(const char* path)
{
    bool exists = false;
    if (std::FILE* f = std::fopen(path, "rb")) { std::fseek(f, 0, SEEK_END); exists = std::ftell(f) > 0; std::fclose(f); }
    g_csv = std::fopen(path, "ab");
    if (!g_csv) { std::perror("csv"); std::exit(1); }
    if (!exists)
        std::fprintf(g_csv,
            "phase,trace,config,context,learner,window,waste,eval_begin,eval_end,"
            "accuracy,coverage,net_per_1k,strict_acc,strict_cov,strict_net_per_1k,"
            "mean_lead_data,mean_lead_total,issued,useful,useful_miss,scored_refs,scored_misses,"
            "action_rate,self_issued,self_useful,dedup_suppressed,"
            "state_bytes,table_bytes,update_ops,ns_per_ref,inserts,evictions,unresolved\n");
}

static void csv_row(const char* phase, const std::string& trace, const std::string& config,
                    const std::string& ctx, const std::string& learner,
                    const RunCfg& rc, const Metrics& m)
{
    std::fprintf(g_csv,
        "%s,%s,%s,%s,%s,%llu,%.3f,%.4f,%.4f,"
        "%.6f,%.6f,%.3f,%.6f,%.6f,%.3f,"
        "%.3f,%.1f,%llu,%llu,%llu,%llu,%llu,"
        "%.6f,%llu,%llu,%llu,"
        "%zu,%zu,%llu,%.2f,%llu,%llu,%llu\n",
        phase, trace.c_str(), config.c_str(), ctx.c_str(), learner.c_str(),
        (unsigned long long)rc.window, rc.waste, rc.eval_begin, rc.eval_end,
        m.accuracy(), m.coverage(), m.net_per_1k(rc.value, rc.waste),
        m.strict_accuracy(), m.strict_coverage(), m.strict_net_per_1k(rc.value, rc.waste),
        m.mean_lead_data(), m.mean_lead_total(),
        (unsigned long long)m.issued, (unsigned long long)m.useful, (unsigned long long)m.useful_miss,
        (unsigned long long)m.scored_refs, (unsigned long long)m.scored_misses,
        m.action_rate(), (unsigned long long)m.self_issued, (unsigned long long)m.self_useful,
        (unsigned long long)m.dedup_suppressed,
        m.state_bytes, m.table_bytes, (unsigned long long)m.update_ops, m.ns_per_ref,
        (unsigned long long)m.inserts, (unsigned long long)m.evictions,
        (unsigned long long)m.unresolved);
    std::fflush(g_csv);
}

static void report(const char* phase, const Trace& tr, const std::string& cfgname,
                   Predictor& p, const RunCfg& rc, const std::string& ctxl, const std::string& lrl)
{
    Engine eng;
    Metrics m = eng.run(tr, p, rc);
    csv_row(phase, tr.name, cfgname, ctxl, lrl, rc, m);
    std::printf("  %-26s acc %6.1f%%  cov %6.1f%%  net/1K %+8.1f  | strict acc %6.1f%%  cov %6.1f%%  net %+8.1f | lead %4.1fd/%6.0ft  act %5.1f%%  %6zuB  %5.1fns\n",
                cfgname.c_str(), 100 * m.accuracy(), 100 * m.coverage(), m.net_per_1k(rc.value, rc.waste),
                100 * m.strict_accuracy(), 100 * m.strict_coverage(), m.strict_net_per_1k(rc.value, rc.waste),
                m.mean_lead_data(), m.mean_lead_total(), 100 * m.action_rate(),
                m.state_bytes, m.ns_per_ref);
    std::fflush(stdout);
}

// ------------------------------------------------------------- pc-base ------

static CtxCfg pcbase_ctx()
{
    CtxCfg c;
    c.kind = CtxCfg::Kind::Fading;
    c.taus = {2, 8, 32, 128};
    c.heads = 8;
    c.quant = CtxCfg::Quant::Ternary;
    c.thresh = 0.20f;
    c.label = "bitty-4x8-ternary";
    return c;
}

static CtxCfg explicit4_ctx()
{
    CtxCfg c;
    c.kind = CtxCfg::Kind::ExplicitHist;
    c.hist_len = 4;
    c.label = "explicit-4";
    return c;
}

static LearnCfg pcbase_learn(float waste)
{
    LearnCfg l;
    l.kind = LearnCfg::Kind::Counts;
    l.gate = LearnCfg::Gate::ExpectedUtility;
    l.min_evidence = 6;
    l.waste = waste;
    l.label = "counts-eu";
    return l;
}

// =========================================================== Phase A ========

// A predictor that issues one prefetch for a chosen line at a chosen data index
// and nothing else; used to pin the expiry/credit ordering by hand.
class FixedIssue : public Predictor {
public:
    FixedIssue(uint64_t at, uint64_t line) : at_(at), line_(line) {}
    void reset() override {}
    Proposal onData(uint64_t, int, uint64_t di) override
    {
        Proposal p;
        if (di == at_) { p.act = true; p.target = line_; }
        return p;
    }
    std::string label() const override { return "fixed-issue"; }
private:
    uint64_t at_, line_;
};

// A third, deliberately naive scorer: for each issue, scan forward over the
// demand stream.  O(n*W) so it is only used on a prefix.
static void brute_force_score(const EventLog& log, uint64_t& issued, uint64_t& useful, double& lead_sum)
{
    issued = log.issue_line.size();
    useful = 0; lead_sum = 0;
    std::vector<char> consumed(log.demand_line.size(), 0);
    for (size_t e = 0; e < log.issue_line.size(); ++e) {
        uint64_t at = log.issue_index[e], L = log.issue_line[e];
        for (uint64_t k = at + 1; k <= at + log.window && k < log.demand_line.size(); ++k) {
            if (log.demand_line[k] == L && !consumed[k]) {
                consumed[k] = 1; ++useful; lead_sum += double(k - at);
                break;
            }
        }
    }
}

static int g_fail = 0;
static void check(bool ok, const char* name, const std::string& detail = "")
{
    std::printf("  [%s] %-46s %s\n", ok ? "PASS" : "FAIL", name, detail.c_str());
    if (!ok) ++g_fail;
}

static uint64_t ctx_id_at(const Trace& tr, const CtxCfg& cc, uint64_t target_data_index,
                          uint64_t corrupt_after_pos)
{
    ContextEngine ctx;
    ctx.configure(cc);
    uint64_t di = 0;
    for (uint64_t i = 0; i < tr.n; ++i) {
        uint64_t ad = tr.addr(i);
        if (corrupt_after_pos && i >= corrupt_after_pos) ad = (ad ^ mix64(i)) & ((1ull << 48) - 1);
        if (tr.type(i) == kIFetch) { ctx.observeInstr(ad); continue; }
        ctx.observeData(ad >> kLineShift, tr.type(i));
        if (di == target_data_index) return ctx.id();
        ++di;
    }
    return 0;
}

static void phase_a(const Trace& tr)
{
    std::printf("\n=== Phase A: freeze, reproduce, audit -- %s ===\n", tr.name.c_str());

    RunCfg rc;
    rc.log_events = true;
    EventLog log;
    PrepChef pchef(pcbase_ctx(), pcbase_learn(rc.waste));
    Engine eng;
    Metrics m = eng.run(tr, pchef, rc, &log);

    std::printf("  refs %llu (instr %llu, data %llu)  scored %llu  misses %llu\n",
                (unsigned long long)m.total_refs, (unsigned long long)m.instr_refs,
                (unsigned long long)m.data_refs, (unsigned long long)m.scored_refs,
                (unsigned long long)m.scored_misses);
    std::printf("  engine: issued %llu useful %llu lead %.3f\n",
                (unsigned long long)m.issued, (unsigned long long)m.useful, m.mean_lead_data());

    // --- Gate A: an independent scorer reproduces the counts ---------------
    IndepResult ir = score_independently(log);
    std::printf("  indep : issued %llu useful %llu lead %.3f\n",
                (unsigned long long)ir.issued, (unsigned long long)ir.useful,
                ir.useful ? ir.lead_sum / double(ir.useful) : 0.0);
    check(ir.issued == m.issued && ir.useful == m.useful, "Gate A: independent scorer matches engine");
    check(std::fabs(ir.lead_sum - m.lead_data_sum) < 1e-6, "Gate A: independent lead sum matches");
    check(ir.dup_outstanding_violations == 0, "duplicate outstanding fetches suppressed",
          "violations=" + std::to_string(ir.dup_outstanding_violations));

    // --- third implementation, on a prefix ---------------------------------
    {
        EventLog pre;
        pre.window = log.window;
        uint64_t cut = std::min<uint64_t>(200000, log.demand_line.size());
        pre.demand_line.assign(log.demand_line.begin(), log.demand_line.begin() + long(cut));
        for (size_t e = 0; e < log.issue_index.size(); ++e)
            if (log.issue_index[e] + log.window < cut) {
                pre.issue_index.push_back(log.issue_index[e]);
                pre.issue_line.push_back(log.issue_line[e]);
            }
        uint64_t bi, bu; double bl;
        brute_force_score(pre, bi, bu, bl);
        IndepResult pr = score_independently(pre);
        check(bu == pr.useful, "brute-force scorer matches independent scorer",
              "brute=" + std::to_string(bu) + " indep=" + std::to_string(pr.useful));
    }

    // --- no prefetch receives credit twice ---------------------------------
    check(m.useful <= m.issued && m.useful <= m.scored_refs, "no prefetch credited twice (bounds)");

    // --- no future information enters the context --------------------------
    {
        bool all_same = true;
        std::string detail;
        for (int s = 0; s < 8; ++s) {
            uint64_t k = uint64_t(double(m.data_refs) * (0.3 + 0.07 * s));
            // find the total-ref position of data index k, then corrupt after it
            uint64_t di = 0, pos = 0;
            for (uint64_t i = 0; i < tr.n; ++i) {
                if (tr.type(i) != kIFetch) { if (di == k) { pos = i; break; } ++di; }
            }
            uint64_t a = ctx_id_at(tr, pcbase_ctx(), k, 0);
            uint64_t b = ctx_id_at(tr, pcbase_ctx(), k, pos + 1);
            if (a != b) { all_same = false; detail = "at data idx " + std::to_string(k); break; }
        }
        check(all_same, "no future information enters the context", detail);
    }

    // --- scored metrics are unaffected by the far future -------------------
    {
        RunCfg a = rc; a.eval_end = 0.60;
        RunCfg b = a;
        // corrupt everything after 80% of the trace: strictly beyond the scored
        // region plus its usefulness window
        b.corrupt_after_pos = uint64_t(double(tr.n) * 0.80);
        PrepChef pa(pcbase_ctx(), pcbase_learn(rc.waste)), pb(pcbase_ctx(), pcbase_learn(rc.waste));
        Metrics ma = eng.run(tr, pa, a), mb = eng.run(tr, pb, b);
        check(ma.issued == mb.issued && ma.useful == mb.useful,
              "scored region invariant to post-hoc trace corruption",
              "a=" + std::to_string(ma.useful) + " b=" + std::to_string(mb.useful));
    }

    // --- warm-up cannot leak scored reward ---------------------------------
    {
        RunCfg w = rc; w.warmup_frac = 1.0; w.eval_begin = 1.0;
        PrepChef pw(pcbase_ctx(), pcbase_learn(rc.waste));
        Metrics mw = eng.run(tr, pw, w);
        check(mw.issued == 0 && mw.useful == 0, "warm-up issues nothing and earns nothing");
    }

    // --- instruction and data indices are not mixed ------------------------
    check(m.instr_refs + m.data_refs == m.total_refs, "instruction/data reference accounting closes");
    {
        int differ = 0;
        for (int j = 0; j < 8; ++j)
            if (signed_feature(0x1234, j, kInstrSalt) != signed_feature(0x1234, j, kDataSalt)) ++differ;
        check(differ > 0, "instruction and data feature namespaces are separated",
              "differing heads=" + std::to_string(differ));
    }

    // --- expiration / demand ordering is defined ---------------------------
    {
        // Synthetic: instruction, then data refs on distinct lines, with line
        // 0xF00 demanded at a controlled distance after a single issue.
        for (int variant = 0; variant < 2; ++variant) {
            const uint64_t W = 32, issue_at = 100;
            const uint64_t demand_at = issue_at + W + uint64_t(variant);   // exactly W, then W+1
            std::vector<uint64_t> recs;
            uint64_t hdr[3] = {0, 0, 0};
            std::memcpy(hdr, "VTRACE01", 8);
            for (uint64_t d = 0; d < 400; ++d) {
                recs.push_back((uint64_t(kIFetch) << 61) | (0x400000 + d * 4));
                uint64_t line = (d == demand_at) ? 0xF00 : (0x9000 + d);
                recs.push_back((uint64_t(kRead) << 61) | (line << kLineShift));
            }
            hdr[1] = recs.size();
            std::string path = "/tmp/pc_order_" + std::to_string(variant) + ".vtr";
            std::FILE* f = std::fopen(path.c_str(), "wb");
            std::fwrite(hdr, 24, 1, f); std::fwrite(recs.data(), 8, recs.size(), f); std::fclose(f);
            Trace st; st.open(path);
            RunCfg sc; sc.window = W; sc.warmup_frac = 0.0; sc.eval_begin = 0.0;
            FixedIssue fi(issue_at, 0xF00);
            Metrics sm = eng.run(st, fi, sc);
            bool want_useful = (variant == 0);
            check(sm.useful == (want_useful ? 1u : 0u),
                  variant == 0 ? "demand at exactly the window edge is useful"
                               : "demand one past the window edge is waste",
                  "useful=" + std::to_string(sm.useful));
            std::remove(path.c_str());
        }
    }

    std::printf("  audit result: %s\n", g_fail ? "FAILURES PRESENT" : "all checks passed");
}

// =========================================================== Phase B ========

static void phase_b(const Trace& tr)
{
    std::printf("\n=== Phase B: context ablation -- %s ===\n", tr.name.c_str());
    const float prices[] = {0.10f, 0.25f, 1.00f};

    std::vector<CtxCfg> cfgs;
    {
        CtxCfg c; c.kind = CtxCfg::Kind::TypeDelta; c.label = "type+delta"; cfgs.push_back(c);
    }
    {
        CtxCfg c; c.kind = CtxCfg::Kind::PcOnly; c.label = "pc-only"; cfgs.push_back(c);
    }
    for (int h : {1, 2, 4, 8, 16}) {
        CtxCfg c; c.kind = CtxCfg::Kind::ExplicitHist; c.hist_len = h;
        c.label = "explicit-" + std::to_string(h); cfgs.push_back(c);
    }
    {
        CtxCfg c = pcbase_ctx(); c.taus = {8}; c.label = "bitty-1x8-tau8"; cfgs.push_back(c);
        CtxCfg d = pcbase_ctx(); d.taus = {32}; d.label = "bitty-1x8-tau32"; cfgs.push_back(d);
    }
    cfgs.push_back(pcbase_ctx());
    {
        CtxCfg c = pcbase_ctx(); c.kind = CtxCfg::Kind::RandomMatched;
        c.random_cardinality = 1u << 20; c.label = "random-matched"; cfgs.push_back(c);
    }
    {
        CtxCfg c = pcbase_ctx(); c.shuffle_instr = true; c.label = "bitty-shuffled-instr"; cfgs.push_back(c);
    }
    {
        CtxCfg c = pcbase_ctx(); c.use_delta = false; c.label = "bitty-no-delta"; cfgs.push_back(c);
        CtxCfg d = pcbase_ctx(); d.use_type = false; d.label = "bitty-no-type"; cfgs.push_back(d);
    }
    {
        CtxCfg c = pcbase_ctx(); c.feed_instr = true;  c.feed_data = false; c.label = "bitty-instr-only"; cfgs.push_back(c);
        CtxCfg d = pcbase_ctx(); d.feed_instr = false; d.feed_data = true;  d.label = "bitty-data-only"; cfgs.push_back(d);
        CtxCfg e = pcbase_ctx(); e.feed_instr = true;  e.feed_data = true;  e.label = "bitty-instr+data"; cfgs.push_back(e);
    }

    for (float w : prices) {
        std::printf(" -- waste price %.2f\n", w);
        for (const auto& c : cfgs) {
            RunCfg rc; rc.waste = w;
            PrepChef p(c, pcbase_learn(w));
            report("B", tr, c.label, p, rc, c.label, "counts-eu");
        }
    }
}

// =========================================================== Phase C ========

static void phase_c(const Trace& tr)
{
    std::printf("\n=== Phase C: Bitty representation sweep -- %s ===\n", tr.name.c_str());
    RunCfg rc;

    const std::vector<std::vector<int>> tausets = {
        {1, 2, 4, 8}, {2, 8, 32, 128}, {4, 16, 64, 256},
        {2, 4, 8, 16, 32, 64}, {1, 4, 16, 64, 256, 1024},
    };
    for (const auto& ts : tausets) {
        CtxCfg c = pcbase_ctx(); c.taus = ts;
        // '|' rather than ',' so the label survives the CSV
        c.label = "tau{"; for (size_t i = 0; i < ts.size(); ++i) c.label += (i ? "|" : "") + std::to_string(ts[i]);
        c.label += "}";
        PrepChef p(c, pcbase_learn(rc.waste));
        report("C-tau", tr, c.label, p, rc, c.label, "counts-eu");
    }
    for (int h : {1, 2, 4, 8, 16, 32, 64}) {
        CtxCfg c = pcbase_ctx(); c.heads = h; c.label = "heads-" + std::to_string(h);
        PrepChef p(c, pcbase_learn(rc.waste));
        report("C-heads", tr, c.label, p, rc, c.label, "counts-eu");
    }
    struct Q { CtxCfg::Quant q; const char* n; };
    for (const Q& q : {Q{CtxCfg::Quant::Sign, "sign"}, Q{CtxCfg::Quant::Ternary, "ternary"},
                       Q{CtxCfg::Quant::Q2, "2-bit"}, Q{CtxCfg::Quant::Q3, "3-bit"},
                       Q{CtxCfg::Quant::Int8, "int8"}, Q{CtxCfg::Quant::Float, "float"}}) {
        CtxCfg c = pcbase_ctx(); c.quant = q.q; c.label = std::string("quant-") + q.n;
        PrepChef p(c, pcbase_learn(rc.waste));
        report("C-quant", tr, c.label, p, rc, c.label, "counts-eu");
    }
    for (float th : {0.05f, 0.10f, 0.20f, 0.35f, 0.50f}) {
        CtxCfg c = pcbase_ctx(); c.thresh = th;
        char b[64]; std::snprintf(b, sizeof b, "ternary-thresh-%.2f", th); c.label = b;
        PrepChef p(c, pcbase_learn(rc.waste));
        report("C-thresh", tr, c.label, p, rc, c.label, "counts-eu");
    }
    {
        CtxCfg c = pcbase_ctx(); c.integer_update = true; c.label = "integer-q8.8-shift";
        PrepChef p(c, pcbase_learn(rc.waste));
        report("C-int", tr, c.label, p, rc, c.label, "counts-eu");
    }
}

// =========================================================== Phase D ========

static void phase_d(const Trace& tr)
{
    std::printf("\n=== Phase D: learner ablation -- %s ===\n", tr.name.c_str());
    RunCfg rc;
    CtxCfg c = pcbase_ctx();

    std::vector<LearnCfg> ls;
    { LearnCfg l = pcbase_learn(rc.waste); ls.push_back(l); }
    { LearnCfg l = pcbase_learn(rc.waste); l.kind = LearnCfg::Kind::DecayCounts; l.decay = 0.99f;  l.label = "decay-0.99";  ls.push_back(l); }
    { LearnCfg l = pcbase_learn(rc.waste); l.kind = LearnCfg::Kind::DecayCounts; l.decay = 0.999f; l.label = "decay-0.999"; ls.push_back(l); }
    { LearnCfg l = pcbase_learn(rc.waste); l.kind = LearnCfg::Kind::Bandit; l.label = "bandit"; ls.push_back(l); }
    for (int k : {1, 2, 4, 8}) {
        LearnCfg l = pcbase_learn(rc.waste); l.max_actions = k;
        l.label = "topk-" + std::to_string(k); ls.push_back(l);
    }
    for (float t : {0.25f, 0.50f, 0.75f, 0.90f}) {
        LearnCfg l = pcbase_learn(rc.waste); l.gate = LearnCfg::Gate::FixedConf; l.conf_threshold = t;
        char b[64]; std::snprintf(b, sizeof b, "fixed-conf-%.2f", t); l.label = b; ls.push_back(l);
    }
    { LearnCfg l = pcbase_learn(rc.waste); l.gate = LearnCfg::Gate::AlwaysBest; l.label = "no-gate"; ls.push_back(l); }
    for (int e : {1, 2, 6, 16, 64}) {
        LearnCfg l = pcbase_learn(rc.waste); l.min_evidence = e;
        l.label = "min-evidence-" + std::to_string(e); ls.push_back(l);
    }
    { LearnCfg l = pcbase_learn(rc.waste); l.outcome_delta = false; l.label = "absolute-line-outcome"; ls.push_back(l); }
    for (int bits : {14, 16, 18, 20, 22}) {
        LearnCfg l = pcbase_learn(rc.waste); l.log2_slots = bits;
        l.label = "slots-2^" + std::to_string(bits); ls.push_back(l);
    }

    for (const auto& l : ls) {
        PrepChef p(c, l);
        report("D", tr, l.label, p, rc, c.label, l.label);
    }
}

// =========================================================== Phase E ========

static void phase_e(const Trace& tr)
{
    std::printf("\n=== Phase E: strong baselines at matched action budget -- %s ===\n", tr.name.c_str());
    RunCfg rc;

    std::vector<std::unique_ptr<Predictor>> bs;
    bs.emplace_back(new NextLine());
    bs.emplace_back(new LastStride());
    bs.emplace_back(new PcStride());
    bs.emplace_back(new DeltaMarkov());
    bs.emplace_back(new Ghb());
    bs.emplace_back(new SppLite());
    for (auto& b : bs) report("E", tr, b->label(), *b, rc, "-", b->label());

    // PrepChef across the price sweep: the frontier the baselines are compared
    // against, including points at matched action rates.
    for (float w : {0.05f, 0.10f, 0.25f, 0.50f, 1.00f, 2.00f, 4.00f}) {
        RunCfg r = rc; r.waste = w;
        { PrepChef p(pcbase_ctx(), pcbase_learn(w));
          char b[64]; std::snprintf(b, sizeof b, "prepchef-bitty-w%.2f", w);
          report("E", tr, b, p, r, "bitty-4x8-ternary", "counts-eu"); }
        { PrepChef p(explicit4_ctx(), pcbase_learn(w));
          char b[64]; std::snprintf(b, sizeof b, "prepchef-explicit4-w%.2f", w);
          report("E", tr, b, p, r, "explicit-4", "counts-eu"); }
    }
    // Registered variant: forbid the degenerate "prefetch the line I just
    // touched" action, which the crude protocol would otherwise reward.
    for (float w : {0.25f, 1.00f}) {
        RunCfg r = rc; r.waste = w; r.allow_self_prefetch = false;
        PrepChef p(pcbase_ctx(), pcbase_learn(w));
        char b[64]; std::snprintf(b, sizeof b, "prepchef-noself-w%.2f", w);
        report("E-noself", tr, b, p, r, "bitty-4x8-ternary", "counts-eu");
    }
}

// ================================================== splits / rolling =========

static void phase_splits(const Trace& tr)
{
    std::printf("\n=== Chronological splits and rolling windows -- %s ===\n", tr.name.c_str());
    for (double wu : {0.10, 0.20, 0.35, 0.50}) {
        RunCfg rc; rc.warmup_frac = wu; rc.eval_begin = wu;
        char b[64]; std::snprintf(b, sizeof b, "split-warmup-%.2f", wu);
        PrepChef p(pcbase_ctx(), pcbase_learn(rc.waste));
        report("A-split", tr, b, p, rc, "bitty-4x8-ternary", "counts-eu");
        PrepChef q(explicit4_ctx(), pcbase_learn(rc.waste));
        std::snprintf(b, sizeof b, "split-warmup-%.2f-explicit4", wu);
        report("A-split", tr, b, q, rc, "explicit-4", "counts-eu");
    }
    for (int k = 0; k < 8; ++k) {
        RunCfg rc; rc.warmup_frac = 0.20;
        rc.eval_begin = 0.20 + 0.10 * k;
        rc.eval_end = std::min(1.0, rc.eval_begin + 0.10);
        if (rc.eval_begin >= 1.0) break;
        char b[64]; std::snprintf(b, sizeof b, "roll-%.2f-%.2f", rc.eval_begin, rc.eval_end);
        PrepChef p(pcbase_ctx(), pcbase_learn(rc.waste));
        report("A-roll", tr, b, p, rc, "bitty-4x8-ternary", "counts-eu");
        PrepChef q(explicit4_ctx(), pcbase_learn(rc.waste));
        std::snprintf(b, sizeof b, "roll-%.2f-%.2f-explicit4", rc.eval_begin, rc.eval_end);
        report("A-roll", tr, b, q, rc, "explicit-4", "counts-eu");
    }
}

// =========================================================== Phase G-lite ===

// The handoff's third load-bearing rule is "charge real prices".  The window
// usefulness rule does not: it pays +1 for a prefetch of a line that was
// already sitting in L1.  Here the learner is told the *miss-filtered* reward
// instead -- +1 only when the prefetch saved a real miss, -waste when it was
// redundant -- and we ask whether the null gate then throttles itself.
static void phase_g(const Trace& tr)
{
    std::printf("\n=== Phase G-lite: economic gate under real prices -- %s ===\n", tr.name.c_str());
    CtxCfg c = pcbase_ctx();

    auto ev_learner = [](float waste) {
        LearnCfg l = pcbase_learn(waste);
        l.kind = LearnCfg::Kind::RealizedEV;
        l.label = "realized-ev";
        return l;
    };

    // Reference points under the crude window reward.
    { RunCfg rc; PrepChef p(c, pcbase_learn(rc.waste));
      report("G", tr, "counts-eu/window-reward", p, rc, c.label, "counts-eu"); }
    { RunCfg rc; PrepChef p(c, ev_learner(rc.waste));
      report("G", tr, "realized-ev/window-reward", p, rc, c.label, "realized-ev"); }

    // The same two gates, now paid the miss-filtered reward.  counts-eu cannot
    // see it (it gates on predicted p, not on realised reward), so it is the
    // registered control for "does the feedback channel matter".
    { RunCfg rc; rc.reward_mode = RunCfg::Reward::MissFiltered;
      PrepChef p(c, pcbase_learn(rc.waste));
      report("G", tr, "counts-eu/miss-reward", p, rc, c.label, "counts-eu"); }

    // Does the action rate move smoothly with the price of being wrong?
    for (float w : {0.05f, 0.10f, 0.25f, 0.50f, 1.00f, 2.00f, 4.00f, 8.00f}) {
        RunCfg rc; rc.waste = w; rc.reward_mode = RunCfg::Reward::MissFiltered;
        PrepChef p(c, ev_learner(w));
        char b[64]; std::snprintf(b, sizeof b, "realized-ev/miss-reward-w%.2f", w);
        report("G", tr, b, p, rc, c.label, "realized-ev");
    }
    // ... and under the window reward, for the same price sweep, so the two
    // curves can be read against each other.
    for (float w : {0.05f, 0.25f, 1.00f, 4.00f}) {
        RunCfg rc; rc.waste = w;
        PrepChef p(c, ev_learner(w));
        char b[64]; std::snprintf(b, sizeof b, "realized-ev/window-reward-w%.2f", w);
        report("G", tr, b, p, rc, c.label, "realized-ev");
    }
    // Baselines under the miss-filtered reward, for the same honest comparison.
    { RunCfg rc; rc.reward_mode = RunCfg::Reward::MissFiltered;
      NextLine nl;   report("G", tr, "next-line/miss-reward", nl, rc, "-", "next-line");
      PcStride ps;   report("G", tr, "pc-stride/miss-reward", ps, rc, "-", "pc-stride");
      Ghb gh;        report("G", tr, "ghb-gdc/miss-reward", gh, rc, "-", "ghb-gdc");
      SppLite sp;    report("G", tr, "spp-lite/miss-reward", sp, rc, "-", "spp-lite"); }
}

// =========================================================== Phase H ========

// The handoff asks how far ahead the primitive actually sees.  Mean useful lead
// is reported in data references *and* in total references (a proxy for
// instructions retired).  Here we also ask whether the primitive can be aimed
// further ahead simply by moving its label: learn the line demanded `horizon`
// data references later instead of the very next one.
static void phase_h(const Trace& tr)
{
    std::printf("\n=== Phase H: how far ahead can it see? -- %s ===\n", tr.name.c_str());
    for (int h : {1, 2, 4, 8, 16, 32}) {
        for (int mode = 0; mode < 2; ++mode) {
            RunCfg rc;
            if (mode) rc.reward_mode = RunCfg::Reward::MissFiltered;
            LearnCfg l = pcbase_learn(rc.waste);
            l.horizon = h;
            l.label = "horizon-" + std::to_string(h) + (mode ? "/miss-reward" : "/window-reward");
            PrepChef p(pcbase_ctx(), l);
            report("H", tr, l.label, p, rc, pcbase_ctx().label, l.label);
        }
    }
}

// ====================================== the combination worth carrying on ===

// Phases G-lite and H each fix one half of the problem: the realised-reward
// gate stops paying for redundant prefetches, and a longer label horizon buys
// lead time.  Neither was run with the other.  This is that combination,
// against next-line at its own (unchosen) action rate.
static void phase_best(const Trace& tr)
{
    std::printf("\n=== Combined: realised-reward gate x label horizon -- %s ===\n", tr.name.c_str());
    for (int h : {1, 2, 4, 8, 16}) {
        for (float w : {0.05f, 0.25f, 1.00f}) {
            RunCfg rc; rc.waste = w; rc.reward_mode = RunCfg::Reward::MissFiltered;
            LearnCfg l = pcbase_learn(w);
            l.kind = LearnCfg::Kind::RealizedEV;
            l.horizon = h;
            l.label = "ev-h" + std::to_string(h);
            char b[64]; std::snprintf(b, sizeof b, "ev-horizon-%d-w%.2f", h, w);
            PrepChef p(pcbase_ctx(), l);
            report("best", tr, b, p, rc, pcbase_ctx().label, l.label);
        }
    }
    // Same treatment for the explicit-history context, so the comparison
    // between representations survives into the configuration that matters.
    for (int h : {1, 4}) {
        RunCfg rc; rc.waste = 0.05f; rc.reward_mode = RunCfg::Reward::MissFiltered;
        LearnCfg l = pcbase_learn(0.05f);
        l.kind = LearnCfg::Kind::RealizedEV; l.horizon = h; l.label = "ev-h" + std::to_string(h);
        char b[64]; std::snprintf(b, sizeof b, "explicit4-ev-horizon-%d-w0.05", h);
        PrepChef p(explicit4_ctx(), l);
        report("best", tr, b, p, rc, "explicit-4", l.label);
    }
    { RunCfg rc; rc.reward_mode = RunCfg::Reward::MissFiltered;
      NextLine nl; report("best", tr, "next-line", nl, rc, "-", "next-line"); }
}

// ================================================= hot-path cost (Phase I) ==

// Hot-path cost, measured in a dedicated single-process run: every other phase
// may be run in parallel across traces, which inflates ns/ref.  The null
// predictor measures the evaluation loop itself, so the difference is what the
// primitive actually costs.
static void phase_cost(const Trace& tr)
{
    std::printf("\n=== Hot-path cost (single process) -- %s ===\n", tr.name.c_str());
    RunCfg rc;
    { NullPredictor np; report("cost", tr, "null-loop", np, rc, "-", "null"); }
    { NextLine nl;      report("cost", tr, "next-line", nl, rc, "-", "next-line"); }
    { PcStride ps;      report("cost", tr, "pc-stride", ps, rc, "-", "pc-stride"); }
    { Ghb gh;           report("cost", tr, "ghb-gdc", gh, rc, "-", "ghb-gdc"); }
    { PrepChef p(explicit4_ctx(), pcbase_learn(rc.waste));
      report("cost", tr, "prepchef-explicit4", p, rc, "explicit-4", "counts-eu"); }
    for (int h : {1, 2, 4, 8, 16, 32}) {
        CtxCfg c = pcbase_ctx(); c.heads = h;
        c.label = "bitty-4x" + std::to_string(h);
        PrepChef p(c, pcbase_learn(rc.waste));
        report("cost", tr, "prepchef-" + c.label, p, rc, c.label, "counts-eu");
    }
    { CtxCfg c = pcbase_ctx(); c.integer_update = true; c.label = "bitty-4x8-int";
      PrepChef p(c, pcbase_learn(rc.waste));
      report("cost", tr, "prepchef-bitty-4x8-int", p, rc, c.label, "counts-eu"); }
    // The leaky update runs on every instruction fetch, which is where the
    // hot-path cost lives.  Most consecutive fetches repeat a line.
    { CtxCfg c = pcbase_ctx(); c.distinct_instr_only = true; c.label = "bitty-4x8-distinct";
      PrepChef p(c, pcbase_learn(rc.waste));
      report("cost", tr, "prepchef-bitty-4x8-distinct", p, rc, c.label, "counts-eu"); }
    { CtxCfg c = pcbase_ctx(); c.distinct_instr_only = true; c.heads = 4;
      c.label = "bitty-4x4-distinct";
      PrepChef p(c, pcbase_learn(rc.waste));
      report("cost", tr, "prepchef-bitty-4x4-distinct", p, rc, c.label, "counts-eu"); }
    { CtxCfg c = pcbase_ctx(); c.distinct_instr_only = true; c.heads = 4; c.integer_update = true;
      c.label = "bitty-4x4-distinct-int";
      PrepChef p(c, pcbase_learn(rc.waste));
      report("cost", tr, "prepchef-bitty-4x4-distinct-int", p, rc, c.label, "counts-eu"); }
    for (int st : {2, 4, 8, 16}) {
        CtxCfg c = pcbase_ctx(); c.instr_stride = st;
        c.label = "bitty-4x8-every-" + std::to_string(st);
        PrepChef p(c, pcbase_learn(rc.waste));
        report("cost", tr, "prepchef-" + c.label, p, rc, c.label, "counts-eu");
    }
    // ... and the combination that Phases G-lite and H selected, costed.
    { LearnCfg l = pcbase_learn(0.05f); l.kind = LearnCfg::Kind::RealizedEV; l.horizon = 4;
      l.label = "ev-h4"; l.log2_slots = 16; l.max_actions = 4;
      CtxCfg c = pcbase_ctx(); c.distinct_instr_only = true; c.heads = 4; c.integer_update = true;
      c.label = "bitty-4x4-distinct-int";
      RunCfg r = rc; r.waste = 0.05f; r.reward_mode = RunCfg::Reward::MissFiltered;
      PrepChef p(c, l);
      report("cost", tr, "prepchef-selected", p, r, c.label, l.label); }
    for (int bits : {12, 14, 16, 18, 20, 22}) {
        LearnCfg l = pcbase_learn(rc.waste); l.log2_slots = bits; l.max_actions = 4;
        l.label = "bounded-2^" + std::to_string(bits) + "x4";
        PrepChef p(pcbase_ctx(), l);
        report("cost", tr, "prepchef-" + l.label, p, rc, "bitty-4x8-ternary", l.label);
    }
}

// ============================================ concept drift / phase change ==

// Run on a trace built by concatenating two different workloads: the seam is a
// hard phase change, and the question is which learner notices.
static void phase_drift(const Trace& tr)
{
    std::printf("\n=== Concept drift across a workload seam -- %s ===\n", tr.name.c_str());
    std::vector<LearnCfg> ls;
    { LearnCfg l = pcbase_learn(0.25f); ls.push_back(l); }
    { LearnCfg l = pcbase_learn(0.25f); l.kind = LearnCfg::Kind::DecayCounts; l.decay = 0.99f;  l.label = "decay-0.99";  ls.push_back(l); }
    { LearnCfg l = pcbase_learn(0.25f); l.kind = LearnCfg::Kind::DecayCounts; l.decay = 0.999f; l.label = "decay-0.999"; ls.push_back(l); }
    { LearnCfg l = pcbase_learn(0.25f); l.kind = LearnCfg::Kind::Bandit; l.label = "bandit"; ls.push_back(l); }
    { LearnCfg l = pcbase_learn(0.25f); l.max_actions = 4; l.label = "topk-4"; ls.push_back(l); }

    for (const auto& l : ls) {
        for (int k = 0; k < 18; ++k) {
            RunCfg rc; rc.warmup_frac = 0.05;
            rc.eval_begin = 0.10 + 0.05 * k;
            rc.eval_end = rc.eval_begin + 0.05;
            if (rc.eval_end > 1.0001) break;
            char b[96]; std::snprintf(b, sizeof b, "%s@%.2f", l.label.c_str(), rc.eval_begin);
            PrepChef p(pcbase_ctx(), l);
            report("drift", tr, b, p, rc, "bitty-4x8-ternary", l.label);
        }
    }
}

// ================================================================= main =====

int main(int argc, char** argv)
{
    if (argc < 3) {
        std::fprintf(stderr,
            "usage: prepchef <audit|base|ctx|rep|learn|base2|splits|econ|lead|best|cost|drift|all> <trace.vtr...>\n");
        return 2;
    }
    const std::string cmd = argv[1];
    const char* csv_path = std::getenv("PREPCHEF_CSV");
    csv_open(csv_path ? csv_path : "results/runs.csv");

    for (int a = 2; a < argc; ++a) {
        Trace tr;
        if (!tr.open(argv[a])) { std::fprintf(stderr, "cannot open %s\n", argv[a]); return 1; }
        std::printf("\n##### trace %s: %llu refs #####\n", tr.name.c_str(), (unsigned long long)tr.n);

        if (cmd == "audit" || cmd == "all") phase_a(tr);
        if (cmd == "base"  || cmd == "all") {
            std::printf("\n=== PC-BASE headline + price sweep -- %s ===\n", tr.name.c_str());
            for (float w : {0.05f, 0.10f, 0.25f, 0.50f, 1.00f, 2.00f}) {
                RunCfg rc; rc.waste = w;
                char b[64];
                { PrepChef p(pcbase_ctx(), pcbase_learn(w));
                  std::snprintf(b, sizeof b, "bitty-w%.2f", w);
                  report("base", tr, b, p, rc, "bitty-4x8-ternary", "counts-eu"); }
                { PrepChef p(explicit4_ctx(), pcbase_learn(w));
                  std::snprintf(b, sizeof b, "explicit4-w%.2f", w);
                  report("base", tr, b, p, rc, "explicit-4", "counts-eu"); }
                { NextLine nl; std::snprintf(b, sizeof b, "next-line-w%.2f", w);
                  report("base", tr, b, nl, rc, "-", "next-line"); }
                { LastStride ls; std::snprintf(b, sizeof b, "last-stride-w%.2f", w);
                  report("base", tr, b, ls, rc, "-", "last-stride"); }
            }
        }
        if (cmd == "splits" || cmd == "all") phase_splits(tr);
        if (cmd == "ctx"    || cmd == "all") phase_b(tr);
        if (cmd == "rep"    || cmd == "all") phase_c(tr);
        if (cmd == "learn"  || cmd == "all") phase_d(tr);
        if (cmd == "base2"  || cmd == "all") phase_e(tr);
        if (cmd == "drift") phase_drift(tr);
        if (cmd == "econ" || cmd == "all") phase_g(tr);
        if (cmd == "lead" || cmd == "all") phase_h(tr);
        if (cmd == "cost") phase_cost(tr);
        if (cmd == "best" || cmd == "all") phase_best(tr);
    }
    if (g_csv) std::fclose(g_csv);
    return g_fail ? 1 : 0;
}
