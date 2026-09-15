// An independent scorer.
//
// This deliberately shares no code with Engine.  It consumes only the event log
// -- the demanded line at each scored data index and the (index, line) of each
// issued prefetch -- and re-derives the headline counts by a different
// algorithm: build a per-line index of demand positions, then binary-search the
// first unconsumed demand inside the window for each issue, in issue order.
//
// Gate A is "an independent scorer reproduces the effect", so agreement between
// this and Engine is a precondition for reading any number in this lab.
#pragma once

#include "engine.hpp"

#include <algorithm>
#include <map>

namespace pc {

struct IndepResult {
    uint64_t issued = 0, useful = 0;
    double lead_sum = 0;
    uint64_t dup_outstanding_violations = 0;
    bool ok = true;
    std::string note;
};

inline IndepResult score_independently(const EventLog& log)
{
    IndepResult r;
    r.issued = log.issue_line.size();

    // line -> sorted demand positions
    std::map<uint64_t, std::vector<uint64_t>> pos;
    for (uint64_t k = 0; k < log.demand_line.size(); ++k)
        pos[log.demand_line[k]].push_back(k);

    // Consumption cursor per line: a demand may pay for at most one prefetch.
    std::map<uint64_t, size_t> cursor;
    // Independent check of the "one outstanding fetch per line" invariant:
    // the previous issue of a line must have been resolved (credited within the
    // window, or expired) before the next issue of that line.
    std::map<uint64_t, uint64_t> last_issue;

    for (size_t e = 0; e < log.issue_line.size(); ++e) {
        const uint64_t L = log.issue_line[e];
        const uint64_t at = log.issue_index[e];

        auto li = last_issue.find(L);
        if (li != last_issue.end()) {
            // Resolution time of the previous issue: first demand of L strictly
            // after it, or expiry, whichever came first.
            uint64_t prev = li->second;
            uint64_t resolve = prev + log.window;      // expiry
            auto& v = pos[L];
            auto it = std::upper_bound(v.begin(), v.end(), prev);
            if (it != v.end() && *it <= prev + log.window) resolve = *it;
            if (at <= resolve && at != prev) {
                // An issue landing on the very demand that resolves the previous
                // one is legal; anything strictly inside the window is not.
                if (at < resolve) ++r.dup_outstanding_violations;
            }
        }
        last_issue[L] = at;

        auto pit = pos.find(L);
        if (pit == pos.end()) continue;
        auto& v = pit->second;
        size_t& cur = cursor[L];
        // first demand strictly after `at`, not already consumed
        size_t lo = std::lower_bound(v.begin() + long(cur), v.end(), at + 1) - v.begin();
        if (lo >= v.size()) continue;
        if (v[lo] - at <= log.window) {
            ++r.useful;
            r.lead_sum += double(v[lo] - at);
            cur = lo + 1;
        }
    }
    return r;
}

}  // namespace pc
