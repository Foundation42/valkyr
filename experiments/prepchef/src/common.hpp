// Shared primitives for the PrepChef lab: deterministic hashing, the binary
// trace reader, and the 64-byte-line conventions every experiment uses.
#pragma once

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace pc {

constexpr int kLineShift = 6;           // 64-byte lines throughout
constexpr uint64_t kLineSize = 1ull << kLineShift;

enum RefType : uint8_t { kRead = 0, kWrite = 1, kIFetch = 2 };

static inline uint64_t mix64(uint64_t z)
{
    z += 0x9E3779B97F4A7C15ull;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

static inline uint64_t hash_combine(uint64_t h, uint64_t v)
{
    return mix64(h ^ (v + 0x9E3779B97F4A7C15ull + (h << 6) + (h >> 2)));
}

// Signed +-1 features for a line address; `head` selects the feature index and
// `salt` separates namespaces (instruction lines vs data lines) so the two can
// never be silently mixed.
static inline float signed_feature(uint64_t line, int head, uint64_t salt)
{
    uint64_t h = mix64(line ^ salt ^ (uint64_t(head >> 6) * 0xD6E8FEB86659FD93ull));
    return ((h >> (head & 63)) & 1) ? 1.0f : -1.0f;
}

constexpr uint64_t kInstrSalt = 0x51ED270B4D1FE9A1ull;
constexpr uint64_t kDataSalt  = 0x2545F4914F6CDD1Dull;

// Signed log2 bucket of a line delta: 0, +-1, +-2, +-3.. up to +-13.
static inline int delta_bucket(int64_t d)
{
    if (d == 0) return 0;
    uint64_t a = uint64_t(d < 0 ? -d : d);
    int b = 0;
    while (a) { ++b; a >>= 1; }         // b = floor(log2|d|) + 1, >= 1
    if (b > 13) b = 13;
    return d < 0 ? -b : b;
}

// ------------------------------------------------------------------ trace ---

struct Trace {
    const uint64_t* rec = nullptr;
    uint64_t n = 0;
    void* map = nullptr;
    size_t map_len = 0;
    std::string name;

    bool open(const std::string& path)
    {
        int fd = ::open(path.c_str(), O_RDONLY);
        if (fd < 0) return false;
        struct stat st;
        if (fstat(fd, &st) != 0 || st.st_size < 24) { ::close(fd); return false; }
        map_len = size_t(st.st_size);
        map = mmap(nullptr, map_len, PROT_READ, MAP_PRIVATE, fd, 0);
        ::close(fd);
        if (map == MAP_FAILED) { map = nullptr; return false; }
        const uint64_t* h = (const uint64_t*)map;
        if (std::memcmp(h, "VTRACE01", 8) != 0) return false;
        n = h[1];
        rec = h + 3;
        uint64_t avail = (map_len - 24) / 8;
        if (n > avail) n = avail;
        size_t slash = path.find_last_of('/');
        name = path.substr(slash == std::string::npos ? 0 : slash + 1);
        if (name.size() > 4 && name.compare(name.size() - 4, 4, ".vtr") == 0)
            name.resize(name.size() - 4);
        madvise(map, map_len, MADV_SEQUENTIAL);
        return true;
    }
    ~Trace() { if (map) munmap(map, map_len); }

    inline uint8_t  type(uint64_t i) const { return uint8_t(rec[i] >> 61); }
    inline uint64_t addr(uint64_t i) const { return rec[i] & ((1ull << 61) - 1); }
};

// --------------------------------------------------------------- L1 model ---
// A plain LRU cache over 64-byte lines.  Used only as a *counterfactual*: it
// sees demand references and nothing else, so "this demand would have missed"
// is well defined independently of what the prefetcher did.
class LruCache {
public:
    void configure(size_t sets, int ways)
    {
        sets_ = sets; ways_ = ways;
        tag_.assign(sets * ways, ~0ull);
        age_.assign(sets * ways, 0);
        clock_ = 0;
    }
    void reset() { std::fill(tag_.begin(), tag_.end(), ~0ull); std::fill(age_.begin(), age_.end(), 0); clock_ = 0; }

    // Returns true on hit.  Always installs the line (demand fill).
    bool access(uint64_t line)
    {
        size_t s = size_t(mix64(line) & (sets_ - 1));
        uint64_t* t = &tag_[s * ways_];
        uint64_t* a = &age_[s * ways_];
        ++clock_;
        for (int w = 0; w < ways_; ++w)
            if (t[w] == line) { a[w] = clock_; return true; }
        int victim = 0;
        uint64_t oldest = ~0ull;
        for (int w = 0; w < ways_; ++w)
            if (t[w] == ~0ull) { victim = w; break; }
            else if (a[w] < oldest) { oldest = a[w]; victim = w; }
        t[victim] = line; a[victim] = clock_;
        return false;
    }

private:
    size_t sets_ = 64;
    int ways_ = 8;
    std::vector<uint64_t> tag_, age_;
    uint64_t clock_ = 0;
};

}  // namespace pc
