// din2vtr — convert a valgrind/lackey memory trace (or a Dinero .din text
// trace) on stdin into the compact binary trace format used by the PrepChef
// lab.  Reads until EOF or until `max_refs` records have been written, then
// exits (which SIGPIPEs the tracer, so long runs can be capped cheaply).
//
//   lackey:  "I  0401f540,3"   instruction fetch
//            " L 1ffeffebf8,8" data load
//            " S 1ffeffebf8,8" data store
//            " M 1ffeffebf8,8" data modify  -> emitted as load then store
//   dinero:  "2 0401f540"      0=read 1=write 2=ifetch
//
// Record layout: uint64_t = (type << 61) | addr,  type 0=read 1=write 2=ifetch.

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>

static const char kMagic[8] = {'V','T','R','A','C','E','0','1'};

static inline uint64_t parse_hex(const char* p, const char** end)
{
    uint64_t v = 0;
    for (;; ++p) {
        unsigned c = (unsigned char)*p;
        unsigned d;
        if (c >= '0' && c <= '9')      d = c - '0';
        else if (c >= 'a' && c <= 'f') d = c - 'a' + 10;
        else if (c >= 'A' && c <= 'F') d = c - 'A' + 10;
        else break;
        v = (v << 4) | d;
    }
    *end = p;
    return v;
}

int main(int argc, char** argv)
{
    if (argc < 2) {
        std::fprintf(stderr, "usage: din2vtr <out.vtr> [max_refs]\n");
        return 2;
    }
    const uint64_t max_refs = (argc > 2) ? std::strtoull(argv[2], nullptr, 10) : UINT64_MAX;

    std::FILE* out = std::fopen(argv[1], "wb");
    if (!out) { std::perror("fopen"); return 1; }

    uint64_t header[3] = {0, 0, 0};
    std::memcpy(header, kMagic, 8);
    std::fwrite(header, sizeof(header), 1, out);      // patched on exit

    static uint64_t buf[1 << 16];
    size_t n_buf = 0;
    uint64_t n_out = 0, n_lines = 0, n_skipped = 0;

    char line[512];
    while (n_out < max_refs && std::fgets(line, sizeof(line), stdin)) {
        ++n_lines;
        int type = -1;
        const char* p = line;

        if (line[0] == 'I') { type = 2; p = line + 1; }
        else if (line[0] == ' ') {
            char c = line[1];
            if      (c == 'L') type = 0;
            else if (c == 'S') type = 1;
            else if (c == 'M') type = 3;              // modify: load + store
            else { ++n_skipped; continue; }
            p = line + 2;
        }
        else if (line[0] >= '0' && line[0] <= '2' && line[1] == ' ') {
            type = line[0] - '0';                     // raw dinero .din
            p = line + 1;
        }
        else { ++n_skipped; continue; }

        while (*p == ' ' || *p == '\t') ++p;
        const char* end;
        uint64_t addr = parse_hex(p, &end);
        if (end == p) { ++n_skipped; continue; }
        if (addr >> 61) { ++n_skipped; continue; }    // address must fit 61 bits

        if (type == 3) {
            buf[n_buf++] = (uint64_t(0) << 61) | addr;
            if (n_buf == (1 << 16)) { std::fwrite(buf, 8, n_buf, out); n_buf = 0; }
            ++n_out;
            type = 1;
        }
        buf[n_buf++] = (uint64_t(type) << 61) | addr;
        if (n_buf == (1 << 16)) { std::fwrite(buf, 8, n_buf, out); n_buf = 0; }
        ++n_out;
    }
    if (n_buf) std::fwrite(buf, 8, n_buf, out);

    header[1] = n_out;
    std::fseek(out, 0, SEEK_SET);
    std::fwrite(header, sizeof(header), 1, out);
    std::fclose(out);

    std::fprintf(stderr, "din2vtr: %llu refs written (%llu lines read, %llu skipped)\n",
                 (unsigned long long)n_out, (unsigned long long)n_lines,
                 (unsigned long long)n_skipped);
    return 0;
}
