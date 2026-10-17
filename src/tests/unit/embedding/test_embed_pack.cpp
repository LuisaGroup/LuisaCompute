// test_embed_pack.cpp -- model.bgm packer/package byte-level integrity round-trip
// (embedding_model_test_plan.md §2 L0 row "test_embed_pack": "tensor checksums vs
// golden manifest; round-trip: pack -> load -> checksum equality", plus the §4.5
// corrupt-fixture rejection parity). Host-only test (LUISA_EMBED_HOST_ONLY=1):
// it creates no LuisaCompute device and takes no backend argument.
//
// Coverage
// (1) REAL packed model (default D:/compute/bge-m3/bge-m3/model.bgm, argv[1]
//     override): every one of the 296 tensor blobs is hashed with FNV-1a-64 over
//     its bytes, in package emission order. The 296 per-tensor checksums are
//     aggregated into ONE committed constant, defined as
//         aggregate = FNV-1a-64( concat_i u64le(per_tensor_checksum_i) )
//     i.e. the per-tensor checksums written little-endian back-to-back in table
//     order, hashed starting from the FNV offset basis. Seven pinned per-tensor
//     checksums (emb.word, emb.pos, layer0.qkv, layer12.ffn.down,
//     layer23.attn.out, head.sparse.w, head.colbert.w) guard individual regions
//     so a corrupted constant is localizable rather than only "the aggregate
//     changed". Size/dims coverage is deliberately NOT repeated here --
//     test_embed_package owns it; this test is about byte-level integrity.
//     The same checksums are then recomputed by an independent streaming pass
//     over the raw file (in-test minimal v1 table reader, no engine code) and
//     required to match what embed::Package::load served, tensor by tensor: that
//     is the pack -> load -> checksum equality gate.
// (2) synthetic .bgm fixtures written in-test per the format spec (the header
//     comment of examples/embedding/tools/pack_model.cpp IS the spec: magic
//     "BGM3", version 1, precision 0, 48-byte entries {fnv1a64(name), offset,
//     size_bytes, rank, dims[4]}, 256-byte aligned blobs, zero padding) and
//     parsed back: header fields, name hashes, rank/dims, offsets and alignment,
//     the file-size formula, padding bytes, payload bytes and the checksum
//     aggregation all round-trip. Three fixtures (1 / 3 / 4 tensors) cover
//     rank-1, rank-2, the 4-byte blob (maximum padding) and the exact-256-byte
//     blob (zero padding).
// (3) corrupt fixtures through embed::Package::load in a SPAWNED CHILD process
//     (the rejection path is LUISA_ERROR -> std::abort, which is noreturn, so it
//     cannot be observed in-process): bad magic, unsupported version,
//     precision=1 (bf16, reserved for plan M7), a wrong tensor_count and a
//     truncated table must each abort loudly with their own diagnostic; a
//     control child loading the real package must exit 0. Child exit codes and
//     log substrings are both checked, and the codes are printed as evidence.
//
// DEVIATION (documented): the task brief suggested loading the tiny synthetic
// fixtures of (2) through embed::Package::load. That is impossible with the
// implemented loader, by design: package.cpp is fail-closed against the bge-m3 v1
// package -- tensor_count must be 296, every entry must carry its canonical
// name_hash and the dims derived from the architecture constants, size_bytes must
// equal prod(dims)*4 and the blobs must be exactly packed at 256-aligned offsets
// inside the file. Any file the loader accepts therefore carries the real ~2.27
// GB of payloads -- precisely the "too slow for a unit test" case the brief
// excludes. Hence (2) round-trips the fixtures against an in-test minimal v1
// reader (an independent transcription of the format spec, and the same writer/
// reader pair (3) uses to build the corrupt fixtures), while the pack -> load ->
// checksum equality against genuine packer output runs through Package::load in
// (1).
//
// Constant derivation (frozen; never regenerate goldens to pass -- plan §8 (a)):
//   python examples/embedding/tools/pack_model_dev.py dump <model.bgm> <name> <out.bin>
//   followed by a stdlib-only FNV-1a-64 pass over the dumped blobs, cross-checked
//   against a second independent whole-package reader; both agree with this C++
//   test. Derived 2026-10-17 against model.bgm = 2271033600 bytes, 296 tensors.
//
// Run: xmake run test_embed_pack [D:/compute/bge-m3/bge-m3/model.bgm]
#include "ut/ut.hpp"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <initializer_list>
#include <system_error>

#include <luisa/core/logging.h>
#include <luisa/core/stl/filesystem.h>
#include <luisa/core/stl/format.h>
#include <luisa/core/stl/string.h>
#include <luisa/core/stl/vector.h>

#include "embedding/embed/package.h"

using namespace luisa;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {
// ---------------------------------------------------------------------
// FNV-1a-64 exactly as the format spec defines it (the same constants the
// packer uses for the name_hash column, re-transcribed here independently)
// ---------------------------------------------------------------------
constexpr uint64_t kFnvBasis = 0xcbf29ce484222325ull;
constexpr uint64_t kFnvPrime = 0x00000100000001b3ull;
constexpr uint64_t kBgmAlign = 256ull;
constexpr uint64_t kBgmHeaderBytes = 16ull;
constexpr uint64_t kBgmEntryBytes = 48ull;

uint64_t fnv1a64(uint64_t h, const uint8_t *p, size_t n) noexcept {
    for (size_t i = 0u; i < n; i++) {
        h ^= p[i];
        h *= kFnvPrime;
    }
    return h;
}
uint64_t fnv1a64_bytes(const void *p, size_t n) noexcept {
    return fnv1a64(kFnvBasis, static_cast<const uint8_t *>(p), n);
}
uint64_t fnv1a64_name(luisa::string_view s) noexcept {
    return fnv1a64_bytes(s.data(), s.size());
}
// feed the 8 little-endian bytes of v into a running FNV state
uint64_t fnv1a64_mix_u64(uint64_t h, uint64_t v) noexcept {
    uint8_t b[8];
    for (int i = 0; i < 8; i++) {
        b[i] = static_cast<uint8_t>(v >> (8u * static_cast<size_t>(i)));
    }
    return fnv1a64(h, b, 8u);
}
uint64_t align_up(uint64_t v, uint64_t a) noexcept { return (v + a - 1ull) / a * a; }

// ---------------------------------------------------------------------
// committed expectations (derivation: see the file header)
// ---------------------------------------------------------------------
constexpr uint32_t kExpectedTensorCount = 296u;
// FNV-1a-64 over the 296 per-tensor checksums, LE u64, in emission order
constexpr uint64_t kExpectedAggregateChecksum = 0xccb886ac364d5764ull;
struct PinnedChecksum {
    luisa::string_view name;
    uint64_t checksum;
};
constexpr PinnedChecksum kPinnedChecksums[] = {
    {"emb.word", 0x9e683ddd2cddf3c3ull},
    {"emb.pos", 0xc97fd00352b82b9bull},
    {"layer0.qkv", 0xb6602fe02911f20bull},
    {"layer12.ffn.down", 0xd7d2311e861f9260ull},
    {"layer23.attn.out", 0xe8c67926e569a1e3ull},
    {"head.sparse.w", 0xd9f8a6c91b91dd6dull},
    {"head.colbert.w", 0xb06ee6c57f53f029ull},
};

// ---------------------------------------------------------------------
// io helpers
// ---------------------------------------------------------------------
luisa::vector<uint8_t> read_whole_file(const std::filesystem::path &p) {
    luisa::vector<uint8_t> out;
    std::error_code ec;
    if (!std::filesystem::exists(p, ec) || ec) return out;
    if (auto *f = std::fopen(p.string().c_str(), "rb"); f != nullptr) {
        uint8_t buf[64u << 10u];// keep the stack footprint modest
        size_t got = 0u;
        while ((got = std::fread(buf, 1u, sizeof(buf), f)) > 0u) {
            out.insert(out.end(), buf, buf + static_cast<ptrdiff_t>(got));
        }
        std::fclose(f);
    }
    return out;
}
bool write_whole_file(const std::filesystem::path &p, const luisa::vector<uint8_t> &bytes) {
    if (auto *f = std::fopen(p.string().c_str(), "wb"); f != nullptr) {
        if (!bytes.empty() && std::fwrite(bytes.data(), 1u, bytes.size(), f) != bytes.size()) {
            std::fclose(f);
            return false;
        }
        std::fclose(f);
        return true;
    }
    return false;
}
luisa::string read_text_file(const std::filesystem::path &p) {
    auto b = read_whole_file(p);
    return luisa::string{luisa::string_view{reinterpret_cast<const char *>(b.data()), b.size()}};
}
// 64-bit-safe seek: plain fseek takes a 32-bit long on Windows and the model
// blobs live far past 2 GiB
void seek64(std::FILE *f, uint64_t off) {
#ifdef _WIN32
    if (_fseeki64(f, static_cast<long long>(off), SEEK_SET) != 0) {
#else
    if (std::fseek(f, static_cast<long>(off), SEEK_SET) != 0) {
#endif
        LUISA_ERROR("seek to {} failed", off);
    }
}
void append_le32(luisa::vector<uint8_t> &img, uint32_t v) {
    for (int i = 0; i < 4; i++) {
        img.push_back(static_cast<uint8_t>(v >> (8u * static_cast<size_t>(i))));
    }
}
void append_le64(luisa::vector<uint8_t> &img, uint64_t v) {
    for (int i = 0; i < 8; i++) {
        img.push_back(static_cast<uint8_t>(v >> (8u * static_cast<size_t>(i))));
    }
}
uint32_t read_le32_at(const luisa::vector<uint8_t> &img, size_t at) {
    return static_cast<uint32_t>(img[at]) | (static_cast<uint32_t>(img[at + 1u]) << 8u) |
           (static_cast<uint32_t>(img[at + 2u]) << 16u) | (static_cast<uint32_t>(img[at + 3u]) << 24u);
}
uint64_t read_le64_at(const luisa::vector<uint8_t> &img, size_t at) {
    return static_cast<uint64_t>(read_le32_at(img, at)) |
           (static_cast<uint64_t>(read_le32_at(img, at + 4u)) << 32u);
}

// ---------------------------------------------------------------------
// minimal in-test BGM3 v1 table representation + writer/reader (independent
// transcriptions of the format spec; structural problems are returned as a
// message and never fatal -- unlike the engine loader, this must not abort
// the test process)
// ---------------------------------------------------------------------
struct FixtureTensor {
    luisa::string name;
    uint32_t rank;
    uint32_t dims[4];
    luisa::vector<float> values;
};
struct RawTable {
    luisa::string magic{};
    uint32_t version{};
    uint32_t precision{};
    uint32_t count{};
    luisa::vector<uint64_t> hashes, offsets, sizes;
    luisa::vector<uint32_t> ranks, dims;// 4 per entry
    // the bytes this table was parsed from: the whole image for in-test
    // fixtures, header + table only for the streamed real file
    luisa::vector<uint8_t> image{};

    [[nodiscard]] uint64_t entry_elems(uint32_t i) const noexcept {
        uint64_t n = 1ull;
        for (uint32_t d = 0u; d < ranks[i]; d++) {
            n *= dims[static_cast<size_t>(i) * 4u + d];
        }
        return n;
    }
    [[nodiscard]] uint64_t first_blob() const noexcept {
        return align_up(kBgmHeaderBytes + kBgmEntryBytes * count, kBgmAlign);
    }
};
uint64_t fixture_elems(const FixtureTensor &t) {
    uint64_t n = 1ull;
    for (uint32_t i = 0u; i < t.rank; i++) n *= t.dims[i];
    return n;
}
// Builds a spec-conformant image unless a field is deliberately corrupted
// (magic / version / precision / tensor_count) or the image is cut short.
luisa::vector<uint8_t> build_bgm_image(const luisa::vector<FixtureTensor> &tensors,
                                       luisa::string_view magic = "BGM3",
                                       uint32_t version = 1u,
                                       uint32_t precision = 0u,
                                       uint32_t count_override = 0xFFFFFFFFu,
                                       uint64_t truncate_to = 0ull) {
    luisa::vector<uint8_t> img;
    const auto count = count_override == 0xFFFFFFFFu ? static_cast<uint32_t>(tensors.size()) : count_override;
    for (size_t i = 0u; i < 4u; i++) img.push_back(static_cast<uint8_t>(magic[i]));
    append_le32(img, version);
    append_le32(img, precision);
    append_le32(img, count);
    // blob placement first: exact packing, 256-byte aligned
    luisa::vector<uint64_t> offsets, sizes;
    auto running = align_up(kBgmHeaderBytes + kBgmEntryBytes * tensors.size(), kBgmAlign);
    for (const auto &t : tensors) {
        const auto sz = fixture_elems(t) * 4ull;
        offsets.push_back(running);
        sizes.push_back(sz);
        running = align_up(running + sz, kBgmAlign);
    }
    for (size_t i = 0u; i < tensors.size(); i++) {
        const auto &t = tensors[i];
        append_le64(img, fnv1a64_name(t.name));
        append_le64(img, offsets[i]);
        append_le64(img, sizes[i]);
        append_le32(img, t.rank);
        for (uint32_t d = 0u; d < 4u; d++) {
            append_le32(img, d < t.rank ? t.dims[d] : 0u);
        }
        // entry stride is 48 bytes while the documented fields occupy 44
        // (8+8+8+4+16): the trailing 4 bytes are zero padding, as the real
        // packer writes them (verified against model.bgm's table)
        append_le32(img, 0u);
    }
    while (img.size() < offsets.front()) img.push_back(0u);
    for (size_t i = 0u; i < tensors.size(); i++) {
        while (img.size() < offsets[i] + sizes[i]) {
            const auto *src = reinterpret_cast<const uint8_t *>(tensors[i].values.data());
            img.push_back(src[img.size() - offsets[i]]);
        }
        const auto next = align_up(offsets[i] + sizes[i], kBgmAlign);
        while (img.size() < next) img.push_back(0u);
    }
    if (truncate_to > 0ull && truncate_to < img.size()) {
        img.resize(static_cast<size_t>(truncate_to));
    }
    return img;
}
// parses header + table out of an in-memory image (fixtures are small)
bool parse_bgm_image(const luisa::vector<uint8_t> &img, RawTable &out, luisa::string &err) {
    out = RawTable{};
    if (img.size() < kBgmHeaderBytes + kBgmEntryBytes) {
        err = "image shorter than a BGM3 header plus one 48-byte entry";
        return false;
    }
    out.magic = luisa::string{luisa::string_view{reinterpret_cast<const char *>(img.data()), 4u}};
    out.version = read_le32_at(img, 4u);
    out.precision = read_le32_at(img, 8u);
    out.count = read_le32_at(img, 12u);
    if (static_cast<uint64_t>(out.count) > (img.size() - kBgmHeaderBytes) / kBgmEntryBytes) {
        err = luisa::format("tensor_count {} does not fit in a {}-byte image", out.count, img.size());
        return false;
    }
    out.image = img;
    out.hashes.resize(out.count);
    out.offsets.resize(out.count);
    out.sizes.resize(out.count);
    out.ranks.resize(out.count);
    out.dims.resize(static_cast<size_t>(out.count) * 4u);
    for (uint32_t i = 0u; i < out.count; i++) {
        const auto base = static_cast<size_t>(kBgmHeaderBytes + kBgmEntryBytes * i);
        out.hashes[i] = read_le64_at(img, base);
        out.offsets[i] = read_le64_at(img, base + 8u);
        out.sizes[i] = read_le64_at(img, base + 16u);
        out.ranks[i] = read_le32_at(img, base + 24u);
        for (uint32_t d = 0u; d < 4u; d++) {
            out.dims[static_cast<size_t>(i) * 4u + d] = read_le32_at(img, base + 28u + 4u * d);
        }
    }
    err.clear();
    return true;
}
// reads only header + table from a file (the 2.27 GB payloads are streamed
// separately, so no second whole-file copy is held)
bool read_bgm_table(const luisa::string &path, RawTable &out, luisa::string &err) {
    auto *f = std::fopen(path.c_str(), "rb");
    if (f == nullptr) {
        err = luisa::format("cannot open \"{}\"", path);
        return false;
    }
    uint8_t head[kBgmHeaderBytes];
    bool ok = std::fread(head, 1u, sizeof(head), f) == sizeof(head);
    luisa::vector<uint8_t> hdr{head, head + sizeof(head)};
    std::fclose(f);
    if (!ok) {
        err = "short header read";
        return false;
    }
    out = RawTable{};
    out.magic = luisa::string{luisa::string_view{reinterpret_cast<const char *>(hdr.data()), 4u}};
    out.version = read_le32_at(hdr, 4u);
    out.precision = read_le32_at(hdr, 8u);
    out.count = read_le32_at(hdr, 12u);
    if (auto *ff = std::fopen(path.c_str(), "rb"); ff != nullptr) {
        luisa::vector<uint8_t> tbl(static_cast<size_t>(kBgmEntryBytes * out.count));
        seek64(ff, kBgmHeaderBytes);
        ok = tbl.empty() || std::fread(tbl.data(), 1u, tbl.size(), ff) == tbl.size();
        std::fclose(ff);
        if (!ok) {
            err = "short table read";
            return false;
        }
        out.image = hdr;
        out.image.insert(out.image.end(), tbl.begin(), tbl.end());
        out.hashes.resize(out.count);
        out.offsets.resize(out.count);
        out.sizes.resize(out.count);
        out.ranks.resize(out.count);
        out.dims.resize(static_cast<size_t>(out.count) * 4u);
        for (uint32_t i = 0u; i < out.count; i++) {
            const auto base = static_cast<size_t>(kBgmEntryBytes * i);
            out.hashes[i] = read_le64_at(tbl, base);
            out.offsets[i] = read_le64_at(tbl, base + 8u);
            out.sizes[i] = read_le64_at(tbl, base + 16u);
            out.ranks[i] = read_le32_at(tbl, base + 24u);
            for (uint32_t d = 0u; d < 4u; d++) {
                out.dims[static_cast<size_t>(i) * 4u + d] = read_le32_at(tbl, base + 28u + 4u * d);
            }
        }
        err.clear();
        return true;
    }
    err = luisa::format("cannot reopen \"{}\"", path);
    return false;
}
// verifies the structural guarantees the packer documents (alignment, exact
// packing, blob containment, size == prod(dims)*4, dims beyond rank == 0,
// file size == align_up(16+48N,256) + sum(align_up(size_i,256)))
bool validate_table_structure(const RawTable &t, uint64_t file_size, luisa::string &err) {
    const auto &img = t.image;
    auto expect_off = t.first_blob();
    for (uint32_t i = 0u; i < t.count; i++) {
        const auto tail = static_cast<size_t>(kBgmHeaderBytes + kBgmEntryBytes * i + 44ull);
        const uint32_t entry_tail_pad = tail + 4u <= img.size() ? read_le32_at(img, tail) : 0u;
        if (t.ranks[i] < 1u || t.ranks[i] > 4u) {
            err = luisa::format("entry {}: rank {} out of range", i, t.ranks[i]);
            return false;
        }
        for (uint32_t d = t.ranks[i]; d < 4u; d++) {
            if (t.dims[static_cast<size_t>(i) * 4u + d] != 0u) {
                err = luisa::format("entry {}: dims beyond rank are not zero", i);
                return false;
            }
        }
        if (entry_tail_pad != 0u) {
            err = luisa::format("entry {}: the 4 pad bytes at offset 44 of the 48-byte entry "
                                "(file offset {}) are not zero",
                                i, kBgmHeaderBytes + kBgmEntryBytes * i + 44ull);
            return false;
        }
        if (t.entry_elems(i) * 4ull != t.sizes[i]) {
            err = luisa::format("entry {}: size {} != prod(dims) {} * 4", i, t.sizes[i], t.entry_elems(i));
            return false;
        }
        if (t.offsets[i] % kBgmAlign != 0ull) {
            err = luisa::format("entry {}: offset {} is not 256-byte aligned", i, t.offsets[i]);
            return false;
        }
        if (t.offsets[i] != expect_off) {
            err = luisa::format("entry {}: offset {} != exact-packing {}", i, t.offsets[i], expect_off);
            return false;
        }
        if (t.offsets[i] + t.sizes[i] > file_size) {
            err = luisa::format("entry {}: blob [{}, {}) runs past the {}-byte file",
                                i, t.offsets[i], t.offsets[i] + t.sizes[i], file_size);
            return false;
        }
        expect_off = align_up(t.offsets[i] + t.sizes[i], kBgmAlign);
    }
    if (expect_off != file_size) {
        err = luisa::format("file size {} != computed align_up(header+table,256) + padded blobs {}",
                            file_size, expect_off);
        return false;
    }
    err.clear();
    return true;
}
// per-tensor checksums of an in-memory image + the aggregate (file header)
void checksum_image(const luisa::vector<uint8_t> &img, const RawTable &t,
                    luisa::vector<uint64_t> &per_tensor, uint64_t &aggregate) {
    per_tensor.clear();
    per_tensor.reserve(t.count);
    aggregate = kFnvBasis;
    for (uint32_t i = 0u; i < t.count; i++) {
        const auto *p = img.data() + static_cast<size_t>(t.offsets[i]);
        const auto h = fnv1a64_bytes(p, static_cast<size_t>(t.sizes[i]));
        per_tensor.push_back(h);
        aggregate = fnv1a64_mix_u64(aggregate, h);
    }
}
// same, streaming one blob at a time from an open file
bool checksum_file(const luisa::string &path, const RawTable &t,
                   luisa::vector<uint64_t> &per_tensor, uint64_t &aggregate, luisa::string &err) {
    per_tensor.clear();
    per_tensor.reserve(t.count);
    aggregate = kFnvBasis;
    auto *f = std::fopen(path.c_str(), "rb");
    if (f == nullptr) {
        err = luisa::format("cannot open \"{}\"", path);
        return false;
    }
    luisa::vector<uint8_t> buf;
    bool ok = true;
    for (uint32_t i = 0u; ok && i < t.count; i++) {
        buf.resize(static_cast<size_t>(t.sizes[i]));
        seek64(f, t.offsets[i]);
        ok = buf.empty() || std::fread(buf.data(), 1u, buf.size(), f) == buf.size();
        if (!ok) {
            err = luisa::format("short blob read at entry {}", i);
            break;
        }
        const auto h = fnv1a64_bytes(buf.data(), buf.size());
        per_tensor.push_back(h);
        aggregate = fnv1a64_mix_u64(aggregate, h);
    }
    std::fclose(f);
    return ok;
}

// ---------------------------------------------------------------------
// (1) real package: committed checksum manifest + loader parity
// ---------------------------------------------------------------------
void test_real_package_checksums(const luisa::embed::Package &pkg,
                                 const luisa::string &path,
                                 const RawTable &raw,
                                 const luisa::vector<uint64_t> &raw_checksums,
                                 uint64_t raw_aggregate) {
    const auto table = pkg.tensors();
    expect(table.size() == kExpectedTensorCount)
        << luisa::format("the committed aggregate covers exactly {} tensors, but the package holds {}",
                         kExpectedTensorCount, table.size());
    uint32_t mismatch_count = 0u;
    // (a) aggregate over the loader-served bytes
    luisa::vector<uint64_t> loader_checksums;
    loader_checksums.reserve(table.size());
    uint64_t loader_aggregate = kFnvBasis;
    for (const auto &t : table) {
        const auto h = fnv1a64_bytes(t.data, static_cast<size_t>(t.size_bytes));
        loader_checksums.push_back(h);
        loader_aggregate = fnv1a64_mix_u64(loader_aggregate, h);
    }
    if (loader_aggregate != kExpectedAggregateChecksum) {
        LUISA_WARNING("aggregate byte checksum 0x{:016x} != committed 0x{:016x}",
                      loader_aggregate, kExpectedAggregateChecksum);
        mismatch_count++;
    }
    // (c) pinned per-tensor checksums
    for (const auto &p : kPinnedChecksums) {
        const auto *t = pkg.find(p.name);
        if (t == nullptr) {
            LUISA_WARNING("pinned tensor '{}' is missing from the package", p.name);
            mismatch_count++;
            continue;
        }
        const auto h = fnv1a64_bytes(t->data, static_cast<size_t>(t->size_bytes));
        if (h != p.checksum) {
            LUISA_WARNING("pinned tensor '{}': byte checksum 0x{:016x} != committed 0x{:016x}",
                          p.name, h, p.checksum);
            mismatch_count++;
        }
    }
    // (1, second half) pack -> load -> checksum equality against the raw file
    if (raw.count == static_cast<uint32_t>(loader_checksums.size())) {
        for (uint32_t i = 0u; i < raw.count; i++) {
            if (raw_checksums[i] != loader_checksums[i]) {
                LUISA_WARNING("tensor {}: raw-file checksum 0x{:016x} != loader-served 0x{:016x}",
                              i, raw_checksums[i], loader_checksums[i]);
                mismatch_count++;
            }
        }
        if (raw_aggregate != loader_aggregate) {
            LUISA_WARNING("raw-file aggregate 0x{:016x} != loader aggregate 0x{:016x}",
                          raw_aggregate, loader_aggregate);
            mismatch_count++;
        }
        // the raw pass resolves the pinned names by their own FNV hashing
        for (const auto &p : kPinnedChecksums) {
            const auto nh = fnv1a64_name(p.name);
            uint32_t found = 0xFFFFFFFFu;
            for (uint32_t i = 0u; i < raw.count; i++) {
                if (raw.hashes[i] == nh) {
                    found = i;
                    break;
                }
            }
            if (found == 0xFFFFFFFFu) {
                LUISA_WARNING("raw pass: no table entry hashes to '{}' (0x{:016x})", p.name, nh);
                mismatch_count++;
            } else if (raw_checksums[found] != p.checksum) {
                LUISA_WARNING("raw pass: '{}' checksum 0x{:016x} != committed 0x{:016x}",
                              p.name, raw_checksums[found], p.checksum);
                mismatch_count++;
            }
        }
    } else {
        LUISA_WARNING("raw pass saw {} tensors, the loader served {}", raw.count, loader_checksums.size());
        mismatch_count++;
    }
    LUISA_INFO("test_embed_pack: {} tensors of \"{}\" hashed; aggregate 0x{:016x} (committed 0x{:016x}); "
               "{} mismatches",
               table.size(), path, loader_aggregate, kExpectedAggregateChecksum, mismatch_count);
    expect(mismatch_count == 0u)
        << "real package bytes must match the committed checksum manifest, and Package::load must serve identical bytes";
}

// ---------------------------------------------------------------------
// (2) synthetic fixture round-trip
// ---------------------------------------------------------------------
luisa::vector<FixtureTensor> make_fixture(uint32_t kind) {
    // NOTE: kinds 1/2 deliberately reuse canonical bge-m3 tensor NAMES (with
    // tiny synthetic dims) so the writer/reader exercise the same FNV name
    // hashing the packer uses; these images are only ever parsed by this
    // test's own minimal reader and by the fail-closed loader's rejection
    // path (group 3) -- never accepted as a real package.
    luisa::vector<FixtureTensor> ts;
    // deterministic, non-symmetric bit patterns (also NaN-free so a byte
    // compare is well defined)
    const auto fill = [](FixtureTensor &t) {
        t.values.resize(static_cast<size_t>(fixture_elems(t)));
        for (size_t i = 0u; i < t.values.size(); i++) {
            t.values[i] = static_cast<float>(static_cast<int>(i) * 7 - 13) / 3.f;
        }
    };
    const auto add = [&](luisa::string name, std::initializer_list<uint32_t> dims) {
        FixtureTensor t;
        t.name = std::move(name);
        t.rank = static_cast<uint32_t>(dims.size());
        t.dims[0] = t.dims[1] = t.dims[2] = t.dims[3] = 0u;// fully initialized
        uint32_t i = 0u;
        for (const auto d : dims) t.dims[i++] = d;
        fill(t);
        ts.push_back(std::move(t));
    };
    if (kind == 0u) {
        add("fixture.alpha", {8u});
    } else if (kind == 1u) {
        add("emb.ln.gamma", {1u});   // 4-byte blob: maximum padding
        add("layer0.qkv", {3u, 5u}); // rank-2
        add("head.colbert.b", {64u});// exactly one 256-byte blob: zero padding
    } else {
        add("emb.word", {2u, 3u});
        add("layer23.ffn.down.bias", {7u});
        add("x", {1u});// the smallest legal tensor
        add("y.y.y", {9u, 1u});
    }
    return ts;
}
void test_synthetic_fixture_round_trip(uint32_t kind, uint32_t &mismatch_count) {
    const auto tensors = make_fixture(kind);
    const auto img = build_bgm_image(tensors);
    luisa::string err;
    RawTable t;
    if (!parse_bgm_image(img, t, err)) {
        LUISA_WARNING("fixture {}: the image this test wrote does not parse back: {}", kind, err);
        mismatch_count++;
        return;
    }
    if (t.magic != "BGM3" || t.version != 1u || t.precision != 0u ||
        t.count != static_cast<uint32_t>(tensors.size())) {
        LUISA_WARNING("fixture {}: header round-trip mismatch (magic \"{}\", version {}, precision {}, count {})",
                      kind, t.magic, t.version, t.precision, t.count);
        mismatch_count++;
        return;
    }
    // header/table/alignment/file-size formula
    luisa::string serr;
    if (!validate_table_structure(t, img.size(), serr)) {
        LUISA_WARNING("fixture {}: structural check failed: {}", kind, serr);
        mismatch_count++;
        return;
    }
    expect(img.size() == align_up(kBgmHeaderBytes + kBgmEntryBytes * t.count, kBgmAlign) + [&t]() {
        uint64_t s = 0ull;
        for (uint32_t i = 0u; i < t.count; i++) s += align_up(t.sizes[i], kBgmAlign);
        return s;
    }()) << "file size must follow the spec formula align_up(16+48N,256) + sum(align_up(size_i,256))";
    for (uint32_t i = 0u; i < t.count; i++) {
        const auto &src = tensors[i];
        // name hash round-trips
        if (t.hashes[i] != fnv1a64_name(src.name)) {
            LUISA_WARNING("fixture {} entry {}: name_hash 0x{:016x} != FNV-1a-64(\"{}\") 0x{:016x}",
                          kind, i, t.hashes[i], src.name, fnv1a64_name(src.name));
            mismatch_count++;
        }
        if (t.ranks[i] != src.rank) {
            LUISA_WARNING("fixture {} entry '{}': rank {} != {}", kind, src.name, t.ranks[i], src.rank);
            mismatch_count++;
        }
        for (uint32_t d = 0u; d < 4u; d++) {
            const auto want = d < src.rank ? src.dims[d] : 0u;
            if (t.dims[static_cast<size_t>(i) * 4u + d] != want) {
                LUISA_WARNING("fixture {} entry '{}': dims[{}] {} != {}", kind, src.name, d,
                              t.dims[static_cast<size_t>(i) * 4u + d], want);
                mismatch_count++;
            }
        }
        // payload bytes round-trip bit-exactly
        const auto *blob = img.data() + static_cast<size_t>(t.offsets[i]);
        const auto *orig = reinterpret_cast<const uint8_t *>(src.values.data());
        if (std::memcmp(blob, orig, static_cast<size_t>(t.sizes[i])) != 0) {
            LUISA_WARNING("fixture {} entry '{}': payload bytes differ after the round-trip", kind, src.name);
            mismatch_count++;
        }
        // padding between/after blobs is zero
        const auto pad_begin = t.offsets[i] + t.sizes[i];
        const auto pad_end = i + 1u < t.count ? t.offsets[i + 1u] : align_up(pad_begin, kBgmAlign);
        for (auto o = pad_begin; o < pad_end; o++) {
            if (img[static_cast<size_t>(o)] != 0u) {
                LUISA_WARNING("fixture {} entry {}: padding byte at {} is not zero", kind, i, o);
                mismatch_count++;
                break;
            }
        }
    }
    // checksums round-trip, and the aggregation is reproducible
    luisa::vector<uint64_t> cs_a, cs_b;
    uint64_t agg_a = 0ull, agg_b = 0ull;
    checksum_image(img, t, cs_a, agg_a);
    for (uint32_t i = 0u; i < t.count; i++) {
        const auto want = fnv1a64_bytes(tensors[i].values.data(),
                                        static_cast<size_t>(tensors[i].values.size() * sizeof(float)));
        if (cs_a[i] != want) {
            LUISA_WARNING("fixture {} entry {}: blob checksum 0x{:016x} != source-values checksum 0x{:016x}",
                          kind, i, cs_a[i], want);
            mismatch_count++;
        }
    }
    // a rebuilt image hashes identically (writer determinism), and a
    // single-flipped payload byte changes the checksum (checksum sensitivity)
    const auto img2 = build_bgm_image(tensors);
    checksum_image(img2, t, cs_b, agg_b);
    if (cs_b != cs_a || agg_b != agg_a) {
        LUISA_WARNING("fixture {}: re-serializing the same tensors changed the checksums", kind);
        mismatch_count++;
    }
    {
        auto img3 = img;
        img3[static_cast<size_t>(t.offsets[0]) + 1u] = static_cast<uint8_t>(img3[static_cast<size_t>(t.offsets[0]) + 1u] ^ 0x80u);
        RawTable t3;
        if (!parse_bgm_image(img3, t3, err) || [&]() {
                luisa::vector<uint64_t> cs3;
                uint64_t agg3 = 0ull;
                checksum_image(img3, t3, cs3, agg3);
                return cs3[0] == cs_a[0] && agg3 == agg_a;
            }()) {
            LUISA_WARNING("fixture {}: flipping one payload byte left the checksums unchanged", kind);
            mismatch_count++;
        }
    }
    LUISA_INFO("test_embed_pack: synthetic fixture {} ({} tensors) round-tripped; aggregate 0x{:016x}",
               kind, t.count, agg_a);
}

// ---------------------------------------------------------------------
// (3) corrupt fixtures: rejection parity through a spawned child
// ---------------------------------------------------------------------
struct CorruptCase {
    luisa::string_view tag;
    luisa::string_view diagnostic;// substring the child log must contain
};
// the child: load a package path; a fatal LUISA_ERROR surfaces as an abort
int run_child_load(const char *path) {
    auto pkg = luisa::embed::Package::load(std::filesystem::path{path});
    if (pkg == nullptr) {
        std::fprintf(stderr, "child: Package::load returned null for \"%s\"\n", path);
        return 1;
    }
    std::printf("child: loaded %zu tensors\n", pkg->tensors().size());
    std::fflush(stdout);
    return 0;
}
luisa::vector<uint8_t> corrupt_fixture(uint32_t kind, const luisa::vector<FixtureTensor> &base) {
    switch (kind) {
        case 0u:// bad magic
            return build_bgm_image(base, "BGM4");
        case 1u:// unsupported version
            return build_bgm_image(base, "BGM3", 2u);
        case 2u:// precision = 1 (bf16, reserved for plan M7)
            return build_bgm_image(base, "BGM3", 1u, 1u);
        case 3u:// tensor_count inconsistent with the bge-m3 v1 set
            return build_bgm_image(base, "BGM3", 1u, 0u, 3u);
        default:// truncated file: a valid v1 header announcing the full
            // 296-entry table, cut to the minimum size the loader accepts
            // (16 + 48 bytes) so the table itself does not fit
            return build_bgm_image(base, "BGM3", 1u, 0u, kExpectedTensorCount,
                                   kBgmHeaderBytes + kBgmEntryBytes);
    }
}
int spawn_and_wait(const luisa::string &exe, const luisa::string &target,
                   const std::filesystem::path &log_path) {
    auto command = luisa::format("\"{}\" --child-load \"{}\" > \"{}\" 2>&1",
                                 exe, target, log_path.string());
#ifdef _WIN32
    // system() invokes cmd /C, which strips the outer quote pair; enclose the
    // whole line to keep the executable and redirected-path quotes intact
    // (same idiom as test_vk_native_route_guard.cpp)
    command = luisa::format("\"{}\"", command);
#endif
    return std::system(command.c_str());
}
void test_corrupt_fixtures_rejected(const luisa::string &exe,
                                    const std::filesystem::path &work_dir,
                                    const luisa::vector<FixtureTensor> &base,
                                    const luisa::string &real_package,
                                    bool real_package_exists,
                                    uint32_t &mismatch_count,
                                    int &control_exit_code) {
    constexpr CorruptCase cases[] = {
        {"bad-magic", "bad magic"},
        {"bad-version", "unsupported BGM3 version"},
        {"precision-bf16", "unsupported precision"},
        {"wrong-tensor-count", "tensor_count"},
        {"truncated-table", "does not fit"},
    };
    for (uint32_t i = 0u; i < sizeof(cases) / sizeof(cases[0]); i++) {
        const auto &c = cases[i];
        auto fixture_path = (work_dir / luisa::format("corrupt_{}.bgm", i)).string();
        const auto img = corrupt_fixture(i, base);
        if (!write_whole_file(std::filesystem::path{fixture_path}, img)) {
            LUISA_WARNING("cannot write the corrupt fixture \"{}\"", fixture_path);
            mismatch_count++;
            continue;
        }
        auto log_path = work_dir / luisa::format("corrupt_{}.log", i);
        const auto status = spawn_and_wait(exe, fixture_path, log_path);
        const auto log = read_text_file(log_path);
        // evidence: the child exit code of every malformed fixture
        LUISA_INFO("corrupt fixture \"{}\" ({} bytes) -> child exit code {}, log {} bytes",
                   c.tag, img.size(), static_cast<int>(status), log.size());
        if (status == 0) {
            LUISA_WARNING("corrupt fixture \"{}\" was ACCEPTED by Package::load (child exited 0)", c.tag);
            mismatch_count++;
        }
        if (log.find(c.diagnostic.data()) == luisa::string::npos) {
            LUISA_WARNING("corrupt fixture \"{}\": child log does not contain \"{}\"; child output:\n{}",
                          c.tag, c.diagnostic, log);
            mismatch_count++;
        }
    }
    // positive control: the harness reports success correctly, and the real
    // package loads cleanly in a child
    if (real_package_exists) {
        auto log_path = work_dir / "control.log";
        control_exit_code = spawn_and_wait(exe, real_package, log_path);
        const auto log = read_text_file(log_path);
        LUISA_INFO("control child loading the real package -> exit code {}",
                   static_cast<int>(control_exit_code));
        if (control_exit_code != 0) {
            LUISA_WARNING("control child must load the real package successfully; child output:\n{}", log);
            mismatch_count++;
        }
    } else {
        LUISA_INFO("control child skipped (no real package present)");
    }
    expect(mismatch_count == 0u)
        << "every malformed .bgm fixture must be rejected loudly by Package::load (nonzero child exit + its own diagnostic)";
}
}// namespace

int main(int argc, char *argv[]) {
    // child mode (must be handled before Boost.UT touches argv): the rejection
    // path under test is LUISA_ERROR -> std::abort, noreturn, so corrupt fixtures
    // are loaded in a separate process -- see test_corrupt_fixtures_rejected
    if (argc >= 3 && std::strcmp(argv[1], "--child-load") == 0) {
        return run_child_load(argv[2]);
    }

    // this executable, captured before the argv[1] shift below (the corrupt
    // fixtures are loaded by re-spawning this same binary)
    const luisa::string self{argc > 0 && argv != nullptr ? argv[0] : "test_embed_pack"};

    // argv[1] (optional) is the packed model path; the default pins the real
    // bge-m3 package the packer produced (plan §6 M2, dev environment), same
    // convention as test_embed_package.cpp
    luisa::string bgm_path{"D:/compute/bge-m3/bge-m3/model.bgm"};
    if (argc > 1) {
        if (argv[1][0] == '\0') {
            std::fprintf(stderr, "test_embed_pack: empty package path\n");
            return 1;
        }
        bgm_path = luisa::string{argv[1]};
        argc--;
        argv++;
    }
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));

    const auto exe = luisa::filesystem::absolute(self).string();
    std::error_code ec;
    auto work_dir = luisa::filesystem::temp_directory_path(ec) /
                    luisa::string{"luisa_embed_pack_scratch"};
    std::filesystem::create_directories(work_dir, ec);

    uint32_t early_mismatches = 0u;
    // (2) synthetic fixture round-trip -- needs no model file
    for (uint32_t k = 0u; k < 3u; k++) {
        test_synthetic_fixture_round_trip(k, early_mismatches);
    }
    expect(early_mismatches == 0u) << "in-test .bgm fixtures must round-trip the v1 header/table/alignment/byte layout";

    const bool model_present = std::filesystem::exists(std::filesystem::path{bgm_path}, ec) && !ec;

    if (!model_present) {
        // fail loudly (plan §8: the byte-integrity manifest of §2 test_embed_pack
        // cannot be verified without the packed model), but only after the
        // model-independent groups have already been reported
        std::filesystem::remove_all(work_dir, ec);
        LUISA_ERROR("packed model \"{}\" not found (pass its path as argv[1])", bgm_path);
    }

    // (1) the real package through the engine loader + an independent raw pass
    auto pkg = luisa::embed::Package::load(std::filesystem::path{bgm_path});
    if (pkg == nullptr) {
        LUISA_ERROR("Package::load returned null for \"{}\"", bgm_path);
    }
    luisa::string err;
    RawTable raw;
    if (!read_bgm_table(bgm_path, raw, err)) {
        LUISA_ERROR("the in-test minimal reader cannot parse \"{}\": {}", bgm_path, err);
    }
    const auto file_size = static_cast<uint64_t>(std::filesystem::file_size(std::filesystem::path{bgm_path}, ec));
    if (ec) {
        LUISA_ERROR("cannot size the packed model \"{}\"", bgm_path);
    }
    // the packer's own structural guarantees, verified against the raw bytes
    // rather than through the loader (independent of package.cpp)
    luisa::string serr;
    if (raw.magic != "BGM3" || raw.version != 1u || raw.precision != 0u) {
        LUISA_WARNING("raw header: magic \"{}\", version {}, precision {} (expected BGM3/1/0)",
                      raw.magic, raw.version, raw.precision);
    } else if (!validate_table_structure(raw, file_size, serr)) {
        LUISA_WARNING("raw structural validation of \"{}\" failed: {}", bgm_path, serr);
    } else {
        LUISA_INFO("test_embed_pack: raw v1 structure verified independently — {} tensors, {} bytes, "
                   "first blob at {}",
                   raw.count, file_size, raw.first_blob());
    }
    luisa::vector<uint64_t> raw_checksums;
    uint64_t raw_aggregate = 0ull;
    if (!checksum_file(bgm_path, raw, raw_checksums, raw_aggregate, err)) {
        LUISA_ERROR("streaming checksum pass over \"{}\" failed: {}", bgm_path, err);
    }
    test_real_package_checksums(*pkg, bgm_path, raw, raw_checksums, raw_aggregate);

    // (3) corrupt fixtures + the positive control, in spawned child processes
    const auto base = make_fixture(1u);
    uint32_t subprocess_mismatches = 0u;
    int control_exit_code = -1;
    test_corrupt_fixtures_rejected(exe, work_dir, base, bgm_path, model_present,
                                   subprocess_mismatches, control_exit_code);

    std::filesystem::remove_all(work_dir, ec);
    return 0;
}
