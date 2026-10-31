// bge-m3 packed-model loader — implements embed/package.h (embedding_model_plan.md
// §3.3, milestone M2 "loader half"; plan §9.1 D4 consumer).
//
// The on-disk format is model.bgm VERSION 1, specified in the header comment of
// examples/embedding/tools/pack_model.cpp (that comment block IS the spec):
// magic 'B','G','M','3' | u32 version=1 | u32 precision=0 (fp32) |
// u32 tensor_count=296 | 296 * 48-byte table entries
// {fnv1a64(name), offset, size_bytes, rank, dims[4]} in the fixed emission
// order emb.*, layer0.* .. layer23.*, head.* | 256-byte aligned fp32 LE blobs.
//
// VERSION 2 (optimization plan §4 Q1/Q2, O-G): same layout with version=2 and
// precision = 1 (bf16) or 2 (fp16); every blob is 2-byte little-endian
// elements (RNE from fp32 at pack time), size_bytes = prod(dims)*2. The v1
// fp32 path is unchanged byte-for-byte.
//
// VERSION 3 (INT8 quantization, phase 1): same layout with version=3 and
// precision = 3 (per-tensor symmetric int8). Every blob is exactly
// prod(dims) signed int8 bytes (no per-blob header, size_bytes = prod(dims)),
// and the tensor's positive fp32 scale is stored in the reserved trailing
// 4 bytes of its 48-byte table entry (entry offset 44, IEEE-754 LE). v1/v2
// entries keep those bytes zero; this loader reads them only for v3 and
// rejects a scale that is nonfinite or <= 0 (fail-closed). The quantization
// math itself lives in embed/quantization.h; v3 blobs are exposed through
// Tensor::int8_data/int8_scale and never through data/half_data.
//
// Loading strategy (documented choice; package.h: "Blobs stay on disk (mapped)
// or are read on demand — implementation choice"): the whole file is read once
// into a single host allocation (2.27 GiB for bge-m3 fp32 — simplest and fine
// for a dev-time loader; mmap/async upload is a plan §10 optimization that must
// not change this API). The allocation is over-sized by 256 bytes so the file
// image sits at a 256-byte aligned address: every on-disk blob offset is a
// multiple of 256, hence every Tensor::data pointer keeps that alignment and
// the Session can upload blobs into device buffers directly without re-packing.
// _raw is never resized after load(), so the pointers are stable for the
// lifetime of the Package.
//
// Validation is fail-closed and loud: every deviation from the spec
// (magic / unknown version / v1-with-precision!=0 / v2-with-precision==0 /
// precision > 3 / v3-with-precision!=3 / tensor_count != 296 / table
// structural inconsistency / canonical-name or emission-order mismatch / dims
// disagreeing with the bge-m3 hyper-parameters / file size / an invalid v3
// scale) raises LUISA_ERROR with the offending values. Version 2 accepts
// precision 1 (bf16) / 2 (fp16) only; version 3 accepts precision 3 only.
//
// Hyper-parameter cross-check: the packer stores no constants beyond the table,
// so the architecture (24 layers, 1024 hidden, 4096 intermediate, 250002 vocab,
// 8194 positions) is validated by regenerating the canonical (name, rank, dims)
// set FROM those constants and requiring table equality position by position.
// num_heads (16) is not a table dimension — it is implied by the fused-QKV
// width 3*hidden = 3072 and head_dim = hidden/16 = 64 (format spec: inside
// layer{i}.qkv the column index is head-major n = h*64 + d); it is served as a
// fixed constant, matching the packer's "RUNTIME CONSTANTS THE PACKER DOES NOT
// STORE" list.
#include "embedding/embed/package.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <system_error>
#include <utility>

#include <luisa/core/logging.h>
#include <luisa/core/stl/format.h>
#include <luisa/core/stl/string.h>

namespace luisa::embed {

const char *package_precision_name(PackagePrecision p) noexcept {
    switch (p) {
        case PackagePrecision::fp32: return "fp32";
        case PackagePrecision::bf16: return "bf16";
        case PackagePrecision::fp16: return "fp16";
        case PackagePrecision::int8: return "int8";
    }
    return "unknown";
}

uint64_t Tensor::element_count() const noexcept {
    uint64_t n = 1ull;
    for (uint32_t i = 0u; i < rank; i++) {
        n *= dims[i];
    }
    return n;
}

namespace {

// --- format constants (mirror the packer spec; pack_model.cpp kBgm*) -----
constexpr uint32_t kBgmVersion = 1u;
constexpr uint32_t kBgmVersionHalf = 2u;// opt plan §4 Q1/Q2
constexpr uint32_t kBgmVersionInt8 = 3u;// INT8 quantization, phase 1
constexpr uint32_t kBgmPrecisionFp32 = 0u;
constexpr uint32_t kBgmPrecisionBf16 = 1u;
constexpr uint32_t kBgmPrecisionFp16 = 2u;
constexpr uint32_t kBgmPrecisionInt8 = 3u;
constexpr uint64_t kBgmAlign = 256ull;
constexpr uint64_t kBgmHeaderBytes = 16ull;
constexpr uint64_t kBgmEntryBytes = 48ull;
// Byte offset of the reserved trailing 4 bytes inside a 48-byte table entry:
// 8 (hash) + 8 (offset) + 8 (size_bytes) + 4 (rank) + 4*4 (dims) == 44. Zero for
// v1/v2; the per-tensor fp32 INT8 scale for v3.
constexpr uint64_t kBgmEntryReservedOffset = 44ull;

// --- bge-m3 architecture constants (format spec "RUNTIME CONSTANTS THE
// PACKER DOES NOT STORE", verified against bge-m3/config.json, plan §0) ----
constexpr uint32_t kNumLayers = 24u;
constexpr uint32_t kHiddenSize = 1024u;
constexpr uint32_t kNumHeads = 16u;
constexpr uint32_t kIntermediateSize = 4096u;
constexpr uint32_t kVocabSize = 250002u;
constexpr uint32_t kMaxPositions = 8194u;

// FNV-1a-64 exactly as specified by the format spec and implemented by
// pack_model.cpp::fnv1a_64: offset_basis 0xcbf29ce484222325, prime
// 0x00000100000001b3, per byte: h ^= b; h *= prime.
uint64_t fnv1a_64(luisa::string_view name) noexcept {
    uint64_t h = 0xcbf29ce484222325ull;
    for (const auto c : name) {
        h ^= static_cast<uint8_t>(c);
        h *= 0x00000100000001b3ull;
    }
    return h;
}

constexpr uint64_t align_up(uint64_t v, uint64_t a) noexcept {
    return (v + a - 1ull) / a * a;
}

uint32_t read_le32(const uint8_t *p) noexcept {
    return static_cast<uint32_t>(p[0]) |         //
           (static_cast<uint32_t>(p[1]) << 8u) | //
           (static_cast<uint32_t>(p[2]) << 16u) |//
           (static_cast<uint32_t>(p[3]) << 24u);
}
uint64_t read_le64(const uint8_t *p) noexcept {
    return static_cast<uint64_t>(read_le32(p)) |//
           (static_cast<uint64_t>(read_le32(p + 4u)) << 32u);
}

// One canonical tensor of the v1 spec: name + rank + dims, the dims derived
// from the architecture constants above (single arithmetic source for the
// cross-check; e.g. fused qkv N = 3*hidden, head.sparse N = 1).
struct ExpectedTensor {
    luisa::string name;
    uint32_t rank;
    uint32_t dims[4];
};

luisa::vector<ExpectedTensor> expected_table() {
    luisa::vector<ExpectedTensor> table;
    table.reserve(4u + kNumLayers * 12u + 4u);
    const auto add = [&table](luisa::string name, uint32_t rank, const uint32_t *dims) noexcept {
        ExpectedTensor e;
        e.name = std::move(name);
        e.rank = rank;
        for (uint32_t i = 0u; i < 4u; i++) {
            e.dims[i] = i < rank ? dims[i] : 0u;
        }
        table.push_back(std::move(e));
    };
    const uint32_t d_word[2] = {kVocabSize, kHiddenSize};
    const uint32_t d_pos[2] = {kMaxPositions, kHiddenSize};
    const uint32_t d_h[1] = {kHiddenSize};
    const uint32_t d_qkv[2] = {kHiddenSize, 3u * kHiddenSize};
    const uint32_t d_qkv_b[1] = {3u * kHiddenSize};
    const uint32_t d_hh[2] = {kHiddenSize, kHiddenSize};
    const uint32_t d_up[2] = {kHiddenSize, kIntermediateSize};
    const uint32_t d_up_b[1] = {kIntermediateSize};
    const uint32_t d_down[2] = {kIntermediateSize, kHiddenSize};
    const uint32_t d_one[1] = {1u};
    const uint32_t d_h1[2] = {kHiddenSize, 1u};

    add("emb.word", 2u, d_word);
    add("emb.pos", 2u, d_pos);
    add("emb.ln.gamma", 1u, d_h);
    add("emb.ln.beta", 1u, d_h);
    for (uint32_t i = 0u; i < kNumLayers; i++) {
        const auto p = luisa::format("layer{}.", i);
        add(p + "qkv", 2u, d_qkv);
        add(p + "qkv.bias", 1u, d_qkv_b);
        add(p + "attn.out", 2u, d_hh);
        add(p + "attn.out.bias", 1u, d_h);
        add(p + "ln1.gamma", 1u, d_h);
        add(p + "ln1.beta", 1u, d_h);
        add(p + "ffn.up", 2u, d_up);
        add(p + "ffn.up.bias", 1u, d_up_b);
        add(p + "ffn.down", 2u, d_down);
        add(p + "ffn.down.bias", 1u, d_h);
        add(p + "ln2.gamma", 1u, d_h);
        add(p + "ln2.beta", 1u, d_h);
    }
    add("head.sparse.w", 2u, d_h1);
    add("head.sparse.b", 1u, d_one);
    add("head.colbert.w", 2u, d_hh);
    add("head.colbert.b", 1u, d_h);
    return table;
}

// Concrete package. The whole file image lives in _raw (256-byte aligned
// base in _image); _tensors is ordered exactly like the on-disk table
// (emission order) and _hashes is the parallel name_hash column used by
// find(). Pimpl: the host-only loader owns no device resources.
class PackageImpl final : public Package {
public:
    [[nodiscard]] static luisa::unique_ptr<PackageImpl> create(
        const std::filesystem::path &bgm_file);

    [[nodiscard]] const Tensor *find(luisa::string_view canonical_name) const noexcept override {
        const auto h = fnv1a_64(canonical_name);
        for (size_t i = 0u; i < _hashes.size(); i++) {
            if (_hashes[i] == h) {
                return &_tensors[i];
            }
        }
        return nullptr;
    }
    [[nodiscard]] luisa::span<const Tensor> tensors() const noexcept override {
        return {_tensors.data(), _tensors.size()};
    }
    [[nodiscard]] PackagePrecision precision() const noexcept override { return _precision; }
    [[nodiscard]] uint32_t num_layers() const noexcept override { return kNumLayers; }
    [[nodiscard]] uint32_t hidden_size() const noexcept override { return kHiddenSize; }
    [[nodiscard]] uint32_t num_heads() const noexcept override { return kNumHeads; }
    [[nodiscard]] uint32_t intermediate_size() const noexcept override { return kIntermediateSize; }
    [[nodiscard]] uint32_t vocab_size() const noexcept override { return kVocabSize; }
    [[nodiscard]] uint32_t max_positions() const noexcept override { return kMaxPositions; }

private:
    void _fill(const std::filesystem::path &bgm_file);

    luisa::vector<uint8_t> _raw{};// whole file image + alignment slack
    const uint8_t *_image{};      // 256-byte aligned start of the file image
    luisa::vector<Tensor> _tensors{};
    luisa::vector<uint64_t> _hashes{};
    PackagePrecision _precision{PackagePrecision::fp32};
};

void PackageImpl::_fill(const std::filesystem::path &bgm_file) {
    // --- open + size ---------------------------------------------------------
    std::error_code ec;
    const auto fsize_u64 = static_cast<uint64_t>(std::filesystem::file_size(bgm_file, ec));
    if (ec) {
        LUISA_ERROR("cannot determine the size of the packed model \"{}\": {}",
                    bgm_file.string(), ec.message());
    }
    // dev loader: a complete v1 bge-m3 fp32 package is ~2.27 GiB; anything below
    // a plausible header + blob or above 16 GiB is rejected as corrupt input.
    if (fsize_u64 < kBgmHeaderBytes + kBgmEntryBytes || fsize_u64 > (16ull << 30ull)) {
        LUISA_ERROR("packed model \"{}\" has implausible size {} bytes", bgm_file.string(), fsize_u64);
    }
    const size_t fsize = static_cast<size_t>(fsize_u64);

    // --- single bulk read into the 256-byte aligned file image ----------------
    // H7 (host-memory opt, plan §2.3) was MEASURED and NOT landed: see
    // benchmark_results/embed_opt_log.md. The eager ~160 ms in the allocation
    // below is mimalloc's OS page-commit of the 2.27 GiB image (VirtualAlloc
    // MEM_COMMIT zeroes the pages), not the vector zero-fill, so switching to
    // resize_uninitialized changes nothing measurable (<100 ms gate, win ~0).
    // The remaining cost is the bulk fread memcpy; a zero-copy mmap that folds
    // page-faults into the GPU upload (and overlaps kernel compile) lives in the
    // Scheduler/WeightStore upload path -- outside this file's ownership -- and
    // cold start already meets the <3 s target at baseline (1.0-1.3 s).
    _raw.resize(fsize + static_cast<size_t>(kBgmAlign));
    _image = reinterpret_cast<const uint8_t *>(
        align_up(reinterpret_cast<uintptr_t>(_raw.data()), kBgmAlign));
    if (auto *f = std::fopen(bgm_file.string().c_str(), "rb"); f != nullptr) {
        const auto got = std::fread(const_cast<uint8_t *>(_image), 1u, fsize, f);
        std::fclose(f);
        if (got != fsize) [[unlikely]] {
            LUISA_ERROR("short read on the packed model \"{}\": {} of {} bytes",
                        bgm_file.string(), got, fsize);
        }
    } else [[unlikely]] {
        LUISA_ERROR("cannot open the packed model \"{}\" for reading", bgm_file.string());
    }

    // --- header ----------------------------------------------------------------
    if (!(_image[0] == 'B' && _image[1] == 'G' && _image[2] == 'M' && _image[3] == '3')) {
        LUISA_ERROR("bad magic in \"{}\": not a BGM3 package", bgm_file.string());
    }
    const auto version = read_le32(_image + 4u);
    if (version != kBgmVersion && version != kBgmVersionHalf && version != kBgmVersionInt8) {
        LUISA_ERROR("unsupported BGM3 version {} in \"{}\" (this loader implements v1 fp32, "
                    "v2 half-precision and v3 int8 packages)",
                    version, bgm_file.string());
    }
    const auto precision = read_le32(_image + 8u);
    if (version == kBgmVersionHalf && precision == kBgmPrecisionFp32) {
        LUISA_ERROR("unsupported BGM3 version {} with precision 0 in \"{}\": v2 packages are half-precision only "
                    "(1 = bf16, 2 = fp16)",
                    version, bgm_file.string());
    }
    if (version == kBgmVersion && precision != kBgmPrecisionFp32) {
        LUISA_ERROR("unsupported precision {} in \"{}\": v1 is fp32 only "
                    "(bf16 = precision 1 and fp16 = precision 2 packages are version 2, "
                    "int8 = precision 3 packages are version 3)",
                    precision, bgm_file.string());
    }
    if (version == kBgmVersionInt8 && precision != kBgmPrecisionInt8) {
        LUISA_ERROR("unsupported BGM3 version {} with precision {} in \"{}\": v3 packages are int8 only "
                    "(precision 3)",
                    version, precision, bgm_file.string());
    }
    if (version != kBgmVersionInt8 && precision == kBgmPrecisionInt8) {
        LUISA_ERROR("precision 3 (int8) in \"{}\" requires BGM3 version 3, not version {}",
                    bgm_file.string(), version);
    }
    if (precision != kBgmPrecisionFp32 && precision != kBgmPrecisionBf16 &&
        precision != kBgmPrecisionFp16 && precision != kBgmPrecisionInt8) {
        LUISA_ERROR("unsupported precision {} in \"{}\" (0 = fp32, 1 = bf16, 2 = fp16, 3 = int8 are the "
                    "only defined modes)",
                    precision, bgm_file.string());
    }
    const PackagePrecision pkg_precision =
        precision == kBgmPrecisionBf16 ? PackagePrecision::bf16 : precision == kBgmPrecisionFp16 ? PackagePrecision::fp16 :
        precision == kBgmPrecisionInt8                                 ? PackagePrecision::int8 :
                                                                        PackagePrecision::fp32;
    _precision = pkg_precision;
    const uint64_t elem_bytes = pkg_precision == PackagePrecision::int8 ? 1ull :
                                pkg_precision == PackagePrecision::fp32 ? 4ull :
                                                                         2ull;
    const auto count = read_le32(_image + 12u);
    const auto expected = expected_table();
    if (count != static_cast<uint32_t>(expected.size())) {
        LUISA_ERROR("tensor_count {} in \"{}\" is not the bge-m3 v1 tensor set ({} = "
                    "4 emb + 24*12 layer + 4 head)",
                    count, bgm_file.string(), expected.size());
    }

    // --- table + structure + canonical-set validation (fail-closed) ------------
    const uint64_t first_blob = align_up(kBgmHeaderBytes + kBgmEntryBytes * count, kBgmAlign);
    if (first_blob > fsize_u64) {
        LUISA_ERROR("the {}-entry tensor table of \"{}\" does not fit in {} bytes",
                    count, bgm_file.string(), fsize_u64);
    }
    uint64_t running_off = first_blob;
    uint64_t payload = 0ull;
    for (uint32_t i = 0u; i < count; i++) {
        const uint8_t *p = _image + kBgmHeaderBytes + kBgmEntryBytes * i;
        const auto hash = read_le64(p);
        const auto offset = read_le64(p + 8u);
        const auto size_bytes = read_le64(p + 16u);
        const auto rank = read_le32(p + 24u);
        const uint32_t dims[4] = {read_le32(p + 28u), read_le32(p + 32u),
                                  read_le32(p + 36u), read_le32(p + 40u)};
        const auto &e = expected[i];

        // canonical name (via its FNV-1a-64) at its emission-order position
        const auto want_hash = fnv1a_64(e.name);
        if (hash != want_hash) {
            LUISA_ERROR("table entry {} of \"{}\" has name_hash 0x{:016x}, but the v1 "
                        "emission order expects tensor '{}' (0x{:016x}) — the table is "
                        "corrupted or the package is not a bge-m3 model",
                        i, bgm_file.string(), hash, e.name, want_hash);
        }
        // rank / dims / size consistency vs. the hyper-parameter-derived spec
        if (rank != e.rank) {
            LUISA_ERROR("tensor '{}' of \"{}\": rank {} but the bge-m3 v1 spec says {}",
                        e.name, bgm_file.string(), rank, e.rank);
        }
        for (uint32_t d = 0u; d < 4u; d++) {
            if (dims[d] != e.dims[d]) {
                LUISA_ERROR("tensor '{}' of \"{}\": dims [{},{},{},{}] disagree with the "
                            "bge-m3 hyper-parameters (expected [{},{},{},{}])",
                            e.name, bgm_file.string(), dims[0], dims[1], dims[2], dims[3],
                            e.dims[0], e.dims[1], e.dims[2], e.dims[3]);
            }
        }
        uint64_t elems = 1ull;
        for (uint32_t d = 0u; d < rank; d++) {
            elems *= dims[d];
        }
        if (elems * elem_bytes != size_bytes) {
            LUISA_ERROR("tensor '{}' of \"{}\": size_bytes {} != prod(dims) {} * {} ({})",
                        e.name, bgm_file.string(), size_bytes, elems, elem_bytes,
                        package_precision_name(pkg_precision));
        }

        // blob placement: 256-byte aligned, exactly packed, inside the file, in
        // ascending order (guaranteed by the packer; reject anything else loudly)
        if (offset % kBgmAlign != 0ull) {
            LUISA_ERROR("tensor '{}' of \"{}\": blob offset {} is not {}-byte aligned",
                        e.name, bgm_file.string(), offset, kBgmAlign);
        }
        if (offset != running_off) {
            LUISA_ERROR("tensor '{}' of \"{}\": blob offset {} disagrees with the v1 "
                        "exact-packing layout (expected {})",
                        e.name, bgm_file.string(), offset, running_off);
        }
        if (size_bytes > fsize_u64 - offset) {
            LUISA_ERROR("tensor '{}' of \"{}\": blob [{}, {}) runs past the end of the "
                        "{}-byte file",
                        e.name, bgm_file.string(), offset, offset + size_bytes, fsize_u64);
        }
        running_off = align_up(offset + size_bytes, kBgmAlign);
        payload += size_bytes;

        Tensor t;
        if (pkg_precision == PackagePrecision::int8) {
            // v3: the reserved trailing 4 bytes of the entry carry the per-tensor
            // fp32 scale (IEEE-754, little-endian). Reject a scale that is not
            // finite and positive -- fail-closed, since q*scale is the only
            // dequantization the engine ever performs. v1/v2 entries keep these
            // bytes zero and are deliberately not checked for it (that would
            // change the accepted v1/v2 input set for no benefit: the version /
            // precision pair already discriminates the format).
            const auto scale_bits = read_le32(p + kBgmEntryReservedOffset);
            float scale;
            std::memcpy(&scale, &scale_bits, sizeof(scale));
            if (!(std::isfinite(scale) && scale > 0.0f)) {
                LUISA_ERROR("tensor '{}' of \"{}\": the v3 int8 scale {} (0x{:08x}) is not finite and "
                            "positive (fail-closed)",
                            e.name, bgm_file.string(), scale, scale_bits);
            }
            t.int8_data = reinterpret_cast<const int8_t *>(_image + offset);
            t.int8_scale = scale;
        } else if (elem_bytes == 4ull) {
            t.data = reinterpret_cast<const float *>(_image + offset);
        } else {// 256-byte blob offsets keep the uint16 pointer aligned too
            t.half_data = reinterpret_cast<const uint16_t *>(_image + offset);
        }
        t.size_bytes = size_bytes;
        t.rank = rank;
        for (uint32_t d = 0u; d < 4u; d++) {
            t.dims[d] = dims[d];
        }
        _tensors.push_back(t);
        _hashes.push_back(hash);
    }
    if (running_off != fsize_u64) {
        LUISA_ERROR("packed model \"{}\" is {} bytes but the v1 layout computes {} bytes "
                    "(header {} + table {} + {} aligned blobs, payload {}): trailing or "
                    "inserted bytes",
                    bgm_file.string(), fsize_u64, running_off, kBgmHeaderBytes,
                    kBgmEntryBytes * count, count, payload);
    }
}

luisa::unique_ptr<PackageImpl> PackageImpl::create(const std::filesystem::path &bgm_file) {
    auto impl = luisa::make_unique<PackageImpl>();
    impl->_fill(bgm_file);
    return impl;
}

}// namespace

// Package::load is declared against the abstract interface in package.h.
luisa::unique_ptr<Package> Package::load(const std::filesystem::path &bgm_file) {
    return PackageImpl::create(bgm_file);
}

}// namespace luisa::embed
