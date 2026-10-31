// =============================================================================
// pack_model.cpp -- offline weight packer for the bge-m3 sentence-embedding
// engine. Implements the pack half of milestone M2 (embedding_model_plan.md
// §3.3 "Weight packer", §6 M2, §8 review gates).
//
// WHAT IT DOES
//   bge-m3 assets (onnx/model.onnx + onnx/model.onnx_data external fp32 weights,
//   sparse_linear.pt, colbert_linear.pt)
//     -> discover the real XLM-RoBERTa initializer names by graph tracing
//     -> repack (fuse Q/K/V, pre-transpose the .pt head weights to B[K][N],
//        fold embedding constants)
//     -> write one packed fp32 blob: model.bgm  (format spec below)
//   With --check it then reads the weights BACK FROM THE WRITTEN .bgm and runs a
//   naive single-threaded CPU fp32 forward pass of the full 24-layer encoder over
//   the first three corpus sentences (golden token ids from golden_tokens.json),
//   comparing the cosine of each dense vector against golden_dense.bin. The gate
//   is cosine >= 0.9999; any mismatch is a hard error (nonzero exit).
//
// DEPENDENCIES: none but the C++20 standard library. No protobuf, no onnx, no
// python, no LuisaCompute runtime -- the ONNX graph is read with the minimal
// protobuf wire-format reader below, the torch checkpoints with the minimal zip +
// pickle readers below, and the goldens with the minimal JSON reader below.
// Deviation note (CONVENTIONS.md "Style"): this is a host-only tool target that
// links no lc libraries, so luisa:: containers / LUISA_ERROR are not available;
// std::vector / std::string are used inside tools/ (explicitly allowed for
// tools/) and fatal() below plays the role of LUISA_ERROR (message printed in the
// same "[file:line] ..." style, then exit(1)). No exceptions, no RTTI.
//
// =============================================================================
// model.bgm FORMAT -- VERSION 1. THIS COMMENT BLOCK IS THE FORMAT SPECIFICATION;
// Phase B (package.h/.cpp loader, plan §3.3/B3) is written against it.
// All integers little-endian, all fields packed (no implicit padding except the
// explicitly stated 256-byte blob alignment).
// offset  size  field
// ------  ----  -------------------------------------------------------------
// 0 4 magic 'B','G','M','3'
// 4 4 u32 version = 1 (fp32), 2 (half-precision, opt plan §4 Q1/Q2) or
//   3 (per-tensor symmetric int8, INT8 quantization phase 1)
// 8 4 u32 precision = 0 (fp32 little-endian elements; the only v1 mode).
//   Version 2 only: 1 = bf16, 2 = fp16 -- every blob is 2-byte
//   little-endian elements, size_bytes = prod(dims) * 2
//   Version 3 only: 3 = int8 -- every blob is exactly prod(dims)
//   signed int8 bytes (size_bytes = prod(dims)), scale below
//   (v1 must have precision 0; v2 must NOT have precision 0; v3 must
//   have precision 3 and precision 3 requires v3; all loaders reject
//   mismatches loudly)
// 12 4 u32 tensor_count = N
// 16 N*48  tensor table, N entries, in the emission order listed below
// ... blobs: each tensor blob starts at a 256-byte aligned offset
// Tensor table entry (48 bytes):
// u64  name_hash FNV-1a-64 of the canonical name bytes (no NUL):
// offset_basis = 0xcbf29ce484222325, prime = 0x00000100000001b3,
// h = basis; for each byte b: h ^= b; h *= prime;
// u64  offset absolute byte offset of the blob from the start of file
// (always a multiple of 256)
// u64  size_bytes == prod(dims[0..rank-1]) * elem_size (4 for fp32/v1,
// 2 for bf16/fp16/v2)
// u32  rank number of meaningful dims (1 or 2 in v1/v2/v3)
// u32  dims[4] outermost-first; entries beyond rank are 0
// u32  reserved: zero in v1/v2; in v3 the per-tensor INT8 scale as an
// IEEE-754 fp32 (positive, finite) -- see VERSION 3 below
// Blobs are row-major (C-contiguous) tensors. Padding bytes between blobs are
// zero. The file size is align_up(16 + 48*N, 256) + sum(align_up(size_i, 256)).
// VERSION 2 ELEMENT ENCODING (pack_model --precision bf16|fp16; everything
// above -- names, order, dims, alignment -- is identical to v1, and the v1
// fp32 output is byte-identical to the historical packer):
// * bf16: round-to-nearest-even truncation of the fp32 bits (ties-to-even on
//   the discarded low 16 bits); NaN stays NaN, +-0/inf exact.
// * fp16: IEEE binary32 -> binary16 round-to-nearest-even with subnormal and
//   overflow-to-inf handling; exact for every value a bf16 blob could hold
//   within range, but VALUES OUTSIDE +-65504 (|w| > 65504) become inf -- the
//   packer reports the observed max |w| of the model so the range is audited
//   at pack time (bge-m3 weights are far inside fp16 range; see the run log).
// * Runtime: the engine uploads v2 GEMM weight blobs verbatim into packed
//   Buffer<float> device buffers (two LE elements per word) and converts
//   bf16/fp16 -> fp32 at the kernel B-tile load (FMA accumulation stays fp32;
//   see kernels_gemm.h "WEIGHT FORMAT"). Other tensors are up-converted
//   host-side once at upload. The conversion is exact in both directions.
//
// VERSION 3 ELEMENT ENCODING (pack_model --precision int8; names, order, dims,
// alignment and the table layout are identical to v1/v2):
// * Per-tensor symmetric quantization. For each tensor, over the SAME fp32
//   values v1 would write:
//     absmax = max |x|                      (a nonfinite x is a fatal error)
//     scale  = absmax / 127                 (exactly 1.0 for an all-zero tensor)
//     code   = clamp(RNE(x / scale), -127, +127)   (RNE = ties-to-even)
//     dequantized value = code * scale
//   -128 is never emitted, so the code range is symmetric. The blob is exactly
//   prod(dims) signed int8 bytes with no per-blob header; the positive fp32
//   scale is written into the entry's reserved trailing 4 bytes (offset 44).
// The bge-m3 weights are all small (the pack run audits max |w| = 1.85 over the
// whole model), so a per-tensor scale keeps ~7 significant bits of every weight.
// * Runtime: the engine either uploads the codes and dequantizes at the
//   weight-load point (q * scale, fp32 accumulation) or calls the shared
//   host helper once. The math is mirrored bit-for-bit in
//   examples/embedding/embed/quantization.h (that header is host-only engine
//   code and this tool links no lc libraries, so the two copies must agree).
// * --check dequantizes every blob on read-back, i.e. the internal fp32 forward
//   measures the true int8 quantization effect end to end.
//   u32  dims[4]      outermost-first; entries beyond rank are 0
//
// Blobs are row-major (C-contiguous) tensors. Padding bytes between blobs are
// zero. The file size is align_up(16 + 48*N, 256) + sum(align_up(size_i, 256)).
//
// GEMM / LAYOUT CONVENTION (plan §3.3, §4 kernel #3): every linear weight is
// stored as B[K][N] so that C[M,N] = A[M,K] @ B[K,N] reads B coalesced along N.
// torch Linear stores weight as [out,in]; the opset-11 ONNX export of this model
// already stores the graph's MatMul initializers transposed ([in,out]) -- verified
// numerically against pytorch_model.bin -- so they are copied as-is, while the
// .pt head weights (kept in [out,in]) ARE transposed by the packer. All rank-2
// tensors in this file therefore have dims = {K, N} and dims[1] is the contiguous
// (coalesced) dimension.
//
// CANONICAL NAMES (rank, dims, emission order). i = 0..23, hidden H = 1024,
// heads 16, head_dim 64, intermediate 4096, vocab 250002, max_pos 8194.
//
//   "emb.word"            [250002,1024]  word embeddings, token-type row 0 ALREADY
//                                        FOLDED IN (see EMBEDDING FOLDING below)
//   "emb.pos"             [8194,1024]    absolute position embeddings, raw table
//   "emb.ln.gamma"        [1024]         embedding LayerNorm weight
//   "emb.ln.beta"         [1024]         embedding LayerNorm bias
//   "layer{i}.qkv"        [1024,3072]    fused Q,K,V weight; column blocks in row
//                                        order Q=[0,1024), K=[1024,2048),
//                                        V=[2048,3072); within a block the column
//                                        index is head-major: n = h*64 + d
//   "layer{i}.qkv.bias"   [3072]         fused Q,K,V bias, same block order
//   "layer{i}.attn.out"   [1024,1024]    attention output projection B[K][N]
//   "layer{i}.attn.out.bias" [1024]
//   "layer{i}.ln1.gamma"  [1024]         post-attention LayerNorm weight
//   "layer{i}.ln1.beta"   [1024]         post-attention LayerNorm bias
//   "layer{i}.ffn.up"     [1024,4096]    intermediate (up) projection B[K][N]
//   "layer{i}.ffn.up.bias"   [4096]
//   "layer{i}.ffn.down"   [4096,1024]    output (down) projection B[K][N]
//   "layer{i}.ffn.down.bias" [1024]
//   "layer{i}.ln2.gamma"  [1024]         post-FFN LayerNorm weight
//   "layer{i}.ln2.beta"   [1024]         post-FFN LayerNorm bias
//   "head.sparse.w"       [1024,1]       sparse_linear weight, transposed to B[K][N]
//   "head.sparse.b"       [1]            sparse_linear bias
//   "head.colbert.w"      [1024,1024]    colbert_linear weight, transposed to B[K][N]
//   "head.colbert.b"      [1024]         colbert_linear bias
//
// N = 4 + 24 * 12 + 4 = 296 tensors, ~2.27 GB fp32. Emission order is exactly the
// order above (emb.*, layer0.* ... layer23.*, head.*) so a loader may also walk
// the table positionally, but lookup by name_hash is the intended access path.
//
// RUNTIME CONSTANTS THE PACKER DOES NOT STORE (fixed by the bge-m3 architecture,
// verified against config.json in main()/run_pack when the file is
// present):
//   num_layers = 24, hidden = 1024, heads = 16, head_dim = 64, intermediate = 4096,
//   layer_norm_eps = 1e-5 (fp32), attention score scale = 1/8 (the ONNX graph
//   divides the QK^T logits by 8.0 = sqrt(head_dim)), GELU = erf variant
//   0.5*x*(1+erf(x/sqrt(2))) (24 Erf nodes), padding mask is additive with
//   (1-mask)*(-3.4028234663852886e38) before the softmax, dense output = row 0
//   (CLS) of the last layer's ln2 output, L2-normalized (denominator clipped at
//   1e-12), pad token id = 1.
//
// EMBEDDING FOLDING / POSITION NOTE (deviation from plan §3.3, evidence-based):
//   * The plan assumed token_type_embeddings is all-zero (type_vocab_size = 1) and
//     could be dropped. It is NOT zero in this checkpoint (max|w| ~ 0.2). The graph
//     gathers it with a constant index 0 for every token, so the packer folds
//     row 0 into "emb.word" (word'[v] = word[v] + type[0]) -- exact and it keeps
//     the graph's addition order ((word + type) + pos). No separate tensor.
//   * Position ids are NOT plain 0..S-1: the graph builds them from the padding
//     mask, pos = cumsum(id != 1) * (id != 1) + 1, i.e. for an unpadded sequence
//     the first token uses row 2 of "emb.pos" (XLM-R's +2 offset). Kernels must
//     reproduce that index computation.
//   * The retrieval heads are stored float16 inside their .pt checkpoints; the
//     packer converts them to fp32.
//
// CLI
// pack_model <model_dir> <out.bgm> [--golden-dir <dir>] [--check] [--dump]
// [--precision fp32|bf16|fp16|int8]
// <model_dir> directory containing onnx/model.onnx, onnx/model.onnx_data,
// sparse_linear.pt, colbert_linear.pt and config.json (the
// config is read for cross-checking only)
// --golden-dir directory holding golden_tokens.json and golden_dense.bin
// --check after packing, reopen the .bgm, run the internal naive CPU
// fp32 forward over the first three corpus sentences and
// compare against golden_dense.bin (requires --golden-dir).
// Gate: cosine >= 0.9999 for an fp32 package, >= 0.999 for a
// half-precision (v2) one, >= 0.99 for an int8 (v3) one (opt
// plan §4 parity gates; the int8 bound is the measured
// per-tensor-quantization level of the dev model -- see
// run_check)
// --dump print the canonical-name -> source-initializer mapping table
// --precision element format of the OUTPUT file (default fp32 = the
// historical byte-identical v1 output). bf16/fp16 write
// BGM3 v2 (see VERSION 2 above): every blob is
// 2-byte RNE-quantized elements. int8 writes BGM3 v3
// (see VERSION 3 above): every blob is prod(dims) signed
// int8 codes plus one fp32 scale in the entry's reserved
// bytes. The internal --check forward dequantizes on
// read-back, so it measures the true quantization effect
// end to end.
//
// Exit code: 0 on success, 1 on any fatal error (missing or unexpected
// initializer, truncated data, alignment mismatch, parity mismatch, ...).
// =============================================================================

#include <algorithm>
#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

namespace pack_model_detail {
// ---------------------------------------------------------------------------
// small utilities
// ---------------------------------------------------------------------------

// LUISA_ERROR-style fatal reporter (this target links no lc libraries).
[[noreturn]] void fatal(const char *file, int line, const char *fmt, ...) {
    char buf[2048];
    va_list ap;
    va_start(ap, fmt);
    std::vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    std::fprintf(stderr, "[%s:%d] %s\n", file, line, buf);
    std::fflush(stderr);
    std::exit(1);
}
#define PACK_FATAL(...) ::pack_model_detail::fatal(__FILE__, __LINE__, __VA_ARGS__)
#define PACK_CHECK(cond, ...)                 \
    do {                                      \
        if (!(cond)) PACK_FATAL(__VA_ARGS__); \
    } while (0)

void info(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    std::vfprintf(stdout, fmt, ap);
    va_end(ap);
    std::fflush(stdout);
}

// little-endian integer reading
inline uint16_t read_le16(const uint8_t *p) { return static_cast<uint16_t>(p[0] | (p[1] << 8)); }
inline uint32_t read_le32(const uint8_t *p) {
    return static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8) | (static_cast<uint32_t>(p[2]) << 16) | (static_cast<uint32_t>(p[3]) << 24);
}
inline int32_t read_le32s(const uint8_t *p) { return static_cast<int32_t>(read_le32(p)); }
inline uint64_t read_le64(const uint8_t *p) {
    return static_cast<uint64_t>(read_le32(p)) | (static_cast<uint64_t>(read_le32(p + 4)) << 32);
}

uint64_t fnv1a_64(const char *s, size_t n) {
    uint64_t h = 0xcbf29ce484222325ull;
    for (size_t i = 0; i < n; i++) {
        h ^= static_cast<uint8_t>(s[i]);
        h *= 0x00000100000001b3ull;
    }
    return h;
}
inline uint64_t fnv1a_64(const std::string &s) { return fnv1a_64(s.data(), s.size()); }

inline uint64_t align_up(uint64_t v, uint64_t a) { return (v + a - 1) / a * a; }

// IEEE754 binary16 -> float32 (the retrieval heads are stored fp16 in .pt)
float half_to_float(uint16_t h) {
    const uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
    const uint32_t exp = (h >> 10) & 0x1Fu;
    const uint32_t mant = h & 0x3FFu;
    uint32_t bits = 0;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {// subnormal half -> normalized float
            int e = -1;
            uint32_t m = mant;
            do {
                e++;
                m <<= 1;
            } while ((m & 0x400u) == 0);
            const uint32_t fexp = static_cast<uint32_t>(127 - 15 - e);
            bits = sign | (fexp << 23) | ((m & 0x3FFu) << 13);
        }
    } else if (exp == 0x1Fu) {
        bits = sign | 0x7F800000u | (mant << 13);
    } else {
        bits = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
    }
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}

// ---------------------------------------------------------------------------
// 64-bit file access (the external data file and the packed model are >2 GB,
// past the 32-bit seek range of std::fseek on Windows)
// ---------------------------------------------------------------------------

void seek64(std::FILE *f, uint64_t pos) {
#if defined(_WIN32)
    PACK_CHECK(_fseeki64(f, static_cast<long long>(pos), SEEK_SET) == 0, "seek64 failed (%llu)", static_cast<unsigned long long>(pos));
#else
    PACK_CHECK(fseeko(f, static_cast<off_t>(pos), SEEK_SET) == 0, "seek64 failed (%llu)", static_cast<unsigned long long>(pos));
#endif
}
uint64_t tell64(std::FILE *f) {
#if defined(_WIN32)
    const long long p = _ftelli64(f);
#else
    const long long p = ftello(f);
#endif
    PACK_CHECK(p >= 0, "tell64 failed");
    return static_cast<uint64_t>(p);
}
uint64_t file_size64(std::FILE *f) {
    seek64(f, 0);
#if defined(_WIN32)
    PACK_CHECK(_fseeki64(f, 0, SEEK_END) == 0, "size64 failed");
    const long long sz = _ftelli64(f);
#else
    PACK_CHECK(fseeko(f, 0, SEEK_END) == 0, "size64 failed");
    const long long sz = ftello(f);
#endif
    PACK_CHECK(sz >= 0, "size64 failed");
    return static_cast<uint64_t>(sz);
}
uint64_t file_size64_of(const std::string &path) {
    std::FILE *f = std::fopen(path.c_str(), "rb");
    PACK_CHECK(f, "cannot open file '%s'", path.c_str());
    const auto s = file_size64(f);
    std::fclose(f);
    return s;
}

bool ends_with(const std::string &s, const char *suf) {
    const size_t n = std::strlen(suf);
    return s.size() >= n && s.compare(s.size() - n, n, suf) == 0;
}
bool contains(const std::string &s, const char *sub) { return s.find(sub) != std::string::npos; }

std::string read_file_text(const std::string &path, bool optional_ok = false) {
    std::FILE *f = std::fopen(path.c_str(), "rb");
    if (!f) {
        if (optional_ok) return {};
        PACK_FATAL("cannot open file '%s'", path.c_str());
    }
    const uint64_t sz = file_size64(f);
    seek64(f, 0);
    std::string out;
    out.resize(static_cast<size_t>(sz));
    if (sz > 0 && std::fread(out.data(), 1, static_cast<size_t>(sz), f) != static_cast<size_t>(sz)) {
        std::fclose(f);
        PACK_FATAL("short read on '%s'", path.c_str());
    }
    std::fclose(f);
    return out;
}

std::string path_dir(const std::string &p) {
    const auto sep = p.find_last_of("/\\");
    return sep == std::string::npos ? std::string(".") : p.substr(0, sep);
}
std::string path_join(const std::string &a, const std::string &b) {
    if (a.empty()) return b;
    const char c = a.back();
    return (c == '/' || c == '\\') ? a + b : a + "/" + b;
}
bool path_is_absolute(const std::string &p) {
    return !p.empty() && (p[0] == '/' || p[0] == '\\' || (p.size() > 1 && p[1] == ':'));
}

// ---------------------------------------------------------------------------
// minimal protobuf wire-format reader (plan §3.3: ~200 lines, no protobuf dep)
// ---------------------------------------------------------------------------
// Wire types: 0 varint, 1 fixed64, 2 length-delimited, 5 fixed32. Groups (3/4)
// are not used by the ONNX schema and are rejected.
struct pb_view {
    const uint8_t *data = nullptr;
    size_t size = 0;
};

class pb_reader {
public:
    pb_reader(const uint8_t *begin, const uint8_t *end) : _p(begin), _end(end) {}

    bool next() {
        if (_p >= _end) return false;
        uint64_t tag = 0;
        if (!varint(tag)) PACK_FATAL("protobuf: truncated tag");
        _field = static_cast<uint32_t>(tag >> 3);
        _wire = static_cast<uint32_t>(tag & 7);
        switch (_wire) {
            case 0: {
                if (!varint(_varint)) PACK_FATAL("protobuf: truncated varint");
                _payload = {};
                return true;
            }
            case 1: {
                if (_end - _p < 8) PACK_FATAL("protobuf: truncated fixed64");
                _varint = read_le64(_p);
                _payload = {_p, 8};
                _p += 8;
                return true;
            }
            case 2: {
                uint64_t len = 0;
                if (!varint(len) || len > static_cast<uint64_t>(_end - _p)) PACK_FATAL("protobuf: truncated length-delimited field (len=%llu)", static_cast<unsigned long long>(len));
                _payload = {_p, static_cast<size_t>(len)};
                _varint = len;
                _p += len;
                return true;
            }
            case 5: {
                if (_end - _p < 4) PACK_FATAL("protobuf: truncated fixed32");
                _varint = read_le32(_p);
                _payload = {_p, 4};
                _p += 4;
                return true;
            }
            default:
                PACK_FATAL("protobuf: unsupported wire type %u (group?)", _wire);
        }
    }

    uint32_t field() const { return _field; }
    uint32_t wire() const { return _wire; }
    uint64_t varint_value() const { return _varint; }
    pb_view bytes_value() const {
        PACK_CHECK(_wire == 2, "protobuf: field %u is not length-delimited", _field);
        return _payload;
    }
    std::string string_value() const {
        const auto v = bytes_value();
        return {reinterpret_cast<const char *>(v.data), v.size};
    }

    // repeated packed varint (used by TensorProto.dims) / packed float (float_data)
    template<class Fn>
    void packed_varints(Fn fn) const {
        pb_view v = bytes_value();
        pb_reader r{v.data, v.data + v.size};
        while (r._p < r._end) {
            uint64_t x = 0;
            if (!r.varint(x)) PACK_FATAL("protobuf: truncated packed varints");
            fn(x);
        }
    }

private:
    bool varint(uint64_t &out) {
        out = 0;
        int shift = 0;
        while (_p < _end) {
            const uint8_t b = *_p++;
            out |= static_cast<uint64_t>(b & 0x7F) << shift;
            if ((b & 0x80) == 0) return true;
            shift += 7;
            if (shift > 63) return false;
        }
        return false;
    }

    const uint8_t *_p;
    const uint8_t *_end;
    uint32_t _field = 0;
    uint32_t _wire = 0;
    uint64_t _varint = 0;
    pb_view _payload{};
};

// ONNX subset: ModelProto{graph=7}, GraphProto{node=1, initializer=5},
// NodeProto{input=1, output=2, name=3, op_type=4},
// TensorProto{dims=1, data_type=2, float_data=4, name=8, raw_data=9,
//             external_data=13, data_location=14},
// StringStringEntryProto{key=1, value=2}.
struct onnx_node {
    std::string op_type;
    std::string name;
    std::vector<std::string> inputs;
    std::vector<std::string> outputs;
};

struct onnx_tensor {
    std::string name;
    std::vector<int64_t> dims;
    uint32_t data_type = 0;// 1 == FLOAT
    uint32_t data_location = 0;
    std::string ext_location;// "model.onnx_data"
    uint64_t ext_offset = 0;
    uint64_t ext_length = 0;
    std::vector<float> inline_floats;// float_data / raw_data fallback
    bool used = false;

    uint64_t num_elements() const {
        uint64_t n = 1;
        for (auto d : dims) n *= static_cast<uint64_t>(d);
        return n;
    }
};

struct onnx_model {
    std::vector<onnx_node> nodes;
    std::vector<onnx_tensor> initializers;
    std::unordered_map<std::string, size_t> init_index;// name -> initializers[]
};

onnx_model parse_onnx(const std::string &blob) {
    onnx_model model;
    pb_reader top{reinterpret_cast<const uint8_t *>(blob.data()), reinterpret_cast<const uint8_t *>(blob.data()) + blob.size()};
    while (top.next()) {
        if (top.field() != 7) continue;// everything but GraphProto is irrelevant
        pb_reader graph = {top.bytes_value().data, top.bytes_value().data + top.bytes_value().size};
        while (graph.next()) {
            if (graph.field() == 1) {// NodeProto
                const auto nv = graph.bytes_value();
                pb_reader nr{nv.data, nv.data + nv.size};
                onnx_node node;
                while (nr.next()) {
                    switch (nr.field()) {
                        case 1: node.inputs.push_back(nr.string_value()); break;
                        case 2: node.outputs.push_back(nr.string_value()); break;
                        case 3: node.name = nr.string_value(); break;
                        case 4: node.op_type = nr.string_value(); break;
                        default: break;// attributes unused by the packer
                    }
                }
                model.nodes.push_back(std::move(node));
            } else if (graph.field() == 5) {// TensorProto
                const auto tv = graph.bytes_value();
                pb_reader tr{tv.data, tv.data + tv.size};
                onnx_tensor t;
                std::vector<std::pair<std::string, std::string>> ext;
                while (tr.next()) {
                    switch (tr.field()) {
                        case 1:
                            if (tr.wire() == 2)
                                tr.packed_varints([&](uint64_t x) { t.dims.push_back(static_cast<int64_t>(x)); });
                            else
                                t.dims.push_back(static_cast<int64_t>(tr.varint_value()));
                            break;
                        case 2: t.data_type = static_cast<uint32_t>(tr.varint_value()); break;
                        case 4: {
                            const auto v = tr.bytes_value();// packed float_data
                            for (size_t o = 0; o + 4 <= v.size; o += 4) {
                                uint32_t bits = read_le32(v.data + o);
                                float f;
                                std::memcpy(&f, &bits, 4);
                                t.inline_floats.push_back(f);
                            }
                            break;
                        }
                        case 8: t.name = tr.string_value(); break;
                        case 9: {
                            const auto v = tr.bytes_value();
                            for (size_t o = 0; o + 4 <= v.size; o += 4) {
                                uint32_t bits = read_le32(v.data + o);
                                float f;
                                std::memcpy(&f, &bits, 4);
                                t.inline_floats.push_back(f);
                            }
                            break;
                        }
                        case 13: {
                            const auto ev = tr.bytes_value();
                            pb_reader er{ev.data, ev.data + ev.size};
                            std::pair<std::string, std::string> e;
                            while (er.next()) {
                                if (er.field() == 1)
                                    e.first = er.string_value();
                                else if (er.field() == 2)
                                    e.second = er.string_value();
                            }
                            ext.push_back(std::move(e));
                            break;
                        }
                        case 14: t.data_location = static_cast<uint32_t>(tr.varint_value()); break;
                        default: break;
                    }
                }
                for (const auto &e : ext) {
                    if (e.first == "location")
                        t.ext_location = e.second;
                    else if (e.first == "offset")
                        t.ext_offset = std::strtoull(e.second.c_str(), nullptr, 10);
                    else if (e.first == "length")
                        t.ext_length = std::strtoull(e.second.c_str(), nullptr, 10);
                }
                PACK_CHECK(!t.name.empty(), "protobuf: TensorProto without a name");
                model.init_index[t.name] = model.initializers.size();
                model.initializers.push_back(std::move(t));
            }
        }
        break;// only one GraphProto
    }
    return model;
}

}// namespace pack_model_detail

namespace pack_model_detail {
// ---------------------------------------------------------------------------
// minimal zip reader (plan §3.3: torch checkpoints are zip archives)
// ---------------------------------------------------------------------------
// Only STORED (method 0) entries are supported: torch.save() writes the tensor
// archives uncompressed, and anything else is a hard error rather than silent
// garbage.

struct zip_entry {
    std::string name;
    uint16_t method = 0;
    uint64_t comp_size = 0;
    uint64_t uncomp_size = 0;
    uint64_t local_header = 0;
};

struct zip_archive {
    std::vector<uint8_t> blob;
    std::vector<zip_entry> entries;

    const zip_entry *find(std::string_view suffix) const {
        for (const auto &e : entries) {
            if (e.name.size() >= suffix.size() && e.name.compare(e.name.size() - suffix.size(), suffix.size(), suffix) == 0) return &e;
        }
        return nullptr;
    }
    std::string member_name(std::string_view suffix) const {
        const auto *e = find(suffix);
        return e ? e->name : std::string{};
    }
};

zip_archive zip_open(const std::string &path) {
    zip_archive z;
    z.blob.resize(static_cast<size_t>(file_size64_of(path)));
    {
        std::FILE *f = std::fopen(path.c_str(), "rb");
        PACK_CHECK(f, "cannot open zip '%s'", path.c_str());
        const size_t n = std::fread(z.blob.data(), 1, z.blob.size(), f);
        std::fclose(f);
        PACK_CHECK(n == z.blob.size(), "short read on zip '%s'", path.c_str());
    }
    PACK_CHECK(z.blob.size() >= 22, "'%s' is too small to be a zip", path.c_str());
    // End of central directory: signature 0x06054b50, within the last 64 KiB.
    size_t eocd = 0;
    bool found = false;
    const size_t lo = z.blob.size() > 65557 ? z.blob.size() - 65557 : 0;
    for (size_t i = z.blob.size() - 22;;) {
        if (read_le32(z.blob.data() + i) == 0x06054b50u) {
            eocd = i;
            found = true;
            break;
        }
        if (i == lo) break;
        --i;
    }
    PACK_CHECK(found, "'%s': end-of-central-directory record not found", path.c_str());
    const uint16_t total = read_le16(z.blob.data() + eocd + 10);
    const uint32_t cd_off = read_le32(z.blob.data() + eocd + 16);
    PACK_CHECK(cd_off < z.blob.size(), "'%s': corrupt central directory offset", path.c_str());
    size_t p = cd_off;
    for (uint16_t i = 0; i < total; i++) {
        PACK_CHECK(p + 46 <= z.blob.size() && read_le32(z.blob.data() + p) == 0x02014b50u,
                   "'%s': corrupt central directory entry %u", path.c_str(), i);
        zip_entry e;
        e.method = read_le16(z.blob.data() + p + 10);
        e.comp_size = read_le32(z.blob.data() + p + 20);
        e.uncomp_size = read_le32(z.blob.data() + p + 24);
        const uint16_t name_len = read_le16(z.blob.data() + p + 28);
        const uint16_t extra_len = read_le16(z.blob.data() + p + 30);
        const uint16_t cmt_len = read_le16(z.blob.data() + p + 32);
        e.local_header = read_le32(z.blob.data() + p + 42);
        PACK_CHECK(p + 46 + name_len <= z.blob.size(), "'%s': central directory name overflow", path.c_str());
        e.name.assign(reinterpret_cast<const char *>(z.blob.data() + p + 46), name_len);
        z.entries.push_back(e);
        p += 46 + name_len + extra_len + cmt_len;
    }
    return z;
}

std::vector<uint8_t> zip_read(const zip_archive &z, const zip_entry &e, const std::string &path) {
    PACK_CHECK(e.method == 0, "'%s' entry '%s': compressed (method %u) entries are not supported",
               path.c_str(), e.name.c_str(), e.method);
    const uint8_t *lh = z.blob.data() + e.local_header;
    PACK_CHECK(static_cast<size_t>(lh - z.blob.data()) + 30 <= z.blob.size() && read_le32(lh) == 0x04034b50u,
               "'%s': corrupt local header for '%s'", path.c_str(), e.name.c_str());
    const uint16_t name_len = read_le16(lh + 26);
    const uint16_t extra_len = read_le16(lh + 28);
    const size_t data = static_cast<size_t>(lh - z.blob.data()) + 30 + name_len + extra_len;
    // Prefer the central-directory size; a zero local size with a data descriptor
    // is legal, and torch writes the sizes in both places.
    const uint64_t size = e.comp_size != 0 ? e.comp_size : read_le32(lh + 18);
    PACK_CHECK(data + size <= z.blob.size(), "'%s': entry '%s' extends past the archive",
               path.c_str(), e.name.c_str());
    return {z.blob.data() + data, z.blob.data() + data + size};
}

// ---------------------------------------------------------------------------
// minimal pickle reader (protocol 2, the fixed opcode stream torch uses for
// state_dict checkpoints). Only what sparse_linear.pt / colbert_linear.pt use:
// OrderedDict + _rebuild_tensor_v2 + persistent ('storage') references.
// ---------------------------------------------------------------------------

struct pickle_tensor {
    std::string storage_key; // "0" / "1" -> zip member <dir>/data/<key>
    std::string storage_type;// "HalfStorage" / "FloatStorage" / ...
    int64_t offset = 0;
    std::vector<int64_t> sizes;
};

class pickle_reader {
public:
    pickle_reader(const uint8_t *begin, const uint8_t *end, std::string archive_name)
        : _begin(begin), _end(end), _p(begin), _name(std::move(archive_name)) {}

    // Returns the dict's tensor entries for the given keys ("weight", "bias").
    void load_state_dict(std::unordered_map<std::string, pickle_tensor> &out) {
        run();
        PACK_CHECK(_root.kind == v_kind::dict, "%s: pickle root is not a dict", _name.c_str());
        for (const auto &[key, val] : _root.map) {
            PACK_CHECK(val.kind == v_kind::tensor, "%s: entry '%s' is not a tensor", _name.c_str(), key.c_str());
            out[key] = val.tensor;
        }
    }

private:
    enum class v_kind { none,
                        boolean,
                        integer,
                        string,
                        mark,
                        tuple,
                        dict,
                        global,
                        storage,
                        tensor,
                        list };
    struct map_entry;
    struct value {
        value() noexcept;
        value(v_kind kind) noexcept;
        ~value() noexcept;
        value(const value &);
        value(value &&) noexcept;
        value &operator=(const value &);
        value &operator=(value &&) noexcept;

        v_kind kind = v_kind::none;
        bool b = false;
        int64_t i = 0;
        std::string s;         // global id / storage key
        std::vector<value> seq;// tuple, list, dict keys (values are implicit)
        std::vector<map_entry> map;
        pickle_tensor tensor;
        std::string storage_type;// storage value
    };
    // vector supports an incomplete element type; pair's constraints can
    // instantiate value's special members before the recursive pair is complete.
    struct map_entry {
        std::string first;
        value second;
    };

    [[noreturn]] void fail(const char *what) { PACK_FATAL("%s: pickle error: %s", _name.c_str(), what); }

    uint8_t byte() {
        PACK_CHECK(_p < _end, "%s: pickle truncated", _name.c_str());
        return *_p++;
    }
    int64_t u16() {
        need(2);
        const int64_t v = read_le16(_p);
        _p += 2;
        return v;
    }
    int64_t i32() {
        need(4);
        const int64_t v = read_le32s(_p);
        _p += 4;
        return v;
    }
    uint64_t u64le() {
        uint64_t v = 0;
        for (int k = 0; k < 8; k++) v |= static_cast<uint64_t>(byte()) << (8 * k);
        return v;
    }
    void need(size_t n) { PACK_CHECK(static_cast<size_t>(_end - _p) >= n, "%s: pickle truncated", _name.c_str()); }
    std::string line() {
        std::string s;
        for (;;) {
            PACK_CHECK(_p < _end, "%s: unterminated pickle newline string", _name.c_str());
            const char c = static_cast<char>(*(_p++));
            if (c == '\n') break;
            s.push_back(c);
        }
        return s;
    }
    std::string bytes_n(size_t n) {
        need(n);
        std::string s(reinterpret_cast<const char *>(_p), n);
        _p += n;
        return s;
    }
    static int64_t le_signed(const std::string &raw) {
        int64_t v = 0;
        for (size_t k = 0; k < raw.size(); k++) v |= static_cast<int64_t>(static_cast<uint8_t>(raw[k])) << (8 * k);
        if (!raw.empty() && (static_cast<uint8_t>(raw.back()) & 0x80) != 0 && raw.size() < 8) {
            v |= -(static_cast<int64_t>(1) << (8 * raw.size()));// sign extend
        }
        return v;
    }

    value pop() {
        PACK_CHECK(!_stack.empty(), "%s: pickle stack underflow", _name.c_str());
        value v = std::move(_stack.back());
        _stack.pop_back();
        return v;
    }
    size_t mark_pos() {
        for (size_t n = _stack.size(); n-- > 0;) {
            if (_stack[n].kind == v_kind::mark) return n;
        }
        fail("mark not found");
    }

    void run() {
        bool done = false;
        while (!done && _p < _end) {
            const uint8_t op = byte();
            switch (op) {
                case 0x80: byte(); break;                          // PROTO
                case 0x91: u64le(); break;                         // FRAME (protocol 5)
                case '(': _stack.push_back({v_kind::mark}); break; // MARK
                case ')': _stack.push_back({v_kind::tuple}); break;// EMPTY_TUPLE
                case ']': _stack.push_back({v_kind::list}); break; // EMPTY_LIST
                case '}': _stack.push_back({v_kind::dict}); break; // EMPTY_DICT
                case 'N': _stack.push_back({v_kind::none}); break; // NONE
                case 0x88: {
                    value v;
                    v.kind = v_kind::boolean;
                    v.b = true;
                    _stack.push_back(std::move(v));
                    break;
                }// NEWTRUE
                case 0x89: {
                    value v;
                    v.kind = v_kind::boolean;
                    v.b = false;
                    _stack.push_back(std::move(v));
                    break;
                }// NEWFALSE
                case 'K': {
                    value v;
                    v.kind = v_kind::integer;
                    v.i = byte();
                    _stack.push_back(std::move(v));
                    break;
                }// BININT1
                case 'M': {
                    value v;
                    v.kind = v_kind::integer;
                    v.i = u16();
                    _stack.push_back(std::move(v));
                    break;
                }// BININT2
                case 'J': {
                    value v;
                    v.kind = v_kind::integer;
                    v.i = i32();
                    _stack.push_back(std::move(v));
                    break;
                }// BININT
                case 0x8a: {
                    value v;
                    v.kind = v_kind::integer;
                    v.i = le_signed(bytes_n(byte()));
                    _stack.push_back(std::move(v));
                    break;
                }// LONG1
                case 0x8b: {
                    value v;
                    v.kind = v_kind::integer;
                    v.i = le_signed(bytes_n(static_cast<size_t>(i32())));
                    _stack.push_back(std::move(v));
                    break;
                }// LONG4
                case 'L': {// LONG (ascii)
                    std::string t = line();
                    if (!t.empty() && (t.back() == 'L' || t.back() == 'l')) t.pop_back();
                    value v;
                    v.kind = v_kind::integer;
                    v.i = std::strtoll(t.c_str(), nullptr, 10);
                    _stack.push_back(std::move(v));
                    break;
                }
                case 'I': {// INT (ascii)
                    const std::string t = line();
                    value v;
                    v.kind = v_kind::integer;
                    if (t == "01")
                        v.b = true, v.kind = v_kind::boolean;
                    else if (t == "00")
                        v.b = false, v.kind = v_kind::boolean;
                    else
                        v.i = std::strtoll(t.c_str(), nullptr, 10);
                    _stack.push_back(std::move(v));
                    break;
                }
                case 'F': {// FLOAT (ascii) -- only used for values we do not need
                    const std::string t = line();
                    value v;
                    v.kind = v_kind::integer;
                    v.i = static_cast<int64_t>(std::strtod(t.c_str(), nullptr));// approx; never used
                    _stack.push_back(std::move(v));
                    break;
                }
                case 'X': {// BINUNICODE
                    value v;
                    v.kind = v_kind::string;
                    v.s = bytes_n(static_cast<size_t>(i32()));
                    _stack.push_back(std::move(v));
                    break;
                }
                case 0x8c: {// SHORT_BINUNICODE
                    value v;
                    v.kind = v_kind::string;
                    v.s = bytes_n(byte());
                    _stack.push_back(std::move(v));
                    break;
                }
                case 'V': {// UNICODE (ascii)
                    value v;
                    v.kind = v_kind::string;
                    v.s = line();
                    _stack.push_back(std::move(v));
                    break;
                }
                case 'c': {// GLOBAL
                    const std::string mod = line();
                    const std::string nm = line();
                    value v;
                    v.kind = v_kind::global;
                    v.s = mod + " " + nm;
                    _stack.push_back(std::move(v));
                    break;
                }
                case 'q': {// BINPUT
                    const uint8_t idx = byte();
                    memo_put(idx, _stack.empty() ? value{} : _stack.back());
                    break;
                }
                case 0x8e: {// SHORT_BINPUT
                    const uint8_t idx = byte();
                    memo_put(idx, _stack.empty() ? value{} : _stack.back());
                    break;
                }
                case 'h': {// BINGET
                    _stack.push_back(memo_get(byte()));
                    break;
                }
                case 0x8f: {// SHORT_BINGET
                    _stack.push_back(memo_get(byte()));
                    break;
                }
                case 't':// TUPLE
                {
                    const size_t m = mark_pos();
                    value v;
                    v.kind = v_kind::tuple;
                    for (size_t i = m + 1; i < _stack.size(); i++) v.seq.push_back(_stack[i]);
                    _stack.resize(m);
                    _stack.push_back(std::move(v));
                    break;
                }
                case 0x85: {// TUPLE1
                    value v;
                    v.kind = v_kind::tuple;
                    v.seq.push_back(pop());
                    _stack.push_back(std::move(v));
                    break;
                }
                case 0x86: {// TUPLE2
                    auto b = pop();
                    auto a = pop();
                    value v;
                    v.kind = v_kind::tuple;
                    v.seq.push_back(std::move(a));
                    v.seq.push_back(std::move(b));
                    _stack.push_back(std::move(v));
                    break;
                }
                case 0x87: {// TUPLE3
                    auto c = pop();
                    auto b = pop();
                    auto a = pop();
                    value v;
                    v.kind = v_kind::tuple;
                    v.seq.push_back(std::move(a));
                    v.seq.push_back(std::move(b));
                    v.seq.push_back(std::move(c));
                    _stack.push_back(std::move(v));
                    break;
                }
                case 'l': {// LIST
                    const size_t m = mark_pos();
                    value v;
                    v.kind = v_kind::list;
                    for (size_t i = m + 1; i < _stack.size(); i++) v.seq.push_back(_stack[i]);
                    _stack.resize(m);
                    _stack.push_back(std::move(v));
                    break;
                }
                case 'a': {// APPEND
                    auto x = pop();
                    PACK_CHECK(!_stack.empty(), "%s: APPEND on empty stack", _name.c_str());
                    _stack.back().seq.push_back(std::move(x));
                    break;
                }
                case 'e': {// APPENDS
                    const size_t m = mark_pos();
                    auto &dst = _stack[m - 1];
                    PACK_CHECK(m >= 1 && (dst.kind == v_kind::list || dst.kind == v_kind::tuple),
                               "%s: APPENDS target is not a list", _name.c_str());
                    for (size_t i = m + 1; i < _stack.size(); i++) dst.seq.push_back(_stack[i]);
                    _stack.resize(m);
                    break;
                }
                case 'u': {// SETITEMS
                    const size_t m = mark_pos();
                    auto &dst = _stack[m - 1];
                    PACK_CHECK(m >= 1 && dst.kind == v_kind::dict, "%s: SETITEMS target is not a dict", _name.c_str());
                    for (size_t i = m + 1; i + 1 < _stack.size(); i += 2) {
                        PACK_CHECK(_stack[i].kind == v_kind::string, "%s: dict keys must be strings in this subset",
                                   _name.c_str());
                        dst.map.emplace_back(_stack[i].s, _stack[i + 1]);
                    }
                    _stack.resize(m);
                    break;
                }
                case 'd': {// DICT
                    const size_t m = mark_pos();
                    value v;
                    v.kind = v_kind::dict;
                    for (size_t i = m + 1; i + 1 < _stack.size(); i += 2) {
                        PACK_CHECK(_stack[i].kind == v_kind::string, "%s: dict keys must be strings in this subset",
                                   _name.c_str());
                        v.map.emplace_back(_stack[i].s, _stack[i + 1]);
                    }
                    _stack.resize(m);
                    _stack.push_back(std::move(v));
                    break;
                }
                case 'Q': {// BINPERSID
                    auto pid = pop();
                    PACK_CHECK(pid.kind == v_kind::tuple && !pid.seq.empty(), "%s: bad persistent id", _name.c_str());
                    if (pid.seq[0].s != "storage") fail("unsupported persistent id (expected 'storage')");
                    PACK_CHECK(pid.seq.size() >= 3, "%s: incomplete storage persistent id", _name.c_str());
                    value v;
                    v.kind = v_kind::storage;
                    // ( "storage", <StorageClass>, key, device[, size] )
                    // ( "storage", <StorageClass>, dtype_int, key, device[, size] ) in newer torch
                    size_t cls_idx = 1;
                    if (pid.seq[1].kind == v_kind::integer) {// dtype-by-number form: not used by bge-m3
                        fail("numeric storage dtype is not supported by this packer");
                    }
                    // GLOBAL values carry "module name"; keep just the class name
                    {
                        const std::string &g = pid.seq[cls_idx].s;
                        const auto sp = g.rfind(' ');
                        v.storage_type = sp == std::string::npos ? g : g.substr(sp + 1);
                    }
                    v.s = pid.seq[cls_idx + 1].s;
                    if (v.s.empty()) fail("storage key is not a string");
                    _stack.push_back(std::move(v));
                    break;
                }
                case 'R': {// REDUCE
                    auto args = pop();
                    auto callee = pop();
                    PACK_CHECK(callee.kind == v_kind::global, "%s: REDUCE target is not a global", _name.c_str());
                    if (callee.s == "collections OrderedDict" || callee.s == "builtins dict") {
                        value v;
                        v.kind = v_kind::dict;
                        _stack.push_back(std::move(v));
                    } else if (callee.s == "torch._utils _rebuild_tensor_v2" || callee.s == "torch._utils _rebuild_tensor") {
                        PACK_CHECK(args.kind == v_kind::tuple && args.seq.size() >= 4, "%s: bad _rebuild_tensor args", _name.c_str());
                        PACK_CHECK(args.seq[0].kind == v_kind::storage, "%s: tensor argument 0 is not a storage", _name.c_str());
                        pickle_tensor t;
                        t.storage_key = args.seq[0].s;
                        t.storage_type = args.seq[0].storage_type;
                        t.offset = args.seq[1].i;
                        PACK_CHECK(args.seq[2].kind == v_kind::tuple, "%s: tensor sizes are not a tuple", _name.c_str());
                        for (const auto &s : args.seq[2].seq) {
                            PACK_CHECK(s.kind == v_kind::integer, "%s: non-integer tensor dimension", _name.c_str());
                            t.sizes.push_back(s.i);
                        }
                        value v;
                        v.kind = v_kind::tensor;
                        v.tensor = std::move(t);
                        _stack.push_back(std::move(v));
                    } else {
                        fail(("unsupported reduce target '" + callee.s + "'").c_str());
                    }
                    break;
                }
                case 'b':
                    pop();
                    pop();
                    break;                   // BUILD (ignored)
                case '.': done = true; break;// STOP
                default: {
                    char msg[64];
                    std::snprintf(msg, sizeof(msg), "unsupported opcode 0x%02x at %zu", op, static_cast<size_t>(_p - _begin - 1));
                    fail(msg);
                }
            }
        }
        PACK_CHECK(done, "%s: pickle did not terminate (no STOP)", _name.c_str());
        _root = pop();
    }

    void memo_put(uint8_t idx, value v) {
        if (_memo.size() <= idx) _memo.resize(idx + 1);
        _memo[idx] = std::move(v);
    }
    value memo_get(uint8_t idx) {
        PACK_CHECK(idx < _memo.size(), "%s: pickle memo index %u out of range", _name.c_str(), idx);
        return _memo[idx];
    }

    const uint8_t *_begin;
    const uint8_t *_end;
    const uint8_t *_p;
    std::string _name;
    std::vector<value> _stack;
    std::vector<value> _memo;
    value _root;
};

// Define special members only after both recursive value types are complete.
// Default copying keeps pickle memo entries independent; moving stays noexcept.
pickle_reader::value::value() noexcept = default;
pickle_reader::value::value(v_kind kind) noexcept : kind{kind} {}
pickle_reader::value::~value() noexcept = default;
pickle_reader::value::value(const value &) = default;
pickle_reader::value::value(value &&) noexcept = default;
pickle_reader::value &pickle_reader::value::operator=(const value &) = default;
pickle_reader::value &pickle_reader::value::operator=(value &&) noexcept = default;

// One torch .pt checkpoint holding a state_dict of {weight:[out,in], bias:[out]}.
struct pt_head {
    std::vector<float> weight;// [out,in] converted to fp32
    std::vector<int64_t> weight_sizes;
    std::vector<float> bias;
    std::vector<int64_t> bias_sizes;
};

pt_head load_pt_head(const std::string &path) {
    const auto zip = zip_open(path);
    const auto *pkl_entry = zip.find("data.pkl");
    PACK_CHECK(pkl_entry, "'%s': no data.pkl member", path.c_str());
    const auto pkl = zip_read(zip, *pkl_entry, path);
    // byteorder member must say "little" (the reader assumes LE below)
    if (const auto *bo = zip.find("byteorder"); bo != nullptr) {
        const auto bo_data = zip_read(zip, *bo, path);
        const std::string s(reinterpret_cast<const char *>(bo_data.data()), bo_data.size());
        PACK_CHECK(s == "little", "'%s': byteorder '%s' is not supported", path.c_str(), s.c_str());
    }
    std::unordered_map<std::string, pickle_tensor> tensors;
    pickle_reader pr{pkl.data(), pkl.data() + pkl.size(), path};
    pr.load_state_dict(tensors);
    PACK_CHECK(tensors.count("weight") == 1 && tensors.count("bias") == 1,
               "'%s': expected exactly 'weight' and 'bias' entries, found %zu", path.c_str(), tensors.size());

    pt_head head;
    for (const auto &[key, dst] : {std::pair{"weight", &head.weight}, std::pair{"bias", &head.bias}}) {
        const auto it = tensors.find(key);
        PACK_CHECK(it != tensors.end(), "'%s': missing '%s'", path.c_str(), key);
        const auto &t = it->second;
        PACK_CHECK(t.offset == 0, "'%s': non-zero storage offset on '%s'", path.c_str(), key);
        const size_t elems = static_cast<size_t>([&] {
            int64_t n = 1;
            for (auto d : t.sizes) n *= d;
            return static_cast<uint64_t>(n);
        }());
        const size_t esz = t.storage_type == "HalfStorage" ? 2u : t.storage_type == "FloatStorage" ? 4u :
                                                                                                     0u;
        PACK_CHECK(esz != 0, "'%s': unsupported storage type '%s' for '%s'", path.c_str(), t.storage_type.c_str(), key);
        const std::string member = zip.member_name(std::string("/data/") + t.storage_key);
        PACK_CHECK(!member.empty(), "'%s': storage member data/%s not found", path.c_str(), t.storage_key.c_str());
        const auto raw = zip_read(zip, *zip.find(member), path);
        PACK_CHECK(raw.size() >= elems * esz, "'%s': member '%s' holds %zu bytes, expected >= %zu",
                   path.c_str(), member.c_str(), raw.size(), elems * esz);
        dst->resize(elems);
        if (esz == 4) {
            for (size_t i = 0; i < elems; i++) {
                uint32_t bits = read_le32(raw.data() + i * 4);
                float f;
                std::memcpy(&f, &bits, 4);
                (*dst)[i] = f;
            }
        } else {
            for (size_t i = 0; i < elems; i++) (*dst)[i] = half_to_float(read_le16(raw.data() + i * 2));
        }
        (key[0] == 'w' ? head.weight_sizes : head.bias_sizes) = t.sizes;
    }
    return head;
}

}// namespace pack_model_detail

namespace pack_model_detail {
// ---------------------------------------------------------------------------
// fp32 tensor access over the ONNX graph (external data or inline raw bytes)
// ---------------------------------------------------------------------------

class onnx_weights {
public:
    onnx_weights(onnx_model &model, std::string data_path) : _model(&model), _data_path(std::move(data_path)) {
        if (!_data_path.empty()) {
            _f = std::fopen(_data_path.c_str(), "rb");
            PACK_CHECK(_f, "cannot open external data '%s'", _data_path.c_str());
            _data_size = file_size64(_f);
        }
    }
    onnx_weights(const onnx_weights &) = delete;
    onnx_weights &operator=(const onnx_weights &) = delete;
    ~onnx_weights() {
        if (_f) std::fclose(_f);
    }

    const onnx_tensor *find(const std::string &name) const {
        const auto it = _model->init_index.find(name);
        return it == _model->init_index.end() ? nullptr : &_model->initializers[it->second];
    }
    const onnx_tensor &require(const std::string &name) const {
        const auto *t = find(name);
        PACK_CHECK(t, "initializer '%s' is missing from the ONNX graph", name.c_str());
        return *t;
    }
    // Marked used + shape-checked; returns the fp32 payload.
    std::vector<float> load(const std::string &name, std::initializer_list<int64_t> shape) {
        const auto &t = require(name);
        check(t, name, shape);
        auto *mut = const_cast<onnx_tensor *>(&t);
        mut->used = true;
        return payload(t);
    }
    void mark_used(const std::string &name) {
        auto *mut = const_cast<onnx_tensor *>(find(name));
        PACK_CHECK(mut, "initializer '%s' is missing", name.c_str());
        mut->used = true;
    }
    // Row-range read of a [rows, cols] tensor (used to stream the 1 GB word table
    // without holding it in memory).
    void load_rows(const onnx_tensor &t, uint64_t row_begin, uint64_t row_count, std::vector<float> &dst) {
        PACK_CHECK(t.dims.size() == 2, "load_rows expects a rank-2 tensor ('%s')", t.name.c_str());
        const uint64_t cols = static_cast<uint64_t>(t.dims[1]);
        PACK_CHECK(row_begin + row_count <= static_cast<uint64_t>(t.dims[0]), "row range out of bounds");
        dst.resize(static_cast<size_t>(row_count * cols));
        if (t.data_location == 1) {
            PACK_CHECK(_f, "external data file is not open while reading '%s'", t.name.c_str());
            seek64(_f, t.ext_offset + row_begin * cols * 4);
            const size_t want = static_cast<size_t>(row_count * cols * 4);
            PACK_CHECK(std::fread(dst.data(), 1, want, _f) == want, "short external-data read on '%s'", t.name.c_str());
        } else {
            PACK_CHECK(t.inline_floats.size() == t.num_elements(), "'%s': inline payload size mismatch", t.name.c_str());
            std::memcpy(dst.data(), t.inline_floats.data() + row_begin * cols, want_bytes(row_count, cols));
        }
    }
    static size_t want_bytes(uint64_t rows, uint64_t cols) { return static_cast<size_t>(rows * cols * 4); }

    void check(const onnx_tensor &t, const std::string &name, std::initializer_list<int64_t> shape) const {
        PACK_CHECK(t.data_type == 1, "initializer '%s' has data_type %u, expected 1 (float32)", name.c_str(), t.data_type);
        const std::vector<int64_t> want(shape);
        PACK_CHECK(t.dims == want, "initializer '%s' has dims [%s], expected [%s]", name.c_str(), dims_str(t.dims).c_str(), dims_str(want).c_str());
        if (t.data_location == 1) {// EXTERNAL
            PACK_CHECK(!t.ext_location.empty(), "initializer '%s' is EXTERNAL without a location", name.c_str());
            PACK_CHECK(t.ext_length == t.num_elements() * 4, "initializer '%s': external length %llu != %llu bytes",
                       name.c_str(), static_cast<unsigned long long>(t.ext_length), static_cast<unsigned long long>(t.num_elements() * 4));
            PACK_CHECK(t.ext_offset + t.ext_length <= _data_size, "initializer '%s' external range [%llu,%llu) is past '%s' (%llu bytes)",
                       name.c_str(), static_cast<unsigned long long>(t.ext_offset),
                       static_cast<unsigned long long>(t.ext_offset + t.ext_length), _data_path.c_str(),
                       static_cast<unsigned long long>(_data_size));
        } else {
            PACK_CHECK(t.inline_floats.size() == t.num_elements(), "initializer '%s': inline payload is %zu floats, expected %llu",
                       name.c_str(), t.inline_floats.size(), static_cast<unsigned long long>(t.num_elements()));
        }
    }

    std::string dims_str(const std::vector<int64_t> &d) const {
        std::string s;
        for (size_t i = 0; i < d.size(); i++) {
            if (i) s += ",";
            s += std::to_string(d[i]);
        }
        return s;
    }

private:
    std::vector<float> payload(const onnx_tensor &t) {
        std::vector<float> out;
        const auto n = static_cast<size_t>(t.num_elements());
        out.resize(n);
        if (t.data_location == 1) {
            seek64(_f, t.ext_offset);
            PACK_CHECK(std::fread(out.data(), 1, n * 4, _f) == n * 4, "short external-data read on '%s'", t.name.c_str());
        } else {
            std::memcpy(out.data(), t.inline_floats.data(), n * 4);
        }
        return out;
    }

    onnx_model *_model;
    std::string _data_path;
    std::FILE *_f = nullptr;
    uint64_t _data_size = 0;
};

// ---------------------------------------------------------------------------
// weight-name discovery (plan §3.3 "dump/mapping step")
// ---------------------------------------------------------------------------
// The graph's own names are sentence-transformers flavoured
// ("0.auto_model.encoder.layer.7.attention.self.query.bias"); the six projection
// matrices per layer are NOT named that way -- the exporter replaced them with
// hash-style "onnx::MatMul_<id>" initializers holding the transposed [in,out]
// payload. They are recovered structurally: every bias is added by an Add node
// fed by the corresponding MatMul, so Add(bias, MatMul(x, W)) gives (layer, role)
// -> W. Anything that is not consumed by this mapping is a hard error.

struct layer_sources {
    std::string q, k, v, attn_out, ffn_up, ffn_down;            // weights
    std::string q_b, k_b, v_b, attn_out_b, ffn_up_b, ffn_down_b;// biases
    std::string ln1_g, ln1_b, ln2_g, ln2_b;                     // layer norms
};

struct model_sources {
    std::string word, pos, token_type, emb_ln_g, emb_ln_b;
    std::vector<layer_sources> layers;// size == num_layers
    std::vector<std::string> unused;  // initializers the mapping did not claim
};

struct config_info {
    int64_t num_hidden_layers = 24;
    int64_t hidden_size = 1024;
    int64_t intermediate_size = 4096;
    int64_t num_attention_heads = 16;
    int64_t vocab_size = 250002;
    int64_t max_position_embeddings = 8194;
    int64_t type_vocab_size = 1;
    double layer_norm_eps = 1e-5;
    std::string hidden_act = "gelu";
    bool hidden_act_found = false;
    bool loaded = false;
};

// Parse only the keys we cross-check. Deliberately small: integers, one float,
// one string. A missing key keeps the default (config.json is not mandatory for
// packing -- the shapes are verified against the actual initializers anyway).
void parse_config(const std::string &json, config_info &cfg) {
    auto get_int = [&](const char *key, int64_t &dst) {
        const std::string k = std::string("\"") + key + "\"";
        const auto p = json.find(k);
        if (p == std::string::npos) return;
        const auto c = json.find(':', p + k.size());
        if (c == std::string::npos) return;
        size_t i = c + 1;
        while (i < json.size() && (json[i] == ' ' || json[i] == '\t' || json[i] == '\n' || json[i] == '\r')) i++;
        const size_t e = json.find_first_of(",}] \n\r\t", i);
        if (e == std::string::npos) return;
        dst = std::strtoll(json.substr(i, e - i).c_str(), nullptr, 10);
    };
    get_int("num_hidden_layers", cfg.num_hidden_layers);
    get_int("hidden_size", cfg.hidden_size);
    get_int("intermediate_size", cfg.intermediate_size);
    get_int("num_attention_heads", cfg.num_attention_heads);
    get_int("vocab_size", cfg.vocab_size);
    get_int("max_position_embeddings", cfg.max_position_embeddings);
    get_int("type_vocab_size", cfg.type_vocab_size);
    {// eps is written as 1e-05 -> read it as text
        const std::string k = "\"layer_norm_eps\"";
        const auto p = json.find(k);
        if (p != std::string::npos) {
            const auto c = json.find(':', p + k.size());
            if (c != std::string::npos) {
                size_t i = c + 1;
                while (i < json.size() && (json[i] == ' ' || json[i] == '\t')) i++;
                const size_t e = json.find_first_of(",}\n\r ", i);
                if (e != std::string::npos) cfg.layer_norm_eps = std::strtod(json.substr(i, e - i).c_str(), nullptr);
            }
        }
    }
    {// hidden_act is a string
        const std::string k = "\"hidden_act\"";
        const auto p = json.find(k);
        if (p != std::string::npos) {
            const auto q = json.find('"', json.find(':', p + k.size()));
            if (q != std::string::npos) {
                const auto r = json.find('"', q + 1);
                if (r != std::string::npos) {
                    cfg.hidden_act = json.substr(q + 1, r - q - 1);
                    cfg.hidden_act_found = true;
                }
            }
        }
    }
    cfg.loaded = true;
}

// Extract the layer index and the role tail from a torch-style parameter name,
// e.g. "0.auto_model.encoder.layer.12.attention.self.key.bias" -> (12, "key").
bool split_layer_param(const std::string &name, int &layer, std::string &tail) {
    const std::string marker = ".encoder.layer.";
    const auto p = name.find(marker);
    if (p == std::string::npos) return false;
    size_t i = p + marker.size();
    size_t j = i;
    while (j < name.size() && name[j] >= '0' && name[j] <= '9') j++;
    if (j == i || j >= name.size() || name[j] != '.') return false;
    layer = static_cast<int>(std::strtol(name.substr(i, j - i).c_str(), nullptr, 10));
    tail = name.substr(j + 1);// everything after the layer index, e.g. "attention.self.key.bias"
    return true;
}

model_sources discover_sources(const onnx_model &model, const onnx_weights &w, int num_layers) {
    auto &m = const_cast<onnx_model &>(model);
    model_sources src;
    src.layers.resize(static_cast<size_t>(num_layers));

    // ---- named (torch-style) initializers -----------------------------------
    std::unordered_map<std::string, std::string> layer_tensors;// "layer<i>/<tail>" -> name
    for (const auto &t : m.initializers) {
        int layer = -1;
        std::string tail;
        if (split_layer_param(t.name, layer, tail)) {
            PACK_CHECK(layer >= 0 && layer < num_layers,
                       "initializer '%s': layer index %d out of range (config says %d layers)", t.name.c_str(), layer, num_layers);
            const std::string key = "layer" + std::to_string(layer) + "/" + tail;
            PACK_CHECK(layer_tensors.emplace(key, t.name).second, "duplicate initializer for '%s'", key.c_str());
            continue;
        }
        if (ends_with(t.name, "embeddings.word_embeddings.weight") || ends_with(t.name, "embeddings.wte.weight")) {
            PACK_CHECK(src.word.empty(), "two word-embedding initializers found ('%s', '%s')", src.word.c_str(), t.name.c_str());
            src.word = t.name;
        } else if (ends_with(t.name, "embeddings.position_embeddings.weight") || ends_with(t.name, "embeddings.wpe.weight")) {
            PACK_CHECK(src.pos.empty(), "two position-embedding initializers found");
            src.pos = t.name;
        } else if (contains(t.name, "embeddings.token_type_embeddings.weight")) {
            PACK_CHECK(src.token_type.empty(), "two token-type embedding initializers found");
            src.token_type = t.name;
        } else if (contains(t.name, "embeddings.LayerNorm.weight")) {
            PACK_CHECK(src.emb_ln_g.empty(), "two embedding LayerNorm weights found");
            src.emb_ln_g = t.name;
        } else if (contains(t.name, "embeddings.LayerNorm.bias")) {
            PACK_CHECK(src.emb_ln_b.empty(), "two embedding LayerNorm biases found");
            src.emb_ln_b = t.name;
        }
        // everything else (onnx::MatMul_* etc.) is claimed by the graph trace
    }

    auto take_layer = [&](int layer, const char *tail) -> std::string {
        const std::string key = "layer" + std::to_string(layer) + "/" + tail;
        const auto it = layer_tensors.find(key);
        PACK_CHECK(it != layer_tensors.end(), "missing initializer for the expected parameter '%s'", key.c_str());
        return it->second;
    };

    for (int i = 0; i < num_layers; i++) {
        auto &L = src.layers[static_cast<size_t>(i)];
        L.q_b = take_layer(i, "attention.self.query.bias");
        L.k_b = take_layer(i, "attention.self.key.bias");
        L.v_b = take_layer(i, "attention.self.value.bias");
        L.attn_out_b = take_layer(i, "attention.output.dense.bias");
        L.ln1_g = take_layer(i, "attention.output.LayerNorm.weight");
        L.ln1_b = take_layer(i, "attention.output.LayerNorm.bias");
        L.ffn_up_b = take_layer(i, "intermediate.dense.bias");
        L.ffn_down_b = take_layer(i, "output.dense.bias");
        L.ln2_g = take_layer(i, "output.LayerNorm.weight");
        L.ln2_b = take_layer(i, "output.LayerNorm.bias");
    }

    // ---- graph trace: bias-Add -> MatMul -> weight initializer ---------------
    std::unordered_map<std::string, size_t> producer;// value name -> node index
    for (size_t n = 0; n < m.nodes.size(); n++) {
        for (const auto &o : m.nodes[n].outputs) producer[o] = n;
    }
    auto bias_role = [&](const std::string &name, int &layer, std::string &role) -> bool {
        std::string tail;
        if (!split_layer_param(name, layer, tail)) return false;
        if (tail == "attention.self.query.bias")
            role = "q";
        else if (tail == "attention.self.key.bias")
            role = "k";
        else if (tail == "attention.self.value.bias")
            role = "v";
        else if (tail == "attention.output.dense.bias")
            role = "attn_out";
        else if (tail == "intermediate.dense.bias")
            role = "ffn_up";
        else if (tail == "output.dense.bias")
            role = "ffn_down";
        else
            return false;
        return true;
    };

    for (const auto &node : m.nodes) {
        if (node.op_type != "Add" || node.inputs.size() != 2) continue;
        for (int side = 0; side < 2; side++) {
            int layer = -1;
            std::string role;
            const std::string &bias_in = node.inputs[static_cast<size_t>(side)];
            if (!bias_role(bias_in, layer, role)) continue;
            const auto other = node.inputs[static_cast<size_t>(1 - side)];
            const auto pit = producer.find(other);
            if (pit == producer.end()) continue;// LayerNorm bias adds reach here too
            const auto &mm = m.nodes[pit->second];
            if (mm.op_type != "MatMul" || mm.outputs.size() != 1 || mm.outputs[0] != other) continue;
            std::string weight;
            for (const auto &in : mm.inputs) {
                if (in == other) continue;
                if (w.find(in) != nullptr) {
                    PACK_CHECK(weight.empty(), "MatMul '%s' has more than one initializer input", mm.name.c_str());
                    weight = in;
                }
            }
            PACK_CHECK(!weight.empty(), "MatMul feeding the bias-add of layer %d '%s' has no initializer weight input (node '%s')",
                       layer, role.c_str(), mm.name.c_str());
            auto &L = src.layers[static_cast<size_t>(layer)];
            auto &dst = role == "q" ? L.q : role == "k"    ? L.k :
                                        role == "v"        ? L.v :
                                        role == "attn_out" ? L.attn_out :
                                        role == "ffn_up"   ? L.ffn_up :
                                                             L.ffn_down;
            PACK_CHECK(dst.empty(), "layer %d '%s' weight claimed twice ('%s' and '%s')", layer, role.c_str(), dst.c_str(), weight.c_str());
            dst = weight;
            break;
        }
    }

    for (int i = 0; i < num_layers; i++) {
        const auto &L = src.layers[static_cast<size_t>(i)];
        PACK_CHECK(!L.q.empty() && !L.k.empty() && !L.v.empty() && !L.attn_out.empty() && !L.ffn_up.empty() && !L.ffn_down.empty(),
                   "layer %d: unresolved projection weight(s) (q='%s' k='%s' v='%s' attn.out='%s' ffn.up='%s' ffn.down='%s')",
                   i, L.q.c_str(), L.k.c_str(), L.v.c_str(), L.attn_out.c_str(), L.ffn_up.c_str(), L.ffn_down.c_str());
    }
    const std::pair<const char *, std::string *> emb_fields[] = {
        {"word", &src.word}, {"pos", &src.pos}, {"token_type", &src.token_type}, {"emb_ln_g", &src.emb_ln_g}, {"emb_ln_b", &src.emb_ln_b}};
    for (const auto &[label, field] : emb_fields) {
        PACK_CHECK(!field->empty(), "embedding initializer '%s' was not found in the ONNX graph", label);
    }

    // ---- nothing may be left over ------------------------------------------
    {
        std::unordered_map<std::string, bool> claimed;
        for (const auto *n : {&src.word, &src.pos, &src.token_type, &src.emb_ln_g, &src.emb_ln_b}) claimed[*n] = true;
        for (const auto &L : src.layers) {
            for (const auto *n : {&L.q, &L.k, &L.v, &L.attn_out, &L.ffn_up, &L.ffn_down, &L.q_b, &L.k_b, &L.v_b,
                                  &L.attn_out_b, &L.ln1_g, &L.ln1_b, &L.ffn_up_b, &L.ffn_down_b, &L.ln2_g, &L.ln2_b})
                claimed[*n] = true;
        }
        size_t leftovers = 0;
        for (const auto &t : m.initializers) {
            if (claimed.count(t.name) == 0) {
                if (leftovers < 10) src.unused.push_back(t.name);
                leftovers++;
            }
        }
        PACK_CHECK(leftovers == 0, "%zu ONNX initializer(s) are not part of the expected XLM-RoBERTa weight set (first: '%s')",
                   leftovers, src.unused.empty() ? "?" : src.unused[0].c_str());
    }
    return src;
}

}// namespace pack_model_detail

namespace pack_model_detail {
// ---------------------------------------------------------------------------
// model.bgm writer (format spec at the top of this file)
// ---------------------------------------------------------------------------

constexpr uint32_t kBgmVersion = 1;
constexpr uint32_t kBgmVersionHalf = 2;// opt plan §4 Q1/Q2 (bf16/fp16 blobs)
constexpr uint32_t kBgmVersionInt8 = 3;// INT8 quantization phase 1 (int8 blobs)
constexpr uint32_t kBgmPrecisionFp32 = 0;
constexpr uint32_t kBgmPrecisionBf16 = 1;
constexpr uint32_t kBgmPrecisionFp16 = 2;
constexpr uint32_t kBgmPrecisionInt8 = 3;
constexpr uint64_t kBgmAlign = 256;
// reserved trailing 4 bytes of the 48-byte entry (8+8+8+4+16 == 44): zero for
// v1/v2, the per-tensor fp32 INT8 scale for v3
constexpr uint64_t kBgmEntryScaleOffset = 44;
inline uint64_t bgm_elem_bytes(uint32_t precision) {
    return precision == kBgmPrecisionFp32 ? 4ull : precision == kBgmPrecisionInt8 ? 1ull : 2ull;
}
inline const char *bgm_precision_name(uint32_t precision) {
    switch (precision) {
        case kBgmPrecisionFp32: return "fp32";
        case kBgmPrecisionBf16: return "bf16";
        case kBgmPrecisionFp16: return "fp16";
        case kBgmPrecisionInt8: return "int8";
        default: return "invalid";
    }
}
// ---------------------------------------------------------------------------
// int8 quantization codec (format spec "VERSION 3 ELEMENT ENCODING"). Mirrors
// examples/embedding/embed/quantization.h bit for bit (that header is host-only
// engine code; this target links no lc libraries, so the math is transcribed
// here -- the same arrangement as the half-precision codecs below).
// ---------------------------------------------------------------------------
constexpr int kInt8QuantMax = 127;// symmetric range [-127, 127]; -128 unused
constexpr float kInt8UnitScale = 1.0f;// scale of an all-zero tensor
// round-to-nearest-even, done in double so the tie test is exact
inline double int8_round_rne(double x) {
    const double base = std::floor(x);
    const double frac = x - base;
    if (frac > 0.5) return base + 1.0;
    if (frac < 0.5) return base;
    return std::fmod(base, 2.0) == 0.0 ? base : base + 1.0;
}
inline int8_t int8_from_float(float v, float scale) {
    const double q = int8_round_rne(static_cast<double>(v) / static_cast<double>(scale));
    if (q > static_cast<double>(kInt8QuantMax)) return static_cast<int8_t>(kInt8QuantMax);
    if (q < -static_cast<double>(kInt8QuantMax)) return static_cast<int8_t>(-kInt8QuantMax);
    return static_cast<int8_t>(q);
}
inline float int8_to_float(int8_t q, float scale) { return static_cast<float>(q) * scale; }
inline bool int8_scale_is_valid(float scale) { return std::isfinite(scale) && scale > 0.0f; }
// absmax -> per-tensor scale (1.0 for an all-zero tensor); fail-closed
inline float int8_scale_from_absmax(float absmax) {
    PACK_CHECK(std::isfinite(absmax) && absmax >= 0.0f, "int8: invalid absmax %g (must be finite and >= 0)",
               static_cast<double>(absmax));
    if (absmax == 0.0f) return kInt8UnitScale;
    const float scale = absmax / static_cast<float>(kInt8QuantMax);
    PACK_CHECK(int8_scale_is_valid(scale), "int8: absmax %g yields the unrepresentable scale %g",
               static_cast<double>(absmax), static_cast<double>(scale));
    return scale;
}
// ---------------------------------------------------------------------------
// half-precision codecs (format spec above). Quantization is round-to-nearest-
// even; the upcasts are exact bit operations (mirror of the engine's device-
// side load path in kernels_gemm.cpp and the upload conversion in
// executor.cpp -- all three must agree bit for bit).
// ---------------------------------------------------------------------------
inline uint16_t f32_to_bf16_rne(float v) {
    uint32_t u;
    std::memcpy(&u, &v, sizeof(u));
    if ((u & 0x7f800000u) == 0x7f800000u) {// inf: high bits exact; NaN: keep a
        // quieting payload bit so the round-trip stays NaN
        if ((u & 0x007fffffu) != 0u) u |= 0x00400000u;
        return static_cast<uint16_t>(u >> 16);
    }
    const uint32_t lsb = (u >> 16) & 1u;
    u += 0x7fffu + lsb;// RNE (also rounds finite overflow to inf, correctly)
    return static_cast<uint16_t>(u >> 16);
}
inline uint16_t f32_to_f16_rne(float v) {
    uint32_t u;
    std::memcpy(&u, &v, sizeof(u));
    const uint32_t sign = (u >> 16) & 0x8000u;
    const uint32_t E = (u >> 23) & 0xffu;
    const uint32_t m = u & 0x007fffffu;
    if (E == 0xffu) return static_cast<uint16_t>(sign | 0x7c00u | (m != 0u ? (0x200u | ((m >> 13) & 0x1ffu)) : 0u));
    if (E >= 143u) return static_cast<uint16_t>(sign | 0x7c00u);// |v| >= 2^16 -> inf
    if (E >= 113u) {                                            // normal: round the 13 discarded mantissa bits (RNE)
        const uint32_t r = m & 0x1fffu;
        uint32_t h = m >> 13;
        if (r > 0x1000u || (r == 0x1000u && (h & 1u))) ++h;
        if (h == 0x400u) return static_cast<uint16_t>(sign | ((E - 111u) << 10u));// carry to exp+1 (inf at the top)
        return static_cast<uint16_t>(sign | ((E - 112u) << 10u) | h);
    }
    if (E >= 103u) {// fp16 subnormal: q = M >> (126 - E), RNE on the residue
        const uint32_t M = m | 0x800000u;
        const uint32_t sh = 126u - E;
        const uint32_t r = M & ((1u << sh) - 1u);
        uint32_t q = M >> sh;
        const uint32_t half = 1u << (sh - 1u);
        if (r > half || (r == half && (q & 1u))) ++q;
        return static_cast<uint16_t>(sign | q);// q == 0x400 is the min normal (correct bits)
    }
    return static_cast<uint16_t>(sign);// |v| < 2^-25 -> signed zero
}
inline float bf16_to_f32(uint16_t h) {
    const uint32_t u = static_cast<uint32_t>(h) << 16u;
    float f;
    std::memcpy(&f, &u, sizeof(f));
    return f;
}
inline float f16_to_f32(uint16_t h) {
    const uint32_t mag = h & 0x7fffu;
    const bool neg = (h >> 15u) != 0u;
    float pos;
    if (mag >= 0x7c00u) {
        const uint32_t u = 0x7f800000u | ((mag & 0x03ffu) << 13u);
        std::memcpy(&pos, &u, sizeof(pos));
    } else if (mag >= 0x0400u) {
        const uint32_t u = (mag << 13u) + 0x38000000u;// exact rebias, e16 <= 30
        std::memcpy(&pos, &u, sizeof(pos));
    } else {
        pos = static_cast<float>(mag) * 5.9604644775390625e-8f;// 2^-24
    }
    return neg ? -pos : pos;
}
constexpr int kNumLayers = 24;
constexpr int kHidden = 1024;
constexpr int kIntermediate = 4096;
constexpr int kHeads = 16;
constexpr int kHeadDim = 64;
constexpr int kVocab = 250002;
constexpr int kMaxPos = 8194;
constexpr double kLnEps = 1e-5;
constexpr float kPadBias = -3.4028234663852886e38f;// (1 - mask) * finfo.min

std::string dims_str(const std::vector<int64_t> &d) {
    std::string s = "[";
    for (size_t i = 0; i < d.size(); i++) {
        if (i) s += ",";
        s += std::to_string(d[i]);
    }
    s += "]";
    return s;
}

struct bgm_plan_entry {
    std::string name;
    uint32_t rank = 0;
    uint32_t dims[4] = {0, 0, 0, 0};
    uint64_t size_bytes = 0;
    uint64_t offset = 0;
};

class bgm_writer {
public:
    // precision: kBgmPrecisionFp32 (version 1, byte-identical to the
    // historical output) | bf16 | fp16 (version 2, 2-byte RNE blobs) | int8
    // (version 3, 1-byte symmetric codes + a per-tensor fp32 scale registered
    // through set_int8_absmax() before open()).
    explicit bgm_writer(std::string path, uint32_t precision = kBgmPrecisionFp32)
        : _path(std::move(path)), _precision(precision), _esz(bgm_elem_bytes(precision)) {}
    ~bgm_writer() { close(); }
    bgm_writer(const bgm_writer &) = delete;
    bgm_writer &operator=(const bgm_writer &) = delete;
    uint32_t precision() const { return _precision; }
    // max |value| observed over all written fp32 weights (fp16 range audit;
    // format spec "VERSION 2 ELEMENT ENCODING").
    double max_abs() const { return _max_abs; }
    // Assigns every blob offset (256-byte aligned) and remembers the total size.
    void plan(std::vector<bgm_plan_entry> entries) {
        _entries = std::move(entries);
        _absmax.assign(_entries.size(), 0.0f);
        _have_absmax.assign(_entries.size(), 0u);
        uint64_t off = align_up(16 + 48 * static_cast<uint64_t>(_entries.size()), kBgmAlign);
        for (auto &e : _entries) {
            e.offset = off;
            off = align_up(off + e.size_bytes, kBgmAlign);
        }
        _total = off;
    }
    // INT8 only: the tensor's absmax, measured by the caller over exactly the
    // values write_entry() will produce. The scale (absmax/127) goes into the
    // table entry, so it must be known BEFORE open() -- that is why the packer
    // computes every absmax up front. write_entry() re-measures the absmax while
    // quantizing and refuses to continue if it disagrees.
    void set_int8_absmax(size_t idx, float absmax) {
        PACK_CHECK(_precision == kBgmPrecisionInt8, "set_int8_absmax on a %s package",
                   bgm_precision_name(_precision));
        PACK_CHECK(idx < _absmax.size(), "set_int8_absmax: entry %zu out of range", idx);
        (void)int8_scale_from_absmax(absmax);// validates (fail-closed)
        _absmax[idx] = absmax;
        _have_absmax[idx] = 1u;
    }
    float int8_scale_of(size_t idx) const { return int8_scale_from_absmax(_absmax[idx]); }

    uint64_t total_size() const { return _total; }
    size_t tensor_count() const { return _entries.size(); }
    const bgm_plan_entry &entry(size_t i) const { return _entries[i]; }
    void open() {
        _f = std::fopen(_path.c_str(), "wb");
        PACK_CHECK(_f, "cannot open the output file '%s' for writing", _path.c_str());
        uint8_t header[16] = {'B', 'G', 'M', '3', 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        write_u32(header + 4, _precision == kBgmPrecisionFp32 ? kBgmVersion : _precision == kBgmPrecisionInt8 ? kBgmVersionInt8 :
                                                                                                                kBgmVersionHalf);
        write_u32(header + 8, _precision);
        write_u32(header + 12, static_cast<uint32_t>(_entries.size()));
        put(header, sizeof(header));
        std::vector<uint8_t> table(48 * _entries.size());// zero-filled: v1/v2 reserved bytes stay 0
        for (size_t i = 0; i < _entries.size(); i++) {
            const auto &e = _entries[i];
            uint8_t *p = table.data() + i * 48;
            const uint64_t hash = fnv1a_64(e.name);
            std::memcpy(p, &hash, 8);
            std::memcpy(p + 8, &e.offset, 8);
            std::memcpy(p + 16, &e.size_bytes, 8);
            write_u32(p + 24, e.rank);
            for (int d = 0; d < 4; d++) write_u32(p + 28 + 4 * static_cast<size_t>(d), e.dims[d]);
            if (_precision == kBgmPrecisionInt8) {// v3: scale in the reserved bytes
                PACK_CHECK(_have_absmax[i] != 0u, "int8 package: no absmax was registered for entry %zu ('%s')",
                           i, e.name.c_str());
                const float scale = int8_scale_of(i);
                PACK_CHECK(int8_scale_is_valid(scale), "int8 package: '%s' got the invalid scale %g", e.name.c_str(),
                           static_cast<double>(scale));
                std::memcpy(p + kBgmEntryScaleOffset, &scale, sizeof(scale));
            }
        }
        put(table.data(), table.size());
        pad_to(_entries.empty() ? kBgmAlign : _entries.front().offset);
    }
    // Writes the blob of plan entry `idx`; `fill` produces exactly the expected
    // number of floats, so the 1 GB word table can be streamed in row chunks.
    // For a half package the floats are RNE-quantized to 2-byte elements here
    // (and the running max |w| is audited for the fp16 range check); for an int8
    // package they are quantized to 1-byte symmetric codes with the scale
    // registered by set_int8_absmax() (and the absmax is re-measured to prove the
    // registered scale belongs to this data).
    template<class Fn>
    void write_entry(size_t idx, uint64_t rows_per_call, Fn &&fill) {
        const auto &e = _entries[idx];
        PACK_CHECK(tell64(_f) == e.offset, "'%s': the writer is at %llu but the plan says %llu", e.name.c_str(),
                   static_cast<unsigned long long>(tell64(_f)), static_cast<unsigned long long>(e.offset));
        const bool int8 = _precision == kBgmPrecisionInt8;
        const float int8_scale = int8 ? int8_scale_of(idx) : 1.0f;// also asserts it was registered
        if (int8) {
            PACK_CHECK(_have_absmax[idx] != 0u, "int8 package: no absmax was registered for '%s'", e.name.c_str());
        }
        float entry_absmax = 0.0f;
        const uint64_t total_rows = e.dims[0];
        std::vector<float> buf;
        for (uint64_t row = 0; row < total_rows; row += rows_per_call) {
            const uint64_t count = std::min(rows_per_call, total_rows - row);
            buf.clear();
            fill(row, count, buf);
            const uint64_t cols = e.rank > 1 ? e.dims[1] : 1;
            PACK_CHECK(static_cast<uint64_t>(buf.size()) == count * cols,
                       "'%s': chunk at row %llu produced %zu floats, expected %llu", e.name.c_str(),
                       static_cast<unsigned long long>(row), buf.size(), static_cast<unsigned long long>(count * cols));
            for (const float x : buf) {
                const double ax = std::abs(static_cast<double>(x));
                if (ax > _max_abs) _max_abs = ax;
            }
            if (int8) {
                _ibuf.resize(buf.size());
                for (size_t j = 0; j < buf.size(); j++) {
                    const float x = buf[j];
                    PACK_CHECK(std::isfinite(x),
                               "'%s': nonfinite value at row %llu element %zu -- int8 quantization is fail-closed",
                               e.name.c_str(), static_cast<unsigned long long>(row), j);
                    _ibuf[j] = int8_from_float(x, int8_scale);
                    const float a = std::fabs(x);
                    if (a > entry_absmax) entry_absmax = a;
                }
                put(reinterpret_cast<const uint8_t *>(_ibuf.data()), _ibuf.size());
            } else if (_esz == 2ull) {
                _hbuf.resize(buf.size());
                for (size_t j = 0; j < buf.size(); j++) {
                    _hbuf[j] = _precision == kBgmPrecisionBf16 ? f32_to_bf16_rne(buf[j]) : f32_to_f16_rne(buf[j]);
                }
                put(reinterpret_cast<const uint8_t *>(_hbuf.data()), _hbuf.size() * 2);
            } else {
                put(reinterpret_cast<const uint8_t *>(buf.data()), buf.size() * 4);
            }
        }
        if (int8) {// the registered scale must belong to exactly this data
            PACK_CHECK(entry_absmax == _absmax[idx],
                       "int8 package: '%s' was registered with absmax %.9g but the quantized data has absmax %.9g",
                       e.name.c_str(), static_cast<double>(_absmax[idx]), static_cast<double>(entry_absmax));
        }
        pad_to(align_up(e.offset + e.size_bytes, kBgmAlign));
    }

    template<class Fn>
    void write_entry(size_t idx, Fn &&fill) { write_entry(idx, 1u << 30, std::forward<Fn>(fill)); }
    void close() {
        if (!_f) return;
        std::fflush(_f);
        const uint64_t written = tell64(_f);
        PACK_CHECK(written == _total, "'%s': wrote %llu bytes, the plan said %llu", _path.c_str(),
                   static_cast<unsigned long long>(written), static_cast<unsigned long long>(_total));
        if (_esz == 2ull && _precision == kBgmPrecisionFp16 && _max_abs > 65504.0) [[unlikely]] {
            info("WARNING: fp16 package: max |w| = %.6g exceeds the fp16 finite range (+-65504); such weights "
                 "became inf -- do not deploy this package.\n",
                 _max_abs);
        }
        std::fclose(_f);
        _f = nullptr;
    }

    const std::string &path() const { return _path; }

private:
    static void write_u32(uint8_t *p, uint32_t x) {
        p[0] = static_cast<uint8_t>(x);
        p[1] = static_cast<uint8_t>(x >> 8);
        p[2] = static_cast<uint8_t>(x >> 16);
        p[3] = static_cast<uint8_t>(x >> 24);
    }

    void put(const uint8_t *p, size_t n) {
        if (n == 0) return;
        PACK_CHECK(std::fwrite(p, 1, n, _f) == n, "write failed on '%s'", _path.c_str());
    }
    void pad_to(uint64_t target) {
        static const uint8_t zeros[256] = {};
        uint64_t cur = tell64(_f);
        PACK_CHECK(target >= cur, "alignment target %llu is behind the writer (%llu)", static_cast<unsigned long long>(target), static_cast<unsigned long long>(cur));
        while (cur < target) {
            const size_t n = static_cast<size_t>(std::min<uint64_t>(target - cur, 256));
            put(zeros, n);
            cur += n;
        }
    }

    std::string _path;
    std::FILE *_f = nullptr;
    std::vector<bgm_plan_entry> _entries;
    uint64_t _total = 0;
        uint32_t _precision = kBgmPrecisionFp32;
        uint64_t _esz = 4;
        double _max_abs = 0.0;
        std::vector<uint16_t> _hbuf;// half-chunk staging (_esz == 2)
        std::vector<int8_t> _ibuf;// int8-chunk staging (_esz == 1)
        std::vector<float> _absmax;// per-entry absmax (int8 scale source)
        std::vector<uint8_t> _have_absmax;// 1 when set_int8_absmax() ran
    };

// ---------------------------------------------------------------------------
// emission plan: canonical names, ranks and dims in file order
// ---------------------------------------------------------------------------

std::vector<bgm_plan_entry> build_plan(uint32_t precision = kBgmPrecisionFp32) {
    const uint64_t esz = bgm_elem_bytes(precision);
    auto mk = [esz](const std::string &name, std::initializer_list<uint32_t> dims) {
        bgm_plan_entry e;
        e.name = name;
        e.rank = static_cast<uint32_t>(dims.size());
        size_t i = 0;
        uint64_t n = 1;
        for (auto d : dims) {
            e.dims[i++] = d;
            n *= d;
        }
        e.size_bytes = n * esz;
        return e;
    };
    std::vector<bgm_plan_entry> plan;
    plan.reserve(296);
    plan.push_back(mk("emb.word", {static_cast<uint32_t>(kVocab), static_cast<uint32_t>(kHidden)}));
    plan.push_back(mk("emb.pos", {static_cast<uint32_t>(kMaxPos), static_cast<uint32_t>(kHidden)}));
    plan.push_back(mk("emb.ln.gamma", {static_cast<uint32_t>(kHidden)}));
    plan.push_back(mk("emb.ln.beta", {static_cast<uint32_t>(kHidden)}));
    for (int i = 0; i < kNumLayers; i++) {
        const std::string p = "layer" + std::to_string(i) + ".";
        plan.push_back(mk(p + "qkv", {static_cast<uint32_t>(kHidden), static_cast<uint32_t>(3 * kHidden)}));
        plan.push_back(mk(p + "qkv.bias", {static_cast<uint32_t>(3 * kHidden)}));
        plan.push_back(mk(p + "attn.out", {static_cast<uint32_t>(kHidden), static_cast<uint32_t>(kHidden)}));
        plan.push_back(mk(p + "attn.out.bias", {static_cast<uint32_t>(kHidden)}));
        plan.push_back(mk(p + "ln1.gamma", {static_cast<uint32_t>(kHidden)}));
        plan.push_back(mk(p + "ln1.beta", {static_cast<uint32_t>(kHidden)}));
        plan.push_back(mk(p + "ffn.up", {static_cast<uint32_t>(kHidden), static_cast<uint32_t>(kIntermediate)}));
        plan.push_back(mk(p + "ffn.up.bias", {static_cast<uint32_t>(kIntermediate)}));
        plan.push_back(mk(p + "ffn.down", {static_cast<uint32_t>(kIntermediate), static_cast<uint32_t>(kHidden)}));
        plan.push_back(mk(p + "ffn.down.bias", {static_cast<uint32_t>(kHidden)}));
        plan.push_back(mk(p + "ln2.gamma", {static_cast<uint32_t>(kHidden)}));
        plan.push_back(mk(p + "ln2.beta", {static_cast<uint32_t>(kHidden)}));
    }
    plan.push_back(mk("head.sparse.w", {static_cast<uint32_t>(kHidden), 1}));
    plan.push_back(mk("head.sparse.b", {1}));
    plan.push_back(mk("head.colbert.w", {static_cast<uint32_t>(kHidden), static_cast<uint32_t>(kHidden)}));
    plan.push_back(mk("head.colbert.b", {static_cast<uint32_t>(kHidden)}));
    return plan;
}

// torch Linear weight [out,in] (row-major) -> packed B[K=in][N=out] (row-major)
std::vector<float> transpose_kn(const std::vector<float> &w, uint64_t out, uint64_t in) {
    PACK_CHECK(static_cast<size_t>(out * in) == w.size(), "head weight has %zu elements, expected %llu*%llu", w.size(),
               static_cast<unsigned long long>(out), static_cast<unsigned long long>(in));
    std::vector<float> t(static_cast<size_t>(in * out));
    for (uint64_t o = 0; o < out; o++) {
        for (uint64_t i = 0; i < in; i++) t[static_cast<size_t>(i * out + o)] = w[static_cast<size_t>(o * in + i)];
    }
    return t;
}

// ---------------------------------------------------------------------------
// the pack step
// ---------------------------------------------------------------------------

void run_pack(const std::string &model_dir, const std::string &out_path, const config_info &cfg, bool dump,
              uint32_t precision = kBgmPrecisionFp32) {
    const std::string onnx_path = path_join(path_join(model_dir, "onnx"), "model.onnx");
    const auto blob = read_file_text(onnx_path);
    info("read %s (%zu bytes)\n", onnx_path.c_str(), blob.size());
    auto model = parse_onnx(blob);
    info("graph: %zu nodes, %zu initializers\n", model.nodes.size(), model.initializers.size());
    PACK_CHECK(!model.nodes.empty() && !model.initializers.empty(), "the ONNX graph parsed as empty (bad protobuf?)");

    // external data file: TensorProto.external_data.location is relative to the
    // directory of model.onnx
    std::string data_path;
    for (const auto &t : model.initializers) {
        if (t.data_location != 1) continue;
        const std::string cand = path_is_absolute(t.ext_location) ? t.ext_location : path_join(path_dir(onnx_path), t.ext_location);
        if (data_path.empty()) data_path = cand;
        PACK_CHECK(cand == data_path, "initializers reference more than one external data file ('%s' vs '%s')",
                   cand.c_str(), data_path.c_str());
    }
    onnx_weights weights(model, data_path);
    info("external data: %s (%llu bytes)\n", data_path.empty() ? "<none, all inline>" : data_path.c_str(),
         static_cast<unsigned long long>(data_path.empty() ? 0 : file_size64_of(data_path)));

    PACK_CHECK(cfg.num_hidden_layers == kNumLayers, "config.json says %lld layers, this packer is written for %d",
               static_cast<long long>(cfg.num_hidden_layers), kNumLayers);
    PACK_CHECK(cfg.hidden_size == kHidden && cfg.intermediate_size == kIntermediate && cfg.num_attention_heads == kHeads,
               "config.json architecture (%lld hidden / %lld intermediate / %lld heads) does not match %d/%d/%d",
               static_cast<long long>(cfg.hidden_size), static_cast<long long>(cfg.intermediate_size),
               static_cast<long long>(cfg.num_attention_heads), kHidden, kIntermediate, kHeads);

    const auto src = discover_sources(model, weights, kNumLayers);
    const auto sparse = load_pt_head(path_join(model_dir, "sparse_linear.pt"));
    const auto colbert = load_pt_head(path_join(model_dir, "colbert_linear.pt"));
    PACK_CHECK(sparse.weight_sizes.size() == 2 && sparse.weight_sizes[0] == 1 && sparse.weight_sizes[1] == kHidden,
               "sparse_linear weight has unexpected shape %s", dims_str(sparse.weight_sizes).c_str());
    PACK_CHECK(sparse.bias_sizes.size() == 1 && sparse.bias_sizes[0] == 1, "sparse_linear bias has unexpected shape %s",
               dims_str(sparse.bias_sizes).c_str());
    PACK_CHECK(colbert.weight_sizes.size() == 2 && colbert.weight_sizes[0] == kHidden && colbert.weight_sizes[1] == kHidden,
               "colbert_linear weight has unexpected shape %s", dims_str(colbert.weight_sizes).c_str());
    PACK_CHECK(colbert.bias_sizes.size() == 1 && colbert.bias_sizes[0] == kHidden, "colbert_linear bias has unexpected shape %s",
               dims_str(colbert.bias_sizes).c_str());

    if (dump) {
        info("mapping (canonical name -> source initializer):\n");
        info("  emb.word          <- %s\n", src.word.c_str());
        info("  emb.pos           <- %s\n", src.pos.c_str());
        info("  emb.word(+type)   <- %s%s row 0 folded in\n", src.token_type.c_str(), "");
        info("  emb.ln.gamma      <- %s\n", src.emb_ln_g.c_str());
        info("  emb.ln.beta       <- %s\n", src.emb_ln_b.c_str());
        for (int i = 0; i < kNumLayers; i++) {
            const auto &L = src.layers[static_cast<size_t>(i)];
            info("  layer%d.qkv        <- %s %s %s\n", i, L.q.c_str(), L.k.c_str(), L.v.c_str());
            info("  layer%d.qkv.bias   <- %s %s %s\n", i, L.q_b.c_str(), L.k_b.c_str(), L.v_b.c_str());
            info("  layer%d.attn.out   <- %s + %s\n", i, L.attn_out.c_str(), L.attn_out_b.c_str());
            info("  layer%d.ln1        <- %s + %s\n", i, L.ln1_g.c_str(), L.ln1_b.c_str());
            info("  layer%d.ffn.up     <- %s + %s\n", i, L.ffn_up.c_str(), L.ffn_up_b.c_str());
            info("  layer%d.ffn.down   <- %s + %s\n", i, L.ffn_down.c_str(), L.ffn_down_b.c_str());
            info("  layer%d.ln2        <- %s + %s\n", i, L.ln2_g.c_str(), L.ln2_b.c_str());
        }
        info("  head.sparse.w/b   <- sparse_linear.pt %s %s\n", dims_str(sparse.weight_sizes).c_str(), dims_str(sparse.bias_sizes).c_str());
        info("  head.colbert.w/b  <- colbert_linear.pt %s %s\n", dims_str(colbert.weight_sizes).c_str(), dims_str(colbert.bias_sizes).c_str());
    }

    // ---- source payloads -----------------------------------------------------
    const auto &word_t = weights.require(src.word);
    weights.check(word_t, src.word, {kVocab, kHidden});
    const auto &type_t = weights.require(src.token_type);
    weights.check(type_t, src.token_type, {1, kHidden});
    PACK_CHECK(cfg.type_vocab_size == 1, "config.json says type_vocab_size=%lld; the packer only folds a single token-type row",
               static_cast<long long>(cfg.type_vocab_size));
    auto type_row = weights.load(src.token_type, {1, kHidden});
    weights.mark_used(src.word);// streamed below, not via load()

    auto pos = weights.load(src.pos, {kMaxPos, kHidden});
    auto emb_ln_g = weights.load(src.emb_ln_g, {kHidden});
    auto emb_ln_b = weights.load(src.emb_ln_b, {kHidden});

    std::vector<float> sparse_w = transpose_kn(sparse.weight, 1, static_cast<uint64_t>(kHidden));
    std::vector<float> colbert_w = transpose_kn(colbert.weight, static_cast<uint64_t>(kHidden), static_cast<uint64_t>(kHidden));

    struct layer_payload {
        std::vector<float> qkv, qkv_b, attn_out, attn_out_b, ln1_g, ln1_b, ffn_up, ffn_up_b, ffn_down, ffn_down_b, ln2_g, ln2_b;
    };
    std::vector<layer_payload> layers(static_cast<size_t>(kNumLayers));
    // QKV fusion: the opset-11 export stores the projection weights as [in,out]
    // (verified against pytorch_model.bin), i.e. already B[K][N]; concatenating
    // the three matrices along N produces the fused [1024,3072] block Q|K|V.
    for (int i = 0; i < kNumLayers; i++) {
        const auto &L = src.layers[static_cast<size_t>(i)];
        auto &P = layers[static_cast<size_t>(i)];
        auto q = weights.load(L.q, {kHidden, kHidden});
        auto k = weights.load(L.k, {kHidden, kHidden});
        auto v = weights.load(L.v, {kHidden, kHidden});
        P.qkv.resize(static_cast<size_t>(kHidden) * 3 * static_cast<size_t>(kHidden));
        for (int row = 0; row < kHidden; row++) {
            float *dst = P.qkv.data() + static_cast<size_t>(row) * 3 * static_cast<size_t>(kHidden);
            std::memcpy(dst, q.data() + static_cast<size_t>(row) * kHidden, static_cast<size_t>(kHidden) * 4);
            std::memcpy(dst + kHidden, k.data() + static_cast<size_t>(row) * kHidden, static_cast<size_t>(kHidden) * 4);
            std::memcpy(dst + 2 * kHidden, v.data() + static_cast<size_t>(row) * kHidden, static_cast<size_t>(kHidden) * 4);
        }
        auto qb = weights.load(L.q_b, {kHidden});
        auto kb = weights.load(L.k_b, {kHidden});
        auto vb = weights.load(L.v_b, {kHidden});
        P.qkv_b.reserve(3 * static_cast<size_t>(kHidden));
        P.qkv_b.insert(P.qkv_b.end(), qb.begin(), qb.end());
        P.qkv_b.insert(P.qkv_b.end(), kb.begin(), kb.end());
        P.qkv_b.insert(P.qkv_b.end(), vb.begin(), vb.end());
        P.attn_out = weights.load(L.attn_out, {kHidden, kHidden});
        P.attn_out_b = weights.load(L.attn_out_b, {kHidden});
        P.ln1_g = weights.load(L.ln1_g, {kHidden});
        P.ln1_b = weights.load(L.ln1_b, {kHidden});
        P.ffn_up = weights.load(L.ffn_up, {kHidden, kIntermediate});
        P.ffn_up_b = weights.load(L.ffn_up_b, {kIntermediate});
        P.ffn_down = weights.load(L.ffn_down, {kIntermediate, kHidden});
        P.ffn_down_b = weights.load(L.ffn_down_b, {kHidden});
        P.ln2_g = weights.load(L.ln2_g, {kHidden});
        P.ln2_b = weights.load(L.ln2_b, {kHidden});
        q.clear();
        q.shrink_to_fit();
        k.clear();
        k.shrink_to_fit();
        v.clear();
        v.shrink_to_fit();
        qb.clear();
        qb.shrink_to_fit();
        kb.clear();
        kb.shrink_to_fit();
        vb.clear();
        vb.shrink_to_fit();
    }
    // Every initializer must have been claimed and read exactly once.
    {
        size_t unread = 0;
        std::string first;
        for (const auto &t : model.initializers) {
            if (!t.used) {
                if (first.empty()) first = t.name;
                unread++;
            }
        }
        PACK_CHECK(unread == 0, "%zu claimed initializers were never read (first: '%s') -- the mapping is incomplete",
                   unread, first.c_str());
    }

    // ---- write ---------------------------------------------------------------
    bgm_writer writer(out_path, precision);
    writer.plan(build_plan(precision));
    info("packing %zu tensors as %s, %llu bytes of payload, %llu bytes total (256-byte aligned)\n",
         writer.tensor_count(), bgm_precision_name(precision),
         static_cast<unsigned long long>([&] {
             uint64_t s = 0;
             for (size_t i = 0; i < writer.tensor_count(); i++) s += writer.entry(i).size_bytes;
             return s;
         }()),
         static_cast<unsigned long long>(writer.total_size()));

    // emb.word row-chunk fill (token-type row 0 folded in). Shared by the int8
    // absmax pre-pass below and the writer's streaming write, so both see
    // exactly the same values.
    const auto fill_emb_word = [&](uint64_t row, uint64_t rows, std::vector<float> &buf) {
        weights.load_rows(word_t, row, rows, buf);
        for (uint64_t r = 0; r < rows; r++) {
            float *dst = buf.data() + static_cast<size_t>(r) * kHidden;
            for (int c = 0; c < kHidden; c++) dst[c] += type_row[static_cast<size_t>(c)];
        }
    };

    if (precision == kBgmPrecisionInt8) {
        // VERSION 3 element encoding: per-tensor symmetric int8. The scale lives
        // in the table entry, so every absmax must be known BEFORE open() writes
        // the table. All payloads except emb.word are already in memory; emb.word
        // is streamed a second time through the same fill lambda (one extra ~1 GB
        // read of the source, an offline cost only).
        const auto plan = build_plan(precision);
        std::vector<const std::vector<float> *> payloads(plan.size(), nullptr);
        payloads[1] = &pos;
        payloads[2] = &emb_ln_g;
        payloads[3] = &emb_ln_b;
        for (int i = 0; i < kNumLayers; i++) {
            const auto &P = layers[static_cast<size_t>(i)];
            const size_t base = static_cast<size_t>(4 + i * 12);
            payloads[base + 0] = &P.qkv;
            payloads[base + 1] = &P.qkv_b;
            payloads[base + 2] = &P.attn_out;
            payloads[base + 3] = &P.attn_out_b;
            payloads[base + 4] = &P.ln1_g;
            payloads[base + 5] = &P.ln1_b;
            payloads[base + 6] = &P.ffn_up;
            payloads[base + 7] = &P.ffn_up_b;
            payloads[base + 8] = &P.ffn_down;
            payloads[base + 9] = &P.ffn_down_b;
            payloads[base + 10] = &P.ln2_g;
            payloads[base + 11] = &P.ln2_b;
        }
        const size_t head_base = 4 + static_cast<size_t>(kNumLayers) * 12;
        payloads[head_base + 0] = &sparse_w;
        payloads[head_base + 1] = &sparse.bias;
        payloads[head_base + 2] = &colbert_w;
        payloads[head_base + 3] = &colbert.bias;

        const auto absmax_of = [](const std::vector<float> &v) {
            float a = 0.0f;
            for (const float x : v) {
                PACK_CHECK(std::isfinite(x), "int8 quantization: nonfinite weight value %g (fail-closed)",
                           static_cast<double>(x));
                const float ax = std::fabs(x);
                if (ax > a) a = ax;
            }
            return a;
        };
        // emb.word: the only streamed tensor (entry 0)
        {
            float a = 0.0f;
            std::vector<float> buf;
            const uint64_t rows = static_cast<uint64_t>(kVocab);
            const uint64_t chunk = 4096;
            for (uint64_t row = 0; row < rows; row += chunk) {
                const uint64_t n = std::min(chunk, rows - row);
                buf.clear();
                fill_emb_word(row, n, buf);
                const float c = absmax_of(buf);
                if (c > a) a = c;
            }
            writer.set_int8_absmax(0u, a);
        }
        for (size_t i = 1; i < plan.size(); i++) {
            PACK_CHECK(payloads[i] != nullptr, "int8 quantization: no source payload for entry %zu ('%s')", i,
                       plan[i].name.c_str());
            PACK_CHECK(static_cast<uint64_t>(payloads[i]->size()) == plan[i].size_bytes,
                       "int8 quantization: '%s' has %zu elements but the plan says %llu bytes",
                       plan[i].name.c_str(), payloads[i]->size(),
                       static_cast<unsigned long long>(plan[i].size_bytes));
            writer.set_int8_absmax(i, absmax_of(*payloads[i]));
        }
        info("int8: per-tensor absmax measured for all %zu tensors (scale = absmax/127, stored in the "
             "table entry's reserved 4 bytes)\n",
             plan.size());
    }
    writer.open();

    // emb.word: streamed in row chunks with the token-type row folded in
    writer.write_entry(0, 4096, fill_emb_word);
    writer.write_entry(1, [&](uint64_t, uint64_t, std::vector<float> &buf) { buf = pos; });
    writer.write_entry(2, [&](uint64_t, uint64_t, std::vector<float> &buf) { buf = emb_ln_g; });
    writer.write_entry(3, [&](uint64_t, uint64_t, std::vector<float> &buf) { buf = emb_ln_b; });
    for (int i = 0; i < kNumLayers; i++) {
        const auto &P = layers[static_cast<size_t>(i)];
        const size_t base = static_cast<size_t>(4 + i * 12);
        writer.write_entry(base + 0, [&](uint64_t, uint64_t, std::vector<float> &buf) { buf = P.qkv; });
        writer.write_entry(base + 1, [&](uint64_t, uint64_t, std::vector<float> &buf) { buf = P.qkv_b; });
        writer.write_entry(base + 2, [&](uint64_t, uint64_t, std::vector<float> &buf) { buf = P.attn_out; });
        writer.write_entry(base + 3, [&](uint64_t, uint64_t, std::vector<float> &buf) { buf = P.attn_out_b; });
        writer.write_entry(base + 4, [&](uint64_t, uint64_t, std::vector<float> &buf) { buf = P.ln1_g; });
        writer.write_entry(base + 5, [&](uint64_t, uint64_t, std::vector<float> &buf) { buf = P.ln1_b; });
        writer.write_entry(base + 6, [&](uint64_t, uint64_t, std::vector<float> &buf) { buf = P.ffn_up; });
        writer.write_entry(base + 7, [&](uint64_t, uint64_t, std::vector<float> &buf) { buf = P.ffn_up_b; });
        writer.write_entry(base + 8, [&](uint64_t, uint64_t, std::vector<float> &buf) { buf = P.ffn_down; });
        writer.write_entry(base + 9, [&](uint64_t, uint64_t, std::vector<float> &buf) { buf = P.ffn_down_b; });
        writer.write_entry(base + 10, [&](uint64_t, uint64_t, std::vector<float> &buf) { buf = P.ln2_g; });
        writer.write_entry(base + 11, [&](uint64_t, uint64_t, std::vector<float> &buf) { buf = P.ln2_b; });
    }
    writer.write_entry(4 + static_cast<size_t>(kNumLayers) * 12 + 0, [&](uint64_t, uint64_t, std::vector<float> &buf) { buf = sparse_w; });
    writer.write_entry(4 + static_cast<size_t>(kNumLayers) * 12 + 1, [&](uint64_t, uint64_t, std::vector<float> &buf) { buf = sparse.bias; });
    writer.write_entry(4 + static_cast<size_t>(kNumLayers) * 12 + 2, [&](uint64_t, uint64_t, std::vector<float> &buf) { buf = colbert_w; });
    writer.write_entry(4 + static_cast<size_t>(kNumLayers) * 12 + 3, [&](uint64_t, uint64_t, std::vector<float> &buf) { buf = colbert.bias; });
    writer.close();
    info("weight range audit: max |w| = %.6f over all packed fp32 source values\n", writer.max_abs());
    info("wrote %s (%llu bytes)\n", out_path.c_str(), static_cast<unsigned long long>(file_size64_of(out_path)));
}

}// namespace pack_model_detail

namespace pack_model_detail {
// ---------------------------------------------------------------------------
// minimal JSON reader (plan §3.3 --check path: golden_tokens.json, config.json)
// ---------------------------------------------------------------------------

struct json_entry;
struct json_value {
    json_value() noexcept;
    ~json_value() noexcept;
    json_value(const json_value &);
    json_value(json_value &&) noexcept;
    json_value &operator=(const json_value &);
    json_value &operator=(json_value &&) noexcept;

    enum class kind { null,
                      boolean,
                      number,
                      string,
                      array,
                      object } type = kind::null;
    double num = 0;
    bool boolean = false;
    std::string str;
    std::vector<json_value> arr;
    std::vector<json_entry> obj;

    const json_value *at(const std::string &key) const;
};

struct json_entry {
    std::string first;
    json_value second;
};

json_value::json_value() noexcept = default;
json_value::~json_value() noexcept = default;
json_value::json_value(const json_value &) = default;
json_value::json_value(json_value &&) noexcept = default;
json_value &json_value::operator=(const json_value &) = default;
json_value &json_value::operator=(json_value &&) noexcept = default;

const json_value *json_value::at(const std::string &key) const {
    for (const auto &[k, v] : obj) {
        if (k == key) return &v;
    }
    return nullptr;
}

class json_parser {
public:
    json_parser(const std::string &text) : _s(text) {}
    json_value parse() {
        skip_ws();
        auto v = value();
        skip_ws();
        return v;
    }

private:
    [[noreturn]] void fail(const char *what, size_t pos) { PACK_FATAL("JSON error at offset %zu: %s", pos, what); }
    void skip_ws() {
        while (_i < _s.size() && (_s[_i] == ' ' || _s[_i] == '\t' || _s[_i] == '\n' || _s[_i] == '\r')) _i++;
    }
    char peek() {
        skip_ws();
        PACK_CHECK(_i < _s.size(), "JSON: unexpected end of input");
        return _s[_i];
    }
    bool literal(const char *lit) {
        const size_t n = std::strlen(lit);
        if (_s.compare(_i, n, lit) != 0) return false;
        _i += n;
        return true;
    }
    json_value value() {
        skip_ws();
        const size_t start = _i;
        PACK_CHECK(_i < _s.size(), "JSON: unexpected end of input");
        const char c = _s[_i];
        json_value v;
        if (c == '{') {
            v.type = json_value::kind::object;
            _i++;
            if (peek() == '}') {
                _i++;
                return v;
            }
            for (;;) {
                skip_ws();
                PACK_CHECK(_i < _s.size() && _s[_i] == '"', "JSON: object key must be a string");
                auto key = string_body();
                v.obj.emplace_back(std::move(key), value_after_colon());
                const char n = peek();
                if (n == ',') {
                    _i++;
                    continue;
                }
                if (n == '}') {
                    _i++;
                    break;
                }
                fail("expected ',' or '}' in object", start);
            }
            return v;
        }
        if (c == '[') {
            v.type = json_value::kind::array;
            _i++;
            if (peek() == ']') {
                _i++;
                return v;
            }
            for (;;) {
                v.arr.push_back(value());
                const char n = peek();
                if (n == ',') {
                    _i++;
                    continue;
                }
                if (n == ']') {
                    _i++;
                    break;
                }
                fail("expected ',' or ']' in array", start);
            }
            return v;
        }
        if (c == '"') {
            v.type = json_value::kind::string;
            v.str = string_body();
            return v;
        }
        if (c == 't') {
            PACK_CHECK(literal("true"), "JSON: bad literal");
            v.type = json_value::kind::boolean;
            v.boolean = true;
            return v;
        }
        if (c == 'f') {
            PACK_CHECK(literal("false"), "JSON: bad literal");
            v.type = json_value::kind::boolean;
            v.boolean = false;
            return v;
        }
        if (c == 'n') {
            PACK_CHECK(literal("null"), "JSON: bad literal");
            v.type = json_value::kind::null;
            return v;
        }
        // number
        const char *begin = _s.c_str() + _i;
        char *end = nullptr;
        const double d = std::strtod(begin, &end);
        PACK_CHECK(end != begin, "JSON: malformed value");
        v.type = json_value::kind::number;
        v.num = d;
        _i = static_cast<size_t>(end - _s.c_str());
        return v;
    }
    json_value value_after_colon() {
        skip_ws();
        PACK_CHECK(_i < _s.size() && _s[_i] == ':', "JSON: expected ':' after object key");
        _i++;
        return value();
    }
    std::string string_body() {
        PACK_CHECK(_s[_i] == '"', "JSON: expected a string");
        _i++;
        std::string out;
        while (_i < _s.size() && _s[_i] != '"') {
            if (_s[_i] == '\\') {
                _i++;
                if (_i >= _s.size()) break;
                switch (const char e = _s[_i++]; e) {
                    case 'n': out.push_back('\n'); break;
                    case 't': out.push_back('\t'); break;
                    case 'r': out.push_back('\r'); break;
                    case 'b': out.push_back('\b'); break;
                    case 'f': out.push_back('\f'); break;
                    case '"': out.push_back('"'); break;
                    case '\\': out.push_back('\\'); break;
                    case '/': out.push_back('/'); break;
                    case 'u': {// basic plane only (the goldens are ASCII)
                        PACK_CHECK(_i + 4 <= _s.size(), "JSON: truncated \\u escape");
                        const unsigned cp = static_cast<unsigned>(std::strtoul(_s.substr(_i, 4).c_str(), nullptr, 16));
                        _i += 4;
                        if (cp < 0x80)
                            out.push_back(static_cast<char>(cp));
                        else if (cp < 0x800) {
                            out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
                            out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
                        } else {
                            out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
                            out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
                            out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
                        }
                        break;
                    }
                    default: PACK_FATAL("JSON: unsupported escape '\\%c'", e);
                }
            } else {
                out.push_back(_s[_i++]);
            }
        }
        PACK_CHECK(_i < _s.size() && _s[_i] == '"', "JSON: unterminated string");
        _i++;
        return out;
    }

    const std::string &_s;
    size_t _i = 0;
};

// ---------------------------------------------------------------------------
// model.bgm reader (mirrors the format spec at the top of this file)
// ---------------------------------------------------------------------------

struct bgm_entry_view {
    uint64_t hash = 0;
    uint64_t offset = 0;
    uint64_t size_bytes = 0;
    uint32_t rank = 0;
    uint32_t dims[4] = {0, 0, 0, 0};
    float scale = 1.0f;// v3 only: per-tensor int8 scale (1 otherwise)
    uint64_t num_elements() const {
        uint64_t n = 1;
        for (uint32_t d = 0; d < rank; d++) n *= dims[d];
        return n;
    }
};

class bgm_reader {
public:
    explicit bgm_reader(const std::string &path) {
        _f = std::fopen(path.c_str(), "rb");
        PACK_CHECK(_f, "--check: cannot reopen the packed file '%s'", path.c_str());
        _size = file_size64(_f);
        uint8_t header[16];
        read_at(0, header, sizeof(header));
        PACK_CHECK(header[0] == 'B' && header[1] == 'G' && header[2] == 'M' && header[3] == '3', "bad magic (not a BGM3 file)");
        _version = read_le32(header + 4);
        _precision = read_le32(header + 8);
        _count = read_le32(header + 12);
        // v1 = fp32 blobs; v2 = half blobs (bf16/fp16); v3 = int8 codes + a
        // per-tensor scale in the entry's reserved bytes -- format spec above.
        // Mirrors embed::Package::load (package.cpp) so --check reads exactly
        // what the engine accepts.
        PACK_CHECK(_version == kBgmVersion || _version == kBgmVersionHalf || _version == kBgmVersionInt8,
                   "unsupported BGM3 version %u", _version);
        if (_version == kBgmVersion) {
            PACK_CHECK(_precision == kBgmPrecisionFp32,
                       "unsupported precision %u (0 = fp32 is the only v1 mode; bf16/fp16 are version 2, "
                       "int8 is version 3)",
                       _precision);
        } else if (_version == kBgmVersionHalf) {
            PACK_CHECK(_precision == kBgmPrecisionBf16 || _precision == kBgmPrecisionFp16,
                       "unsupported precision %u for version 2 (1 = bf16, 2 = fp16)", _precision);
        } else {
            PACK_CHECK(_precision == kBgmPrecisionInt8,
                       "unsupported precision %u for version 3 (3 = int8 is the only v3 mode)", _precision);
        }
        PACK_CHECK(_precision != kBgmPrecisionInt8 || _version == kBgmVersionInt8,
                   "precision 3 (int8) requires BGM3 version 3, not version %u", _version);
        _esz = bgm_elem_bytes(_precision);
        PACK_CHECK(_count > 0 && _count < (1u << 20), "implausible tensor_count %u", _count);
        PACK_CHECK(static_cast<uint64_t>(16) + 48ull * _count <= _size, "the tensor table does not fit in the file");
        _entries.resize(_count);
        std::vector<uint8_t> table(48ull * _count);
        read_at(16, table.data(), table.size());
        uint64_t expect_off = align_up(16 + 48ull * _count, kBgmAlign);
        for (uint32_t i = 0; i < _count; i++) {
            const uint8_t *p = table.data() + 48ull * i;
            auto &e = _entries[i];
            std::memcpy(&e.hash, p, 8);
            std::memcpy(&e.offset, p + 8, 8);
            std::memcpy(&e.size_bytes, p + 16, 8);
            e.rank = read_le32(p + 24);
            for (int d = 0; d < 4; d++) e.dims[d] = read_le32(p + 28 + 4 * static_cast<size_t>(d));
            if (_precision == kBgmPrecisionInt8) {
                // v3: the reserved trailing 4 bytes hold the per-tensor fp32 scale
                const uint32_t bits = read_le32(p + kBgmEntryScaleOffset);
                std::memcpy(&e.scale, &bits, sizeof(e.scale));
                PACK_CHECK(int8_scale_is_valid(e.scale),
                           "entry %u: int8 scale %g (0x%08x) is not finite and positive", i,
                           static_cast<double>(e.scale), bits);
            }
            // table sanity: alignment, rank, size/dims consistency, ordering
            PACK_CHECK(e.rank >= 1 && e.rank <= 4, "entry %u: rank %u out of range", i, e.rank);
            PACK_CHECK(e.offset % kBgmAlign == 0, "entry %u: offset %llu is not 256-byte aligned", i,
                       static_cast<unsigned long long>(e.offset));
            PACK_CHECK(e.offset >= expect_off || i == 0, "entry %u: offset %llu overlaps the table", i,
                       static_cast<unsigned long long>(e.offset));
            PACK_CHECK(e.offset + e.size_bytes <= _size, "entry %u: blob [%llu,%llu) is past the file (%llu bytes)", i,
                       static_cast<unsigned long long>(e.offset), static_cast<unsigned long long>(e.offset + e.size_bytes),
                       static_cast<unsigned long long>(_size));
            const uint64_t elems = e.num_elements();
            PACK_CHECK(elems * _esz == e.size_bytes, "entry %u: dims %ux%ux%ux%u * %llu != size_bytes %llu", i, e.dims[0], e.dims[1],
                       e.dims[2], e.dims[3], static_cast<unsigned long long>(_esz), static_cast<unsigned long long>(e.size_bytes));
            for (uint32_t d = e.rank; d < 4; d++) PACK_CHECK(e.dims[d] == 0, "entry %u: dims beyond rank are not zero", i);
            expect_off = align_up(e.offset + e.size_bytes, kBgmAlign);
        }
        for (uint32_t i = 0; i < _count; i++) {
            const auto &e = _entries[i];
            _index.emplace(e.hash, i);
        }
    }
    ~bgm_reader() {
        if (_f) std::fclose(_f);
    }
    bgm_reader(const bgm_reader &) = delete;
    bgm_reader &operator=(const bgm_reader &) = delete;

    uint32_t count() const { return _count; }
    uint32_t version() const { return _version; }
    uint32_t precision() const { return _precision; }
    const char *precision_name() const { return bgm_precision_name(_precision); }
    uint64_t elem_bytes() const { return _esz; }
    uint64_t file_size() const { return _size; }
    const bgm_entry_view &entry(size_t i) const { return _entries[i]; }
    const bgm_entry_view *lookup(const std::string &name) const {
        const auto it = _index.find(fnv1a_64(name));
        return it == _index.end() ? nullptr : &_entries[it->second];
    }
    const bgm_entry_view &require(const std::string &name) const {
        const auto *e = lookup(name);
        PACK_CHECK(e, "the packed file has no tensor named '%s' (name_hash %llu)", name.c_str(),
                   static_cast<unsigned long long>(fnv1a_64(name)));
        return *e;
    }
    std::vector<float> read_all(const std::string &name) const {
        const auto &e = require(name);
        const uint64_t elems = e.num_elements();
        std::vector<float> out(static_cast<size_t>(elems));
        if (_esz == 4ull) {
            read_at(e.offset, out.data(), static_cast<size_t>(e.size_bytes));
        } else if (_esz == 1ull) {// v3: dequantize with the tensor's own scale
            _raw.assign(static_cast<size_t>(e.size_bytes), 0u);
            read_at(e.offset, _raw.data(), static_cast<size_t>(e.size_bytes));
            decode_int8(_raw, out.data(), e.scale);
        } else {
            _raw.assign(static_cast<size_t>(e.size_bytes), 0u);
            read_at(e.offset, _raw.data(), static_cast<size_t>(e.size_bytes));
            decode_halves(_raw, out.data());
        }
        return out;
    }
    void read_rows(const std::string &name, uint64_t row, uint64_t rows, float *dst) const {
        const auto &e = require(name);
        PACK_CHECK(e.rank == 2, "'%s' is not rank-2", name.c_str());
        const uint64_t cols = e.dims[1];
        PACK_CHECK(row + rows <= e.dims[0], "'%s': row range out of bounds", name.c_str());
        const uint64_t nelem = rows * cols;
        if (_esz == 4ull) {
            read_at(e.offset + row * cols * 4, dst, static_cast<size_t>(nelem * 4));
        } else if (_esz == 1ull) {
            _raw.assign(static_cast<size_t>(nelem), 0u);
            read_at(e.offset + row * cols, _raw.data(), static_cast<size_t>(nelem));
            decode_int8(_raw, dst, e.scale);
        } else {
            _raw.assign(static_cast<size_t>(nelem * 2), 0u);
            read_at(e.offset + row * cols * 2, _raw.data(), static_cast<size_t>(nelem * 2));
            decode_halves(_raw, dst);
        }
    }

private:
    // v3 blob bytes -> fp32 (q * scale; the exact dequantization the engine
    // performs, mirrored from examples/embedding/embed/quantization.h)
    void decode_int8(const std::vector<uint8_t> &bytes, float *dst, float scale) const {
        for (size_t i = 0; i < bytes.size(); i++) {
            dst[i] = int8_to_float(static_cast<int8_t>(bytes[i]), scale);
        }
    }
    // v2 blob bytes -> fp32 (the exact upcasts of the format spec; the
    // engine's kernel-side load path performs the same conversion)
    void decode_halves(const std::vector<uint8_t> &bytes, float *dst) const {
        const size_t n = bytes.size() / 2;
        for (size_t i = 0; i < n; i++) {
            const uint16_t h = static_cast<uint16_t>(bytes[2 * i] | (static_cast<uint16_t>(bytes[2 * i + 1]) << 8));
            dst[i] = _precision == kBgmPrecisionBf16 ? bf16_to_f32(h) : f16_to_f32(h);
        }
    }

    void read_at(uint64_t off, void *dst, size_t n) const {
        PACK_CHECK(off + n <= _size, "read [%llu,%llu) is past the file", static_cast<unsigned long long>(off),
                   static_cast<unsigned long long>(off + n));
        seek64(_f, off);
        PACK_CHECK(std::fread(dst, 1, n, _f) == n, "read failed at %llu of the packed file",
                   static_cast<unsigned long long>(off));
    }

    std::FILE *_f = nullptr;
    uint64_t _size = 0;
    uint32_t _version = 0;
    uint32_t _precision = 0;
    uint64_t _esz = 4;
    uint32_t _count = 0;
    std::vector<bgm_entry_view> _entries;
    std::unordered_map<uint64_t, uint32_t> _index;
    mutable std::vector<uint8_t> _raw;// half-blob staging for read_all/read_rows
};

void summarize_package(const bgm_reader &r) {
    const auto plan = build_plan();
    info("model.bgm: %llu bytes, version %u, precision %s, tensor_count %u\n",
         static_cast<unsigned long long>(r.file_size()), r.version(), r.precision_name(), r.count());
    auto name_of = [&](uint64_t hash) -> std::string {
        for (const auto &p : plan) {
            if (fnv1a_64(p.name) == hash) return p.name;
        }
        return "<unknown>";
    };
    const uint32_t show = std::min<uint32_t>(r.count(), 6u);
    for (uint32_t i = 0; i < show; i++) {
        const auto &e = r.entry(i);
        if (r.precision() == kBgmPrecisionInt8) {
            info("  [%3u] %-14s hash 0x%016llx offset 0x%08llx size %llu B rank %u dims [%u,%u,%u,%u] int8 scale %.9g\n", i,
                 name_of(e.hash).c_str(), static_cast<unsigned long long>(e.hash), static_cast<unsigned long long>(e.offset),
                 static_cast<unsigned long long>(e.size_bytes), e.rank, e.dims[0], e.dims[1], e.dims[2], e.dims[3],
                 static_cast<double>(e.scale));
        } else {
            info("  [%3u] %-14s hash 0x%016llx offset 0x%08llx size %llu B rank %u dims [%u,%u,%u,%u]\n", i,
                 name_of(e.hash).c_str(), static_cast<unsigned long long>(e.hash), static_cast<unsigned long long>(e.offset),
                 static_cast<unsigned long long>(e.size_bytes), e.rank, e.dims[0], e.dims[1], e.dims[2], e.dims[3]);
        }
    }
    info("  ... (%u entries total; the last one is '%s')\n", r.count(), name_of(r.entry(r.count() - 1).hash).c_str());
    uint64_t aligned = 0;
    for (uint32_t i = 0; i < r.count(); i++) aligned += align_up(r.entry(i).size_bytes, kBgmAlign);
    info("  payload %llu bytes, %llu bytes including 256-byte alignment padding\n",
         static_cast<unsigned long long>([&] {
             uint64_t s = 0;
             for (uint32_t i = 0; i < r.count(); i++) s += r.entry(i).size_bytes;
             return s;
         }()),
         static_cast<unsigned long long>(align_up(16 + 48ull * r.count(), kBgmAlign) + aligned - 16 - 48ull * r.count()));
}

}// namespace pack_model_detail

namespace pack_model_detail {
// ---------------------------------------------------------------------------
// naive CPU fp32 forward (plan §3.3 verification step)
// ---------------------------------------------------------------------------
// Same math order as the ONNX graph: fused QKV projection, per-head scaled
// QK^T (the graph divides the logits by 8.0 = sqrt(head_dim)), additive padding
// mask, softmax, P*V, output projection, residual + LayerNorm, up-projection +
// Erf-GELU, down-projection, residual + LayerNorm; dense = L2-normalized CLS row
// of the last layer. All weights are read back from the packed .bgm, so this
// exercises the package rather than the in-memory copy.

inline void gemm_f32(const float *a, uint64_t m, const float *b, uint64_t k, uint64_t n, const float *bias, float *c) {
    // C[M,N] = A[M,K] @ B[K,N] (+ bias[N]); B is the packed row-major [K][N].
    for (uint64_t i = 0; i < m; i++) {
        float *row = c + i * n;
        if (bias) {
            std::memcpy(row, bias, static_cast<size_t>(n) * sizeof(float));
        } else {
            std::memset(row, 0, static_cast<size_t>(n) * sizeof(float));
        }
    }
    for (uint64_t j = 0; j < k; j++) {
        const float *brow = b + j * n;
        for (uint64_t i = 0; i < m; i++) {
            const float av = a[i * k + j];
            float *crow = c + i * n;
            for (uint64_t t = 0; t < n; t++) crow[t] += av * brow[t];
        }
    }
}

// y = (x - mean) / sqrt(var + eps) * gamma + beta, per row of `cols` elements
void layernorm_f32(const float *x, float *y, uint64_t rows, uint64_t cols, const float *gamma, const float *beta) {
    for (uint64_t r = 0; r < rows; r++) {
        const float *src = x + r * cols;
        float *dst = y + r * cols;
        double sum = 0.0;
        for (uint64_t c = 0; c < cols; c++) sum += src[c];
        const double mean = sum / static_cast<double>(cols);
        double sq = 0.0;
        for (uint64_t c = 0; c < cols; c++) {
            const double d = src[c] - mean;
            sq += d * d;
        }
        const float inv = static_cast<float>(1.0 / std::sqrt(sq / static_cast<double>(cols) + kLnEps));
        for (uint64_t c = 0; c < cols; c++) dst[c] = (src[c] - mean) * inv * gamma[c] + beta[c];
    }
}

float gelu_erf_f32(float v) { return 0.5f * v * (1.0f + static_cast<float>(std::erf(static_cast<double>(v) * 0.70710678118654752))); }

struct forward_state {
    std::vector<float> x;   // [S,H] current activations
    std::vector<float> tmp; // [S,3H] fused QKV output
    std::vector<float> next;// [S,H] scratch
};

void encode_sequence(const std::vector<int64_t> &ids, const bgm_reader &r, forward_state &st) {
    const uint64_t S = ids.size();
    st.x.assign(S * kHidden, 0.0f);
    st.tmp.assign(S * 3 * static_cast<size_t>(kHidden), 0.0f);
    st.next.assign(S * kHidden, 0.0f);

    // --- embeddings: word (+ folded token-type) + position, then LayerNorm ---
    std::vector<float> word_row(static_cast<size_t>(kHidden));
    std::vector<float> pos_row(static_cast<size_t>(kHidden));
    uint64_t cum = 0;
    for (uint64_t t = 0; t < S; t++) {
        const bool real = ids[t] != 1;// pad token id
        cum += real ? 1 : 0;
        const uint64_t pos = real ? cum + 1 : 1;// graph: cumsum(non-pad)*(non-pad) + 1
        r.read_rows("emb.word", static_cast<uint64_t>(ids[t]), 1, word_row.data());
        r.read_rows("emb.pos", pos, 1, pos_row.data());
        float *dst = st.x.data() + t * kHidden;
        for (int c = 0; c < kHidden; c++) dst[c] = word_row[c] + pos_row[c];
    }
    {
        const auto gamma = r.read_all("emb.ln.gamma");
        const auto beta = r.read_all("emb.ln.beta");
        layernorm_f32(st.x.data(), st.next.data(), S, kHidden, gamma.data(), beta.data());
        std::swap(st.x, st.next);
    }

    // padding mask: additive bias on the key axis, (1 - mask) * finfo.min
    std::vector<float> mask_bias(S, 0.0f);
    for (uint64_t t = 0; t < S; t++) {
        if (ids[t] == 1) mask_bias[t] = kPadBias;
    }

    std::vector<float> attn(static_cast<size_t>(S) * S);
    std::vector<float> ctx(static_cast<size_t>(S) * kHidden);
    std::vector<float> up(static_cast<size_t>(S) * kIntermediate);
    for (int layer = 0; layer < kNumLayers; layer++) {
        const std::string p = "layer" + std::to_string(layer) + ".";
        const auto qkv_w = r.read_all(p + "qkv");
        const auto qkv_b = r.read_all(p + "qkv.bias");
        gemm_f32(st.x.data(), S, qkv_w.data(), kHidden, 3 * static_cast<uint64_t>(kHidden), qkv_b.data(), st.tmp.data());

        // attention: per head scores = (Q K^T)/8 + mask_bias, softmax, ctx = P V
        std::fill(attn.begin(), attn.end(), 0.0f);
        std::fill(ctx.begin(), ctx.end(), 0.0f);
        const uint64_t qkv_stride = 3 * static_cast<uint64_t>(kHidden);
        for (int h = 0; h < kHeads; h++) {
            const float *q = st.tmp.data() + h * kHeadDim;
            const float *k = st.tmp.data() + static_cast<size_t>(kHidden) + h * kHeadDim;
            const float *v = st.tmp.data() + 2 * static_cast<size_t>(kHidden) + h * kHeadDim;
            for (uint64_t i = 0; i < S; i++) {
                float *ap = attn.data() + i * S;
                float row_max = -3.0e38f;
                for (uint64_t j = 0; j < S; j++) {
                    const float *qq = q + i * qkv_stride;
                    const float *kk = k + j * qkv_stride;
                    float acc = 0.0f;
                    for (int d = 0; d < kHeadDim; d++) acc += qq[d] * kk[d];
                    const float s = acc / 8.0f + mask_bias[j];// the graph divides by 8.0
                    ap[j] = s;
                    row_max = s > row_max ? s : row_max;
                }
                float sum = 0.0f;
                for (uint64_t j = 0; j < S; j++) {
                    const float e = std::exp(ap[j] - row_max);
                    ap[j] = e;
                    sum += e;
                }
                const float inv = sum > 0.0f ? 1.0f / sum : 0.0f;
                float *dst = ctx.data() + i * kHidden + h * kHeadDim;
                for (uint64_t j = 0; j < S; j++) {
                    const float pr = ap[j] * inv;
                    if (pr == 0.0f) continue;// fully masked keys contribute nothing
                    const float *vj = v + j * qkv_stride;
                    for (int d = 0; d < kHeadDim; d++) dst[d] += pr * vj[d];
                }
            }
        }

        const auto out_w = r.read_all(p + "attn.out");
        const auto out_b = r.read_all(p + "attn.out.bias");
        gemm_f32(ctx.data(), S, out_w.data(), kHidden, kHidden, out_b.data(), st.next.data());
        for (uint64_t t = 0; t < S * static_cast<uint64_t>(kHidden); t++) st.next[t] += st.x[t];// residual
        {
            const auto g = r.read_all(p + "ln1.gamma");
            const auto b = r.read_all(p + "ln1.beta");
            layernorm_f32(st.next.data(), st.x.data(), S, kHidden, g.data(), b.data());
        }
        const auto up_w = r.read_all(p + "ffn.up");
        const auto up_b = r.read_all(p + "ffn.up.bias");
        const auto down_w = r.read_all(p + "ffn.down");
        const auto down_b = r.read_all(p + "ffn.down.bias");
        gemm_f32(st.x.data(), S, up_w.data(), kHidden, kIntermediate, up_b.data(), up.data());
        for (auto &u : up) u = gelu_erf_f32(u);
        std::fill(st.next.begin(), st.next.end(), 0.0f);
        gemm_f32(up.data(), S, down_w.data(), kIntermediate, kHidden, down_b.data(), st.next.data());
        for (uint64_t t = 0; t < S * static_cast<uint64_t>(kHidden); t++) st.next[t] += st.x[t];// residual
        {
            const auto g = r.read_all(p + "ln2.gamma");
            const auto b = r.read_all(p + "ln2.beta");
            layernorm_f32(st.next.data(), st.x.data(), S, kHidden, g.data(), b.data());
        }
    }
}

std::vector<float> dense_of(const forward_state &st) {
    // CLS pool (row 0) + L2 normalization with the graph's 1e-12 floor
    std::vector<float> out(static_cast<size_t>(kHidden));
    double sq = 0.0;
    for (int c = 0; c < kHidden; c++) {
        const double v = st.x[c];
        sq += v * v;
        out[static_cast<size_t>(c)] = static_cast<float>(v);
    }
    const double norm = std::max(std::sqrt(sq), 1e-12);
    for (auto &v : out) v = static_cast<float>(static_cast<double>(v) / norm);
    return out;
}

double cosine(const float *a, const float *b, uint64_t n) {
    double dot = 0.0, na = 0.0, nb = 0.0;
    for (uint64_t i = 0; i < n; i++) {
        dot += static_cast<double>(a[i]) * b[i];
        na += static_cast<double>(a[i]) * a[i];
        nb += static_cast<double>(b[i]) * b[i];
    }
    return dot / (std::sqrt(na) * std::sqrt(nb) + 1e-30);
}

void run_check(const std::string &out_path, const std::string &golden_dir, bool dump) {
    const bgm_reader r(out_path);
    summarize_package(r);

    // ---- golden token ids (the first three corpus sentences) -----------------
    const auto tokens_json = read_file_text(path_join(golden_dir, "golden_tokens.json"));
    const auto root = json_parser(tokens_json).parse();
    const auto *entries = root.at("entries");
    PACK_CHECK(entries != nullptr && entries->type == json_value::kind::array, "golden_tokens.json: no 'entries' array");
    std::vector<std::vector<int64_t>> ids_by_index(3);
    for (const auto &e : entries->arr) {
        const auto *idx = e.at("i");
        const auto *ids = e.at("ids");
        if (idx == nullptr || ids == nullptr || idx->num >= 3) continue;
        PACK_CHECK(ids->type == json_value::kind::array, "golden_tokens.json: entry %d has no 'ids' array",
                   static_cast<int>(idx->num));
        auto &dst = ids_by_index[static_cast<size_t>(idx->num)];
        PACK_CHECK(dst.empty(), "golden_tokens.json: duplicate entry i=%d", static_cast<int>(idx->num));
        for (const auto &id : ids->arr) {
            PACK_CHECK(id.type == json_value::kind::number, "golden_tokens.json: non-numeric token id");
            dst.push_back(static_cast<int64_t>(id.num));
        }
    }
    for (size_t i = 0; i < ids_by_index.size(); i++) {
        PACK_CHECK(!ids_by_index[i].empty(), "golden_tokens.json: sentence %zu is missing", i);
        PACK_CHECK(ids_by_index[i].size() >= 1, "golden_tokens.json: sentence %zu has no tokens", i);
        if (dump) {
            info("  sentence %zu: %zu tokens, ids[0..3] = %lld %lld %lld %lld\n", i, ids_by_index[i].size(),
                 static_cast<long long>(ids_by_index[i][0]), static_cast<long long>(ids_by_index[i][1]),
                 static_cast<long long>(ids_by_index[i][2]), static_cast<long long>(ids_by_index[i][3]));
        }
    }

    // ---- golden dense ---------------------------------------------------------
    const auto dense_path = path_join(golden_dir, "golden_dense.bin");
    const auto dense_size = file_size64_of(dense_path);
    PACK_CHECK(dense_size % (static_cast<uint64_t>(kHidden) * 4) == 0, "%s: size %llu is not a multiple of %d bytes",
               dense_path.c_str(), static_cast<unsigned long long>(dense_size), kHidden * 4);
    const uint64_t golden_rows = dense_size / (static_cast<uint64_t>(kHidden) * 4);
    PACK_CHECK(golden_rows >= 3, "%s: only %llu rows", dense_path.c_str(), static_cast<unsigned long long>(golden_rows));
    std::vector<float> golden(3 * static_cast<size_t>(kHidden));
    {
        std::FILE *f = std::fopen(dense_path.c_str(), "rb");
        PACK_CHECK(f, "cannot open '%s'", dense_path.c_str());
        for (size_t row = 0; row < 3; row++) {
            seek64(f, static_cast<uint64_t>(row) * kHidden * 4);
            PACK_CHECK(std::fread(golden.data() + row * kHidden, 1, static_cast<size_t>(kHidden) * 4, f) == static_cast<size_t>(kHidden) * 4,
                       "short read on '%s'", dense_path.c_str());
        }
        std::fclose(f);
    }

    // ---- forward + parity ------------------------------------------------------
    double worst = 1.0;
    double worst_abs = 0.0;
    for (size_t s = 0; s < 3; s++) {
        forward_state st;
        encode_sequence(ids_by_index[s], r, st);
        const auto out = dense_of(st);
        const float *gt = golden.data() + s * kHidden;
        const double c = cosine(out.data(), gt, kHidden);
        double max_abs = 0.0;
        for (int d = 0; d < kHidden; d++) max_abs = std::max(max_abs, std::abs(static_cast<double>(out[d]) - gt[d]));
        info("sentence %zu (S=%zu): cosine vs golden_dense = %.10f, max-abs-diff = %.3e\n", s, ids_by_index[s].size(), c, max_abs);
        worst = std::min(worst, c);
        worst_abs = std::max(worst_abs, max_abs);
    }
    // Parity gate (opt plan §4 / §5): fp32 v1 >= 0.9999; v2 half >= 0.999; v3
    // per-tensor int8 >= 0.99. The int8 gate is lower ON PURPOSE and measured, not
    // guessed: every tensor (LayerNorm weights, biases and the embedding tables
    // included) is quantized to 7-bit symmetric levels, which the dev model
    // measures at 0.9920-0.9937 worst cosine against the SAME frozen fp32
    // goldens, with the dequantized weights feeding the identical fp32 forward.
    // 0.99 still catches any real regression (a broken scale, a byte swap or a
    // mis-dequantized blob lands far below it).
    const double gate = r.precision() == kBgmPrecisionFp32 ? 0.9999 :
                        r.precision() == kBgmPrecisionInt8 ? 0.99 :
                                                            0.999;
    PACK_CHECK(worst >= gate, "internal CPU forward parity FAILED (%s weights): worst cosine %.10f < %.4f (worst max-abs-diff %.3e)",
               r.precision_name(), worst, gate, worst_abs);
    info("parity gate passed (%s weights): worst cosine %.10f >= %.4f, worst max-abs-diff %.3e\n",
         r.precision_name(), worst, gate, worst_abs);
}

}// namespace pack_model_detail

// ---------------------------------------------------------------------------
// entry point
// ---------------------------------------------------------------------------

int main_impl(int argc, char **argv) {
    using namespace pack_model_detail;
    std::string golden_dir;
    bool check = false;
    bool dump = false;
    uint32_t precision = kBgmPrecisionFp32;
    std::vector<std::string> positional;
    for (int i = 1; i < argc; i++) {
        const std::string a = argv[i];
        if (a == "--check") {
            check = true;
        } else if (a == "--dump") {
            dump = true;
        } else if (a == "--golden-dir") {
            PACK_CHECK(i + 1 < argc, "--golden-dir needs a directory argument");
            golden_dir = argv[++i];
        } else if (a.rfind("--golden-dir=", 0) == 0) {
            golden_dir = a.substr(std::strlen("--golden-dir="));
        } else if (a == "--precision" || a.rfind("--precision=", 0) == 0) {
            std::string p = a.rfind("--precision=", 0) == 0 ? a.substr(std::strlen("--precision=")) : std::string{};
            if (p.empty()) {
                PACK_CHECK(i + 1 < argc, "--precision needs an argument: fp32|bf16|fp16|int8");
                p = argv[++i];
            }
            if (p == "fp32") {
                precision = kBgmPrecisionFp32;
            } else if (p == "bf16") {
                precision = kBgmPrecisionBf16;
            } else if (p == "fp16") {
                precision = kBgmPrecisionFp16;
            } else if (p == "int8") {
                precision = kBgmPrecisionInt8;
            } else {
                PACK_CHECK(false, "--precision '%s' is not fp32|bf16|fp16|int8", p.c_str());
            }
        } else if (a == "-h" || a == "--help") {
            info("usage: pack_model <model_dir> <out.bgm> [--golden-dir <dir>] [--check] [--dump] "
                 "[--precision fp32|bf16|fp16|int8]\n");
            return 0;
        } else {
            positional.push_back(a);
        }
    }
    PACK_CHECK(positional.size() == 2,
               "expected <model_dir> <out.bgm> [--golden-dir <dir>] [--check] [--dump] "
               "[--precision fp32|bf16|fp16|int8], "
               "got %zu positional argument(s)",
               positional.size());
    std::string model_dir = positional[0];
    const std::string out_path = positional[1];
    while (!model_dir.empty() && (model_dir.back() == '/' || model_dir.back() == '\\')) model_dir.pop_back();
    if (check) PACK_CHECK(!golden_dir.empty(), "--check requires --golden-dir <dir>");

    config_info cfg;
    const auto cfg_text = read_file_text(path_join(model_dir, "config.json"), true);
    if (!cfg_text.empty()) {
        parse_config(cfg_text, cfg);
        info("config.json: %lld layers, hidden %lld, intermediate %lld, heads %lld, vocab %lld, max_pos %lld, "
             "type_vocab %lld, eps %g, act %s\n",
             static_cast<long long>(cfg.num_hidden_layers), static_cast<long long>(cfg.hidden_size),
             static_cast<long long>(cfg.intermediate_size), static_cast<long long>(cfg.num_attention_heads),
             static_cast<long long>(cfg.vocab_size), static_cast<long long>(cfg.max_position_embeddings),
             static_cast<long long>(cfg.type_vocab_size), cfg.layer_norm_eps, cfg.hidden_act.c_str());
        PACK_CHECK(cfg.vocab_size == kVocab, "config.json vocab_size %lld != %d", static_cast<long long>(cfg.vocab_size), kVocab);
        PACK_CHECK(cfg.max_position_embeddings == kMaxPos, "config.json max_position_embeddings %lld != %d",
                   static_cast<long long>(cfg.max_position_embeddings), kMaxPos);
        PACK_CHECK(!cfg.hidden_act_found || cfg.hidden_act == "gelu",
                   "config.json hidden_act '%s' is not the Erf-GELU (0.5*x*(1+erf(x/sqrt(2)))) the internal check implements",
                   cfg.hidden_act.c_str());
    } else {
        info("config.json not found in '%s'; using the architecture constants of the format spec\n", model_dir.c_str());
    }

    run_pack(model_dir, out_path, cfg, dump, precision);

    if (check) {
        info("--- internal CPU forward check (weights read back from '%s') ---\n", out_path.c_str());
        run_check(out_path, golden_dir, dump);
    } else {
        info("(pass --golden-dir <dir> --check to run the internal CPU forward parity gate)\n");
    }
    return 0;
}

int main(int argc, char **argv) {
    return main_impl(argc, argv);
}
