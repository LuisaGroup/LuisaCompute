// mm_embedding engine -- package loader implementation.
#include "mm_embedding/embed/package.h"

#include <cstdio>
#include <cstring>

#include <luisa/core/logging.h>
#include <luisa/core/stl/format.h>
#include <luisa/core/stl/unordered_map.h>

namespace luisa::mm_embed {

const char *package_precision_name(PackagePrecision p) noexcept {
    switch (p) {
        case PackagePrecision::fp32: return "fp32";
        case PackagePrecision::bf16: return "bf16";
        case PackagePrecision::fp16: return "fp16";
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

constexpr uint32_t kEgmMagic = 0x324d4745u;// "EGM2" in LE
constexpr uint32_t kEgmVersion = 1u;
constexpr uint64_t kEgmAlign = 256ull;

constexpr uint32_t kNumLayers = 24u;
constexpr uint32_t kHiddenSize = 512u;
constexpr uint32_t kNumHeads = 4u;
constexpr uint32_t kNumKvHeads = 2u;
constexpr uint32_t kHeadDim = 256u;
constexpr uint32_t kIntermediateSize = 2048u;
constexpr uint32_t kVocabSize = 262144u;
constexpr uint32_t kEmbeddingDim = 768u;
constexpr uint32_t kMaxPositions = 8192u;

uint64_t fnv1a_64(luisa::string_view name) noexcept {
    uint64_t h = 0xcbf29ce484222325ull;
    for (const auto c : name) {
        h ^= static_cast<uint8_t>(c);
        h *= 0x00000100000001b3ull;
    }
    return h;
}

#pragma pack(push, 1)
struct EgmHeader {
    uint32_t magic;
    uint32_t version;
    uint32_t precision;
    uint32_t num_tensors;
    uint32_t num_layers;
    uint32_t hidden_size;
    uint32_t intermediate_size;
    uint32_t embedding_dim;
};

struct EgmEntry {
    uint64_t name_hash;
    uint64_t offset;
    uint64_t size_bytes;
    uint32_t rank;
    uint32_t dims[4];
};
#pragma pack(pop)

class PackageImpl final : public Package {
public:
    luisa::vector<uint8_t> raw_buffer;
    const uint8_t *aligned_base{};
    PackagePrecision prec{PackagePrecision::fp32};

    luisa::vector<Tensor> tensor_list;
    luisa::unordered_map<uint64_t, uint32_t> hash_to_index;

    [[nodiscard]] const Tensor *find(luisa::string_view canonical_name) const noexcept override {
        auto h = fnv1a_64(canonical_name);
        auto it = hash_to_index.find(h);
        if (it == hash_to_index.end()) {
            return nullptr;
        }
        return &tensor_list[it->second];
    }

    [[nodiscard]] luisa::span<const Tensor> tensors() const noexcept override {
        return tensor_list;
    }

    [[nodiscard]] PackagePrecision precision() const noexcept override {
        return prec;
    }

    uint32_t _num_layers{kNumLayers};
    uint32_t _hidden_size{kHiddenSize};
    uint32_t _intermediate_size{kIntermediateSize};
    uint32_t _embedding_dim{kEmbeddingDim};

    [[nodiscard]] uint32_t num_layers() const noexcept override { return _num_layers; }
    [[nodiscard]] uint32_t hidden_size() const noexcept override { return _hidden_size; }
    [[nodiscard]] uint32_t num_heads() const noexcept override { return kNumHeads; }
    [[nodiscard]] uint32_t num_kv_heads() const noexcept override { return kNumKvHeads; }
    [[nodiscard]] uint32_t head_dim() const noexcept override { return kHeadDim; }
    [[nodiscard]] uint32_t intermediate_size() const noexcept override { return _intermediate_size; }
    [[nodiscard]] uint32_t vocab_size() const noexcept override { return kVocabSize; }
    [[nodiscard]] uint32_t embedding_dim() const noexcept override { return _embedding_dim; }
    [[nodiscard]] uint32_t max_positions() const noexcept override { return kMaxPositions; }
};

} // namespace

luisa::unique_ptr<Package> Package::load(const std::filesystem::path &egm_file) {
    std::error_code ec;
    auto file_size = std::filesystem::file_size(egm_file, ec);
    if (ec) {
        LUISA_ERROR("Package::load: failed to get file size for '{}': {}", egm_file.string(), ec.message());
    }
    if (file_size < sizeof(EgmHeader)) {
        LUISA_ERROR("Package::load: file '{}' is too small ({} bytes)", egm_file.string(), file_size);
    }

    FILE *fp = std::fopen(egm_file.string().c_str(), "rb");
    if (!fp) {
        LUISA_ERROR("Package::load: failed to open file '{}'", egm_file.string());
    }

    auto pkg = luisa::make_unique<PackageImpl>();
    // Allocate with padding for 256-byte alignment
    pkg->raw_buffer.resize(static_cast<size_t>(file_size) + kEgmAlign);
    auto raw_ptr = reinterpret_cast<uintptr_t>(pkg->raw_buffer.data());
    auto aligned_ptr = (raw_ptr + kEgmAlign - 1ull) & ~(kEgmAlign - 1ull);
    pkg->aligned_base = reinterpret_cast<const uint8_t *>(aligned_ptr);

    size_t n_read = std::fread(const_cast<uint8_t *>(pkg->aligned_base), 1, static_cast<size_t>(file_size), fp);
    std::fclose(fp);

    if (n_read != static_cast<size_t>(file_size)) {
        LUISA_ERROR("Package::load: failed to read complete file '{}'", egm_file.string());
    }

    const auto &hdr = *reinterpret_cast<const EgmHeader *>(pkg->aligned_base);
    if (hdr.magic != kEgmMagic) {
        LUISA_ERROR("Package::load: invalid magic in '{}' (expected 0x{:08x}, got 0x{:08x})",
                    egm_file.string(), kEgmMagic, hdr.magic);
    }
    if (hdr.version != kEgmVersion) {
        LUISA_ERROR("Package::load: unsupported version {} in '{}'", hdr.version, egm_file.string());
    }

    pkg->_num_layers = hdr.num_layers;
    pkg->_hidden_size = hdr.hidden_size;
    pkg->_intermediate_size = hdr.intermediate_size;
    pkg->_embedding_dim = hdr.embedding_dim;

    pkg->prec = static_cast<PackagePrecision>(hdr.precision);
    size_t table_offset = sizeof(EgmHeader);
    size_t table_bytes = hdr.num_tensors * sizeof(EgmEntry);

    if (table_offset + table_bytes > static_cast<size_t>(file_size)) {
        LUISA_ERROR("Package::load: tensor table exceeds file bounds in '{}'", egm_file.string());
    }

    const auto *entries = reinterpret_cast<const EgmEntry *>(pkg->aligned_base + table_offset);
    pkg->tensor_list.reserve(hdr.num_tensors);

    for (uint32_t i = 0u; i < hdr.num_tensors; i++) {
        const auto &e = entries[i];
        if (e.offset + e.size_bytes > static_cast<uint64_t>(file_size)) {
            LUISA_ERROR("Package::load: tensor {} blob out of bounds in '{}'", i, egm_file.string());
        }

        Tensor t{};
        t.size_bytes = e.size_bytes;
        t.rank = e.rank;
        for (uint32_t d = 0; d < 4; d++) {
            t.dims[d] = e.dims[d];
        }

        const uint8_t *blob_ptr = pkg->aligned_base + e.offset;
        if (pkg->prec == PackagePrecision::fp32) {
            t.data = reinterpret_cast<const float *>(blob_ptr);
        } else {
            t.half_data = reinterpret_cast<const uint16_t *>(blob_ptr);
        }

        pkg->tensor_list.push_back(t);
        pkg->hash_to_index[e.name_hash] = i;
    }

    LUISA_INFO("Package::load: successfully loaded '{}' ({} tensors, precision={})",
               egm_file.string(), hdr.num_tensors, package_precision_name(pkg->prec));

    return pkg;
}

}// namespace luisa::mm_embed
