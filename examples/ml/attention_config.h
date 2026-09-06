// Multi-Head Latent Attention (MLA) Example -- shared configuration.
//
// This header collects the compile-time problem dimensions, derived sizes,
// host-side index helpers, and RoPE utilities shared by the DSL kernels, the
// runtime driver, and the CPU reference.
//
// The constants are kept at namespace scope so they remain constant
// expressions inside DSL lambdas (they are not captured, so they can be used
// as template arguments such as in $array<float, latent_dim>).

#pragma once

#include <array>
#include <cmath>
#include <cstdint>
#include <utility>

namespace mla {

// -- Problem dimensions ----------------------------------------------------
constexpr uint32_t batch = 16u;
constexpr uint32_t num_heads = 8u;
constexpr uint32_t seq_len = 256u;
constexpr uint32_t head_dim = 64u;

// MLA-specific dimensions.
constexpr uint32_t hidden_dim = 512u;// input hidden size h_t
constexpr uint32_t latent_dim = 64u; // compressed KV dim d_c
constexpr uint32_t rope_dim = 16u;   // decoupled RoPE dim d_h^R
constexpr uint32_t content_dim = head_dim - rope_dim;
constexpr float rope_theta = 10000.0f;

static_assert(content_dim > 0u, "head_dim must be larger than rope_dim");
static_assert(rope_dim % 2u == 0u, "rope_dim must be even for pair-wise RoPE");

// Block sizes for the block-per-token projection kernels. The host must
// dispatch (batch * seq_len * block_size) threads so each block covers
// exactly one token; kernels index the token via block_id().x.
constexpr uint32_t project_q_block_size = 256u;  // 2 outputs (one d-pair) per thread
constexpr uint32_t project_kv_block_size = 128u; // one cKV output or Krope pair per thread
static_assert(num_heads * head_dim == 2u * project_q_block_size);
static_assert(project_kv_block_size == latent_dim + num_heads * rope_dim / 2u);

// Cooperative vector chunk size (max supported by CoopVecConstructor codegen).
constexpr uint32_t kCoopChunk = 16u;
constexpr uint32_t kLatentChunks = latent_dim / kCoopChunk;
constexpr uint32_t kHiddenChunks = hidden_dim / kCoopChunk;
constexpr uint32_t kContentChunks = content_dim / kCoopChunk;
static_assert(latent_dim % kCoopChunk == 0u);
static_assert(hidden_dim % kCoopChunk == 0u);
static_assert(content_dim % kCoopChunk == 0u);

// -- Derived buffer sizes --------------------------------------------------
constexpr uint32_t hidden_size = batch * seq_len * hidden_dim;
constexpr uint32_t latent_size = batch * seq_len * latent_dim;
constexpr uint32_t rope_size = batch * num_heads * seq_len * rope_dim;
constexpr uint32_t qkv_size = batch * num_heads * seq_len * head_dim;
constexpr uint32_t scores_size = batch * num_heads * seq_len * seq_len;

// Weight sizes.
constexpr uint32_t wq_size = num_heads * head_dim * hidden_dim;
constexpr uint32_t wdkv_size = latent_dim * hidden_dim;
constexpr uint32_t wuk_size = num_heads * content_dim * latent_dim;
constexpr uint32_t wuv_size = num_heads * head_dim * latent_dim;
constexpr uint32_t wkr_size = num_heads * rope_dim * hidden_dim;

// -- Attention scale (1 / sqrt(head_dim), constexpr) -----------------------
namespace detail {
constexpr float constexpr_sqrt(float x) noexcept {
    float curr = x;
    float prev = 0.0f;
    while (curr != prev) {
        prev = curr;
        curr = 0.5f * (curr + x / curr);
    }
    return curr;
}
}// namespace detail
constexpr float attention_scale = 1.0f / detail::constexpr_sqrt(static_cast<float>(head_dim));

// -- Host-side index helpers (CPU reference) -------------------------------
constexpr auto qkv_index = [](uint32_t b, uint32_t h, uint32_t i, uint32_t d) constexpr noexcept {
    return ((b * num_heads + h) * seq_len + i) * head_dim + d;
};
constexpr auto score_index = [](uint32_t b, uint32_t h, uint32_t i, uint32_t j) constexpr noexcept {
    return ((b * num_heads + h) * seq_len + i) * seq_len + j;
};
constexpr auto hidden_index = [](uint32_t b, uint32_t i, uint32_t d) constexpr noexcept {
    return (b * seq_len + i) * hidden_dim + d;
};
constexpr auto latent_index = [](uint32_t b, uint32_t i, uint32_t d) constexpr noexcept {
    return (b * seq_len + i) * latent_dim + d;
};
constexpr auto rope_index = [](uint32_t b, uint32_t h, uint32_t i, uint32_t d) constexpr noexcept {
    return ((b * num_heads + h) * seq_len + i) * rope_dim + d;
};

// -- RoPE utilities --------------------------------------------------------
// Precompute per-pair inverse frequencies (host-side) to avoid pow() in every
// kernel call. For realistic seq_len >= 2048 this matters.
[[nodiscard]] inline std::array<float, rope_dim / 2u> make_rope_inv_freqs() noexcept {
    std::array<float, rope_dim / 2u> f{};
    for (uint32_t p = 0u; p < rope_dim / 2u; ++p) {
        f[p] = 1.0f / std::pow(rope_theta,
                               (2.0f * static_cast<float>(p)) / static_cast<float>(rope_dim));
    }
    return f;
}

// CPU-side pair-wise RoPE rotation.
[[nodiscard]] inline std::pair<float, float> apply_rope_pair_cpu(float x0, float x1, uint32_t pair, uint32_t pos) noexcept {
    float freq = 1.0f / std::pow(rope_theta, (2.0f * static_cast<float>(pair)) / static_cast<float>(rope_dim));
    float angle = static_cast<float>(pos) * freq;
    float c = std::cos(angle);
    float s = std::sin(angle);
    return std::make_pair(x0 * c - x1 * s, x0 * s + x1 * c);
}

// -- Paged KV cache (vLLM-style) -------------------------------------------
// The paged path models vLLM's PagedAttention: the KV cache lives in
// fixed-size physical pages (one sparse-buffer tile each). Every sequence's
// tokens are cut into same-sized logical pages, and a block table maps each
// logical page to a (scattered) physical page -- the OS "page table" analog.
//
// Physical page layout is [h][t_in_page][d]: a page holds all `num_heads`
// heads for `tokens_per_page` tokens. Because the tile size is chosen by the
// device at runtime, the page geometry (tokens_per_page, pages_per_seq,
// elems_per_page) is discovered on the host and passed to the kernels as
// runtime `uint` arguments -- it cannot be constexpr.
//
// In-page element offset of (head h, in-page token t, dim d):
constexpr auto paged_page_offset = [](uint32_t h, uint32_t t, uint32_t d, uint32_t tokens_per_page) constexpr noexcept {
    return (h * tokens_per_page + t) * head_dim + d;
};
// Physical element index of a page-local offset within the paged pool:
constexpr auto paged_kv_index = [](uint32_t phys_page, uint32_t page_offset, uint32_t elems_per_page) constexpr noexcept {
    return phys_page * elems_per_page + page_offset;
};

// Shared-memory sub-tile cap for the paged attention kernel. The real
// sub-tile is the largest divisor of the runtime tokens_per_page that is <=
// this cap, so the shared K/V tile stays small. 16 tokens/sub-tile pairs with
// the kernel's 128-thread block: 16*64 floats = 8 loads/thread/fill and two
// barriers per 16 tokens (half the barrier rate of the 8-token MHA tile).
constexpr uint32_t paged_sub_tile_max = 16u;
// Thread-block size of the paged attention kernel (4 warps). A block spans
// consecutive i of one (b, h), so the block-table read is block-uniform.
constexpr uint32_t paged_attention_block_size = 128u;
static_assert(seq_len % paged_attention_block_size == 0u);

// Largest divisor of `value` that is <= `cap`. Requires cap >= 1; always
// returns at least 1 (1 divides everything), so the result is a safe loop
// bound even when the device tile is much smaller than requested.
[[nodiscard]] inline uint32_t paged_largest_divisor(uint32_t value, uint32_t cap) noexcept {
    uint32_t limit = cap < value ? cap : value;
    while (limit > 1u && value % limit != 0u) {
        --limit;
    }
    return limit;
}

// Tokens per shared-memory sub-tile for the paged attention kernel: the
// largest divisor of the runtime tokens_per_page that is <= paged_sub_tile_max.
[[nodiscard]] inline uint32_t paged_sub_tile(uint32_t tokens_per_page) noexcept {
    return paged_largest_divisor(tokens_per_page, paged_sub_tile_max);
}

}// namespace mla
