#include "simd_embree_packet_support.h"

using luisa::compute::simd::SIMDEmbreeNativeRayPacketSupport;

constexpr auto no_packets = SIMDEmbreeNativeRayPacketSupport{};
static_assert(no_packets.supports(1u));
static_assert(!no_packets.supports(2u));
static_assert(!no_packets.supports(4u));
static_assert(!no_packets.supports(8u));
static_assert(!no_packets.supports(16u));

constexpr auto w4_only = SIMDEmbreeNativeRayPacketSupport{.w4 = true};
static_assert(w4_only.supports(2u));
static_assert(w4_only.supports(4u));
static_assert(!w4_only.supports(8u));
static_assert(!w4_only.supports(16u));

constexpr auto w4_w8 = SIMDEmbreeNativeRayPacketSupport{
    .w4 = true,
    .w8 = true,
};
static_assert(w4_w8.supports(2u));
static_assert(w4_w8.supports(4u));
static_assert(w4_w8.supports(8u));
static_assert(!w4_w8.supports(16u));

constexpr auto all_packets = SIMDEmbreeNativeRayPacketSupport{
    .w4 = true,
    .w8 = true,
    .w16 = true,
};
static_assert(all_packets.supports(16u));
static_assert(!all_packets.supports(0u));
static_assert(!all_packets.supports(32u));

int main() noexcept { return 0; }
