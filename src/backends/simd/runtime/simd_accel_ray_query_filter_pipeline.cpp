#include "simd_accel.h"
#include "simd_accel_ray_query.h"

#include <array>
#include <bit>
#include <cstring>
#include <cstdlib>
#include <limits>

#include <luisa/core/logging.h>

namespace luisa::compute::simd::triangle_ray_query {

using RayQueryRTCContext = detail::RayQueryRTCContext;

struct SurfaceFilterPipelineAccelAccess {
    [[nodiscard]] static RTCScene scene(SIMDAccel &accel) noexcept {
        return accel._scene;
    }
    [[nodiscard]] static const SIMDHostAccelInstanceTable &instances(
        SIMDAccel &accel) noexcept {
        return accel._instance_table;
    }
    [[nodiscard]] static bool direct_surface_filter_candidate(
        SIMDAccel &accel) noexcept {
        return accel._enable_direct_surface_filter_candidate;
    }
};

namespace {

struct SurfaceFilterPipelineContext {
    RayQueryRTCContext rtc{};
    uint32_t lane_count{0u};
    const SIMDHostAccelInstanceTable *instances{nullptr};
    std::array<SIMDHostRayQueryState *, 16u> states{};
    const SIMDPacketLaunchConfig *launch_config{nullptr};
    SIMDHostRayQuerySurfaceFilterHandler *on_surface{nullptr};
    SIMDHostRayQueryDirectSurfaceFilterHandler *on_surface_direct{nullptr};
    bool use_direct_surface_candidate{false};
};
static_assert(offsetof(SurfaceFilterPipelineContext, rtc) == 0u);

[[nodiscard]] constexpr float embree_tnear(float tnear) noexcept {
    static_assert(
        sizeof(float) == sizeof(uint32_t) &&
        std::numeric_limits<float>::is_iec559);
    constexpr auto sign_bit = 0x80000000u;
    constexpr auto magnitude_mask = 0x7fffffffu;
    constexpr auto minimum_normal_bits = 0x00800000u;
    constexpr auto infinity_bits = 0x7f800000u;
    auto bits = std::bit_cast<uint32_t>(tnear);
    auto magnitude = bits & magnitude_mask;
    if (magnitude >= infinity_bits) { return tnear; }
    if (magnitude == 0u) {
        return std::bit_cast<float>(sign_bit | minimum_normal_bits);
    }
    bits = (bits & sign_bit) != 0u ? bits + 1u : bits - 1u;
    if ((bits & magnitude_mask) < minimum_normal_bits) {
        bits = sign_bit | minimum_normal_bits;
    }
    return std::bit_cast<float>(bits);
}

[[nodiscard]] bool instance_is_opaque(
    const SIMDHostAccelInstanceTable &instances,
    uint32_t instance) noexcept {
    LUISA_ASSERT(
        instances.committed_instances != nullptr &&
            instance < instances.committed_size,
        "SIMD in-filter ray query returned invalid instance {}.",
        instance);
    return instances.data != nullptr && instance < instances.size ?
               instances.data[instance].opaque != 0u :
               instances.committed_instances[instance].opaque != 0u;
}

void commit_surface_candidate(
    SIMDHostRayQueryState &state,
    const SIMDHostRayQuerySurfaceHit &candidate) noexcept {
    state.committed = SIMDHostRayQueryCommittedHit{
        .inst = candidate.inst,
        .prim = candidate.prim,
        .bary = {candidate.bary[0u], candidate.bary[1u]},
        .kind = static_cast<uint32_t>(
            SIMDHostRayQueryCandidateKind::surface),
        .t = candidate.t,
    };
    state.world_ray[7u] = candidate.t;
    state.candidate_committed = 0u;
}

template<size_t packet_width>
[[nodiscard]] uint32_t valid_mask(const int *valid) noexcept {
    auto mask = uint32_t{0u};
    for (auto lane = uint32_t{0u}; lane < packet_width; lane++) {
        mask |= static_cast<uint32_t>(valid[lane] == -1) << lane;
    }
    return mask;
}

template<size_t packet_width>
void surface_filter_pipeline(
    const RTCFilterFunctionNArguments *arguments) noexcept {
    auto *context = reinterpret_cast<SurfaceFilterPipelineContext *>(
        arguments->context);
    if (context == nullptr || context->instances == nullptr ||
        context->launch_config == nullptr ||
        (context->use_direct_surface_candidate ?
             context->on_surface_direct == nullptr :
             context->on_surface == nullptr) ||
        arguments->valid == nullptr ||
        arguments->ray == nullptr || arguments->hit == nullptr ||
        arguments->N != packet_width) [[unlikely]] {
        std::abort();
    }
    auto candidates = uint64_t{0u};
    auto packet_candidates = uint64_t{0u};
    std::array<uint32_t, 16u> packet_lanes{};
    packet_lanes.fill(std::numeric_limits<uint32_t>::max());
    auto load_candidate = [&](uint32_t packet_lane) noexcept {
        return SIMDHostRayQuerySurfaceHit{
            .inst = RTCHitN_instID(
                arguments->hit, packet_width, packet_lane, 0u),
            .prim = RTCHitN_primID(
                arguments->hit, packet_width, packet_lane),
            .bary = {
                RTCHitN_u(
                    arguments->hit, packet_width, packet_lane),
                RTCHitN_v(
                    arguments->hit, packet_width, packet_lane)},
            .t = RTCRayN_tfar(arguments->ray, packet_width, packet_lane),
        };
    };
    auto remaining = valid_mask<packet_width>(arguments->valid);
    while (remaining != 0u) {
        auto packet_lane = static_cast<uint32_t>(
            std::countr_zero(remaining));
        remaining &= remaining - 1u;
        auto lane = RTCRayN_id(
            arguments->ray, packet_width, packet_lane);
        if (lane >= context->lane_count) {
            arguments->valid[packet_lane] = 0;
            continue;
        }
        auto *state = context->states[lane];
        if (state == nullptr || state->terminated != 0u) {
            arguments->valid[packet_lane] = 0;
            continue;
        }
        auto candidate = load_candidate(packet_lane);
        if (!(candidate.t >= state->world_ray[3u] &&
              candidate.t <= state->world_ray[7u]) ||
            candidate.inst == RTC_INVALID_GEOMETRY_ID ||
            candidate.prim == RTC_INVALID_GEOMETRY_ID) {
            arguments->valid[packet_lane] = 0;
            continue;
        }
        if (instance_is_opaque(*context->instances, candidate.inst)) {
            commit_surface_candidate(*state, candidate);
            continue;
        }
        packet_lanes[lane] = packet_lane;
        candidates |= uint64_t{1u} << lane;
        packet_candidates |= uint64_t{1u} << packet_lane;
        if (!context->use_direct_surface_candidate) {
            state->candidate = candidate;
            state->candidate_kind = static_cast<uint32_t>(
                SIMDHostRayQueryCandidateKind::surface);
            state->candidate_committed = 0u;
        }
    }
    if (candidates == 0u) { return; }
    auto committed_packet_lanes = uint64_t{0u};
    if (context->use_direct_surface_candidate) {
        context->on_surface_direct(
            context->lane_count, packet_candidates,
            arguments->ray, arguments->hit,
            &committed_packet_lanes);
        LUISA_ASSERT(
            (committed_packet_lanes & ~packet_candidates) == 0u,
            "SIMD direct surface filter committed inactive packet lanes.");
    } else {
        context->on_surface(
            context->lane_count, candidates,
            context->states.data(), context->launch_config);
    }
    remaining = candidates;
    while (remaining != 0u) {
        auto lane = static_cast<uint32_t>(
            std::countr_zero(remaining));
        remaining &= remaining - 1u;
        auto packet_lane = packet_lanes[lane];
        auto *state = context->states[lane];
        LUISA_ASSERT(
            state != nullptr && packet_lane < packet_width &&
                state->terminated == 0u &&
                (context->use_direct_surface_candidate ||
                 state->candidate_kind == static_cast<uint32_t>(
                                              SIMDHostRayQueryCandidateKind::surface)),
            "SIMD in-filter surface handler violated its audited ABI in lane {}.",
            lane);
        auto committed = context->use_direct_surface_candidate ?
                             ((committed_packet_lanes >> packet_lane) & 1u) != 0u :
                             state->candidate_committed != 0u;
        if (committed) {
            if (context->use_direct_surface_candidate) {
                commit_surface_candidate(
                    *state, load_candidate(packet_lane));
            } else {
                commit_surface_candidate(*state, state->candidate);
            }
        } else {
            arguments->valid[packet_lane] = 0;
        }
        if (!context->use_direct_surface_candidate) {
            state->candidate_kind = static_cast<uint32_t>(
                SIMDHostRayQueryCandidateKind::none);
            state->candidate_committed = 0u;
        }
    }
}

void surface_filter_pipeline_dispatch(
    const RTCFilterFunctionNArguments *arguments) noexcept {
    if (arguments == nullptr || arguments->N == 0u ||
        arguments->N > 16u) [[unlikely]] {
        std::abort();
    }
    switch (arguments->N) {
        case 4u: surface_filter_pipeline<4u>(arguments); break;
        case 8u: surface_filter_pipeline<8u>(arguments); break;
        case 16u: surface_filter_pipeline<16u>(arguments); break;
        default: std::abort();
    }
}

template<size_t packet_width, typename RayPacket>
void initialize_ray_packet(
    RayPacket &ray, std::array<int, packet_width> &valid,
    uint32_t lane_count, uint64_t active_mask_bits,
    SIMDHostRayQueryState *const *states) noexcept {
    auto lane_mask = (uint64_t{1u} << lane_count) - 1u;
    auto active = active_mask_bits & lane_mask;
    for (auto lane = uint32_t{0u}; lane < packet_width; lane++) {
        if (lane >= lane_count ||
            (active & (uint64_t{1u} << lane)) == 0u) {
            valid[lane] = 0;
            ray.org_x[lane] = 0.0f;
            ray.org_y[lane] = 0.0f;
            ray.org_z[lane] = 0.0f;
            ray.tnear[lane] = 0.0f;
            ray.dir_x[lane] = 0.0f;
            ray.dir_y[lane] = 0.0f;
            ray.dir_z[lane] = 1.0f;
            ray.time[lane] = 0.0f;
            ray.tfar[lane] = 0.0f;
            ray.mask[lane] = 0u;
            ray.id[lane] = 0u;
            ray.flags[lane] = 0u;
            continue;
        }
        auto *state = states[lane];
        LUISA_ASSERT(
            state != nullptr,
            "SIMD in-filter ray-query packet contains a null active state.");
        valid[lane] = -1;
        ray.org_x[lane] = state->world_ray[0u];
        ray.org_y[lane] = state->world_ray[1u];
        ray.org_z[lane] = state->world_ray[2u];
        ray.tnear[lane] = embree_tnear(state->world_ray[3u]);
        ray.dir_x[lane] = state->world_ray[4u];
        ray.dir_y[lane] = state->world_ray[5u];
        ray.dir_z[lane] = state->world_ray[6u];
        ray.time[lane] = state->time;
        ray.tfar[lane] = state->world_ray[7u];
        ray.mask[lane] = state->visibility_mask;
        ray.id[lane] = lane;
        ray.flags[lane] = 0u;
    }
}

template<size_t packet_width, typename HitPacket>
void initialize_hit_packet(HitPacket &hit) noexcept {
    for (auto lane = uint32_t{0u}; lane < packet_width; lane++) {
        hit.Ng_x[lane] = 0.0f;
        hit.Ng_y[lane] = 0.0f;
        hit.Ng_z[lane] = 0.0f;
        hit.u[lane] = 0.0f;
        hit.v[lane] = 0.0f;
        hit.primID[lane] = RTC_INVALID_GEOMETRY_ID;
        hit.geomID[lane] = RTC_INVALID_GEOMETRY_ID;
        for (auto level = 0u;
             level < RTC_MAX_INSTANCE_LEVEL_COUNT; level++) {
            hit.instID[level][lane] = RTC_INVALID_GEOMETRY_ID;
        }
    }
}

template<size_t packet_width>
void initialize_valid_packet(
    std::array<int, packet_width> &valid,
    uint32_t lane_count, uint64_t active_mask_bits) noexcept {
    auto lane_mask = (uint64_t{1u} << lane_count) - 1u;
    auto active = active_mask_bits & lane_mask;
    for (auto lane = uint32_t{0u}; lane < packet_width; lane++) {
        valid[lane] = lane < lane_count &&
                              (active & (uint64_t{1u} << lane)) != 0u ?
                          -1 :
                          0;
    }
}

template<size_t packet_width, typename RayPacket>
void occluded_packet(
    RTCScene scene, SurfaceFilterPipelineContext &context,
    std::array<int, packet_width> &valid,
    RayPacket &packet) noexcept {
#if LUISA_COMPUTE_SIMD_EMBREE_VERSION == 3
    if constexpr (packet_width == 4u) {
        rtcOccluded4(valid.data(), scene, &context.rtc, &packet);
    } else if constexpr (packet_width == 8u) {
        rtcOccluded8(valid.data(), scene, &context.rtc, &packet);
    } else {
        rtcOccluded16(valid.data(), scene, &context.rtc, &packet);
    }
#else
    RTCOccludedArguments arguments{};
    rtcInitOccludedArguments(&arguments);
    arguments.context = &context.rtc;
    arguments.flags = static_cast<RTCRayQueryFlags>(
        arguments.flags |
        RTC_RAY_QUERY_FLAG_INVOKE_ARGUMENT_FILTER);
    arguments.filter = surface_filter_pipeline_dispatch;
    if constexpr (packet_width == 4u) {
        rtcOccluded4(valid.data(), scene, &packet, &arguments);
    } else if constexpr (packet_width == 8u) {
        rtcOccluded8(valid.data(), scene, &packet, &arguments);
    } else {
        rtcOccluded16(valid.data(), scene, &packet, &arguments);
    }
#endif
}

template<size_t packet_width, typename RayHitPacket>
void intersect_packet(
    RTCScene scene, SurfaceFilterPipelineContext &context,
    std::array<int, packet_width> &valid,
    RayHitPacket &packet) noexcept {
#if LUISA_COMPUTE_SIMD_EMBREE_VERSION == 3
    if constexpr (packet_width == 4u) {
        rtcIntersect4(valid.data(), scene, &context.rtc, &packet);
    } else if constexpr (packet_width == 8u) {
        rtcIntersect8(valid.data(), scene, &context.rtc, &packet);
    } else {
        rtcIntersect16(valid.data(), scene, &context.rtc, &packet);
    }
#else
    RTCIntersectArguments arguments{};
    rtcInitIntersectArguments(&arguments);
    arguments.context = &context.rtc;
    arguments.flags = static_cast<RTCRayQueryFlags>(
        arguments.flags |
        RTC_RAY_QUERY_FLAG_INVOKE_ARGUMENT_FILTER);
    arguments.filter = surface_filter_pipeline_dispatch;
    if constexpr (packet_width == 4u) {
        rtcIntersect4(valid.data(), scene, &packet, &arguments);
    } else if constexpr (packet_width == 8u) {
        rtcIntersect8(valid.data(), scene, &packet, &arguments);
    } else {
        rtcIntersect16(valid.data(), scene, &packet, &arguments);
    }
#endif
}

void initialize_context(
    SurfaceFilterPipelineContext &context,
    const SIMDHostAccelInstanceTable &instances,
    uint32_t lane_count, uint64_t active_mask_bits,
    SIMDHostRayQueryState *const *states,
    const SIMDPacketLaunchConfig *launch_config,
    SIMDHostRayQuerySurfaceFilterHandler *on_surface,
    SIMDHostRayQueryDirectSurfaceFilterHandler *on_surface_direct,
    bool use_direct_surface_candidate) noexcept {
    context.lane_count = lane_count;
    context.instances = &instances;
    context.launch_config = launch_config;
    context.on_surface = on_surface;
    context.on_surface_direct = on_surface_direct;
    context.use_direct_surface_candidate =
        use_direct_surface_candidate && on_surface_direct != nullptr;
    for (auto lane = uint32_t{0u}; lane < lane_count; lane++) {
        if (((active_mask_bits >> lane) & 1u) != 0u) {
            context.states[lane] = states[lane];
        }
    }
#if LUISA_COMPUTE_SIMD_EMBREE_VERSION == 3
    rtcInitIntersectContext(&context.rtc);
    context.rtc.filter = surface_filter_pipeline_dispatch;
#else
    rtcInitRayQueryContext(&context.rtc);
#endif
}

template<bool packet_input, size_t packet_width,
         typename RayHitPacket, typename RayPacket>
void trace_group(
    RTCScene scene, const SIMDHostAccelInstanceTable &instances,
    uint32_t lane_count, uint64_t active_mask_bits,
    SIMDHostRayQueryState *const *states,
    bool terminate_on_first,
    void *ray_packet,
    const SIMDPacketLaunchConfig *launch_config,
    SIMDHostRayQuerySurfaceFilterHandler *on_surface,
    SIMDHostRayQueryDirectSurfaceFilterHandler *on_surface_direct,
    bool use_direct_surface_candidate) noexcept {
    alignas(64) SurfaceFilterPipelineContext context{};
    initialize_context(
        context, instances, lane_count, active_mask_bits, states,
        launch_config, on_surface, on_surface_direct,
        use_direct_surface_candidate);
    alignas(64) std::array<int, packet_width> valid{};
    if constexpr (packet_input) {
        initialize_valid_packet(
            valid, lane_count, active_mask_bits);
    }
    if (terminate_on_first) {
        if constexpr (packet_input) {
            auto *packet = static_cast<RayPacket *>(ray_packet);
            occluded_packet<packet_width>(
                scene, context, valid, *packet);
        } else {
            alignas(64) RayPacket packet{};
            initialize_ray_packet<packet_width>(
                packet, valid, lane_count,
                active_mask_bits, states);
            occluded_packet<packet_width>(
                scene, context, valid, packet);
        }
    } else {
        alignas(64) RayHitPacket packet{};
        if constexpr (packet_input) {
            std::memcpy(
                &packet.ray, ray_packet,
                sizeof(RayPacket));
        } else {
            initialize_ray_packet<packet_width>(
                packet.ray, valid, lane_count,
                active_mask_bits, states);
        }
        initialize_hit_packet<packet_width>(packet.hit);
        intersect_packet<packet_width>(
            scene, context, valid, packet);
    }
    auto remaining =
        active_mask_bits & ((uint64_t{1u} << lane_count) - 1u);
    while (remaining != 0u) {
        auto lane = static_cast<uint32_t>(
            std::countr_zero(remaining));
        remaining &= remaining - 1u;
        auto *state = states[lane];
        LUISA_ASSERT(
            lane < lane_count && state != nullptr,
            "SIMD in-filter ray query lost an active lane state.");
        state->candidate_kind = static_cast<uint32_t>(
            SIMDHostRayQueryCandidateKind::none);
        state->candidate_committed = 0u;
        state->terminated = 1u;
    }
}

template<bool packet_input>
void trace_group_for_width(
    SIMDAccel &accel, uint32_t lane_count,
    uint64_t active_mask_bits,
    SIMDHostRayQueryState *const *states,
    bool terminate_on_first,
    void *ray_packet,
    const SIMDPacketLaunchConfig *launch_config,
    SIMDHostRayQuerySurfaceFilterHandler *on_surface,
    SIMDHostRayQueryDirectSurfaceFilterHandler *on_surface_direct) noexcept {
    auto scene = SurfaceFilterPipelineAccelAccess::scene(accel);
    auto &instances =
        SurfaceFilterPipelineAccelAccess::instances(accel);
    auto use_direct_surface_candidate =
        SurfaceFilterPipelineAccelAccess::
            direct_surface_filter_candidate(accel);
    switch (lane_count) {
        case 2u:
        case 4u:
            trace_group<packet_input, 4u, RTCRayHit4, RTCRay4>(
                scene, instances, lane_count, active_mask_bits,
                states, terminate_on_first, ray_packet, launch_config,
                on_surface, on_surface_direct,
                use_direct_surface_candidate);
            break;
        case 8u:
            trace_group<packet_input, 8u, RTCRayHit8, RTCRay8>(
                scene, instances, lane_count, active_mask_bits,
                states, terminate_on_first, ray_packet, launch_config,
                on_surface, on_surface_direct,
                use_direct_surface_candidate);
            break;
        case 16u:
            trace_group<packet_input, 16u, RTCRayHit16, RTCRay16>(
                scene, instances, lane_count, active_mask_bits,
                states, terminate_on_first, ray_packet, launch_config,
                on_surface, on_surface_direct,
                use_direct_surface_candidate);
            break;
        default:
            LUISA_ERROR_WITH_LOCATION(
                "Unsupported SIMD in-filter ray-query width {}.",
                lane_count);
    }
}

template<bool packet_input>
void ray_query_surface_filter_pipeline_triangle_only_impl(
    uint32_t lane_count, uint64_t active_mask_bits,
    SIMDHostRayQueryState *const *states,
    void *ray_packet,
    const SIMDPacketLaunchConfig *launch_config,
    SIMDHostRayQuerySurfaceFilterHandler *on_surface,
    SIMDHostRayQueryDirectSurfaceFilterHandler *on_surface_direct) noexcept {
    LUISA_ASSERT(
        states != nullptr && launch_config != nullptr &&
            on_surface != nullptr &&
            (lane_count == 2u || lane_count == 4u ||
             lane_count == 8u || lane_count == 16u) &&
            (packet_input ? ray_packet != nullptr && lane_count >= 4u :
                            ray_packet == nullptr),
        "Invalid SIMD in-filter ray-query pipeline invocation at W{}.",
        lane_count);
    auto lane_mask = (uint64_t{1u} << lane_count) - 1u;
    auto pending = active_mask_bits & lane_mask;
    auto *expected_proceed =
        lane_count >= 8u ? ray_query_proceed_wide_triangle_only :
                           ray_query_proceed_triangle_only;
    while (pending != 0u) {
        auto first_lane = static_cast<uint32_t>(
            std::countr_zero(pending));
        auto *first_state = states[first_lane];
        LUISA_ASSERT(
            first_state != nullptr && first_state->accel != nullptr &&
                first_state->proceed == expected_proceed &&
                first_state->terminated == 0u,
            "SIMD in-filter ray query selected a mismatched provider.");
        auto *accel = static_cast<SIMDAccel *>(first_state->accel);
        auto terminate_on_first =
            first_state->terminate_on_first != 0u;
        auto group = uint64_t{0u};
        auto remaining = pending;
        while (remaining != 0u) {
            auto lane = static_cast<uint32_t>(
                std::countr_zero(remaining));
            auto bit = uint64_t{1u} << lane;
            remaining &= remaining - 1u;
            auto *state = states[lane];
            LUISA_ASSERT(
                state != nullptr && state->accel != nullptr &&
                    state->proceed == expected_proceed &&
                    state->terminated == 0u,
                "SIMD in-filter ray query contains a null active state.");
            if (state->accel == accel &&
                (state->terminate_on_first != 0u) ==
                    terminate_on_first) {
                group |= bit;
            }
        }
        LUISA_ASSERT(
            group != 0u,
            "SIMD in-filter ray query formed an empty traversal group.");
        trace_group_for_width<packet_input>(
            *accel, lane_count, group, states,
            terminate_on_first, ray_packet,
            launch_config, on_surface, on_surface_direct);
        pending &= ~group;
    }
}

}// namespace

void ray_query_surface_filter_packet_pipeline_triangle_only(
    uint32_t lane_count, uint64_t active_mask_bits,
    SIMDHostRayQueryState *const *states,
    void *ray_packet,
    const SIMDPacketLaunchConfig *launch_config,
    SIMDHostRayQuerySurfaceFilterHandler *on_surface,
    SIMDHostRayQueryDirectSurfaceFilterHandler *on_surface_direct) noexcept {
    ray_query_surface_filter_pipeline_triangle_only_impl<true>(
        lane_count, active_mask_bits, states, ray_packet,
        launch_config, on_surface, on_surface_direct);
}

void ray_query_surface_filter_pipeline_triangle_only(
    uint32_t lane_count, uint64_t active_mask_bits,
    SIMDHostRayQueryState *const *states,
    const SIMDPacketLaunchConfig *launch_config,
    SIMDHostRayQuerySurfaceFilterHandler *on_surface) noexcept {
    ray_query_surface_filter_pipeline_triangle_only_impl<false>(
        lane_count, active_mask_bits, states, nullptr,
        launch_config, on_surface, nullptr);
}

void ray_query_surface_filter_packet_pipeline_triangle_only_state_oracle(
    uint32_t lane_count, uint64_t active_mask_bits,
    SIMDHostRayQueryState *const *states,
    void *, const SIMDPacketLaunchConfig *launch_config,
    SIMDHostRayQuerySurfaceFilterHandler *on_surface,
    SIMDHostRayQueryDirectSurfaceFilterHandler *on_surface_direct) noexcept {
    ray_query_surface_filter_pipeline_triangle_only_impl<false>(
        lane_count, active_mask_bits, states, nullptr,
        launch_config, on_surface, on_surface_direct);
}

}// namespace luisa::compute::simd::triangle_ray_query
