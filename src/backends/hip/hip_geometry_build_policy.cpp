#include <atomic>
#include <chrono>
#include <cstdlib>
#include <string_view>

#include <luisa/core/logging.h>

#include "hip_geometry_build_policy.h"

#include "hip_check.h"
#include "hip_geometry.h"
#include "hip_geometry_memory_policy.h"

namespace luisa::compute::hip {

namespace {

struct HIPPolicyProfileStats {
    std::atomic<uint64_t> calls{0u};
    std::atomic<uint64_t> get_temp_ns{0u};
    std::atomic<uint64_t> mem_info_ns{0u};
};

[[nodiscard]] bool hip_policy_profile_enabled() noexcept {
    static const bool enabled = [] {
        using namespace std::string_view_literals;
        auto env = std::getenv("LUISA_HIP_PROFILE_SCENE_BUILD");
        return env != nullptr && std::string_view{env} == "1"sv;
    }();
    return enabled;
}

[[nodiscard]] HIPPolicyProfileStats &hip_policy_profile_stats() noexcept {
    static HIPPolicyProfileStats stats;
    static const bool registered = [] {
        if (hip_policy_profile_enabled()) {
            std::atexit([] {
                auto n = stats.calls.load(std::memory_order_relaxed);
                if (n == 0u) { return; }
                LUISA_INFO(
                    "[HIP_SCENE_PROFILE] policy_calls={} get_temp_size={:.2f}ms hipMemGetInfo={:.2f}ms",
                    n,
                    static_cast<double>(stats.get_temp_ns.load(std::memory_order_relaxed)) * 1e-6,
                    static_cast<double>(stats.mem_info_ns.load(std::memory_order_relaxed)) * 1e-6);
            });
        }
        return true;
    }();
    static_cast<void>(registered);
    return stats;
}

}// namespace

HIPGeometryBuildPolicy select_hiprt_geometry_build_policy(
    hiprtContext context,
    const hiprtGeometryBuildInput &input,
    const AccelOption &option) noexcept {
    const auto profile = hip_policy_profile_enabled();
    using ProfileClock = std::chrono::steady_clock;
    auto t0 = profile ? ProfileClock::now() : ProfileClock::time_point{};
    HIPGeometryBuildPolicy result;
    result.options.buildFlags = make_hiprt_build_flags(option);
    LUISA_CHECK_HIPRT(hiprtGetGeometryBuildTemporaryBufferSize(
        context,
        input,
        result.options,
        result.temporary_buffer_size));
    auto t1 = profile ? ProfileClock::now() : ProfileClock::time_point{};
    if (profile) {
        auto &s = hip_policy_profile_stats();
        s.calls.fetch_add(1u, std::memory_order_relaxed);
        s.get_temp_ns.fetch_add(
            static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count()),
            std::memory_order_relaxed);
    }
    if (option.hint != AccelOption::UsageHint::FAST_TRACE) {
        return result;
    }

    result.high_quality_temporary_buffer_size =
        result.temporary_buffer_size;
    size_t total_memory_size{};
    auto t2 = profile ? ProfileClock::now() : ProfileClock::time_point{};
    LUISA_CHECK_HIP(hipMemGetInfo(
        &result.free_memory_size,
        &total_memory_size));
    if (profile) {
        auto t3 = ProfileClock::now();
        hip_policy_profile_stats().mem_info_ns.fetch_add(
            static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(t3 - t2).count()),
            std::memory_order_relaxed);
    }
    static_cast<void>(total_memory_size);
    result.memory_constrained =
        hiprt_high_quality_scratch_exceeds_budget(
            result.high_quality_temporary_buffer_size,
            result.free_memory_size);
    if (!result.memory_constrained) {
        return result;
    }

    result.options.buildFlags = make_hiprt_build_flags(
        option, hiprtBuildFlagBitPreferBalancedBuild);
    LUISA_CHECK_HIPRT(hiprtGetGeometryBuildTemporaryBufferSize(
        context,
        input,
        result.options,
        result.temporary_buffer_size));
    LUISA_INFO(
        "HIPRT geometry build selected BalancedBuild under memory pressure "
        "(free = {} bytes, high-quality scratch = {} bytes, selected scratch "
        "= {} bytes).",
        result.free_memory_size,
        result.high_quality_temporary_buffer_size,
        result.temporary_buffer_size);
    return result;
}

}// namespace luisa::compute::hip
