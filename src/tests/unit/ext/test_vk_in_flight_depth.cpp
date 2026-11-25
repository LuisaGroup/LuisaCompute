// Host-level tests for the Vulkan backend's configurable in-flight depth
// (plan P1.5).
//
// `src/backends/vk/in_flight_depth.h` is device-independent - it touches no
// Vulkan handle and calls no entry point - so the arithmetic that decides when a
// stream throttles the host is covered here, without a Vulkan device:
// - the backend default is 3 frames in flight, and the public
//   `VulkanDeviceConfigExt::max_frames_in_flight()` getter returns the same
//   number (asserted against the header constant, which the getter documents)
// - a configured depth of 0 is rejected instead of being read as "unlimited";
//   every depth >= 1 is accepted verbatim
// - depth 2 reproduces the historical hard-coded rule
//   (`if (last_fence() > 2) sync(last_fence() - 2)`) for every fence value,
//   bit for bit, which is what makes it a compatible value
// - depth 3 admits exactly one more batch before the host throttles, and larger
//   depths keep following the same `F > depth -> target F - depth` engine
// - the target can never be 0 and never exceeds the last fence, i.e. the
//   subtraction cannot underflow for any (depth, fence) pair
// - the dispatch path and the present path share the engine, so the present
//   sequence (one fence value per presented frame) is checked through the same
//   plan; windowed present coverage is not required for the arithmetic.
#include "ut/ut.hpp"
#include "in_flight_depth.h"

#include <cstdint>
#include <initializer_list>
#include <limits>
#include <string_view>

using namespace boost::ut;
using namespace boost::ut::literals;
namespace vk_detail = lc::vk::detail;

namespace {

// The number the backend uses when the config extension keeps its default. The
// public getter's default return value has to match it; the tripwire test
// (src/tests/unit/runtime/test_vk_submit_profile.cpp) checks that on a real
// device through the diagnostic `vk_max_frames_in_flight` query, because the
// public header cannot include this internal header.
static_assert(vk_detail::kDefaultFramesInFlight == 3u,
              "the backend default in-flight depth is 3");

// Number of times a stream throttles itself while submitting `batch_count`
// back-to-back batches at the given depth. The branch is taken while
// `F > depth` with F = 0..batch_count-1, i.e. for the batches at index
// depth+1 .. batch_count-1.
[[nodiscard]] constexpr size_t throttle_entries(size_t batch_count,
                                                uint32_t depth) noexcept {
    auto first = static_cast<size_t>(depth) + 1u;
    return batch_count > first ? batch_count - first : 0u;
}

void reg_default_depth() {
    "vk_in_flight_depth_default_is_three"_test = [] {
        auto plan = vk_detail::plan_frames_in_flight(
            vk_detail::kDefaultFramesInFlight);
        expect(static_cast<bool>(plan))
            << "the backend default depth must be a valid configuration";
        expect(eq(plan.depth, 3u))
            << "the default in-flight depth is three batches";
        // Depth 3 means: batches 1..3 may still be executing while the fourth is
        // recorded, so the host first blocks before submitting batch index 4
        // (the fifth batch).
        expect(!vk_detail::plan_in_flight_throttle(3u, plan.depth).enter)
            << "with depth 3 the fourth submission must not block";
        auto fifth = vk_detail::plan_in_flight_throttle(4u, plan.depth);
        expect(fifth.enter) << "the fifth submission must enter the throttle";
        expect(eq(fifth.target, 1u))
            << "the fifth submission waits for the first batch to complete";
    };
}

void reg_zero_is_rejected() {
    "vk_in_flight_depth_rejects_zero"_test = [] {
        auto rejected = vk_detail::plan_frames_in_flight(0u);
        expect(!static_cast<bool>(rejected))
            << "a depth of 0 is invalid: it is not a synonym for 'unlimited'";
        expect(rejected.status == vk_detail::FramesInFlightStatus::ZERO_DEPTH);
        expect(rejected.depth == 0u)
            << "a rejected plan must not carry a usable depth";
        // The failure is reportable, which is what the device's create-time
        // error message uses.
        auto name = vk_detail::frames_in_flight_status_name(rejected.status);
        expect(std::string_view{name} != "unknown");
        expect(std::string_view{name} != "success")
            << "the zero-depth status must name the actual problem";
    };
}

void reg_every_positive_depth_is_accepted() {
    "vk_in_flight_depth_accepts_every_positive_value"_test = [] {
        for (uint32_t depth = 1u; depth <= 64u; ++depth) {
            auto plan = vk_detail::plan_frames_in_flight(depth);
            expect(static_cast<bool>(plan)) << "a positive depth must be valid";
            expect(eq(plan.depth, depth))
                << "the configured depth must be latched verbatim";
            expect(plan.status == vk_detail::FramesInFlightStatus::SUCCESS);
        }
        // No artificial upper bound: a device whose timeline value window allows
        // it may be configured with any depth, and the Device (not this plan)
        // rejects a value that cannot fit that window.
        auto large = vk_detail::plan_frames_in_flight(
            std::numeric_limits<uint32_t>::max());
        expect(static_cast<bool>(large));
        expect(eq(large.depth, std::numeric_limits<uint32_t>::max()));
    };
}

void reg_depth_two_is_the_legacy_rule() {
    "vk_in_flight_depth_two_matches_legacy_rule"_test = [] {
        // The pre-P1.5 code was, verbatim, in both Stream::dispatch and
        // Stream::present:
        //     if (_evt.last_fence() > 2) _evt.sync(_evt.last_fence() - 2);
        // Depth 2 has to be indistinguishable from it for every fence value a
        // stream can reach, otherwise "depth 2" would silently change the
        // historical behaviour instead of reproducing it.
        constexpr auto legacy_depth = 2u;
        for (uint64_t last_fence = 0u; last_fence <= 4096u; ++last_fence) {
            auto legacy_enter = last_fence > 2u;
            auto plan = vk_detail::plan_in_flight_throttle(last_fence, legacy_depth);
            expect(plan.enter == legacy_enter)
                << "depth 2 must take the branch exactly when last_fence > 2";
            if (legacy_enter) {
                expect(eq(plan.target, last_fence - 2u))
                    << "depth 2 must wait for exactly last_fence - 2";
            } else {
                expect(eq(plan.target, 0u))
                    << "a plan that does not enter carries no target";
            }
        }
    };
}

void reg_depth_three_adds_one_batch() {
    "vk_in_flight_depth_three_adds_one_batch"_test = [] {
        // The same 32-batch submission loop the device tripwire runs: depth 2
        // throttles 32 - 3 = 29 times, depth 3 throttles 32 - 4 = 28 times, i.e.
        // exactly one additional batch is allowed in flight.
        constexpr auto batch_count = size_t{32u};
        auto entries_at = [](uint32_t depth) noexcept {
            auto entries = size_t{0u};
            for (auto i = size_t{0u}; i < batch_count; ++i) {
                if (vk_detail::plan_in_flight_throttle(i, depth).enter) { ++entries; }
            }
            return entries;
        };
        expect(eq(entries_at(2u), size_t{29u}))
            << "depth 2 reproduces the historical 29 stalls in 32 batches";
        expect(eq(entries_at(3u), size_t{28u}))
            << "depth 3 must stall one batch later than depth 2";
        expect(eq(entries_at(1u), size_t{30u}))
            << "depth 1 keeps a single batch in flight";
        // Every depth follows the same engine: entries == batch_count - (depth+1).
        for (uint32_t depth = 1u; depth <= 40u; ++depth) {
            expect(eq(entries_at(depth), throttle_entries(batch_count, depth)))
                << "the throttle count must follow batch_count - (depth + 1)";
        }
        // Depth 4: two batches more than the historical depth 2.
        expect(eq(throttle_entries(batch_count, 4u), size_t{27u}));
    };
}

void reg_targets_are_well_formed() {
    "vk_in_flight_depth_targets_never_underflow"_test = [] {
        // Exhaustive over the low range where an underflow would happen, plus
        // the top of the range where the subtraction could wrap.
        for (uint32_t depth = 1u; depth <= 64u; ++depth) {
            for (uint64_t last_fence = 0u; last_fence <= 66u; ++last_fence) {
                auto plan = vk_detail::plan_in_flight_throttle(last_fence, depth);
                if (!plan.enter) {
                    expect(last_fence <= static_cast<uint64_t>(depth))
                        << "the throttle may only be skipped while there is room";
                    expect(eq(plan.target, 0u));
                    continue;
                }
                expect(last_fence > static_cast<uint64_t>(depth));
                expect(plan.target >= 1u)
                    << "a target of 0 is not a timeline value and must never be "
                       "produced (it would also wait for nothing)";
                expect(plan.target < last_fence)
                    << "the target must be behind the last reserved fence";
                expect(eq(plan.target, last_fence - static_cast<uint64_t>(depth)))
                    << "the target is last_fence - depth";
                expect(last_fence - plan.target == static_cast<uint64_t>(depth))
                    << "exactly `depth` batches stay in flight";
            }
            // Wraparound guard at the top of the fence range: the plan subtracts
            // nothing when the difference is not representable, so it cannot
            // produce a huge target.
            if (depth > 1u) {
                auto top = std::numeric_limits<uint64_t>::max();
                auto plan = vk_detail::plan_in_flight_throttle(top, depth);
                expect(plan.enter);
                expect(eq(plan.target, top - static_cast<uint64_t>(depth)));
            }
        }
        // A depth of 0 is unreachable through the Device (it fails closed at
        // create), and the plan degrades to "no throttle" rather than computing
        // `last_fence - 0`, which would wait for the batch about to be submitted
        // and deadlock the host.
        for (uint64_t last_fence : {uint64_t{0u}, uint64_t{1u}, uint64_t{7u}}) {
            auto plan = vk_detail::plan_in_flight_throttle(last_fence, 0u);
            expect(!plan.enter) << "an unvalidated depth of 0 must not throttle";
            expect(eq(plan.target, 0u));
        }
    };
}

void reg_present_path_shares_the_engine() {
    "vk_in_flight_depth_present_sequence_uses_same_bound"_test = [] {
        // Stream::present reserves one fence value per presented frame, exactly
        // like a dispatch batch, and calls the same throttle before reserving it.
        // Simulate that sequence for the historical depth and for the default so
        // the present logic is pinned to the same arithmetic without a windowed
        // swapchain:
        auto simulate = [](uint32_t depth, size_t hits) noexcept {
            auto entries = size_t{0u};
            auto first_entry = std::numeric_limits<size_t>::max();
            auto first_target = uint64_t{0u};
            auto last_fence = uint64_t{0u};
            for (auto i = size_t{0u}; i < hits; ++i) {
                auto plan = vk_detail::plan_in_flight_throttle(last_fence, depth);
                if (plan.enter) {
                    if (entries == 0u) {
                        first_entry = i;
                        first_target = plan.target;
                    }
                    ++entries;
                }
                last_fence += 1u;// the submission's own signal
            }
            expect(eq(entries, throttle_entries(hits, depth)))
                << "a presented frame throttles exactly like a dispatch batch";
            expect(eq(first_entry, static_cast<size_t>(depth) + 1u))
                << "the first host stall happens one frame after the depth is "
                   "saturated";
            expect(eq(first_target, uint64_t{1u}))
                << "the first stall waits for the very first frame";
        };
        simulate(2u, 8u); // the historical hard-coded depth
        simulate(3u, 8u); // the default
        simulate(5u, 16u);// an explicitly deeper configuration
    };
}

// ---------------------------------------------------------------------------
// Strict spin-budget parsing (`LUISA_VK_SYNC_SPIN_US` / `LUISA_VK_RETIRE_SPIN_US`)
// ---------------------------------------------------------------------------
void reg_spin_pause_budget_parsing() {
    "vk_spin_pause_budget_is_parsed_strictly"_test = [] {
        // The two knobs default to 20 ms (`Event::sync`) and 200 us (the stream's
        // retirement spin); only the parsing is covered here, not the polling.
        constexpr auto default_us = uint64_t{20'000u};
        constexpr auto default_pauses = uint32_t{2'000'000u};
        constexpr auto max_pauses =
            static_cast<uint32_t>(vk_detail::kMaxSpinBudgetUs * vk_detail::kPausesPerUs);
        static_assert(vk_detail::kMaxSpinBudgetUs == 100'000ull);
        static_assert(vk_detail::kPausesPerUs == 100ull);

        // Unset and empty both mean "use the backend default".
        expect(eq(vk_detail::parse_spin_pause_iterations(nullptr, default_us),
                  default_pauses))
            << "an unset variable must fall back to the backend default";
        expect(eq(vk_detail::parse_spin_pause_iterations("", default_us),
                  default_pauses))
            << "an empty variable must fall back to the backend default";

        // "0" is a value, not an absent one: it is the documented way to turn the
        // busy-spin off, and silently replacing it with the default made
        // `LUISA_VK_SYNC_SPIN_US=0` a no-op.
        expect(eq(vk_detail::parse_spin_pause_iterations("0", default_us), 0u))
            << "'0' must disable spinning instead of being read as 'unset'";
        expect(eq(vk_detail::parse_spin_pause_iterations("1", default_us), 100u));
        expect(eq(vk_detail::parse_spin_pause_iterations("500", default_us), 50'000u));
        expect(eq(vk_detail::parse_spin_pause_iterations("0000010", default_us), 1'000u))
            << "leading zeros are still digits";

        // The cap: a value at the cap is accepted, anything above saturates.
        expect(eq(vk_detail::parse_spin_pause_iterations("100000", default_us),
                  max_pauses));
        expect(eq(vk_detail::parse_spin_pause_iterations("100001", default_us),
                  max_pauses))
            << "a budget above the cap must clamp instead of wrapping";
        expect(eq(vk_detail::parse_spin_pause_iterations(
                      "999999999999999999999999", default_us),
                  max_pauses))
            << "a value far beyond 64 bits must clamp, never overflow";

        // Anything that is not a plain unsigned decimal falls back: a negative
        // value used to become UINT64_MAX and then 100 ms of busy-spinning.
        for (auto rejected : {std::string_view{"-1"},
                              std::string_view{"-0"},
                              std::string_view{"abc"},
                              std::string_view{"10abc"},
                              std::string_view{" 10"},
                              std::string_view{"10 "},
                              std::string_view{"+10"},
                              std::string_view{"1e3"},
                              std::string_view{"0x10"}}) {
            expect(eq(vk_detail::parse_spin_pause_iterations(rejected, default_us),
                      default_pauses))
                << "a malformed value must fall back to the default: '" << rejected
                << "'";
        }

        // The fallback is clamped as well, so no caller can sneak a budget above
        // the cap in through the default argument.
        expect(eq(vk_detail::parse_spin_pause_iterations(nullptr, 1'000'000u),
                  max_pauses));
        expect(eq(vk_detail::parse_spin_pause_iterations("", 1'000'000u),
                  max_pauses));
        expect(eq(vk_detail::parse_spin_pause_iterations("junk", 1'000'000u),
                  max_pauses));
    };
}

// ---------------------------------------------------------------------------
// When the orphaned descriptor sets force a descriptor-pool reset
// ---------------------------------------------------------------------------
void reg_descriptor_pool_orphan_reset_policy() {
    "vk_descriptor_pool_orphan_reset_policy"_test = [] {
        constexpr auto threshold = vk_detail::kDescriptorPoolOrphanResetThreshold;
        expect(eq(threshold, size_t{1024u}))
            << "the orphan threshold the backend budgets for is 1024 sets";
        // Rewinding the cursors is only equivalent to a reset while the cache owns
        // every set and no overflow pool was touched.
        expect(!vk_detail::should_reset_descriptor_pools(false, 0u, 0u))
            << "an empty, fully cached state must not pay for a pool reset";
        expect(!vk_detail::should_reset_descriptor_pools(false, 0u, 1023u))
            << "one set below the threshold is still reclaimed by rewinding";
        // The threshold itself forces the reset, i.e. the comparison is >=.
        expect(vk_detail::should_reset_descriptor_pools(false, 0u, threshold));
        expect(vk_detail::should_reset_descriptor_pools(false, 0u, 1024u));
        expect(vk_detail::should_reset_descriptor_pools(false, 0u, 2048u))
            << "the reset stays forced above the threshold";
        // An allocation outside the cache cannot be reclaimed by rewinding.
        expect(vk_detail::should_reset_descriptor_pools(true, 0u, 0u))
            << "an uncached descriptor allocation forces the reset";
        // An overflow pool was used: the cache does not record which pool a set
        // came from, so only a native reset can hand it out again.
        expect(vk_detail::should_reset_descriptor_pools(false, 1u, 0u))
            << "touching the first overflow pool forces the reset";
        expect(vk_detail::should_reset_descriptor_pools(true, 1u, threshold));
    };
}

}// namespace

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(
        argc, const_cast<const char **>(argv));
    reg_default_depth();
    reg_zero_is_rejected();
    reg_every_positive_depth_is_accepted();
    reg_depth_two_is_the_legacy_rule();
    reg_depth_three_adds_one_batch();
    reg_targets_are_well_formed();
    reg_present_path_shares_the_engine();
    reg_spin_pause_budget_parsing();
    reg_descriptor_pool_orphan_reset_policy();
    return 0;
}