// Test for the Vulkan backend's per-command-buffer bind-state tracker
// (src/backends/vk/bind_state.h, plan P1.2).
// This test covers:
// - a pipeline bind is skipped only for the exact same VkPipeline handle at the
//   same bind point; a different handle is always recorded
// - the three bind points (compute / graphics / ray tracing) are independent
// - a descriptor-set bind is skipped only for the identical (layout object,
//   first set, set count, set handles, dynamic offsets) tuple
// - a pipeline bind keeps the descriptor state when the pipeline layout is the
//   same object and drops it when the layout is a different object (the "may be
//   compatible" case is never inferred from the structure)
// - binds the tracker cannot represent (empty, too many sets, too many dynamic
//   offsets) are always recorded and drop the knowledge instead of guessing
// - reset() / invalidate_all() / invalidate(point) forget exactly what they claim
// Everything runs on the host: the class compares handles and never calls a
// Vulkan entry point, so no device and no shader are involved.

#include "ut/ut.hpp"
#include "bind_state.h"

#include <array>
#include <cstdint>

using namespace boost::ut;
using namespace boost::ut::literals;
using lc::vk::detail::BindPoint;
using lc::vk::detail::BindState;
using lc::vk::detail::kMaxTrackedDescriptorSets;
using lc::vk::detail::kMaxTrackedDynamicOffsets;

namespace {

// Fake handles: the tracker only ever compares them, so distinct integers are
// distinct objects as far as it can tell.
constexpr auto kPipelineA = uint64_t{0x1001u};
constexpr auto kPipelineB = uint64_t{0x1002u};
constexpr auto kLayoutA = uint64_t{0x2001u};
constexpr auto kLayoutB = uint64_t{0x2002u};
constexpr auto kSetA = uint64_t{0x3001u};
constexpr auto kSetB = uint64_t{0x3002u};
constexpr auto kSetC = uint64_t{0x3003u};
constexpr auto kSetD = uint64_t{0x3004u};

[[nodiscard]] bool bind_pipeline(BindState &state, BindPoint point,
                                 uint64_t pipeline, uint64_t layout) noexcept {
    return state.bind_pipeline(point, pipeline, layout);
}

[[nodiscard]] bool bind_sets(BindState &state, BindPoint point, uint64_t layout,
                             uint32_t first_set, std::array<uint64_t, 4> const &sets,
                             uint32_t set_count,
                             std::array<uint32_t, 4> const &offsets = {},
                             uint32_t offset_count = 0u) noexcept {
    return state.bind_descriptor_sets(point, layout, first_set, set_count,
                                      sets.data(), offset_count, offsets.data());
}

void test_pipeline_identity() {
    BindState state;
    // Nothing is bound yet, so the first bind of any pipeline is recorded.
    expect(bind_pipeline(state, BindPoint::compute, kPipelineA, kLayoutA))
        << "the first pipeline bind must be recorded";
    expect(!bind_pipeline(state, BindPoint::compute, kPipelineA, kLayoutA))
        << "the same VkPipeline handle must not be bound twice at one bind point";
    expect(bind_pipeline(state, BindPoint::compute, kPipelineB, kLayoutA))
        << "a different VkPipeline handle must always be recorded, even when it "
           "shares the pipeline layout";
    expect(!bind_pipeline(state, BindPoint::compute, kPipelineB, kLayoutA))
        << "the second bind of the new pipeline is redundant again";
    expect(bind_pipeline(state, BindPoint::compute, kPipelineA, kLayoutA))
        << "switching back to an earlier pipeline is a real bind";
    expect(state.point(BindPoint::compute).pipeline == kPipelineA);
    expect(state.point(BindPoint::compute).pipeline_layout == kLayoutA);
    expect(state.point(BindPoint::compute).pipeline_known);
}

void test_bind_points_are_independent() {
    BindState state;
    expect(bind_pipeline(state, BindPoint::compute, kPipelineA, kLayoutA));
    // The same handle at the graphics bind point is a different binding: Vulkan
    // keeps the pipeline state per bind point.
    expect(bind_pipeline(state, BindPoint::graphics, kPipelineA, kLayoutA))
        << "a graphics bind point has its own pipeline binding";
    expect(bind_pipeline(state, BindPoint::ray_tracing, kPipelineA, kLayoutA))
        << "a ray-tracing bind point has its own pipeline binding";
    expect(!bind_pipeline(state, BindPoint::compute, kPipelineA, kLayoutA))
        << "the compute binding is unaffected by the other bind points";
    expect(!bind_pipeline(state, BindPoint::graphics, kPipelineA, kLayoutA));
    expect(!bind_pipeline(state, BindPoint::ray_tracing, kPipelineA, kLayoutA));
}

void test_descriptor_identity() {
    BindState state;
    std::array<uint64_t, 4> sets{kSetA, kSetB, 0u, 0u};
    expect(bind_sets(state, BindPoint::compute, kLayoutA, 0u, sets, 2u))
        << "the first descriptor bind must be recorded";
    expect(!bind_sets(state, BindPoint::compute, kLayoutA, 0u, sets, 2u))
        << "the identical whole-range bind is redundant";
    // Every dimension of the identity must matter on its own.
    expect(bind_sets(state, BindPoint::compute, kLayoutB, 0u, sets, 2u))
        << "a different pipeline layout object must be recorded (never compared "
           "structurally)";
    expect(!bind_sets(state, BindPoint::compute, kLayoutB, 0u, sets, 2u))
        << "the repeat with that layout is redundant again";
    std::array<uint64_t, 4> other_sets{kSetA, kSetC, 0u, 0u};
    expect(bind_sets(state, BindPoint::compute, kLayoutB, 0u, other_sets, 2u))
        << "a different set handle must be recorded";
    // Plan P2.3: the tracker is per set *index*, so a bind whose slots already
    // hold exactly these handles is redundant even when its range differs from the
    // last recorded one. That is precisely the state a pushed set 0 leaves behind
    // (slot 0 changes, the global sets do not), so the narrower bind is skipped
    // here instead of always being recorded.
    std::array<uint64_t, 4> one_set{kSetA, 0u, 0u, 0u};
    expect(!bind_sets(state, BindPoint::compute, kLayoutB, 0u, one_set, 1u))
        << "a sub-range whose slots already hold exactly these handles is redundant";
    expect(bind_sets(state, BindPoint::compute, kLayoutB, 1u,
                     std::array<uint64_t, 4>{kSetB, 0u, 0u, 0u}, 1u))
        << "a different first set is recorded when that slot holds another handle";
    // And the recorded state is exactly the last bind.
    expect(state.point(BindPoint::compute).descriptor_layout == kLayoutB);
    expect(state.point(BindPoint::compute).descriptor_first_set == 1u);
    expect(state.point(BindPoint::compute).descriptor_set_count == 1u);
    expect(state.point(BindPoint::compute).descriptor_sets[0] == kSetB);
}

void test_partial_ranges_are_conservative() {
    BindState state;
    std::array<uint64_t, 4> three{kSetA, kSetB, kSetC, 0u};
    std::array<uint64_t, 4> middle{0u, kSetD, 0u, 0u};
    expect(bind_sets(state, BindPoint::compute, kLayoutA, 0u, three, 3u));
    // A partial bind changes one slot; the others keep their handles. Only an
    // exact per-slot repeat is skipped, and a wider range that would restore a
    // handle the tracker knows is no longer there is recorded.
    expect(bind_sets(state, BindPoint::compute, kLayoutA, 1u,
                     std::array<uint64_t, 4>{kSetD, 0u, 0u, 0u}, 1u))
        << "a partial bind must be recorded";
    expect(!bind_sets(state, BindPoint::compute, kLayoutA, 1u,
                      std::array<uint64_t, 4>{kSetD, 0u, 0u, 0u}, 1u))
        << "the identical repeat of the partial bind is redundant";
    // Same layout object: the slots the partial bind did not touch keep their
    // handles, which is what licenses merging a later wider range with them.
    expect(state.point(BindPoint::compute).slot_known[0u] &&
           state.point(BindPoint::compute).slot_handle[0u] == kSetA &&
           state.point(BindPoint::compute).slot_known[2u] &&
           state.point(BindPoint::compute).slot_handle[2u] == kSetC)
        << "a partial bind must keep the untouched slots of the same layout";
    expect(state.point(BindPoint::compute).slot_known[1u] &&
           !state.point(BindPoint::compute).slot_pushed[1u] &&
           state.point(BindPoint::compute).slot_handle[1u] == kSetD)
        << "the bound slot must describe the new set";
    expect(bind_sets(state, BindPoint::compute, kLayoutA, 0u, three, 3u))
        << "restoring the wider range after a partial bind must be recorded "
           "(the tracker never reconstructs a merged state)";
}


// Plan P2.3 (review fix): a classic bind under a *different* pipeline layout
// object must forget every slot it does not itself rewrite. Those handles were
// bound with another layout, so keeping them would let a later request claim an
// identity that does not exist - and `bind_pipeline` cannot repair it, because
// ordinary encoding binds descriptors before the pipeline. Only a bind under the
// *same* layout may merge with untouched slots.
void test_layout_change_forgets_slots() {
    BindState state;
    std::array<uint64_t, 4> three{kSetA, kSetB, kSetC, 0u};
    // Populate slots 0..2 under layout A.
    expect(bind_sets(state, BindPoint::compute, kLayoutA, 0u, three, 3u));
    expect(!bind_sets(state, BindPoint::compute, kLayoutA, 0u, three, 3u))
        << "the repeat under the same layout is redundant";
    expect(state.point(BindPoint::compute).slot_known[0u] &&
           state.point(BindPoint::compute).slot_known[1u] &&
           state.point(BindPoint::compute).slot_known[2u]);
    // A partial bind under layout B: only slot 1 is rewritten, and the others must
    // be forgotten rather than re-tagged with the new layout.
    expect(bind_sets(state, BindPoint::compute, kLayoutB, 1u,
                     std::array<uint64_t, 4>{kSetD, 0u, 0u, 0u}, 1u))
        << "a bind under a different layout object is always recorded";
    auto const &point = state.point(BindPoint::compute);
    expect(point.descriptor_layout == kLayoutB);
    expect(!point.slot_known[0u] && !point.slot_known[2u])
        << "the slots of the old layout must be forgotten, not inherited";
    expect(point.slot_known[1u] && !point.slot_pushed[1u] &&
           point.slot_handle[1u] == kSetD)
        << "the bound slot describes the new set";
    // No old slot can match: the range that was fully bound under layout A must be
    // recorded again under layout B even though the handles are the ones that were
    // recorded before.
    expect(bind_sets(state, BindPoint::compute, kLayoutB, 0u, three, 3u))
        << "forgotten slots must not satisfy a later request";
    // ... and now that layout B really knows them, the repeat is redundant.
    expect(!bind_sets(state, BindPoint::compute, kLayoutB, 0u, three, 3u));
    expect(!bind_sets(state, BindPoint::compute, kLayoutB, 2u,
                      std::array<uint64_t, 4>{kSetC, 0u, 0u, 0u}, 1u))
        << "a sub-range of the newly recorded state is redundant again";
    // The same rule holds for the transition back to layout A: nothing from B may
    // be reused.
    expect(bind_sets(state, BindPoint::compute, kLayoutA, 2u,
                     std::array<uint64_t, 4>{kSetC, 0u, 0u, 0u}, 1u))
        << "switching back to the first layout must record the bind";
    expect(!state.point(BindPoint::compute).slot_known[0u] &&
           !state.point(BindPoint::compute).slot_known[1u])
        << "and it must forget the slots of the layout it left";
}


void test_dynamic_offsets() {
    BindState state;
    std::array<uint64_t, 4> sets{kSetA, kSetB, 0u, 0u};
    std::array<uint32_t, 4> offsets{16u, 32u, 0u, 0u};
    expect(bind_sets(state, BindPoint::compute, kLayoutA, 0u, sets, 2u,
                     offsets, 2u))
        << "a bind with dynamic offsets must be recorded the first time";
    expect(!bind_sets(state, BindPoint::compute, kLayoutA, 0u, sets, 2u,
                      offsets, 2u))
        << "the same offsets in the same order are redundant";
    // The same sets and offsets under a different layout object are a
    // different binding: the offsets belong to the sets of that layout.
    expect(bind_sets(state, BindPoint::compute, kLayoutB, 0u, sets, 2u,
                     offsets, 2u))
        << "a layout change must record the bind even with identical offsets";
    expect(!bind_sets(state, BindPoint::compute, kLayoutB, 0u, sets, 2u,
                     offsets, 2u));
    std::array<uint32_t, 4> changed{16u, 64u, 0u, 0u};
    expect(bind_sets(state, BindPoint::compute, kLayoutA, 0u, sets, 2u,
                     changed, 2u))
        << "a different dynamic-offset value must be recorded";
    expect(bind_sets(state, BindPoint::compute, kLayoutA, 0u, sets, 2u,
                     offsets, 1u))
        << "a different dynamic-offset count must be recorded";
    std::array<uint32_t, 4> reordered{32u, 16u, 0u, 0u};
    expect(bind_sets(state, BindPoint::compute, kLayoutA, 0u, sets, 2u,
                     reordered, 2u))
        << "reordered dynamic offsets are a different binding";
}

void test_pipeline_layout_gates_descriptors() {
    BindState state;
    std::array<uint64_t, 4> sets{kSetA, kSetB, 0u, 0u};
    expect(bind_pipeline(state, BindPoint::compute, kPipelineA, kLayoutA));
    expect(bind_sets(state, BindPoint::compute, kLayoutA, 0u, sets, 2u));
    expect(!bind_sets(state, BindPoint::compute, kLayoutA, 0u, sets, 2u));
    // Same layout object: the descriptor knowledge survives a pipeline change.
    expect(bind_pipeline(state, BindPoint::compute, kPipelineB, kLayoutA));
    expect(!bind_sets(state, BindPoint::compute, kLayoutA, 0u, sets, 2u))
        << "a pipeline change with the same layout object keeps the descriptor "
           "state (Vulkan keeps the bound sets; nothing about them changed)";
    // Different layout object: the descriptor knowledge is dropped, so the
    // identical bind is recorded again even though the structures may be equal.
    expect(bind_pipeline(state, BindPoint::compute, kPipelineA, kLayoutB));
    expect(!state.point(BindPoint::compute).descriptor_known)
        << "a pipeline bind with a different layout object must drop the "
           "descriptor knowledge instead of assuming compatibility";
    expect(bind_sets(state, BindPoint::compute, kLayoutA, 0u, sets, 2u))
        << "after the drop, the same descriptor bind must be recorded again";
}

void test_unrepresentable_binds_are_recorded() {
    BindState state;
    std::array<uint64_t, 4> sets{kSetA, kSetB, 0u, 0u};
    expect(bind_sets(state, BindPoint::compute, kLayoutA, 0u, sets, 2u));
    // An empty bind is not a state the tracker models.
    expect(bind_sets(state, BindPoint::compute, kLayoutA, 0u, sets, 0u))
        << "an empty descriptor bind must be recorded";
    expect(!state.point(BindPoint::compute).descriptor_known)
        << "an empty bind drops the tracked knowledge";
    expect(bind_sets(state, BindPoint::compute, kLayoutA, 0u, sets, 2u))
        << "the next representable bind is recorded again";

    // More sets than the tracker stores.
    std::array<uint64_t, kMaxTrackedDescriptorSets + 1u> wide{};
    for (auto i = uint64_t{0u}; i < wide.size(); ++i) { wide[i] = kSetA + i; }
    auto const wide_count = static_cast<uint32_t>(wide.size());
    expect(state.bind_descriptor_sets(BindPoint::compute, kLayoutA, 0u,
                                      wide_count, wide.data(), 0u, nullptr))
        << "an oversized set range must be recorded";
    expect(!state.point(BindPoint::compute).descriptor_known)
        << "an oversized set range is not tracked";
    expect(state.bind_descriptor_sets(BindPoint::compute, kLayoutA, 0u,
                                      wide_count, wide.data(), 0u, nullptr))
        << "and it is recorded every time instead of being skipped";

    // A first set that would end outside the tracked range.
    std::array<uint32_t, 2> offsets{0u, 0u};
    expect(state.bind_descriptor_sets(
        BindPoint::compute, kLayoutA,
        static_cast<uint32_t>(kMaxTrackedDescriptorSets) - 1u, 2u,
        sets.data(), 2u, offsets.data()))
        << "a range that runs past the tracked slots must be recorded";
    expect(!state.point(BindPoint::compute).descriptor_known);

    // More dynamic offsets than the tracker stores.
    std::array<uint32_t, kMaxTrackedDynamicOffsets + 1u> many_offsets{};
    expect(state.bind_descriptor_sets(
        BindPoint::compute, kLayoutA, 0u, 1u, sets.data(),
        static_cast<uint32_t>(many_offsets.size()), many_offsets.data()))
        << "an oversized dynamic-offset array must be recorded";
    expect(!state.point(BindPoint::compute).descriptor_known);
}

void test_invalidation() {
    BindState state;
    std::array<uint64_t, 4> sets{kSetA, kSetB, 0u, 0u};
    auto bind_all = [&] {
        static_cast<void>(bind_pipeline(state, BindPoint::compute, kPipelineA, kLayoutA));
        static_cast<void>(bind_pipeline(state, BindPoint::graphics, kPipelineB, kLayoutB));
        static_cast<void>(bind_sets(state, BindPoint::compute, kLayoutA, 0u, sets, 2u));
        static_cast<void>(bind_sets(state, BindPoint::graphics, kLayoutB, 0u, sets, 2u));
    };

    // A reset is what every recording starts with: nothing survives it.
    bind_all();
    state.reset();
    expect(bind_pipeline(state, BindPoint::compute, kPipelineA, kLayoutA))
        << "reset() must forget the compute pipeline";
    expect(bind_pipeline(state, BindPoint::graphics, kPipelineB, kLayoutB))
        << "reset() must forget the graphics pipeline";
    expect(bind_sets(state, BindPoint::compute, kLayoutA, 0u, sets, 2u))
        << "reset() must forget the compute descriptor sets";
    expect(bind_sets(state, BindPoint::graphics, kLayoutB, 0u, sets, 2u))
        << "reset() must forget the graphics descriptor sets";

    // invalidate(point) forgets exactly one bind point.
    bind_all();
    state.invalidate(BindPoint::compute);
    expect(bind_pipeline(state, BindPoint::compute, kPipelineA, kLayoutA));
    expect(bind_sets(state, BindPoint::compute, kLayoutA, 0u, sets, 2u));
    expect(!bind_pipeline(state, BindPoint::graphics, kPipelineB, kLayoutB))
        << "invalidating compute must not touch the graphics binding";
    expect(!bind_sets(state, BindPoint::graphics, kLayoutB, 0u, sets, 2u));

    // invalidate_all() forgets everything: this is what a raw foreign bind on the
    // recorded command buffer forces.
    bind_all();
    state.invalidate_all();
    for (auto point : {BindPoint::compute, BindPoint::graphics, BindPoint::ray_tracing}) {
        expect(!state.point(point).pipeline_known);
        expect(!state.point(point).descriptor_known);
    }
    expect(bind_pipeline(state, BindPoint::compute, kPipelineA, kLayoutA));
    expect(bind_pipeline(state, BindPoint::graphics, kPipelineB, kLayoutB));
}

void test_bind_point_translation() {
    using lc::vk::detail::to_bind_point;
    auto point = BindPoint::count;
    expect(to_bind_point(VK_PIPELINE_BIND_POINT_COMPUTE, point) &&
           point == BindPoint::compute);
    expect(to_bind_point(VK_PIPELINE_BIND_POINT_GRAPHICS, point) &&
           point == BindPoint::graphics);
    expect(to_bind_point(VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, point) &&
           point == BindPoint::ray_tracing);
    // Anything this backend never records into must be rejected so the call site
    // falls back to "flush and bind".
    expect(!to_bind_point(VK_PIPELINE_BIND_POINT_RAY_TRACING_NV, point) ||
           point == BindPoint::ray_tracing)
        << "the NV ray-tracing alias is the same numeric bind point as the KHR one";
}


// Plan P2.3: a vkCmdPushDescriptorSetKHR replaces the descriptor *bank* of one set
// index. The tracker records that as "this slot is a push bank", which keeps every
// later classic bind that names the slot recorded while leaving the other slots'
// knowledge - the global sampler/bindless sets - intact. Without that split a
// pushed set 0 would force the global sets to be re-bound for every dispatch, which
// is exactly the rebinding the push path removes.
void test_push_descriptor_slots() {
    BindState state;
    std::array<uint64_t, 4> global{kSetB, 0u, 0u, 0u};
    // One recorded global-set bind, then its repeat is redundant ...
    expect(bind_sets(state, BindPoint::compute, kLayoutA, 1u, global, 1u));
    expect(!bind_sets(state, BindPoint::compute, kLayoutA, 1u, global, 1u));
    // ... and it stays redundant when set 0 is pushed between two dispatches.
    state.push_descriptor_sets(BindPoint::compute, kLayoutA, 0u);
    expect(!bind_sets(state, BindPoint::compute, kLayoutA, 1u, global, 1u))
        << "a push of set 0 must not force the global sets to be re-bound";
    // The pushed slot itself is not describable by a handle, so a classic bind that
    // names it is recorded.
    expect(bind_sets(state, BindPoint::compute, kLayoutA, 0u,
                     std::array<uint64_t, 4>{kSetA, kSetB, 0u, 0u}, 2u))
        << "a bind naming the pushed slot must be recorded";
    // After that bind slot 0 is a classic handle again.
    expect(!bind_sets(state, BindPoint::compute, kLayoutA, 0u,
                      std::array<uint64_t, 4>{kSetA, kSetB, 0u, 0u}, 2u));
    // A push under another pipeline layout object poisons the push descriptors
    // (Vulkan semantics) and makes the tracked slots incomparable, so the knowledge
    // is dropped and the next bind is recorded.
    state.push_descriptor_sets(BindPoint::compute, kLayoutB, 0u);
    // The slots of the old layout are gone (the tracker now knows only that set 0
    // of kLayoutB is a push bank), so the next bind is recorded.
    expect(!state.point(BindPoint::compute).slot_known[1u] &&
           state.point(BindPoint::compute).slot_pushed[0u]);
    expect(bind_sets(state, BindPoint::compute, kLayoutB, 1u, global, 1u));
    // The Vulkan-bind-point overload maps the same way.
    state.push_descriptor_sets(VK_PIPELINE_BIND_POINT_COMPUTE, kLayoutB, 0u);
    expect(!bind_sets(state, BindPoint::compute, kLayoutB, 1u, global, 1u))
        << "the global sets stay bound across a second push of set 0";
    // A set index the tracker does not model is forgotten rather than guessed.
    state.push_descriptor_sets(VK_PIPELINE_BIND_POINT_COMPUTE, kLayoutB,
                               kMaxTrackedDescriptorSets);
    expect(!state.point(BindPoint::compute).descriptor_known)
        << "an unmodelled set index drops the knowledge";
    // A bind point the backend never records into invalidates everything: a raw
    // push by foreign code is treated exactly like a raw bind.
    auto bind_all = [&] {
        static_cast<void>(bind_pipeline(state, BindPoint::compute, kPipelineA, kLayoutA));
        static_cast<void>(bind_sets(state, BindPoint::compute, kLayoutA, 0u, global, 1u));
        static_cast<void>(bind_pipeline(state, BindPoint::graphics, kPipelineB, kLayoutB));
        static_cast<void>(bind_sets(state, BindPoint::graphics, kLayoutB, 0u, global, 1u));
    };
    bind_all();
    // 0x7ffffffe is not a VkPipelineBindPoint this backend records into (the NV
    // ray-tracing alias is numerically the KHR one, so it is *not* such a value).
    state.push_descriptor_sets(static_cast<VkPipelineBindPoint>(0x7ffffffe), kLayoutA, 0u);
    for (auto point : {BindPoint::compute, BindPoint::graphics, BindPoint::ray_tracing}) {
        expect(!state.point(point).pipeline_known &&
               !state.point(point).descriptor_known)
            << "an untracked bind point drops every binding";
    }
}


}// namespace

static auto test_vk_bind_state_registration = [] {
    "bind_state_pipeline_identity"_test = [] { test_pipeline_identity(); };
    "bind_state_bind_points_are_independent"_test = [] { test_bind_points_are_independent(); };
    "bind_state_descriptor_identity"_test = [] { test_descriptor_identity(); };
    "bind_state_partial_ranges_are_conservative"_test = [] { test_partial_ranges_are_conservative(); };
    "bind_state_dynamic_offsets"_test = [] { test_dynamic_offsets(); };
    "bind_state_pipeline_layout_gates_descriptors"_test = [] { test_pipeline_layout_gates_descriptors(); };
    "bind_state_unrepresentable_binds_are_recorded"_test = [] { test_unrepresentable_binds_are_recorded(); };
    "bind_state_invalidation"_test = [] { test_invalidation(); };
    "bind_state_bind_point_translation"_test = [] { test_bind_point_translation(); };
    "bind_state_push_descriptor_slots"_test = [] { test_push_descriptor_slots(); };
    "bind_state_layout_change_forgets_slots"_test = [] { test_layout_change_forgets_slots(); };
    return 0;
}();

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(
        argc, const_cast<const char **>(argv));
}