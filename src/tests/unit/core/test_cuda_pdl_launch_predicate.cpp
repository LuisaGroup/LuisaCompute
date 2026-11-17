// Host-only matrix test for the stream-form programmatic dependent launch (PDL)
// decision of the CUDA backend (feature B2, extended by D1 with the per-stream
// predecessor state and the launch policy).
//
// The CUDA backend relaxes the one-stream serialization of a dispatch only when
// all six terms of `should_relax_stream_serialization()` hold: the shader waits
// for its programmatic dependency (`grid_dependency_synchronize()`, recorded in
// CUDAShaderMetadata), the immediate predecessor of the launch in the stream is
// a kernel launch (that state comes from `CUDAStream::pdl_predecessor_is_kernel()`,
// i.e. from the `PdlPredecessor` enum of cuda_pdl_launch.h: `unknown` /
// `kernel_launch` / `non_kernel`), the backend policy is on
// (`CUDADeviceConfigExt::enable_programmatic_stream_serialization()` /
// `CUDAPdlStreamExt`, minus the `LUISA_DISABLE_PDL_STREAM` kill switch), the
// per-shader latch still allows the launch attribute, the device is
// Hopper-or-newer (compute capability >= 90) and the driver is 12.0+. The
// predicate is a pure host-side function (it never touches the CUDA API), so the
// full decision matrix is asserted here with no device and no driver at all:
// src/backends/cuda/cuda_pdl_launch.h is the testability seam.
//
// Every case is named after the term it exercises, and each rejection case
// differs from an accepted case in exactly one term, so a dropped or inverted
// term fails at least one line.
//
// TODO(hardware): the *execution* side of stream-form PDL (cuLaunchKernelEx with
// CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION, the per-shader latch
// and the attribute-refusal fallback in cuda_shader_native.cpp) needs sm_90+ and
// cannot be observed here: on this pre-Hopper device the predicate is false by
// construction, so the launch path never reaches the `cuLaunchKernelEx` branch.
// Cover it on Hopper with a two-kernel producer/consumer chain on one stream and
// compare against the sequential result. The end-to-end coverage of the
// per-stream predecessor state lives in
// src/tests/integration/runtime/test_cuda_stream_pdl.cpp (added by Phase 2), not
// here.

#include "ut/ut.hpp"

#include "cuda_pdl_launch.h"

#include <array>
#include <cstdint>

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace test_cuda_pdl_launch_predicate_detail {

using namespace luisa::compute::cuda::cuda_pdl_detail;

struct PredicateCase {
    const char *name;
    bool uses_grid_dependency_synchronize;
    bool predecessor_is_kernel;
    bool stream_policy_enabled;
    bool latch_enabled;
    uint32_t compute_capability;
    uint32_t driver_version;
    bool expected;
};

/// The reference platform of this repository is cc 8.9 with driver 13020; the
/// driver floor is 12000 and the device floor is cc 9.0 (sm_90).
constexpr std::array kPredicateCases{
    // All six terms satisfied: the only configuration that relaxes the launch.
    PredicateCase{"wait+latch+sm90+driver12000", true, true, true, true, 90u, 12000u, true},
    PredicateCase{"wait+latch+sm90+driver13020", true, true, true, true, 90u, 13020u, true},
    PredicateCase{"wait+latch+sm100+driver13020", true, true, true, true, 100u, 13020u, true},
    // Term 1: a shader that does not wait keeps the serialized cuLaunchKernel.
    PredicateCase{"no-wait+latch+sm90+driver13020", false, true, true, true, 90u, 13020u, false},
    PredicateCase{"no-wait+latch+sm100+driver13020", false, true, true, true, 100u, 13020u, false},
    // Term 2: the previous operation of the stream is not a kernel launch (a
    // copy, a build, an event record/wait, a host callback, or unknown after a
    // synchronize), so there is no predecessor kernel to overlap with.
    PredicateCase{"no-kernel-predecessor+policy+wait+latch+sm90+driver13020", true, false, true, true, 90u, 13020u, false},
    // Term 3: the backend policy is off (no device-config opt-in, or the
    // LUISA_DISABLE_PDL_STREAM kill switch), so no launch may be relaxed.
    PredicateCase{"no-stream-policy+kernel-predecessor+wait+latch+sm90+driver13020", true, true, false, true, 90u, 13020u, false},
    // The kernel -> kernel chain the feature exists for: both new terms hold.
    PredicateCase{"kernel-predecessor+policy+wait+latch+sm90+driver13020", true, true, true, true, 90u, 13020u, true},
    // Term 4: the per-shader latch is cleared after the driver refused the
    // attribute once, so the very same shader falls back to cuLaunchKernel.
    PredicateCase{"wait+no-latch+sm90+driver13020", true, true, true, false, 90u, 13020u, false},
    PredicateCase{"wait+no-latch+sm100+driver13020", true, true, true, false, 100u, 13020u, false},
    // Term 5: the cc-89 rejection (this box) and the older architectures.
    PredicateCase{"wait+latch+sm89+driver13020", true, true, true, true, 89u, 13020u, false},
    PredicateCase{"wait+latch+sm80+driver13020", true, true, true, true, 80u, 13020u, false},
    // Term 6: the 12.0 driver floor, including driver 11.x.
    PredicateCase{"wait+latch+sm90+driver11000", true, true, true, true, 90u, 11000u, false},
    PredicateCase{"wait+latch+sm90+driver11999", true, true, true, true, 90u, 11999u, false},
};

[[nodiscard]] bool evaluate(const PredicateCase &test_case) noexcept {
    return should_relax_stream_serialization(PdlLaunchConditions{
        .shader_waits_for_dependency = test_case.uses_grid_dependency_synchronize,
        .predecessor_is_kernel = test_case.predecessor_is_kernel,
        .stream_policy_enabled = test_case.stream_policy_enabled,
        .driver_accepts_attribute = test_case.latch_enabled,
        .compute_capability = test_case.compute_capability,
        .driver_version = test_case.driver_version});
}

}// namespace test_cuda_pdl_launch_predicate_detail

int main(int argc, char *argv[]) {

    boost::ut::detail::cfg::parse_arg_with_fallback(
        argc, const_cast<const char **>(argv));

    using namespace test_cuda_pdl_launch_predicate_detail;

    "cuda_pdl_launch_predicate_matrix"_test = [] {
        for (auto test_case : kPredicateCases) {
            auto got = evaluate(test_case);
            expect(got == test_case.expected)
                << test_case.name << ": should_relax_stream_serialization returned "
                << got << ", expected " << test_case.expected
                << " (wait=" << test_case.uses_grid_dependency_synchronize
                << ", predecessor=" << test_case.predecessor_is_kernel
                << ", policy=" << test_case.stream_policy_enabled
                << ", latch=" << test_case.latch_enabled
                << ", cc=" << test_case.compute_capability
                << ", driver=" << test_case.driver_version << ")";
        }
    };

    // Both directions of the necessary condition, over the whole documented
    // cube: the launch is relaxed if and only if every term holds. This catches a
    // term that was made unconditional (e.g. dropping the cc >= 90u test), which
    // the sparse table above only catches for the sampled configurations.
    "cuda_pdl_launch_predicate_is_conjunctive_over_the_cube"_test = [] {
        constexpr std::array<uint32_t, 4u> capabilities{80u, 89u, 90u, 100u};
        constexpr std::array<uint32_t, 3u> drivers{11000u, 12000u, 13020u};
        // 2^4 boolean terms times 4 capabilities times 3 drivers = 192 cases.
        for (auto uses_wait : {false, true}) {
            for (auto predecessor_is_kernel : {false, true}) {
                for (auto stream_policy_enabled : {false, true}) {
                    for (auto latch_enabled : {false, true}) {
                        for (auto cc : capabilities) {
                            for (auto driver : drivers) {
                                auto got = evaluate(PredicateCase{"cube", uses_wait,
                                                                  predecessor_is_kernel, stream_policy_enabled,
                                                                  latch_enabled, cc,
                                                                  driver, false});
                                auto all_terms = uses_wait && predecessor_is_kernel &&
                                                 stream_policy_enabled && latch_enabled &&
                                                 cc >= 90u && driver >= 12000u;
                                expect(got == all_terms)
                                    << "cube(wait=" << uses_wait
                                    << ", predecessor=" << predecessor_is_kernel
                                    << ", policy=" << stream_policy_enabled
                                    << ", latch=" << latch_enabled
                                    << ", cc=" << cc << ", driver=" << driver << ") = " << got
                                    << ", expected " << all_terms;
                            }
                        }
                    }
                }
            }
        }
    };
}
