#include "hip_shader_link_options.h"
#include "ut/ut.hpp"

#include <cstdint>
#include <string_view>
#include <type_traits>
#include <utility>

using namespace boost::ut;
using namespace boost::ut::literals;
using namespace luisa::compute::hip;

static auto suite = [] {
    "HIPRTC link options remove only the AMDGPU compile-time BB guard"_test = [] {
        HIPRTCLinkOptions options;

        const auto ir_to_isa = options.ir_to_isa_options();
        expect(ir_to_isa.size() == 2u);
        expect(std::string_view{ir_to_isa[0]} == "-mllvm");
        expect(std::string_view{ir_to_isa[1]} ==
               "-amdgpu-inline-max-bb=0");

        const auto jit_options =
            std::as_const(options).jit_options();
        expect(options.jit_option_count() == 2u);
        expect(jit_options.size() == 2u);
        expect(jit_options[0] ==
               HIPRTC_JIT_IR_TO_ISA_OPT_EXT);
        expect(jit_options[1] ==
               HIPRTC_JIT_IR_TO_ISA_OPT_COUNT_EXT);
    };

    "HIPRTC option storage outlives its self-referential pointer view"_test = [] {
        static_assert(
            !std::is_copy_constructible_v<HIPRTCLinkOptions>);
        static_assert(
            !std::is_move_constructible_v<HIPRTCLinkOptions>);

        HIPRTCLinkOptions options;
        const auto ir_to_isa = options.ir_to_isa_options();
        const auto values =
            std::as_const(options).jit_option_values();
        expect(values.size() == 2u);
        expect(values[0] ==
               static_cast<const void *>(ir_to_isa.data()));
        expect(reinterpret_cast<std::uintptr_t>(values[1]) ==
               ir_to_isa.size());
    };
};

int main() {
    return 0;
}
