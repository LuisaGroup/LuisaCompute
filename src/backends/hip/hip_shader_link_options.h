#pragma once

#include <array>
#include <cstdint>
#include <span>
#include <type_traits>

#include <hip/hiprtc.h>

namespace luisa::compute::hip {

// HIPRTC retains the IR-to-ISA option strings until hiprtcLinkComplete. Keep
// every pointer-bearing layer in one non-movable object whose lifetime covers
// the link state; copying this object would leave its value array pointing at
// the source object's string-pointer array.
class HIPRTCLinkOptions final {

private:
    std::array<const char *, 2u> _ir_to_isa_options{
        "-mllvm",
        "-amdgpu-inline-max-bb=0"};
    std::array<hiprtcJIT_option, 2u> _jit_options{
        HIPRTC_JIT_IR_TO_ISA_OPT_EXT,
        HIPRTC_JIT_IR_TO_ISA_OPT_COUNT_EXT};
    std::array<void *, 2u> _jit_option_values{
        static_cast<void *>(_ir_to_isa_options.data()),
        reinterpret_cast<void *>(
            static_cast<std::uintptr_t>(
                _ir_to_isa_options.size()))};

public:
    HIPRTCLinkOptions() noexcept = default;
    HIPRTCLinkOptions(const HIPRTCLinkOptions &) = delete;
    HIPRTCLinkOptions(HIPRTCLinkOptions &&) = delete;
    HIPRTCLinkOptions &operator=(const HIPRTCLinkOptions &) = delete;
    HIPRTCLinkOptions &operator=(HIPRTCLinkOptions &&) = delete;

    [[nodiscard]] unsigned int jit_option_count() const noexcept {
        return static_cast<unsigned int>(_jit_options.size());
    }

    [[nodiscard]] hiprtcJIT_option *jit_options() noexcept {
        return _jit_options.data();
    }

    [[nodiscard]] void **jit_option_values() noexcept {
        return _jit_option_values.data();
    }

    [[nodiscard]] std::span<const char *const>
    ir_to_isa_options() const noexcept {
        return _ir_to_isa_options;
    }

    [[nodiscard]] std::span<const hiprtcJIT_option>
    jit_options() const noexcept {
        return _jit_options;
    }

    [[nodiscard]] std::span<void *const>
    jit_option_values() const noexcept {
        return _jit_option_values;
    }
};

static_assert(!std::is_copy_constructible_v<HIPRTCLinkOptions>);
static_assert(!std::is_move_constructible_v<HIPRTCLinkOptions>);

}// namespace luisa::compute::hip
