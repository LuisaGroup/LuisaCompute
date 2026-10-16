#pragma once

#include <cstdint>

#include <luisa/core/intrin.h>

namespace luisa::compute::cpu {

// Save status as well as control bits so a fast shader does not leak floating-
// point state into its caller, another dispatch, or a user log callback.
struct CPUFloatingPointState {
    uint64_t control{0u};
    uint64_t status{0u};

    [[nodiscard]] static CPUFloatingPointState read() noexcept {
#if defined(LUISA_ARCH_X86_64)
        return {.control = _mm_getcsr()};
#elif defined(LUISA_ARCH_ARM64)
        CPUFloatingPointState state;
        asm volatile("mrs %0, FPCR" : "=r"(state.control) :: "memory");
        asm volatile("mrs %0, FPSR" : "=r"(state.status) :: "memory");
        return state;
#endif
    }

    void apply() const noexcept {
#if defined(LUISA_ARCH_X86_64)
        _mm_setcsr(static_cast<uint32_t>(control));
#elif defined(LUISA_ARCH_ARM64)
        asm volatile("msr FPCR, %0\n\tisb" :: "r"(control) : "memory");
        asm volatile("msr FPSR, %0" :: "r"(status) : "memory");
#endif
    }

    [[nodiscard]] CPUFloatingPointState with_fast_math() const noexcept {
        auto state = *this;
#if defined(LUISA_ARCH_X86_64)
        constexpr auto denormals_are_zero = uint64_t{1u} << 6u;
        constexpr auto flush_to_zero = uint64_t{1u} << 15u;
        state.control |= denormals_are_zero | flush_to_zero;
#elif defined(LUISA_ARCH_ARM64)
        // FPCR.FZ controls single/double precision. When FEAT_AFP's AH mode
        // is active, FIZ separately enables input flushing. AH=1 also proves
        // the FIZ field exists; do not set that reserved bit on older CPUs.
        constexpr auto flush_to_zero = uint64_t{1u} << 24u;
        constexpr auto alternative_handling = uint64_t{1u} << 1u;
        constexpr auto flush_input_to_zero = uint64_t{1u};
        state.control |= flush_to_zero;
        if ((state.control & alternative_handling) != 0u) {
            state.control |= flush_input_to_zero;
        }
        // Preserve FZ16, rounding, traps and the caller's AH selection.
#endif
        return state;
    }
};

class ScopedCPUFloatingPointEnvironment {

private:
    CPUFloatingPointState _saved{};
    bool _enabled{false};

public:
    explicit ScopedCPUFloatingPointEnvironment(bool enable_fast_math) noexcept
        : _enabled{enable_fast_math} {
        if (_enabled) {
            _saved = CPUFloatingPointState::read();
            _saved.with_fast_math().apply();
        }
    }

    // Temporarily expose the original host environment during a callback.
    explicit ScopedCPUFloatingPointEnvironment(const CPUFloatingPointState *state) noexcept
        : _enabled{state != nullptr} {
        if (_enabled) {
            _saved = CPUFloatingPointState::read();
            state->apply();
        }
    }

    ~ScopedCPUFloatingPointEnvironment() noexcept {
        if (_enabled) { _saved.apply(); }
    }

    ScopedCPUFloatingPointEnvironment(const ScopedCPUFloatingPointEnvironment &) = delete;
    ScopedCPUFloatingPointEnvironment &operator=(const ScopedCPUFloatingPointEnvironment &) = delete;

    [[nodiscard]] const CPUFloatingPointState *saved_state() const noexcept {
        return _enabled ? &_saved : nullptr;
    }
};

}// namespace luisa::compute::cpu
