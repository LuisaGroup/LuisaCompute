#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>

namespace llvm {
class LLVMContext;
class Module;
class TargetMachine;
}// namespace llvm

namespace llvm::orc {
class LLJIT;
}// namespace llvm::orc

namespace luisa::compute::simd {

// Host ORC JIT used by the SIMD backend. Schedule lowering deliberately stops
// at target-independent LLVM vector IR; this class is the sole boundary that
// asks LLVM to optimize, legalize, select instructions, allocate registers,
// and machine-schedule for the detected host CPU.
class LLVMJIT {

private:
    std::unique_ptr<::llvm::orc::LLJIT> _jit{};
    std::unique_ptr<::llvm::TargetMachine> _target_machine{};
    std::string _target_triple{};
    std::shared_ptr<std::string> _object{};
    std::string _error{};

private:
    void _fail(std::string message) noexcept;
    [[nodiscard]] bool _prepare_module(::llvm::Module &module) noexcept;
    [[nodiscard]] std::string _emit_assembly(
        ::llvm::Module &module) noexcept;

public:
    // Full O2 is retained for ordinary kernels. Very large scheduler modules
    // use O1 so compile time remains bounded instead of spending minutes in
    // whole-module IPO over a million-instruction state machine.
    [[nodiscard]] static bool selects_size_bounded_pipeline(
        size_t block_count, size_t instruction_count) noexcept;

    explicit LLVMJIT(bool capture_object = false) noexcept;
    ~LLVMJIT() noexcept;
    LLVMJIT(LLVMJIT &&) noexcept;
    LLVMJIT &operator=(LLVMJIT &&) noexcept;
    LLVMJIT(const LLVMJIT &) = delete;
    LLVMJIT &operator=(const LLVMJIT &) = delete;

    [[nodiscard]] bool add_module(
        std::unique_ptr<::llvm::Module> module,
        std::unique_ptr<::llvm::LLVMContext> context) noexcept;
    // Runs the same optimization and host target-machine pipeline as the JIT
    // and returns assembly for native-lowering regression audits.
    [[nodiscard]] std::string emit_assembly(
        std::unique_ptr<::llvm::Module> module,
        std::unique_ptr<::llvm::LLVMContext> context) noexcept;
    // Clones a caller-owned module and synchronously emits host assembly.
    // The source module and its context remain untouched and may subsequently
    // be submitted to this JIT.
    [[nodiscard]] std::string emit_assembly_copy(
        const ::llvm::Module &module) noexcept;
    [[nodiscard]] void *lookup(std::string_view name) noexcept;

    [[nodiscard]] bool succeeded() const noexcept {
        return _jit != nullptr && _error.empty();
    }
    [[nodiscard]] const std::string &error() const noexcept { return _error; }
    [[nodiscard]] const std::string &target_triple() const noexcept {
        return _target_triple;
    }
    // This is deliberately a measured host-policy gate, not merely an LLVM
    // legality query. It selects only the audited W8/ZMM shape; semantic W8
    // remains available when this returns false.
    [[nodiscard]] bool supports_native_paired_leaf_gather(
        uint32_t width) const noexcept;
    // Exact W8 and split-W16 profitability gate for biased 32-bit typed-buffer
    // indices. The portable IR remains valid without this capability;
    // unsupported targets keep the established pointer-width gather lowering.
    [[nodiscard]] bool supports_native_biased_narrow_buffer_gather(
        uint32_t width) const noexcept;
    // The bounded predicated-loop policy is measured only for host targets
    // with native 512-bit fixed vectors and legal masked gathers. Other hosts
    // retain the same portable IR backend through the generic scheduler.
    [[nodiscard]] bool supports_native_predicated_loop(
        uint32_t width) const noexcept;
    // W8 packet-body inlining is profitable only with the measured wide
    // fixed-vector register file. Narrower hosts retain the direct-call batch
    // wrapper to avoid turning cross-packet live ranges into spills.
    [[nodiscard]] bool supports_inlined_packet_batch(
        uint32_t width) const noexcept;
    // W16 sparse ray packets use llvm.masked.compressstore only when the host
    // exposes a native 512-bit register file and legal masked compression.
    // Other hosts retain the full-width portable packet path.
    [[nodiscard]] bool supports_native_vector_compress(
        uint32_t width) const noexcept;
    // W4/W8/W16 HALF4 packets are enabled only when TTI prices both directions
    // as packed conversions. This prevents the portable fpext/fptrunc IR from
    // turning into per-lane compiler-runtime calls on hosts without native
    // half conversion support.
    [[nodiscard]] bool supports_native_half_conversion(
        uint32_t width) const noexcept;
    // Exact relocatable object emitted by ORC's compiler, before JITLink
    // applies relocations. Populated only when requested at construction.
    [[nodiscard]] const std::string &object() const noexcept {
        static const std::string empty;
        return _object == nullptr ? empty : *_object;
    }
};

}// namespace luisa::compute::simd
