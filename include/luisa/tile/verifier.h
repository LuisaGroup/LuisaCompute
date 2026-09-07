#pragma once

#include <luisa/core/dll_export.h>
#include <luisa/core/stl/string.h>
#include <luisa/core/stl/vector.h>
#include <luisa/tile/ir.h>
#include <luisa/tile/target.h>

namespace luisa::compute::tile {

namespace detail {
class Verifier;
}// namespace detail

struct VerificationDiagnostic {
    const Function *function{nullptr};
    const Operation *operation{nullptr};
    luisa::string message;
};

class LUISA_TILE_API VerificationResult final {

private:
    friend class detail::Verifier;
    friend LUISA_TILE_API VerificationResult verify(const Module &, const TargetModel *) noexcept;
    luisa::vector<VerificationDiagnostic> _diagnostics;

public:
    [[nodiscard]] bool ok() const noexcept { return _diagnostics.empty(); }
    [[nodiscard]] explicit operator bool() const noexcept { return ok(); }
    [[nodiscard]] luisa::span<const VerificationDiagnostic> diagnostics() const noexcept { return _diagnostics; }
};

// Checks IR structure, types, lexical dominance, state flow and optional
// target constraints. A successful result is not a general alias/race proof:
// valid PARALLEL instances are independent by language contract. Debug/runtime
// validation may diagnose violated invocation preconditions separately;
// optimizers need not rediscover semantic independence before using it.
[[nodiscard]] LUISA_TILE_API VerificationResult verify(const Module &module, const TargetModel *target = nullptr) noexcept;

}// namespace luisa::compute::tile
