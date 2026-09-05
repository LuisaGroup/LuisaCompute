#pragma once

#include <cstdint>
#include <utility>

#include <tvm/tirx/function.h>

#include <luisa/core/dll_export.h>
#include <luisa/core/stl/string.h>
#include <luisa/tile/ir.h>

namespace luisa::compute::tile::bridge::tirx {

enum class SharedTileMaterialization : uint8_t {
    // Preserve every multi-consumer pure Tile SSA definition as logical TIRx
    // storage. A target mapper may compact or deliberately inline it later.
    PRESERVE,
    // Diagnostic/tuning candidate matching the original conservative policy:
    // preserve shared transcendentals, but recompute shared cheap arithmetic.
    EXPENSIVE_ONLY,
};

struct LowerOptions {
    SharedTileMaterialization shared_tiles{SharedTileMaterialization::PRESERVE};
};

struct NativeFunction {
    tvm::tirx::PrimFunc value;
    luisa::string error;

    [[nodiscard]] bool ok() const noexcept { return error.empty() && value.defined(); }
    [[nodiscard]] explicit operator bool() const noexcept { return ok(); }
};

// Lowers portable Scalar/Tile values and structured control flow to a native
// TIRx PrimFunc. This is an in-memory C++ IR-to-IR bridge: neither Python nor
// TVMScript is involved. Logical parallel domains and execution constraints
// survive as annotations for the later target mapper; structural export does
// not guess or discard a user-requested hardware binding.
[[nodiscard]] LUISA_TILE_TIRX_BRIDGE_API NativeFunction lower(
    const Function &function, const LowerOptions &options = {}) noexcept;

}// namespace luisa::compute::tile::bridge::tirx
