#pragma once

#include <utility>

#include <tvm/tirx/function.h>

#include <luisa/core/dll_export.h>
#include <luisa/core/stl/string.h>
#include <luisa/tile/ir.h>

namespace luisa::compute::tile::bridge::tirx {

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
    const Function &function) noexcept;

}// namespace luisa::compute::tile::bridge::tirx
