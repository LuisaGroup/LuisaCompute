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

// Lowers the portable scalar/control-flow subset of TileIR into a native
// TIRx PrimFunc. This is an in-memory C++ IR-to-IR bridge: neither Python nor
// TVMScript is involved. Target-specific execution/resource binding remains a
// later scheduling concern and is deliberately not guessed here.
[[nodiscard]] LUISA_TILE_TIRX_BRIDGE_API NativeFunction lower(
    const Function &function) noexcept;

}// namespace luisa::compute::tile::bridge::tirx
