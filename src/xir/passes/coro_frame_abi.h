#pragma once

#include <cstddef>
#include <cstdint>

#include <luisa/core/stl/vector.h>

#include "coro_frame_access.h"

namespace luisa::compute::xir::detail {

// ABI decomposition is deliberately bounded. The vector is intentionally
// simple; if profiling later identifies planning allocation as material, this
// is the single container to specialize without changing the analysis model.
inline constexpr size_t CORO_FRAME_ABI_FIELD_LIMIT = 32u;

struct CoroFrameAbiField {
    luisa::vector<uint32_t> access_chain;
    const Type *type{nullptr};
};

struct CoroFrameAbiPlan {
    luisa::vector<CoroFrameAbiField> fields;
    size_t payload_size{0u};
    size_t max_alignment{0u};
    bool decomposed{false};
};

// Computes a storage-preserving partition of one dataflow atom. Whole values
// remain one field unless recursively decomposing them strictly reduces the
// sum of represented type sizes. Packed children are retained as aggregates,
// so the plan removes padding with the fewest fields found by this recursion
// rather than blindly scalarizing every primitive. Only local allocations are
// decomposed; SSA values and plans exceeding field_limit remain whole.
[[nodiscard]] CoroFrameAbiPlan plan_coro_frame_atom_abi(
    const CoroFrameAtomDomain::Atom &atom,
    size_t field_limit = CORO_FRAME_ABI_FIELD_LIMIT) noexcept;

}// namespace luisa::compute::xir::detail
