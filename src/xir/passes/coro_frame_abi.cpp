#include "coro_frame_abi.h"

#include <algorithm>
#include <limits>

#include <luisa/ast/type.h>
#include <luisa/xir/instructions/alloca.h>

namespace luisa::compute::xir::detail {

namespace {

[[nodiscard]] CoroFrameAbiPlan whole_plan(
    const Type *type, luisa::vector<uint32_t> access_chain) noexcept {
    CoroFrameAbiPlan plan;
    if (type != nullptr) {
        plan.fields.emplace_back(CoroFrameAbiField{
            .access_chain = std::move(access_chain),
            .type = type});
        plan.payload_size = type->size();
        plan.max_alignment = type->alignment();
    }
    return plan;
}

[[nodiscard]] bool checked_add(size_t lhs, size_t rhs,
                               size_t &result) noexcept {
    if (rhs > std::numeric_limits<size_t>::max() - lhs) {
        return false;
    }
    result = lhs + rhs;
    return true;
}

[[nodiscard]] CoroFrameAbiPlan plan_type(
    const Type *type, luisa::vector<uint32_t> access_chain,
    size_t field_limit) noexcept {
    auto whole = whole_plan(type, access_chain);
    if (type == nullptr || type->is_scalar() || field_limit < 2u) {
        return whole;
    }

    luisa::vector<const Type *> children;
    switch (type->tag()) {
        case Type::Tag::ARRAY:
        case Type::Tag::VECTOR:
            if (type->dimension() > field_limit) { return whole; }
            children.assign(type->dimension(), type->element());
            break;
        case Type::Tag::MATRIX:
            if (type->dimension() > field_limit) { return whole; }
            children.assign(
                type->dimension(),
                Type::vector(type->element(), type->dimension()));
            break;
        case Type::Tag::STRUCTURE: {
            auto members = type->members();
            if (members.size() > field_limit) { return whole; }
            children.assign(members.begin(), members.end());
            break;
        }
        default: return whole;
    }
    if (children.empty()) { return whole; }

    CoroFrameAbiPlan decomposed;
    decomposed.fields.reserve(children.size());
    for (size_t i = 0u; i < children.size(); ++i) {
        auto child_path = access_chain;
        child_path.emplace_back(static_cast<uint32_t>(i));
        auto child = plan_type(
            children[i], std::move(child_path), field_limit);
        if (child.fields.empty() ||
            child.fields.size() > field_limit - decomposed.fields.size() ||
            !checked_add(decomposed.payload_size,
                         child.payload_size,
                         decomposed.payload_size)) {
            return whole;
        }
        decomposed.max_alignment = std::max(
            decomposed.max_alignment, child.max_alignment);
        for (auto &field : child.fields) {
            decomposed.fields.emplace_back(std::move(field));
        }
    }
    // Equal payload with more load/store operations is never profitable.
    if (decomposed.fields.size() <= 1u ||
        decomposed.payload_size >= whole.payload_size) {
        return whole;
    }
    decomposed.decomposed = true;
    std::stable_sort(
        decomposed.fields.begin(), decomposed.fields.end(),
        [](auto &lhs, auto &rhs) noexcept {
            if (lhs.type->alignment() != rhs.type->alignment()) {
                return lhs.type->alignment() > rhs.type->alignment();
            }
            if (lhs.type->size() != rhs.type->size()) {
                return lhs.type->size() > rhs.type->size();
            }
            return std::lexicographical_compare(
                lhs.access_chain.begin(), lhs.access_chain.end(),
                rhs.access_chain.begin(), rhs.access_chain.end());
        });
    return decomposed;
}

}// namespace

CoroFrameAbiPlan plan_coro_frame_atom_abi(
    const CoroFrameAtomDomain::Atom &atom,
    size_t field_limit) noexcept {
    auto whole = whole_plan(atom.type, atom.access_chain);
    if (atom.root == nullptr || atom.type == nullptr ||
        field_limit < 2u) {
        return whole;
    }
    auto is_local_allocation =
        atom.root->isa<AllocaInst>() &&
        static_cast<AllocaInst *>(atom.root)->is_local();
    // CoroFrameAtomDomain gives every non-lvalue instruction one whole-value
    // atom. Restrict SSA decomposition to that complete atom: recursively
    // partitioning a partial SSA projection would not provide enough leaves
    // to reconstruct the original typed value at a continuation entry.
    auto is_complete_ssa_value =
        !atom.root->is_lvalue() && atom.access_chain.empty();
    if (!is_local_allocation && !is_complete_ssa_value) { return whole; }
    return plan_type(atom.type, atom.access_chain, field_limit);
}

}// namespace luisa::compute::xir::detail
