#include "coro_frame_access.h"

#include <algorithm>
#include <limits>

#include <luisa/ast/type.h>
#include <luisa/core/logging.h>
#include <luisa/xir/constant.h>
#include <luisa/xir/function.h>
#include <luisa/xir/instruction.h>
#include <luisa/xir/instructions/alloca.h>
#include <luisa/xir/instructions/gep.h>
#include <luisa/xir/instructions/load.h>
#include <luisa/xir/instructions/store.h>
#include <luisa/xir/passes/aggregate_field_bitmask.h>

#include "helpers.h"

namespace luisa::compute::xir::detail {

namespace {

struct PointerProjection {
    Value *pointer{nullptr};
    const Type *type{nullptr};
    luisa::vector<luisa::optional<size_t>> access_pattern;
    AggregateFieldBitmask may;
    AggregateFieldBitmask must;
    bool accessed{false};
    bool observed{false};

    PointerProjection(Value *p, const Type *pointer_type,
                      const Type *root_type) noexcept
        : pointer{p}, type{pointer_type}, may{root_type}, must{root_type} {}
};

struct AllocaProjectionAnalysis {
    bool valid{true};
    luisa::vector<PointerProjection> projections;
    luisa::unordered_map<Value *, size_t> projection_indices;
};

struct ProjectedIndex {
    bool valid{false};
    luisa::optional<size_t> constant;
};

struct AtomEndpoint {
    luisa::vector<uint32_t> path;
    const Type *type{nullptr};
};

[[nodiscard]] bool is_integer_index_type(const Type *type) noexcept {
    if (type == nullptr) { return false; }
    switch (type->tag()) {
        case Type::Tag::INT8:
        case Type::Tag::UINT8:
        case Type::Tag::INT16:
        case Type::Tag::UINT16:
        case Type::Tag::INT32:
        case Type::Tag::UINT32:
        case Type::Tag::INT64:
        case Type::Tag::UINT64: return true;
        default: return false;
    }
}

[[nodiscard]] ProjectedIndex decode_projected_index(Value *value) noexcept {
    ProjectedIndex result;
    if (value == nullptr || !is_integer_index_type(value->type())) {
        return result;
    }
    result.valid = true;
    if (!value->isa<Constant>()) { return result; }
    uint64_t index = 0u;
    if (!try_decode_constant_nonnegative_integer(value, index) ||
        index > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
        result.valid = false;
        return result;
    }
    result.constant = static_cast<size_t>(index);
    return result;
}

[[nodiscard]] const Type *project_indexed_type(
    const Type *type, const ProjectedIndex &index,
    bool &selects_one_static_subtree) noexcept {
    if (type == nullptr || !index.valid) { return nullptr; }
    switch (type->tag()) {
        case Type::Tag::VECTOR:
        case Type::Tag::ARRAY: {
            auto dimension = type->dimension();
            if (index.constant) {
                if (*index.constant >= dimension) { return nullptr; }
            } else if (dimension > 1u) {
                selects_one_static_subtree = false;
            }
            return type->element();
        }
        case Type::Tag::MATRIX: {
            auto dimension = type->dimension();
            if (index.constant) {
                if (*index.constant >= dimension) { return nullptr; }
            } else if (dimension > 1u) {
                selects_one_static_subtree = false;
            }
            return Type::vector(type->element(), dimension);
        }
        case Type::Tag::STRUCTURE: {
            if (!index.constant) { return nullptr; }
            auto members = type->members();
            return *index.constant < members.size() ?
                       members[*index.constant] :
                       nullptr;
        }
        default: return nullptr;
    }
}

[[nodiscard]] AllocaProjectionAnalysis analyze_alloca_projections(
    AllocaInst *alloca) noexcept {
    AllocaProjectionAnalysis result;
    if (alloca == nullptr || alloca->type() == nullptr ||
        alloca->type()->is_scalar()) {
        result.valid = false;
        return result;
    }
    auto *root_type = alloca->type();
    result.projections.emplace_back(alloca, root_type, root_type);
    result.projections.front().may.set(true);
    result.projections.front().must.set(true);
    result.projection_indices.emplace(alloca, 0u);

    for (size_t cursor = 0u;
         cursor < result.projections.size() && result.valid; ++cursor) {
        auto *pointer = result.projections[cursor].pointer;
        // The vector may reallocate while discovering child GEPs. Snapshot
        // the immutable parent projection before appending anything.
        auto parent_pattern = result.projections[cursor].access_pattern;
        auto *parent_type = result.projections[cursor].type;
        for (auto *use : pointer->use_list()) {
            auto *user = use == nullptr ? nullptr : use->user();
            if (user == nullptr || !user->isa<Instruction>()) {
                result.valid = false;
                break;
            }
            auto *instruction = static_cast<Instruction *>(user);
            if (instruction->isa<GEPInst>() &&
                static_cast<GEPInst *>(instruction)->base() == pointer) {
                auto *gep = static_cast<GEPInst *>(instruction);
                auto pattern = parent_pattern;
                auto *projected_type = parent_type;
                auto selects_one_static_subtree = true;
                for (size_t i = 0u; i < gep->index_count(); ++i) {
                    auto index = decode_projected_index(gep->index(i));
                    projected_type = project_indexed_type(
                        projected_type, index,
                        selects_one_static_subtree);
                    if (projected_type == nullptr) {
                        result.valid = false;
                        break;
                    }
                    pattern.emplace_back(index.constant);
                }
                if (!result.valid || gep->type() != projected_type) {
                    result.valid = false;
                    break;
                }
                if (auto iter = result.projection_indices.find(gep);
                    iter != result.projection_indices.end()) {
                    if (result.projections[iter->second].access_pattern !=
                        pattern) {
                        result.valid = false;
                    }
                    continue;
                }
                auto projection_index = result.projections.size();
                result.projections.emplace_back(
                    gep, projected_type, root_type);
                auto &projection = result.projections.back();
                projection.access_pattern = std::move(pattern);
                if (!projection.may.mark_access_pattern(
                        projection.access_pattern)) {
                    result.valid = false;
                    break;
                }
                if (selects_one_static_subtree) {
                    projection.must = projection.may;
                }
                result.projection_indices.emplace(
                    gep, projection_index);
                continue;
            }
            auto &projection = result.projections[cursor];
            if (instruction->isa<LoadInst>() &&
                static_cast<LoadInst *>(instruction)->variable() == pointer) {
                projection.accessed = true;
                projection.observed = true;
            } else if (instruction->isa<StoreInst>() &&
                       static_cast<StoreInst *>(instruction)->variable() ==
                           pointer) {
                projection.accessed = true;
            } else {
                // Calls, atomics, and any other address escape may observe
                // and modify any leaf selected by this projection.
                projection.accessed = true;
                projection.observed = true;
            }
        }
    }
    return result;
}

[[nodiscard]] bool span_is_partial(
    const AggregateFieldBitmask &mask,
    luisa::span<const size_t> path) noexcept {
    auto span = mask.access(path);
    return span.any() && !span.all();
}

void collect_atom_partition(
    const Type *type, luisa::vector<size_t> &path,
    const AggregateFieldBitmask &relevant,
    luisa::span<const PointerProjection> projections,
    luisa::vector<AtomEndpoint> &atoms) noexcept {
    auto relevant_span = relevant.access(luisa::span{path});
    if (relevant_span.none()) { return; }
    auto requires_split = !relevant_span.all();
    if (!requires_split) {
        for (auto &&projection : projections) {
            if (!projection.accessed) { continue; }
            if (span_is_partial(projection.may, luisa::span{path}) ||
                span_is_partial(projection.must, luisa::span{path})) {
                requires_split = true;
                break;
            }
        }
    }
    if (!requires_split) {
        luisa::vector<uint32_t> atom_path;
        atom_path.reserve(path.size());
        for (auto index : path) {
            LUISA_DEBUG_ASSERT(
                index <= std::numeric_limits<uint32_t>::max(),
                "Aggregate access index exceeds frame ABI width.");
            atom_path.emplace_back(static_cast<uint32_t>(index));
        }
        atoms.emplace_back(AtomEndpoint{
            .path = std::move(atom_path), .type = type});
        return;
    }

    auto descend = [&](size_t index, const Type *child) noexcept {
        path.emplace_back(index);
        collect_atom_partition(
            child, path, relevant, projections, atoms);
        path.pop_back();
    };
    switch (type->tag()) {
        case Type::Tag::VECTOR:
        case Type::Tag::ARRAY:
            for (size_t i = 0u; i < type->dimension(); ++i) {
                descend(i, type->element());
            }
            break;
        case Type::Tag::MATRIX: {
            auto *column = Type::vector(
                type->element(), type->dimension());
            for (size_t i = 0u; i < type->dimension(); ++i) {
                descend(i, column);
            }
            break;
        }
        case Type::Tag::STRUCTURE: {
            auto members = type->members();
            for (size_t i = 0u; i < members.size(); ++i) {
                descend(i, members[i]);
            }
            break;
        }
        default:
            LUISA_ERROR_WITH_LOCATION(
                "A primitive aggregate leaf cannot be partially masked.");
    }
}

[[nodiscard]] AllocaInst *frame_trace_local_alloca(Value *value) noexcept {
    while (value != nullptr && value->isa<Instruction>()) {
        auto *instruction = static_cast<Instruction *>(value);
        if (instruction->isa<AllocaInst>()) {
            auto *alloca = static_cast<AllocaInst *>(instruction);
            return alloca->is_local() ? alloca : nullptr;
        }
        if (!instruction->isa<GEPInst>()) { return nullptr; }
        value = static_cast<GEPInst *>(instruction)->base();
    }
    return nullptr;
}

}// namespace

CoroFrameAtomDomain::CoroFrameAtomDomain(
    FunctionDefinition *definition,
    luisa::span<Value *const> designated_values) noexcept {
    if (definition == nullptr) { return; }

    luisa::unordered_map<AllocaInst *, AllocaProjectionAnalysis>
        alloca_accesses;
    for (auto *block : definition->basic_blocks()) {
        for (auto *instruction : block->instructions()) {
            if (instruction->isa<AllocaInst>()) {
                auto *alloca = static_cast<AllocaInst *>(instruction);
                if (alloca->is_local()) {
                    alloca_accesses.emplace(
                        alloca, analyze_alloca_projections(alloca));
                }
            }
        }
    }

    for (auto *block : definition->basic_blocks()) {
        for (auto *instruction : block->instructions()) {
            if (instruction->isa<AllocaInst>()) {
                auto *alloca = static_cast<AllocaInst *>(instruction);
                if (!alloca->is_local()) { continue; }
                auto &analysis = alloca_accesses.at(alloca);
                if (analysis.valid) {
                    AggregateFieldBitmask relevant{alloca->type()};
                    for (auto &&projection : analysis.projections) {
                        if (projection.observed) {
                            relevant |= projection.may;
                        }
                    }
                    luisa::vector<AtomEndpoint> endpoints;
                    luisa::vector<size_t> path;
                    collect_atom_partition(
                        alloca->type(), path, relevant,
                        luisa::span{analysis.projections}, endpoints);
                    auto first = _atoms.size();
                    for (auto &endpoint : endpoints) {
                        _atoms.emplace_back(Atom{
                            .root = alloca,
                            .access_chain = endpoint.path,
                            .type = endpoint.type});
                    }
                    if (endpoints.size() > 1u) {
                        ++_split_alloca_count;
                        _split_atom_count += endpoints.size();
                    }
                    // The partition is induced by every memory-access May and
                    // Must mask. Therefore an accessed pointer either covers
                    // an atom completely or misses it completely; no aliasing
                    // precision is lost when the tree is materialized as
                    // disjoint frame values.
                    luisa::vector<size_t> atom_path;
                    for (auto &&projection : analysis.projections) {
                        auto &accesses =
                            _memory_accesses[projection.pointer];
                        for (size_t i = first; i < _atoms.size(); ++i) {
                            atom_path.clear();
                            for (auto index : _atoms[i].access_chain) {
                                atom_path.emplace_back(index);
                            }
                            auto may = projection.may.access(
                                luisa::span{atom_path});
                            if (may.any()) {
                                if (projection.accessed) {
                                    LUISA_DEBUG_ASSERT(
                                        may.all(),
                                        "Frame atom partition does not refine "
                                        "an accessed May mask.");
                                }
                                auto must = projection.must.access(
                                    luisa::span{atom_path});
                                accesses.emplace_back(MemoryAccess{
                                    .atom_index = i,
                                    .covers_atom = must.all()});
                            }
                        }
                    }
                } else {
                    auto index = _atoms.size();
                    _atoms.emplace_back(Atom{
                        .root = alloca,
                        .type = alloca->type()});
                    _memory_accesses[alloca].emplace_back(MemoryAccess{
                        .atom_index = index,
                        .covers_atom = true});
                }
                continue;
            }
            if (instruction->type() != nullptr &&
                !instruction->is_lvalue() &&
                !instruction->is_terminator()) {
                auto index = _atoms.size();
                _atoms.emplace_back(Atom{
                    .root = instruction,
                    .type = instruction->type()});
                _ssa_indices.emplace(instruction, index);
            }
        }
    }

    // Ordinary frame atoms intentionally omit constants, arguments and
    // special registers because they are available without a spill. A
    // scheduler-visible designated value is different: the host observes a
    // stored frame between continuation executions, so even an otherwise
    // replayable/always-available root needs a concrete ABI atom. Preserve
    // instruction atoms already numbered above and append only missing roots.
    for (auto *value : designated_values) {
        if (value == nullptr || value->type() == nullptr ||
            value->is_lvalue() || value->type()->is_resource() ||
            value->type()->is_custom() ||
            _ssa_indices.contains(value)) {
            continue;
        }
        auto index = _atoms.size();
        _atoms.emplace_back(Atom{
            .root = value, .type = value->type()});
        _ssa_indices.emplace(value, index);
    }

    // Every GEP of an unsplit local denotes the same whole-allocation atom.
    for (auto *block : definition->basic_blocks()) {
        for (auto *instruction : block->instructions()) {
            if (!instruction->isa<GEPInst>() ||
                _memory_accesses.contains(instruction)) {
                continue;
            }
            if (auto *alloca = frame_trace_local_alloca(instruction)) {
                if (auto iter = _memory_accesses.find(alloca);
                    iter != _memory_accesses.end()) {
                    auto accesses = iter->second;
                    for (auto &access : accesses) {
                        access.covers_atom = false;
                    }
                    _memory_accesses.emplace(
                        instruction, std::move(accesses));
                }
            }
        }
    }
}

luisa::optional<size_t> CoroFrameAtomDomain::ssa_index(
    Value *value) const noexcept {
    if (auto iter = _ssa_indices.find(value); iter != _ssa_indices.end()) {
        return iter->second;
    }
    return luisa::nullopt;
}

luisa::span<const CoroFrameAtomDomain::MemoryAccess>
CoroFrameAtomDomain::memory_accesses(
    Value *pointer) const noexcept {
    if (auto iter = _memory_accesses.find(pointer);
        iter != _memory_accesses.end()) {
        return luisa::span<const MemoryAccess>{iter->second};
    }
    return {};
}

}// namespace luisa::compute::xir::detail
