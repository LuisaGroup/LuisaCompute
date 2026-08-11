#include "coro_frame_access.h"

#include <algorithm>
#include <limits>

#include <luisa/ast/type.h>
#include <luisa/xir/constant.h>
#include <luisa/xir/function.h>
#include <luisa/xir/instruction.h>
#include <luisa/xir/instructions/alloca.h>
#include <luisa/xir/instructions/gep.h>
#include <luisa/xir/instructions/load.h>
#include <luisa/xir/instructions/store.h>

#include "helpers.h"

namespace luisa::compute::xir::detail {

namespace {

struct StaticAccessEndpoint {
    Value *pointer{nullptr};
    luisa::vector<uint32_t> path;
    const Type *type{nullptr};
};

struct StaticAllocaAccess {
    bool valid{true};
    luisa::vector<StaticAccessEndpoint> endpoints;
    luisa::unordered_map<Value *, luisa::vector<uint32_t>> pointer_paths;
};

[[nodiscard]] bool path_is_prefix(luisa::span<const uint32_t> prefix,
                                  luisa::span<const uint32_t> path) noexcept {
    return prefix.size() <= path.size() &&
           std::equal(prefix.begin(), prefix.end(), path.begin());
}

[[nodiscard]] bool path_less(const StaticAccessEndpoint &lhs,
                             const StaticAccessEndpoint &rhs) noexcept {
    return std::lexicographical_compare(
        lhs.path.begin(), lhs.path.end(), rhs.path.begin(), rhs.path.end());
}

[[nodiscard]] bool same_path(luisa::span<const uint32_t> lhs,
                             luisa::span<const uint32_t> rhs) noexcept {
    return lhs.size() == rhs.size() &&
           std::equal(lhs.begin(), lhs.end(), rhs.begin());
}

void collect_static_accesses(Value *pointer,
                             luisa::vector<uint32_t> path,
                             StaticAllocaAccess &result,
                             luisa::unordered_set<Value *> &visiting) noexcept {
    if (!result.valid || pointer == nullptr) { return; }
    if (auto iter = result.pointer_paths.find(pointer);
        iter != result.pointer_paths.end()) {
        if (!same_path(iter->second, path)) { result.valid = false; }
        return;
    }
    if (!visiting.emplace(pointer).second) {
        result.valid = false;
        return;
    }
    result.pointer_paths.emplace(pointer, path);
    for (auto *use : pointer->use_list()) {
        auto *user = use == nullptr ? nullptr : use->user();
        if (user == nullptr || !user->isa<Instruction>()) {
            result.valid = false;
            break;
        }
        auto *instruction = static_cast<Instruction *>(user);
        switch (instruction->derived_instruction_tag()) {
            case DerivedInstructionTag::GEP: {
                auto *gep = static_cast<GEPInst *>(instruction);
                if (gep->base() != pointer || gep->index_count() == 0u) {
                    result.valid = false;
                    break;
                }
                auto child_path = path;
                for (size_t i = 0u; i < gep->index_count(); ++i) {
                    uint64_t index = 0u;
                    if (!try_decode_constant_nonnegative_integer(
                            gep->index(i), index) ||
                        index > std::numeric_limits<uint32_t>::max()) {
                        result.valid = false;
                        break;
                    }
                    child_path.emplace_back(static_cast<uint32_t>(index));
                }
                if (result.valid) {
                    collect_static_accesses(
                        gep, std::move(child_path), result, visiting);
                }
                break;
            }
            case DerivedInstructionTag::LOAD: {
                auto *load = static_cast<LoadInst *>(instruction);
                if (load->variable() != pointer || path.empty()) {
                    result.valid = false;
                    break;
                }
                result.endpoints.emplace_back(StaticAccessEndpoint{
                    .pointer = pointer,
                    .path = path,
                    .type = pointer->type()});
                break;
            }
            case DerivedInstructionTag::STORE: {
                auto *store = static_cast<StoreInst *>(instruction);
                if (store->variable() != pointer) {
                    result.valid = false;
                }
                // Stores do not define observation granularity. A store to a
                // parent subobject can kill every descendant read atom, while
                // a store inside a coarser read atom kills that enclosing atom.
                // Treating stores as leaf endpoints would make these perfectly
                // representable overlaps force a whole-allocation fallback.
                break;
            }
            default:
                // Atomics, reference calls, and any other address escape can
                // observe or modify an arbitrary overlapping subobject.
                result.valid = false;
                break;
        }
        if (!result.valid) { break; }
    }
    visiting.erase(pointer);
}

[[nodiscard]] StaticAllocaAccess analyze_static_alloca(
    AllocaInst *alloca) noexcept {
    StaticAllocaAccess result;
    if (alloca == nullptr || alloca->type() == nullptr ||
        alloca->type()->is_scalar()) {
        result.valid = false;
        return result;
    }
    luisa::unordered_set<Value *> visiting;
    collect_static_accesses(alloca, {}, result, visiting);
    if (!result.valid || result.endpoints.empty()) {
        result.valid = false;
        return result;
    }
    std::stable_sort(result.endpoints.begin(), result.endpoints.end(), path_less);
    result.endpoints.erase(
        std::unique(result.endpoints.begin(), result.endpoints.end(),
                    [](auto &lhs, auto &rhs) noexcept {
                        return same_path(lhs.path, rhs.path);
                    }),
        result.endpoints.end());
    // Distinct atoms must denote disjoint storage. A whole subobject access
    // overlapping a deeper field is kept as one conservative root atom.
    for (size_t i = 0u; i < result.endpoints.size(); ++i) {
        for (size_t j = i + 1u; j < result.endpoints.size(); ++j) {
            if (path_is_prefix(result.endpoints[i].path,
                               result.endpoints[j].path)) {
                result.valid = false;
                return result;
            }
        }
    }
    return result;
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
    FunctionDefinition *definition) noexcept {
    if (definition == nullptr) { return; }

    luisa::unordered_map<AllocaInst *, StaticAllocaAccess> alloca_accesses;
    for (auto *block : definition->basic_blocks()) {
        for (auto *instruction : block->instructions()) {
            if (instruction->isa<AllocaInst>()) {
                auto *alloca = static_cast<AllocaInst *>(instruction);
                if (alloca->is_local()) {
                    alloca_accesses.emplace(
                        alloca, analyze_static_alloca(alloca));
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
                    ++_split_alloca_count;
                    auto first = _atoms.size();
                    for (auto &endpoint : analysis.endpoints) {
                        _atoms.emplace_back(Atom{
                            .root = alloca,
                            .access_chain = endpoint.path,
                            .type = endpoint.type});
                    }
                    _split_atom_count += _atoms.size() - first;
                    // Interior GEPs are address computations only. Map each
                    // pointer to every overlapping read atom. A parent store
                    // covers every descendant atom, while a child store only
                    // partially updates a coarser enclosing atom. Loads are
                    // non-overlapping by construction, so this relation is
                    // exact and no byte-range approximation is needed.
                    for (auto &[pointer, path] : analysis.pointer_paths) {
                        auto &accesses = _memory_accesses[pointer];
                        for (size_t i = first; i < _atoms.size(); ++i) {
                            if (path_is_prefix(path, _atoms[i].access_chain) ||
                                path_is_prefix(_atoms[i].access_chain, path)) {
                                accesses.emplace_back(MemoryAccess{
                                    .atom_index = i,
                                    .covers_atom = path_is_prefix(
                                        path, _atoms[i].access_chain)});
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
