#include "ast_xir_cache_fingerprint.h"

#include <array>

#include <luisa/core/stl/hash.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/metadata/comment.h>
#include <luisa/xir/metadata/contiguous_copy.h>
#include <luisa/xir/metadata/curve_basis.h>
#include <luisa/xir/metadata/location.h>
#include <luisa/xir/metadata/name.h>
#include <luisa/xir/metadata/reg2mem_spill.h>
#include <luisa/xir/metadata/strided_mma.h>

namespace luisa::compute::xir {

namespace {

// A record of fixed-width words. Variable-length fields have a count or a
// content hash. No allocation addresses or use-list iteration enter the key.
class FingerprintRecord {
private:
    std::array<uint64_t, 32u> _words;
    size_t _count{0u};
    uint64_t _seed{hash64_default_seed};

public:
    [[nodiscard]] uint64_t hash() const noexcept {
        return hash64(_words.data(), _count * sizeof(uint64_t), _seed);
    }

    void add(auto number) noexcept {
        if (_count == _words.size()) {
            _seed = hash();
            _count = 0u;
        }
        _words[_count++] = static_cast<uint64_t>(number);
    }

    void array(luisa::span<const uint64_t> values) noexcept {
        add(values.size());
        for (auto v : values) { add(v); }
    }

    bool value(const Value *v) noexcept {
        add(v->derived_value_tag());
        add(v->type() ? v->type()->hash() : 0u);
        add(v->metadata_list().count_size());
        for (auto *md : v->metadata_list()) {
            auto tag = md->derived_metadata_tag();
            add(tag);
            switch (tag) {
                case DerivedMetadataTag::NAME: add(hash_value(static_cast<const NameMD *>(md)->name())); break;
                case DerivedMetadataTag::COMMENT: add(hash_value(static_cast<const CommentMD *>(md)->comment())); break;
                case DerivedMetadataTag::LOCATION: {
                    auto *loc = static_cast<const LocationMD *>(md);
                    add(hash_value(luisa::to_string(loc->file())));
                    add(loc->line());
                    break;
                }
                case DerivedMetadataTag::CURVE_BASIS: add(static_cast<const CurveBasisMD *>(md)->curve_basis_set().to_u64()); break;
                case DerivedMetadataTag::REG2MEM_SPILL: add(static_cast<const Reg2MemSpillMD *>(md)->kind()); break;
                case DerivedMetadataTag::NO_INLINE:
                case DerivedMetadataTag::SIGNATURE_CONSTRAINT: break;
                case DerivedMetadataTag::CONTIGUOUS_COPY: {
                    auto &d = static_cast<const ContiguousCopyMD *>(md)->descriptor;
                    add(d.element_count);
                    add(d.vector_width);
                    break;
                }
                case DerivedMetadataTag::STRIDED_MMA: {
                    auto &d = static_cast<const StridedMmaMD *>(md)->descriptor;
                    array(d.output_extents);
                    array(d.lhs_output_strides);
                    array(d.rhs_output_strides);
                    add(d.contraction_extent);
                    add(d.lhs_contraction_stride);
                    add(d.rhs_contraction_stride);
                    add(d.vector_width);
                    add(d.allow_reassociation);
                    add(d.vectorization);
                    break;
                }
                default: return false;
            }
        }
        return true;
    }

    // The common operands, types and merge edges are encoded by the caller.
    // Decline instructions outside the normalized XIR-to-AST boundary.
    bool instruction_payload(const Instruction *inst, const auto &local_id) noexcept {
        using enum DerivedInstructionTag;
        auto tag = inst->derived_instruction_tag();
        add(tag);
        switch (tag) {
            case ALLOCA: {
                auto *a = static_cast<const AllocaInst *>(inst);
                add(a->op());
                add(a->coro_return_selector());
                break;
            }
            case ARITHMETIC: add(static_cast<const ArithmeticInst *>(inst)->op()); break;
            case ATOMIC: add(static_cast<const AtomicInst *>(inst)->op()); break;
            case CAST: add(static_cast<const CastInst *>(inst)->op()); break;
            case THREAD_GROUP: add(static_cast<const ThreadGroupInst *>(inst)->op()); break;
            case RAY_QUERY_OBJECT_READ: add(static_cast<const RayQueryObjectReadInst *>(inst)->op()); break;
            case RAY_QUERY_OBJECT_WRITE: add(static_cast<const RayQueryObjectWriteInst *>(inst)->op()); break;
            case RESOURCE_QUERY: resource(static_cast<const ResourceQueryInst *>(inst)); break;
            case RESOURCE_READ: resource(static_cast<const ResourceReadInst *>(inst)); break;
            case RESOURCE_WRITE: resource(static_cast<const ResourceWriteInst *>(inst)); break;
            case ASSERT: add(hash_value(static_cast<const AssertInst *>(inst)->message())); break;
            case ASSUME: add(hash_value(static_cast<const AssumeInst *>(inst)->message())); break;
            case PRINT: add(hash_value(static_cast<const PrintInst *>(inst)->format())); break;
            case UNREACHABLE: add(hash_value(static_cast<const UnreachableInst *>(inst)->message())); break;
            case SWITCH:
            case INDEXED_BRANCH: array(static_cast<const IndexedBranchTerminatorInstruction *>(inst)->case_values()); break;
            case LOOP: {
                auto *loop = static_cast<const LoopInst *>(inst);
                add(local_id(loop->body_block()));
                add(local_id(loop->update_block()));
                break;
            }
            case IF: case SIMPLE_LOOP: case BRANCH: case CONDITIONAL_BRANCH:
            case BREAK: case CONTINUE: case RETURN: case LOAD: case STORE:
            case GEP: case CALL: case CLOCK:
            case RAY_QUERY_LOOP: case RAY_QUERY_DISPATCH: case RASTER_DISCARD: break;
            // RAY_QUERY_PIPELINE is outside the canonical AST-to-XIR
            // preserve_inline_ray_query_loops boundary: backends that lower
            // RayQueryLoopInst (such as native XIR-to-SPIR-V) reject it, and
            // backends that re-analyze inline ray-query loops after pre-CFG
            // mem2reg (such as HIP native closest reduction) rely on seeing
            // canonical RayQueryLoopInst instead of a pre-split pipeline.
            default: return false;
        }
        return true;
    }

    void resource(const auto *inst) noexcept {
        add(inst->op());
        add(inst->bindless_access().encoded());
    }
};

}// namespace

luisa::optional<uint64_t> AstXirCacheFingerprint::_global(const Value *v) noexcept {
    // A recursive edge sees the pending nullopt and declines the entire root.
    if (auto [iter, inserted] = _globals.try_emplace(v, luisa::nullopt); !inserted) {
        return iter->second;
    }
    luisa::optional<uint64_t> result;
    if (v->isa<Function>()) {
        result = _function(static_cast<const Function *>(v));
    } else {
        FingerprintRecord record;
        if (!record.value(v)) { return luisa::nullopt; }
        if (v->isa<Constant>()) {
            auto *c = static_cast<const Constant *>(v);
            record.add(hash64(c->data(), c->type()->size(), hash64_default_seed));
        } else if (v->isa<SpecialRegister>()) {
            record.add(static_cast<const SpecialRegister *>(v)->derived_special_register_tag());
        } else if (!v->isa<Undefined>()) {
            return luisa::nullopt;
        }
        result = record.hash();
    }
    _globals.at(v) = result;
    return result;
}

luisa::optional<uint64_t> AstXirCacheFingerprint::_function(const Function *f) noexcept {
    FingerprintRecord record;
    if (!record.value(f)) { return luisa::nullopt; }
    record.add(f->derived_function_tag());
    if (f->isa<KernelFunction>()) {
        auto size = static_cast<const KernelFunction *>(f)->block_size();
        record.add(size.x);
        record.add(size.y);
        record.add(size.z);
    } else if (f->isa<RasterStageFunction>()) {
        record.add(static_cast<const RasterStageFunction *>(f)->stage());
    }
    // Number locals by definition order, including forward references. Zero
    // represents a missing block or operand.
    luisa::unordered_map<const Value *, uint64_t> locals;
    auto number_local = [&](const Value *v) noexcept { locals.emplace(v, locals.size() + 1u); };
    auto local_id = [&](const Value *v) noexcept { return v ? locals.at(v) : uint64_t{0u}; };
    record.add(f->arguments().count_size());
    for (auto *arg : f->arguments()) {
        number_local(arg);
        if (!record.value(arg)) { return luisa::nullopt; }
        record.add(arg->derived_argument_tag());
    }
    for (auto *block : f->basic_blocks()) {
        number_local(block);
        for (auto *inst : block->instructions()) { number_local(inst); }
    }
    auto *def = f->definition();
    record.add(local_id(def ? def->body_block() : nullptr));
    record.add(f->basic_blocks().count_size());
    for (auto *block : f->basic_blocks()) {
        if (!record.value(block)) { return luisa::nullopt; }
        record.add(block->instructions().count_size());
        for (auto *inst : block->instructions()) {
            if (!record.value(inst) || !record.instruction_payload(inst, local_id)) { return luisa::nullopt; }
            if (auto *merge = inst->control_flow_merge()) { record.add(local_id(merge->merge_block())); }
            record.add(inst->operand_count());
            for (auto *use : inst->operand_uses()) {
                auto *v = use->value();
                auto global = v && v->is_global();
                record.add(global);
                if (global) {
                    auto hash = _global(v);
                    if (!hash) { return luisa::nullopt; }
                    record.add(*hash);
                } else {
                    record.add(local_id(v));
                }
            }
        }
    }
    return record.hash();
}

}// namespace luisa::compute::xir
