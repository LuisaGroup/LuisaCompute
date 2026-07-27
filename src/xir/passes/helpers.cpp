#include <luisa/core/logging.h>
#include <luisa/xir/undefined.h>
#include <luisa/xir/function.h>
#include <luisa/xir/module.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/instructions/resource.h>
#include <luisa/xir/metadata/reg2mem_spill.h>

#include <atomic>

#include "helpers.h"

namespace luisa::compute::xir {

Value *trace_pointer_base_value(Value *pointer) noexcept {
    return pointer != nullptr && pointer->isa<GEPInst>() ?
               trace_pointer_base_value(static_cast<GEPInst *>(pointer)->base()) :
               pointer;
}

AllocaInst *trace_pointer_base_local_alloca_inst(Value *pointer) noexcept {
    if (auto base = trace_pointer_base_value(pointer);
        base != nullptr && base->isa<AllocaInst>() &&
        static_cast<AllocaInst *>(base)->op() == AllocaOp::LOCAL) {
        return static_cast<AllocaInst *>(base);
    }
    return nullptr;
}

InstructionMemoryInfo get_memory_info(Instruction *inst) noexcept {
    auto pointer_scope = [](Value *pointer) noexcept {
        auto base = trace_pointer_base_value(pointer);
        if (base != nullptr && base->isa<AllocaInst>()) {
            return static_cast<AllocaInst *>(base)->is_shared() ?
                       MemoryScope::SHARED :
                       MemoryScope::LOCAL;
        }
        // A ReferenceArgument has no intrinsic address space: callers may
        // bind local or shared storage. NONE paired with a non-NONE effect is
        // the fail-closed "unknown scope" representation used by alias
        // analysis.
        return MemoryScope::NONE;
    };
    switch (inst->derived_instruction_tag()) {
        case DerivedInstructionTag::ARITHMETIC:
        case DerivedInstructionTag::CAST:
        case DerivedInstructionTag::GEP:
        case DerivedInstructionTag::PHI:
            return {MemoryScope::NONE, MemoryEffects::NONE, false};
        case DerivedInstructionTag::CLOCK:
            return {MemoryScope::GLOBAL, MemoryEffects::READ, false};
        case DerivedInstructionTag::RESOURCE_QUERY: {
            auto query = static_cast<ResourceQueryInst *>(inst);
            switch (query->op()) {
                // Resource dimensions, addresses, and explicit-LOD/gradient
                // sampling results are contractually stable for the duration
                // of a shader invocation. In particular, the ResourceQueryOp
                // contract explicitly excludes same-shader texture writes
                // from affecting sampling.
                case ResourceQueryOp::BUFFER_SIZE:
                case ResourceQueryOp::BYTE_BUFFER_SIZE:
                case ResourceQueryOp::TEXTURE2D_SIZE:
                case ResourceQueryOp::TEXTURE3D_SIZE:
                case ResourceQueryOp::BINDLESS_BUFFER_SIZE:
                case ResourceQueryOp::BINDLESS_BYTE_BUFFER_SIZE:
                case ResourceQueryOp::BINDLESS_TEXTURE2D_SIZE:
                case ResourceQueryOp::BINDLESS_TEXTURE3D_SIZE:
                case ResourceQueryOp::BINDLESS_TEXTURE2D_SIZE_LEVEL:
                case ResourceQueryOp::BINDLESS_TEXTURE3D_SIZE_LEVEL:
                case ResourceQueryOp::TEXTURE2D_SAMPLE_LEVEL:
                case ResourceQueryOp::TEXTURE2D_SAMPLE_GRAD:
                case ResourceQueryOp::TEXTURE2D_SAMPLE_GRAD_LEVEL:
                case ResourceQueryOp::TEXTURE3D_SAMPLE_LEVEL:
                case ResourceQueryOp::TEXTURE3D_SAMPLE_GRAD:
                case ResourceQueryOp::TEXTURE3D_SAMPLE_GRAD_LEVEL:
                case ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_LEVEL:
                case ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD:
                case ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD_LEVEL:
                case ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_LEVEL:
                case ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_GRAD:
                case ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_GRAD_LEVEL:
                case ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_LEVEL_SAMPLER:
                case ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD_SAMPLER:
                case ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD_LEVEL_SAMPLER:
                case ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_LEVEL_SAMPLER:
                case ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_GRAD_SAMPLER:
                case ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_GRAD_LEVEL_SAMPLER:
                case ResourceQueryOp::BUFFER_DEVICE_ADDRESS:
                case ResourceQueryOp::BINDLESS_BUFFER_DEVICE_ADDRESS:
                    return {MemoryScope::GLOBAL, MemoryEffects::NONE, false};

                // An implicit-LOD sample computes derivatives at its execution
                // point. Even with identical SSA operands and immutable
                // texture contents, moving or value-numbering it across a
                // divergent CFG boundary can change the derivative/convergence
                // context. Model it as a removable read: DCE may erase an
                // unused result, but CSE/GVN/LICM must leave it in place.
                case ResourceQueryOp::TEXTURE2D_SAMPLE:
                case ResourceQueryOp::TEXTURE3D_SAMPLE:
                case ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE:
                case ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE:
                case ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_SAMPLER:
                case ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_SAMPLER:
                    return {MemoryScope::GLOBAL, MemoryEffects::READ, false};

                // Acceleration-structure instance properties and traces observe
                // global state that ResourceWriteInst may mutate in the same
                // invocation. They are removable when unused, but must not be
                // CSE'd or moved across a potentially aliasing write.
                case ResourceQueryOp::RAY_TRACING_INSTANCE_TRANSFORM:
                case ResourceQueryOp::RAY_TRACING_INSTANCE_USER_ID:
                case ResourceQueryOp::RAY_TRACING_INSTANCE_VISIBILITY_MASK:
                case ResourceQueryOp::RAY_TRACING_TRACE_CLOSEST:
                case ResourceQueryOp::RAY_TRACING_TRACE_ANY:
                case ResourceQueryOp::RAY_TRACING_QUERY_ALL:
                case ResourceQueryOp::RAY_TRACING_QUERY_ANY:
                case ResourceQueryOp::RAY_TRACING_INSTANCE_MOTION_MATRIX:
                case ResourceQueryOp::RAY_TRACING_INSTANCE_MOTION_SRT:
                case ResourceQueryOp::RAY_TRACING_TRACE_CLOSEST_MOTION_BLUR:
                case ResourceQueryOp::RAY_TRACING_TRACE_ANY_MOTION_BLUR:
                case ResourceQueryOp::RAY_TRACING_QUERY_ALL_MOTION_BLUR:
                case ResourceQueryOp::RAY_TRACING_QUERY_ANY_MOTION_BLUR:
                    return {MemoryScope::GLOBAL, MemoryEffects::READ, false};
            }
            // Fail closed if a new query operation is added without a memory
            // contract: treating it as a read prevents unsound value numbering
            // and code motion while still allowing dead-result elimination.
            return {MemoryScope::GLOBAL, MemoryEffects::READ, false};
        }
        case DerivedInstructionTag::RAY_QUERY_OBJECT_READ:
            return {MemoryScope::LOCAL, MemoryEffects::READ, false};
        case DerivedInstructionTag::ALLOCA: {
            auto alloca = static_cast<AllocaInst *>(inst);
            return {alloca->is_shared() ? MemoryScope::SHARED : MemoryScope::LOCAL,
                    MemoryEffects::NONE, false};
        }
        case DerivedInstructionTag::LOAD: {
            auto load = static_cast<LoadInst *>(inst);
            return {pointer_scope(load->variable()), MemoryEffects::READ, false};
        }
        case DerivedInstructionTag::STORE: {
            auto store = static_cast<StoreInst *>(inst);
            return {pointer_scope(store->variable()), MemoryEffects::WRITE, false};
        }
        case DerivedInstructionTag::RESOURCE_READ: {
            auto read = static_cast<ResourceReadInst *>(inst);
            auto is_volatile = read->op() == ResourceReadOp::BUFFER_VOLATILE_READ ||
                               read->op() == ResourceReadOp::BYTE_BUFFER_VOLATILE_READ;
            return {MemoryScope::GLOBAL, MemoryEffects::READ, is_volatile};
        }
        case DerivedInstructionTag::RESOURCE_WRITE: {
            auto write = static_cast<ResourceWriteInst *>(inst);
            auto is_volatile = write->op() == ResourceWriteOp::BUFFER_VOLATILE_WRITE ||
                               write->op() == ResourceWriteOp::BYTE_BUFFER_VOLATILE_WRITE;
            return {MemoryScope::GLOBAL, MemoryEffects::WRITE, is_volatile};
        }
        case DerivedInstructionTag::ATOMIC: {
            auto atomic = static_cast<AtomicInst *>(inst);
            auto base = trace_pointer_base_value(atomic->base());
            auto scope =
                base != nullptr && base->isa<AllocaInst>() ?
                    pointer_scope(base) :
                base != nullptr && base->type() != nullptr &&
                        base->type()->is_resource() ?
                    MemoryScope::GLOBAL :
                    MemoryScope::NONE;
            return {scope, MemoryEffects::READ_WRITE, false};
        }
        case DerivedInstructionTag::RAY_QUERY_OBJECT_WRITE:
            return {MemoryScope::LOCAL, MemoryEffects::WRITE, false};
        case DerivedInstructionTag::THREAD_GROUP:
            return {MemoryScope::SHARED, MemoryEffects::READ_WRITE, true};
        case DerivedInstructionTag::CALL:
            return {MemoryScope::GLOBAL, MemoryEffects::READ_WRITE, false};
        case DerivedInstructionTag::PRINT:
        case DerivedInstructionTag::DEBUG_BREAK:
        case DerivedInstructionTag::ASSERT:
        case DerivedInstructionTag::ASSUME:
            return {MemoryScope::NONE, MemoryEffects::NONE, true};
        case DerivedInstructionTag::AUTODIFF_SCOPE:
        case DerivedInstructionTag::AUTODIFF_INTRINSIC:
            return {MemoryScope::GLOBAL, MemoryEffects::READ_WRITE, true};
        default:
            return {MemoryScope::NONE, MemoryEffects::NONE, true};
    }
}

bool is_arithmetic_op_safe_to_speculate(ArithmeticOp op) noexcept {
    switch (op) {
        // These operations can be undefined for verifier-valid dynamic
        // operands, or may address a dynamic aggregate element outside its
        // valid range. They cannot be made executable on a previously untaken
        // control-flow path without an operand proof.
        case ArithmeticOp::BINARY_DIV:
        case ArithmeticOp::BINARY_MOD:
        case ArithmeticOp::BINARY_SHIFT_LEFT:
        case ArithmeticOp::BINARY_SHIFT_RIGHT:
        case ArithmeticOp::SHUFFLE:
        case ArithmeticOp::INSERT:
        case ArithmeticOp::EXTRACT:
            return false;

        // Exhaustive set of total value computations. Do not replace this
        // list with a permissive default: an enum extension must be audited
        // before LICM, if-conversion, or a future speculative pass can use it.
        case ArithmeticOp::UNARY_MINUS:
        case ArithmeticOp::UNARY_BIT_NOT:
        case ArithmeticOp::BINARY_ADD:
        case ArithmeticOp::BINARY_SUB:
        case ArithmeticOp::BINARY_MUL:
        case ArithmeticOp::BINARY_BIT_AND:
        case ArithmeticOp::BINARY_BIT_OR:
        case ArithmeticOp::BINARY_BIT_XOR:
        case ArithmeticOp::BINARY_ROTATE_LEFT:
        case ArithmeticOp::BINARY_ROTATE_RIGHT:
        case ArithmeticOp::BINARY_LESS:
        case ArithmeticOp::BINARY_GREATER:
        case ArithmeticOp::BINARY_LESS_EQUAL:
        case ArithmeticOp::BINARY_GREATER_EQUAL:
        case ArithmeticOp::BINARY_EQUAL:
        case ArithmeticOp::BINARY_NOT_EQUAL:
        case ArithmeticOp::ALL:
        case ArithmeticOp::ANY:
        case ArithmeticOp::SELECT:
        case ArithmeticOp::CLAMP:
        case ArithmeticOp::SATURATE:
        case ArithmeticOp::LERP:
        case ArithmeticOp::SMOOTHSTEP:
        case ArithmeticOp::STEP:
        case ArithmeticOp::ABS:
        case ArithmeticOp::MIN:
        case ArithmeticOp::MAX:
        case ArithmeticOp::CLZ:
        case ArithmeticOp::CTZ:
        case ArithmeticOp::POPCOUNT:
        case ArithmeticOp::REVERSE:
        case ArithmeticOp::ISINF:
        case ArithmeticOp::ISNAN:
        case ArithmeticOp::ACOS:
        case ArithmeticOp::ACOSH:
        case ArithmeticOp::ASIN:
        case ArithmeticOp::ASINH:
        case ArithmeticOp::ATAN:
        case ArithmeticOp::ATAN2:
        case ArithmeticOp::ATANH:
        case ArithmeticOp::COS:
        case ArithmeticOp::COSH:
        case ArithmeticOp::SIN:
        case ArithmeticOp::SINH:
        case ArithmeticOp::TAN:
        case ArithmeticOp::TANH:
        case ArithmeticOp::EXP:
        case ArithmeticOp::EXP2:
        case ArithmeticOp::EXP10:
        case ArithmeticOp::LOG:
        case ArithmeticOp::LOG2:
        case ArithmeticOp::LOG10:
        case ArithmeticOp::POW:
        case ArithmeticOp::POW_INT:
        case ArithmeticOp::SQRT:
        case ArithmeticOp::RSQRT:
        case ArithmeticOp::CEIL:
        case ArithmeticOp::FLOOR:
        case ArithmeticOp::FRACT:
        case ArithmeticOp::TRUNC:
        case ArithmeticOp::ROUND:
        case ArithmeticOp::RINT:
        case ArithmeticOp::FMA:
        case ArithmeticOp::COPYSIGN:
        case ArithmeticOp::CROSS:
        case ArithmeticOp::DOT:
        case ArithmeticOp::LENGTH:
        case ArithmeticOp::LENGTH_SQUARED:
        case ArithmeticOp::NORMALIZE:
        case ArithmeticOp::FACEFORWARD:
        case ArithmeticOp::REFLECT:
        case ArithmeticOp::REDUCE_SUM:
        case ArithmeticOp::REDUCE_PRODUCT:
        case ArithmeticOp::REDUCE_MIN:
        case ArithmeticOp::REDUCE_MAX:
        case ArithmeticOp::OUTER_PRODUCT:
        case ArithmeticOp::MATRIX_COMP_NEG:
        case ArithmeticOp::MATRIX_COMP_ADD:
        case ArithmeticOp::MATRIX_COMP_SUB:
        case ArithmeticOp::MATRIX_COMP_MUL:
        case ArithmeticOp::MATRIX_COMP_DIV:
        case ArithmeticOp::MATRIX_LINALG_MUL:
        case ArithmeticOp::MATRIX_DETERMINANT:
        case ArithmeticOp::MATRIX_TRANSPOSE:
        case ArithmeticOp::MATRIX_INVERSE:
        case ArithmeticOp::AGGREGATE:
            return true;
    }
    return false;
}

bool contains_structured_control_flow(FunctionDefinition *function) noexcept {
    if (function == nullptr) { return false; }
    // Inspect every block owned by the function, not just blocks reachable
    // from the body. A temporarily unreachable structured region is still a
    // hard boundary for CFG-only transforms and may become reachable again.
    for (auto *block : function->basic_blocks()) {
        for (auto *inst : block->instructions()) {
            switch (inst->derived_instruction_tag()) {
                case DerivedInstructionTag::IF:
                case DerivedInstructionTag::LOOP:
                case DerivedInstructionTag::SIMPLE_LOOP:
                case DerivedInstructionTag::BREAK:
                case DerivedInstructionTag::CONTINUE:
                case DerivedInstructionTag::RAY_QUERY_LOOP:
                case DerivedInstructionTag::RAY_QUERY_DISPATCH:
                case DerivedInstructionTag::AUTODIFF_SCOPE:
                case DerivedInstructionTag::OUTLINE:
                case DerivedInstructionTag::SWITCH:
                    return true;
                default: break;
            }
        }
    }
    return false;
}

bool remove_redundant_phi_instruction(PhiInst *phi) noexcept {
    if (phi == nullptr) { return false; }
    if (phi->use_list().empty()) {
        phi->remove_self();
        return true;
    }
    // A live identity replacement has no unique instruction on which to keep
    // Phi-local metadata. Leave it for a transform that can construct an
    // explicit replacement owner (e.g. reg2mem's generated load).
    if (!phi->metadata_list().empty()) { return false; }
    // A zero-predecessor entry block legitimately has a zero-incoming Phi.
    // Its value is undefined, not absent: unlinking a live Phi without first
    // replacing its uses leaves dangling/null operands in release builds.
    if (phi->incoming_count() == 0u) {
        auto *function = phi->parent_function();
        auto *module = function != nullptr ?
                           function->parent_module() :
                           nullptr;
        if (module == nullptr) { return false; }
        phi->replace_all_uses_with(
            module->create_undefined(phi->type()));
        phi->remove_self();
        return true;
    }
    static constexpr auto is_invariant = [](Value *v) noexcept {
        if (v == nullptr) { return true; }
        switch (v->derived_value_tag()) {
            case DerivedValueTag::UNDEFINED: [[fallthrough]];
            case DerivedValueTag::CONSTANT: [[fallthrough]];
            case DerivedValueTag::ARGUMENT: [[fallthrough]];
            case DerivedValueTag::SPECIAL_REGISTER: return true;
            default: break;
        }
        return false;
    };
    auto all_same_except_undef = true;
    auto undef_incoming = static_cast<Value *>(nullptr);
    auto same_incoming = static_cast<Value *>(nullptr);
    // check if all incoming values are the same
    for (auto value_use : phi->incoming_value_uses()) {
        LUISA_DEBUG_ASSERT(value_use->value() != nullptr, "Invalid incoming value.");
        if (auto value = value_use->value(); value->isa<Undefined>()) {
            undef_incoming = value;
        } else {
            // if we haven't seen any incoming value yet, set it as the same incoming
            if (same_incoming == nullptr) { same_incoming = value; }
            // otherwise, check if the current incoming value is the same as the previous one
            if (same_incoming != value) {
                all_same_except_undef = false;
                break;
            }
        }
    }
    if (all_same_except_undef && is_invariant(same_incoming)) {
        if (same_incoming != nullptr) {
            phi->replace_all_uses_with(same_incoming);
        } else if (undef_incoming != nullptr) {
            phi->replace_all_uses_with(undef_incoming);
        } else {
            LUISA_DEBUG_ASSERT(phi->use_list().empty(), "Invalid phi node.");
        }
        phi->remove_self();
        return true;
    }
    return false;
}

bool simplify_phi_instruction(PhiInst *phi, const DomTree *dom_tree) noexcept {
    if (phi == nullptr) { return false; }
    if (phi->use_list().empty()) {
        phi->remove_self();
        return true;
    }
    if (!phi->metadata_list().empty()) { return false; }
    // Find the unique non-self, non-undef incoming value (if any).
    Value *unique = nullptr;
    for (auto value_use : phi->incoming_value_uses()) {
        auto v = value_use->value();
        if (v == nullptr || v == phi) continue;
        if (v->isa<Undefined>()) continue;
        if (unique == nullptr) {
            unique = v;
        } else if (unique != v) {
            return false;
        }
    }
    if (unique == nullptr) {
        if (auto m = phi->parent_function() ? phi->parent_function()->parent_module() : nullptr) {
            auto undef = m->create_undefined(phi->type());
            phi->replace_all_uses_with(undef);
        }
        phi->remove_self();
        return true;
    }
    // The unique value must dominate the phi's block (and thus all its uses).
    // Loop-carried phis are a classic counter-example: a phi in the loop header
    // may have an undef initial value and a single back-edge value V, but V is
    // defined later in the loop body and does not dominate the header. Replacing
    // the phi with V would create a self-referential cycle across iterations.
    auto dominates_phi = [&]() noexcept -> bool {
        // Constants, function arguments, and special registers dominate everything.
        if (unique->isa<Constant>() || unique->isa<Argument>() || unique->isa<SpecialRegister>()) { return true; }
        auto unique_inst = unique->isa<Instruction>() ? static_cast<Instruction *>(unique) : nullptr;
        if (unique_inst == nullptr) { return false; }
        auto unique_block = unique_inst->parent_block();
        auto phi_block = phi->parent_block();
        if (unique_block == nullptr || phi_block == nullptr) { return false; }
        if (unique_block == phi_block) { return false; }// phis are at the block start
        if (dom_tree != nullptr) { return dom_tree->dominates(unique_block, phi_block); }
        // Without a dominator tree, be conservative and only allow values that
        // are guaranteed to dominate everything (constants/arguments handled above).
        return false;
    };
    if (!dominates_phi()) { return false; }
    phi->replace_all_uses_with(unique);
    phi->remove_self();
    return true;
}

void lower_phi_node_to_local_variable(PhiInst *phi) noexcept {
    if (phi == nullptr) { return; }
    if (!simplify_phi_instruction(phi)) {
        auto f = phi->parent_function();
        LUISA_DEBUG_ASSERT(f != nullptr && f->definition() != nullptr, "Invalid function.");
        XIRBuilder b;
        // create alloca at the beginning of the function
        b.set_insertion_point(f->definition()->body_block()->instructions().head_sentinel());
        auto phi_alloca = b.alloca_local(phi->type());
        static_cast<void>(phi_alloca->create_metadata<Reg2MemSpillMD>());
        phi_alloca->add_comment("alloca to lower phi node");
        static std::atomic_uint64_t phi_counter{0u};
        auto phi_id = phi_counter.fetch_add(1u, std::memory_order_relaxed) + 1u;
        phi_alloca->set_name(luisa::format("_phi_{}", phi_id));
        if (auto m = f->parent_module()) {
            auto undef = m->create_undefined(phi->type());
            b.store(phi_alloca, undef);
        }
        // store incoming values at the end of their respective blocks
        for (size_t i = 0; i < phi->incoming_count(); i++) {
            if (auto incoming = phi->incoming(i); incoming.value != nullptr && !incoming.value->isa<Undefined>()) {
                LUISA_DEBUG_ASSERT(incoming.block != nullptr, "Invalid incoming block.");
                b.set_insertion_point(incoming.block->terminator()->prev());
                b.store(phi_alloca, incoming.value);
            }
        }
        // replace phi uses with local load instructions
        b.set_insertion_point(phi);
        auto phi_load = b.load(phi->type(), phi_alloca);
        for (auto *metadata : phi->metadata_list()) {
            phi_load->metadata_list().push_front(metadata->clone());
        }
        phi->replace_all_uses_with(phi_load);
        phi->remove_self();
    }
}

void hoist_alloca_instructions_to_entry_block(FunctionDefinition *f) noexcept {
    if (f == nullptr || f->body_block() == nullptr) { return; }
    luisa::vector<AllocaInst *> collected;
    f->traverse_instructions([&](Instruction *inst) noexcept {
        if (inst->isa<AllocaInst>()) {
            collected.emplace_back(static_cast<AllocaInst *>(inst));
        }
    });
    if (!collected.empty()) {
        XIRBuilder b;
        b.set_insertion_point(f->body_block()->instructions().head_sentinel());
        for (auto inst : collected) {
            b.append(inst->remove_self());
        }
    }
}

}// namespace luisa::compute::xir
