#include <luisa/core/logging.h>
#include <luisa/dsl/rtx/ray_query.h>
#include <luisa/runtime/rtx/hit.h>
#include <luisa/xir/constant.h>
#include <luisa/xir/instructions/alloca.h>
#include <luisa/xir/instructions/arithmetic.h>
#include <luisa/xir/instructions/load.h>
#include <luisa/xir/instructions/store.h>

#include "../optix_api.h"
#include "cuda_codegen_llvm_impl.h"
#include "cuda_ray_query_callback_sharing.h"

namespace luisa::compute::cuda {

namespace {
constexpr auto context_capture_offset = 2u;
constexpr auto kRayQueryPayloadWordCount = 32u;
constexpr auto kRayQueryPayloadHardwareCaptureOffset = 1u;
constexpr auto kRayQueryPayloadGeneralCaptureOffset = 3u;
// Private protocol between the generated custom IS and AH entry points.
// OptiX reserves hit kinds above 127 for built-in surface intersections.
constexpr auto ray_query_procedural_hit_kind = 1u;
constexpr auto ray_query_procedural_terminated_hit_kind = 2u;

[[nodiscard]] uint32_t ray_query_payload_scalar_bits(llvm::Type *type) noexcept {
    if (type->isPointerTy()) { return 64u; }
    if (type->isIntegerTy()) {
        auto bits = type->getIntegerBitWidth();
        return bits <= 64u ? bits : 0u;
    }
    if (type->isHalfTy() || type->isBFloatTy()) { return 16u; }
    if (type->isFloatTy()) { return 32u; }
    if (type->isDoubleTy()) { return 64u; }
    return 0u;
}

// Counts are saturated above the direct payload budget. Unsupported types and
// large aggregates use the generic context path without partially packing them.
[[nodiscard]] uint32_t ray_query_payload_word_count(llvm::Type *type, const llvm::DataLayout &layout,
                                                        uint32_t budget) noexcept {
    auto overflow = budget + 1u;
    if (auto structure = llvm::dyn_cast<llvm::StructType>(type)) {
        if (structure->isOpaque()) { return overflow; }
        auto count = 0u;
        for (auto field : structure->elements()) {
            auto words = ray_query_payload_word_count(field, layout, budget);
            if (words > budget - count) { return overflow; }
            count += words;
        }
        return count;
    }
    auto repeated_count = [&layout, budget, overflow](llvm::Type *element, uint64_t count) noexcept {
        if (count == 0u) { return 0u; }
        auto words = ray_query_payload_word_count(element, layout, budget);
        if (words == 0u) { return 0u; }
        return count > budget / words ? overflow : static_cast<uint32_t>(count) * words;
    };
    if (auto array = llvm::dyn_cast<llvm::ArrayType>(type)) {
        return repeated_count(array->getElementType(), array->getNumElements());
    }
    if (auto vector = llvm::dyn_cast<llvm::FixedVectorType>(type)) {
        return repeated_count(vector->getElementType(), vector->getNumElements());
    }
    if (type->isPointerTy() && layout.getPointerSizeInBits(type->getPointerAddressSpace()) != 64u) { return overflow; }
    auto bits = ray_query_payload_scalar_bits(type);
    return bits == 0u ? overflow : (bits + 31u) / 32u;
}

[[nodiscard]] bool ray_query_uses_direct_payload(llvm::StructType *context_type, const llvm::DataLayout &layout,
                                                        uint32_t budget) noexcept {
    auto count = 0u;
    for (auto i = context_capture_offset; i < context_type->getNumElements(); i++) {
        auto words = ray_query_payload_word_count(context_type->getElementType(i), layout, budget);
        if (words > budget - count) { return false; }
        count += words;
    }
    return true;
}

void pack_ray_query_payload(llvm::IRBuilder<> &b, llvm::Value *value,
                            llvm::SmallVectorImpl<llvm::Value *> &words) noexcept {
    auto type = value->getType();
    if (type->isEmptyTy()) { return; }
    if (auto structure = llvm::dyn_cast<llvm::StructType>(type)) {
        for (auto i = 0u; i < structure->getNumElements(); i++) {
            pack_ray_query_payload(b, b.CreateExtractValue(value, i), words);
        }
        return;
    }
    if (auto array = llvm::dyn_cast<llvm::ArrayType>(type)) {
        for (auto i = 0u; i < array->getNumElements(); i++) {
            pack_ray_query_payload(b, b.CreateExtractValue(value, i), words);
        }
        return;
    }
    if (auto vector = llvm::dyn_cast<llvm::FixedVectorType>(type)) {
        for (auto i = 0u; i < vector->getNumElements(); i++) {
            pack_ray_query_payload(b, b.CreateExtractElement(value, i), words);
        }
        return;
    }
    auto bits = ray_query_payload_scalar_bits(type);
    LUISA_ASSERT(bits != 0u, "Unsupported direct ray-query payload type.");
    if (type->isPointerTy()) {
        value = b.CreatePtrToInt(value, b.getInt64Ty());
    } else if (type->isFloatingPointTy()) {
        value = b.CreateBitCast(value, b.getIntNTy(bits));
    }
    words.emplace_back(b.CreateZExtOrTrunc(value, b.getInt32Ty()));
    if (bits > 32u) { words.emplace_back(b.CreateZExtOrTrunc(b.CreateLShr(value, 32u), b.getInt32Ty())); }
}

[[nodiscard]] llvm::Value *unpack_ray_query_payload(llvm::IRBuilder<> &b, llvm::Type *type,
                                                    llvm::InlineAsm *get_payload, uint32_t &word_index) noexcept {
    if (type->isEmptyTy()) { return llvm::Constant::getNullValue(type); }
    if (auto structure = llvm::dyn_cast<llvm::StructType>(type)) {
        auto value = static_cast<llvm::Value *>(llvm::Constant::getNullValue(type));
        for (auto i = 0u; i < structure->getNumElements(); i++) {
            auto field = unpack_ray_query_payload(b, structure->getElementType(i), get_payload, word_index);
            value = b.CreateInsertValue(value, field, i);
        }
        return value;
    }
    if (auto array = llvm::dyn_cast<llvm::ArrayType>(type)) {
        auto value = static_cast<llvm::Value *>(llvm::Constant::getNullValue(type));
        for (auto i = 0u; i < array->getNumElements(); i++) {
            auto element = unpack_ray_query_payload(b, array->getElementType(), get_payload, word_index);
            value = b.CreateInsertValue(value, element, i);
        }
        return value;
    }
    if (auto vector = llvm::dyn_cast<llvm::FixedVectorType>(type)) {
        auto value = static_cast<llvm::Value *>(llvm::Constant::getNullValue(type));
        for (auto i = 0u; i < vector->getNumElements(); i++) {
            auto element = unpack_ray_query_payload(b, vector->getElementType(), get_payload, word_index);
            value = b.CreateInsertElement(value, element, i);
        }
        return value;
    }
    auto bits = ray_query_payload_scalar_bits(type);
    LUISA_ASSERT(bits != 0u, "Unsupported direct ray-query payload type.");
    auto read_word = [&]() noexcept -> llvm::Value * {
        LUISA_ASSERT(word_index < kRayQueryPayloadWordCount, "Ray-query payload read exceeds the register budget.");
        return b.CreateCall(get_payload, {b.getInt32(word_index++)});
    };
    auto value = read_word();
    if (bits > 32u) {
        auto lo = b.CreateZExt(value, b.getInt64Ty());
        auto hi = b.CreateZExt(read_word(), b.getInt64Ty());
        value = b.CreateOr(lo, b.CreateShl(hi, 32u));
    }
    // References retain their original addresses, including pointers nested in
    // resource structs. Only actual captured values are reconstructed here.
    if (type->isPointerTy()) { return b.CreateIntToPtr(value, type); }
    value = b.CreateZExtOrTrunc(value, b.getIntNTy(bits));
    return type->isFloatingPointTy() ? b.CreateBitCast(value, type) : value;
}

[[nodiscard]] llvm::Value *unpack_ray_query_payload_from_words(llvm::IRBuilder<> &b, llvm::Type *type,
                                                               llvm::ArrayRef<llvm::Value *> words,
                                                               uint32_t &pos) noexcept {
    if (type->isEmptyTy()) { return llvm::Constant::getNullValue(type); }
    if (auto structure = llvm::dyn_cast<llvm::StructType>(type)) {
        auto value = static_cast<llvm::Value *>(llvm::Constant::getNullValue(type));
        for (auto i = 0u; i < structure->getNumElements(); i++) {
            auto field = unpack_ray_query_payload_from_words(b, structure->getElementType(i), words, pos);
            value = b.CreateInsertValue(value, field, i);
        }
        return value;
    }
    if (auto array = llvm::dyn_cast<llvm::ArrayType>(type)) {
        auto value = static_cast<llvm::Value *>(llvm::Constant::getNullValue(type));
        for (auto i = 0u; i < array->getNumElements(); i++) {
            auto element = unpack_ray_query_payload_from_words(b, array->getElementType(), words, pos);
            value = b.CreateInsertValue(value, element, i);
        }
        return value;
    }
    if (auto vector = llvm::dyn_cast<llvm::FixedVectorType>(type)) {
        auto value = static_cast<llvm::Value *>(llvm::Constant::getNullValue(type));
        for (auto i = 0u; i < vector->getNumElements(); i++) {
            auto element = unpack_ray_query_payload_from_words(b, vector->getElementType(), words, pos);
            value = b.CreateInsertElement(value, element, i);
        }
        return value;
    }
    auto bits = ray_query_payload_scalar_bits(type);
    LUISA_ASSERT(bits != 0u && pos < words.size(), "Invalid direct ray-query RW payload unpack.");
    auto value = words[pos++];
    if (bits > 32u) {
        LUISA_ASSERT(pos < words.size(), "Invalid 64-bit direct ray-query RW payload unpack.");
        auto lo = b.CreateZExt(value, b.getInt64Ty());
        auto hi = b.CreateZExt(words[pos++], b.getInt64Ty());
        value = b.CreateOr(lo, b.CreateShl(hi, 32u));
    }
    if (type->isPointerTy()) { return b.CreateIntToPtr(value, type); }
    value = b.CreateZExtOrTrunc(value, b.getIntNTy(bits));
    return type->isFloatingPointTy() ? b.CreateBitCast(value, type) : value;
}

void validate_ray_query_handler(const xir::Function *function,
                                llvm::DenseSet<const xir::Function *> &visited) noexcept {
    if (function == nullptr || !visited.insert(function).second) { return; }
    if (auto definition = function->definition()) {
        for (auto block : definition->basic_blocks()) {
            for (auto inst : block->instructions()) {
                switch (inst->derived_instruction_tag()) {
                    case xir::DerivedInstructionTag::RAY_QUERY_PIPELINE:
                    case xir::DerivedInstructionTag::RAY_QUERY_LOOP:
                    case xir::DerivedInstructionTag::RAY_QUERY_DISPATCH:
                        LUISA_ERROR_WITH_LOCATION("CUDA LLVM does not support nested ray queries inside a candidate handler.");
                    case xir::DerivedInstructionTag::CALL:
                        validate_ray_query_handler(static_cast<const xir::CallInst *>(inst)->callee(), visited);
                        break;
                    case xir::DerivedInstructionTag::RESOURCE_QUERY: {
                        switch (static_cast<const xir::ResourceQueryInst *>(inst)->op()) {
                            case xir::ResourceQueryOp::RAY_TRACING_TRACE_CLOSEST:
                            case xir::ResourceQueryOp::RAY_TRACING_TRACE_ANY:
                            case xir::ResourceQueryOp::RAY_TRACING_TRACE_CLOSEST_MOTION_BLUR:
                            case xir::ResourceQueryOp::RAY_TRACING_TRACE_ANY_MOTION_BLUR:
                            case xir::ResourceQueryOp::RAY_TRACING_QUERY_ALL:
                            case xir::ResourceQueryOp::RAY_TRACING_QUERY_ANY:
                            case xir::ResourceQueryOp::RAY_TRACING_QUERY_ALL_MOTION_BLUR:
                            case xir::ResourceQueryOp::RAY_TRACING_QUERY_ANY_MOTION_BLUR:
                                LUISA_ERROR_WITH_LOCATION("CUDA LLVM candidate handlers cannot launch nested traversal with the depth-one OptiX pipeline.");
                            default: break;
                        }
                        break;
                    }
                    default: break;
                }
            }
        }
    }
}

[[nodiscard]] bool ray_query_handler_is_empty_impl(
    const xir::Function *function,
    llvm::DenseSet<const xir::Function *> &visiting,
    llvm::DenseSet<const xir::Function *> &empty_cache) noexcept {
    if (function == nullptr) { return true; }
    if (empty_cache.contains(function)) { return true; }
    if (!visiting.insert(function).second) { return false; }
    auto definition = function->definition();
    if (definition == nullptr) {
        visiting.erase(function);
        return false;
    }
    llvm::DenseSet<const xir::BasicBlock *> dfs_visiting;
    llvm::DenseSet<const xir::BasicBlock *> dfs_visited;
    auto is_acyclic = [&](auto &&self, const xir::BasicBlock *bb) noexcept -> bool {
        if (bb == nullptr || dfs_visited.contains(bb)) { return true; }
        if (!dfs_visiting.insert(bb).second) { return false; }
        auto ok = true;
        bb->traverse_successors(false, [&](const xir::BasicBlock *succ) noexcept {
            if (ok && !self(self, succ)) { ok = false; }
        });
        dfs_visiting.erase(bb);
        dfs_visited.insert(bb);
        return ok;
    };
    for (auto block : definition->basic_blocks()) {
        if (!is_acyclic(is_acyclic, block)) {
            visiting.erase(function);
            return false;
        }
        for (auto inst : block->instructions()) {
            switch (inst->derived_instruction_tag()) {
                case xir::DerivedInstructionTag::RETURN:
                    if (static_cast<const xir::ReturnInst *>(inst)->return_value() != nullptr) {
                        visiting.erase(function);
                        return false;
                    }
                    break;
                case xir::DerivedInstructionTag::BRANCH:
                case xir::DerivedInstructionTag::CONDITIONAL_BRANCH:
                case xir::DerivedInstructionTag::IF:
                case xir::DerivedInstructionTag::SWITCH:
                case xir::DerivedInstructionTag::INDEXED_BRANCH:
                case xir::DerivedInstructionTag::UNREACHABLE:
                case xir::DerivedInstructionTag::PHI:
                case xir::DerivedInstructionTag::ALLOCA:
                case xir::DerivedInstructionTag::LOAD:
                case xir::DerivedInstructionTag::GEP:
                case xir::DerivedInstructionTag::ARITHMETIC:
                case xir::DerivedInstructionTag::CAST:
                case xir::DerivedInstructionTag::ASSUME:
                case xir::DerivedInstructionTag::RAY_QUERY_OBJECT_READ:
                    break;
                case xir::DerivedInstructionTag::STORE: {
                    auto store = static_cast<const xir::StoreInst *>(inst);
                    auto var = store->variable();
                    auto is_local_alloca = var != nullptr && var->isa<xir::AllocaInst>() &&
                                           static_cast<const xir::AllocaInst *>(var)->parent_function() == definition;
                    auto is_self_copy = var != nullptr && store->value() != nullptr &&
                                        store->value()->isa<xir::LoadInst>() &&
                                        static_cast<const xir::LoadInst *>(store->value())->variable() == var;
                    if (!is_local_alloca && !is_self_copy) {
                        visiting.erase(function);
                        return false;
                    }
                    break;
                }
                case xir::DerivedInstructionTag::CALL: {
                    auto callee = static_cast<const xir::CallInst *>(inst)->callee();
                    if (!ray_query_handler_is_empty_impl(callee, visiting, empty_cache)) {
                        visiting.erase(function);
                        return false;
                    }
                    break;
                }
                default:
                    visiting.erase(function);
                    return false;
            }
        }
    }
    visiting.erase(function);
    empty_cache.insert(function);
    return true;
}

[[nodiscard]] bool ray_query_handler_is_empty(const xir::Function *function) noexcept {
    llvm::DenseSet<const xir::Function *> visiting;
    llvm::DenseSet<const xir::Function *> empty_cache;
    return ray_query_handler_is_empty_impl(function, visiting, empty_cache);
}

[[nodiscard]] bool alloca_can_stage_inline(const xir::AllocaInst *alloca_inst,
                                           const xir::RayQueryPipelineInst *pipeline) noexcept {
    if (alloca_inst == nullptr || pipeline == nullptr ||
        alloca_inst == pipeline->query_object() ||
        alloca_inst->type()->is_resource()) {
        return false;
    }
    auto capture_matches = [&](const xir::RayQueryPipelineInst *p) noexcept {
        auto count = 0u;
        for (auto capture : p->captured_argument_uses()) {
            if (capture->value() == alloca_inst) { count++; }
        }
        return count;
    };
    if (capture_matches(pipeline) > 1u) { return false; }
    for (auto use : alloca_inst->use_list()) {
        auto user = use->user();
        if (user == nullptr || user->derived_value_tag() != xir::DerivedValueTag::INSTRUCTION) { return false; }
        auto inst = static_cast<const xir::Instruction *>(user);
        switch (inst->derived_instruction_tag()) {
            case xir::DerivedInstructionTag::LOAD:
                if (inst->operand_count() == 0u || inst->operand(0u) != alloca_inst) { return false; }
                break;
            case xir::DerivedInstructionTag::STORE: {
                auto store = static_cast<const xir::StoreInst *>(inst);
                if (store->variable() != alloca_inst || store->value() == alloca_inst) { return false; }
                break;
            }
            case xir::DerivedInstructionTag::RAY_QUERY_PIPELINE: {
                auto other = static_cast<const xir::RayQueryPipelineInst *>(inst);
                if (other->query_object() == alloca_inst || capture_matches(other) > 1u) { return false; }
                break;
            }
            default: return false;
        }
    }
    return true;
}

[[nodiscard]] const xir::Argument *find_forwarded_query_param(
    const xir::CallInst *call, const xir::Value *query) noexcept {
    if (query == nullptr || call == nullptr) { return nullptr; }
    auto callee = call->callee();
    if (callee == nullptr || callee->definition() == nullptr) { return nullptr; }
    const xir::Argument *matched_param = nullptr;
    auto param_it = callee->arguments().begin();
    auto param_end = callee->arguments().end();
    for (auto i = size_t{0u}; i < call->argument_count(); i++) {
        if (param_it == param_end) { return nullptr; }
        const xir::Argument *param = *param_it;
        ++param_it;
        if (call->argument(i) == query) {
            if (matched_param != nullptr || param->type() != query->type()) { return nullptr; }
            matched_param = param;
        }
    }
    return matched_param;
}

[[nodiscard]] bool world_ray_read_uses_only_exact_ah_fields(const xir::Instruction *read_inst) noexcept {
    for (auto use : read_inst->use_list()) {
        auto user = use->user();
        if (user == nullptr || user->derived_value_tag() != xir::DerivedValueTag::INSTRUCTION) { return false; }
        auto inst = static_cast<const xir::Instruction *>(user);
        if (inst->derived_instruction_tag() != xir::DerivedInstructionTag::ARITHMETIC) { return false; }
        auto arith = static_cast<const xir::ArithmeticInst *>(inst);
        if (arith->op() != xir::ArithmeticOp::EXTRACT || arith->operand_count() < 2u || arith->operand(0u) != read_inst) {
            return false;
        }
        uint64_t member = 0u;
        if (!xir::try_decode_constant_nonnegative_integer(arith->operand(1u), member) ||
            member == CUDACodegenLLVMImpl::llvm_ray_type_t_max_index) {
            return false;
        }
    }
    return true;
}

[[nodiscard]] bool ray_query_handler_reads_world_ray(const xir::Function *function, const xir::Value *query,
                                                     llvm::DenseSet<const xir::Function *> &visited) noexcept {
    if (function == nullptr) { return false; }
    if (!visited.insert(function).second) { return false; }
    auto definition = function->definition();
    if (definition == nullptr) { return false; }
    for (auto block : definition->basic_blocks()) {
        for (auto inst : block->instructions()) {
            if (inst->derived_instruction_tag() == xir::DerivedInstructionTag::RAY_QUERY_OBJECT_READ) {
                auto op = static_cast<const xir::RayQueryObjectReadInst *>(inst)->op();
                if (op == xir::RayQueryObjectReadOp::RAY_QUERY_OBJECT_WORLD_SPACE_RAY ||
                    op == xir::RayQueryObjectReadOp::RAY_QUERY_OBJECT_CANDIDATE_OBJECT_SPACE_RAY) {
                    return true;
                }
            } else if (inst->derived_instruction_tag() == xir::DerivedInstructionTag::CALL) {
                auto call = static_cast<const xir::CallInst *>(inst);
                auto forwarded_query = find_forwarded_query_param(call, query);
                if (ray_query_handler_reads_world_ray(call->callee(), forwarded_query, visited)) {
                    return true;
                }
            }
        }
    }
    visited.erase(function);
    return false;
}

[[nodiscard]] bool ray_query_handler_has_hardware_result(const xir::Function *function, const xir::Value *query,
                                                          bool procedural, llvm::DenseSet<const xir::Function *> &visited) noexcept {
    if (function == nullptr) { return true; }
    if (!visited.insert(function).second) { return false; }
    auto definition = function->definition();
    if (definition == nullptr) { return false; }
    if (query != nullptr) {
        // Only query operations (or forwarding query by-ref to an analyzed callee)
        // may observe this invocation's private state.
        // Alias-producing operations keep the original shared query path.
        for (auto use : query->use_list()) {
            auto user = use->user();
            if (user->derived_value_tag() != xir::DerivedValueTag::INSTRUCTION) { return false; }
            auto inst = static_cast<const xir::Instruction *>(user);
            if (inst->derived_instruction_tag() == xir::DerivedInstructionTag::CALL) {
                auto call = static_cast<const xir::CallInst *>(inst);
                if (call->callee() == query || find_forwarded_query_param(call, query) == nullptr) { return false; }
                continue;
            }
            if (inst->operand_count() == 0u || inst->operand(0) != query) { return false; }
            if (inst->derived_instruction_tag() == xir::DerivedInstructionTag::RAY_QUERY_OBJECT_READ) {
                if (inst->operand_count() != 1u) { return false; }
            } else if (inst->derived_instruction_tag() == xir::DerivedInstructionTag::RAY_QUERY_OBJECT_WRITE) {
                auto op = static_cast<const xir::RayQueryObjectWriteInst *>(inst)->op();
                auto count = op == xir::RayQueryObjectWriteOp::RAY_QUERY_OBJECT_COMMIT_PROCEDURAL ? 2u : 1u;
                if (inst->operand_count() != count) { return false; }
            } else {
                return false;
            }
        }
    }
    for (auto block : definition->basic_blocks()) {
        auto surface_committed_in_block = false;
        for (auto inst : block->instructions()) {
            switch (inst->derived_instruction_tag()) {
                case xir::DerivedInstructionTag::RAY_QUERY_OBJECT_READ: {
                    if (query == nullptr || inst->operand(0) != query) { return false; }
                    auto op = static_cast<const xir::RayQueryObjectReadInst *>(inst)->op();
                    if (procedural) {
                        // IS starts with the hardware's committed bound. Local
                        // valid commits tighten it before subsequent reads.
                        if (op != xir::RayQueryObjectReadOp::RAY_QUERY_OBJECT_PROCEDURAL_CANDIDATE_HIT &&
                            op != xir::RayQueryObjectReadOp::RAY_QUERY_OBJECT_WORLD_SPACE_RAY &&
                            op != xir::RayQueryObjectReadOp::RAY_QUERY_OBJECT_CANDIDATE_OBJECT_SPACE_RAY) { return false; }
                    } else if (op != xir::RayQueryObjectReadOp::RAY_QUERY_OBJECT_TRIANGLE_CANDIDATE_HIT &&
                               !(op == xir::RayQueryObjectReadOp::RAY_QUERY_OBJECT_WORLD_SPACE_RAY &&
                                 world_ray_read_uses_only_exact_ah_fields(inst))) {
                        // AH's native TMax is the candidate distance, so it
                        // cannot replace an observable prior committed bound.
                        return false;
                    }
                    break;
                }
                case xir::DerivedInstructionTag::RAY_QUERY_OBJECT_WRITE: {
                    if (query == nullptr || inst->operand(0) != query) { return false; }
                    auto op = static_cast<const xir::RayQueryObjectWriteInst *>(inst)->op();
                    if (procedural) {
                        // A procedural commit can reject its distance. Merely
                        // preceding terminate with a commit is not sufficient.
                        if (op != xir::RayQueryObjectWriteOp::RAY_QUERY_OBJECT_COMMIT_PROCEDURAL) { return false; }
                    } else if (op == xir::RayQueryObjectWriteOp::RAY_QUERY_OBJECT_COMMIT_TRIANGLE) {
                        surface_committed_in_block = true;
                    } else if (op == xir::RayQueryObjectWriteOp::RAY_QUERY_OBJECT_TERMINATE) {
                        if (!surface_committed_in_block) { return false; }
                    } else {
                        return false;
                    }
                    break;
                }
                case xir::DerivedInstructionTag::CALL: {
                    auto call = static_cast<const xir::CallInst *>(inst);
                    auto forwarded_query = find_forwarded_query_param(call, query);
                    if (!ray_query_handler_has_hardware_result(call->callee(), forwarded_query, procedural, visited)) {
                        return false;
                    }
                    break;
                }
                default: break;
            }
        }
    }
    visited.erase(function);
    return true;
}

[[nodiscard]] bool ray_query_pipeline_has_hardware_result(const xir::RayQueryPipelineInst *pipeline) noexcept {
    auto query = pipeline->query_object();
    if (!query->isa<xir::AllocaInst>()) { return false; }
    // Prove the caller's query cannot also reach a callback through a capture.
    // Whole-query loads, derived addresses, and reference calls are conservative
    // fallbacks; ordinary post-traversal query observations remain supported.
    for (auto use : query->use_list()) {
        auto user = use->user();
        if (user->derived_value_tag() != xir::DerivedValueTag::INSTRUCTION) { return false; }
        auto inst = static_cast<const xir::Instruction *>(user);
        switch (inst->derived_instruction_tag()) {
            case xir::DerivedInstructionTag::STORE:
                if (static_cast<const xir::StoreInst *>(inst)->variable() != query ||
                    static_cast<const xir::StoreInst *>(inst)->value() == query) { return false; }
                break;
            case xir::DerivedInstructionTag::RAY_QUERY_OBJECT_READ:
                if (inst->operand_count() != 1u || inst->operand(0) != query) { return false; }
                break;
            case xir::DerivedInstructionTag::RAY_QUERY_PIPELINE: {
                auto other = static_cast<const xir::RayQueryPipelineInst *>(inst);
                if (other->query_object() != query) { return false; }
                for (auto capture : other->captured_argument_uses()) {
                    if (capture->value() == query) { return false; }
                }
                break;
            }
            default: return false;
        }
    }
    llvm::DenseSet<const xir::Function *> visited;
    for (auto procedural : {false, true}) {
        auto callback = procedural ? pipeline->on_procedural_function() : pipeline->on_surface_function();
        if (callback == nullptr) { continue; }
        if (callback->arguments().empty() ||
            !ray_query_handler_has_hardware_result(callback, callback->arguments().front(), procedural, visited)) { return false; }
    }
    return true;
}

enum class CaptureUsage {
    Unused,
    ReadOnly,
    ReadWrite,
};

[[nodiscard]] bool pointer_uses_are_readonly(
    const xir::Value *pointer, llvm::DenseSet<const xir::Value *> &visited) noexcept {
    if (pointer == nullptr || !visited.insert(pointer).second) { return true; }
    for (auto use : pointer->use_list()) {
        auto user = use->user();
        if (user == nullptr || user->derived_value_tag() != xir::DerivedValueTag::INSTRUCTION) { return false; }
        auto inst = static_cast<const xir::Instruction *>(user);
        switch (inst->derived_instruction_tag()) {
            case xir::DerivedInstructionTag::LOAD:
                if (inst->operand_count() == 0u || inst->operand(0u) != pointer) { return false; }
                break;
            case xir::DerivedInstructionTag::RAY_QUERY_OBJECT_READ:
                break;
            case xir::DerivedInstructionTag::GEP: {
                auto gep = static_cast<const xir::GEPInst *>(inst);
                if (gep->base() != pointer || !pointer_uses_are_readonly(gep, visited)) { return false; }
                break;
            }
            case xir::DerivedInstructionTag::CALL: {
                auto call = static_cast<const xir::CallInst *>(inst);
                auto callee = call->callee();
                if (callee == nullptr || callee->definition() == nullptr || call->callee() == pointer) { return false; }
                auto param_it = callee->arguments().begin();
                auto param_end = callee->arguments().end();
                for (auto i = size_t{0u}; i < call->argument_count(); i++) {
                    if (param_it == param_end) { return false; }
                    const xir::Argument *param = *param_it;
                    ++param_it;
                    if (call->argument(i) == pointer) {
                        if (!param->is_reference() || !pointer_uses_are_readonly(param, visited)) { return false; }
                    }
                }
                break;
            }
            default: return false;
        }
    }
    return true;
}

[[nodiscard]] CaptureUsage ray_query_capture_usage(
    const xir::RayQueryPipelineInst *pipeline, size_t capture_idx) noexcept {
    if (pipeline == nullptr) { return CaptureUsage::ReadWrite; }
    auto used = false;
    auto written = false;
    for (auto procedural : {false, true}) {
        auto callback = procedural ? pipeline->on_procedural_function() : pipeline->on_surface_function();
        if (callback == nullptr || ray_query_handler_is_empty(callback)) { continue; }
        auto target_idx = capture_idx + 1u;
        if (target_idx >= callback->arguments().count_size()) {
            return CaptureUsage::ReadWrite;
        }
        const xir::Argument *formal = nullptr;
        size_t idx = 0u;
        for (auto arg : callback->arguments()) {
            if (idx++ == target_idx) {
                formal = arg;
                break;
            }
        }
        if (formal == nullptr) { return CaptureUsage::ReadWrite; }
        if (formal->use_list().empty()) { continue; }
        used = true;
        if (formal->is_reference()) {
            llvm::DenseSet<const xir::Value *> visited;
            if (!pointer_uses_are_readonly(formal, visited)) {
                written = true;
            }
        }
    }
    if (!used) { return CaptureUsage::Unused; }
    return written ? CaptureUsage::ReadWrite : CaptureUsage::ReadOnly;
}

[[nodiscard]] bool ray_query_surface_handler_always_commits_only_impl(
    const xir::Function *function, const xir::Value *query,
    llvm::DenseSet<const xir::Function *> &visiting) noexcept {
    if (function == nullptr || query == nullptr) { return false; }
    if (!visiting.insert(function).second) { return false; }
    auto definition = function->definition();
    if (definition == nullptr || definition->body_block() == nullptr) {
        visiting.erase(function);
        return false;
    }
    llvm::DenseSet<const xir::BasicBlock *> dfs_visiting;
    llvm::DenseSet<const xir::BasicBlock *> dfs_visited;
    auto is_acyclic = [&](auto &&self, const xir::BasicBlock *bb) noexcept -> bool {
        if (bb == nullptr || dfs_visited.contains(bb)) { return true; }
        if (!dfs_visiting.insert(bb).second) { return false; }
        auto ok = true;
        bb->traverse_successors(false, [&](const xir::BasicBlock *succ) noexcept {
            if (ok && !self(self, succ)) { ok = false; }
        });
        dfs_visiting.erase(bb);
        dfs_visited.insert(bb);
        return ok;
    };
    llvm::DenseMap<const xir::BasicBlock *, bool> block_commits;
    for (auto block : definition->basic_blocks()) {
        if (!is_acyclic(is_acyclic, block)) {
            visiting.erase(function);
            return false;
        }
        auto committed = false;
        for (auto inst : block->instructions()) {
            switch (inst->derived_instruction_tag()) {
                case xir::DerivedInstructionTag::RETURN:
                    if (static_cast<const xir::ReturnInst *>(inst)->return_value() != nullptr) {
                        visiting.erase(function);
                        return false;
                    }
                    break;
                case xir::DerivedInstructionTag::BRANCH:
                case xir::DerivedInstructionTag::CONDITIONAL_BRANCH:
                case xir::DerivedInstructionTag::IF:
                case xir::DerivedInstructionTag::SWITCH:
                case xir::DerivedInstructionTag::INDEXED_BRANCH:
                case xir::DerivedInstructionTag::UNREACHABLE:
                case xir::DerivedInstructionTag::PHI:
                case xir::DerivedInstructionTag::ALLOCA:
                case xir::DerivedInstructionTag::LOAD:
                case xir::DerivedInstructionTag::GEP:
                case xir::DerivedInstructionTag::ARITHMETIC:
                case xir::DerivedInstructionTag::CAST:
                case xir::DerivedInstructionTag::ASSUME:
                    break;
                case xir::DerivedInstructionTag::STORE: {
                    auto store = static_cast<const xir::StoreInst *>(inst);
                    auto var = store->variable();
                    auto is_local_alloca = var != nullptr && var->isa<xir::AllocaInst>() &&
                                           static_cast<const xir::AllocaInst *>(var)->parent_function() == definition;
                    auto is_self_copy = var != nullptr && store->value() != nullptr &&
                                        store->value()->isa<xir::LoadInst>() &&
                                        static_cast<const xir::LoadInst *>(store->value())->variable() == var;
                    if (!is_local_alloca && !is_self_copy) {
                        visiting.erase(function);
                        return false;
                    }
                    break;
                }
                case xir::DerivedInstructionTag::RAY_QUERY_OBJECT_WRITE: {
                    auto w = static_cast<const xir::RayQueryObjectWriteInst *>(inst);
                    if (w->op() == xir::RayQueryObjectWriteOp::RAY_QUERY_OBJECT_COMMIT_TRIANGLE &&
                        w->operand_count() == 1u && w->operand(0u) == query) {
                        committed = true;
                        break;
                    }
                    visiting.erase(function);
                    return false;
                }
                case xir::DerivedInstructionTag::RAY_QUERY_OBJECT_READ: {
                    auto op = static_cast<const xir::RayQueryObjectReadInst *>(inst)->op();
                    if (op != xir::RayQueryObjectReadOp::RAY_QUERY_OBJECT_TRIANGLE_CANDIDATE_HIT &&
                        !(op == xir::RayQueryObjectReadOp::RAY_QUERY_OBJECT_WORLD_SPACE_RAY &&
                          world_ray_read_uses_only_exact_ah_fields(inst))) {
                        visiting.erase(function);
                        return false;
                    }
                    break;
                }
                case xir::DerivedInstructionTag::CALL: {
                    auto call = static_cast<const xir::CallInst *>(inst);
                    auto forwarded_query = find_forwarded_query_param(call, query);
                    if (forwarded_query != nullptr &&
                        ray_query_surface_handler_always_commits_only_impl(call->callee(), forwarded_query, visiting)) {
                        committed = true;
                        break;
                    }
                    if (ray_query_handler_is_empty(call->callee())) {
                        break;
                    }
                    visiting.erase(function);
                    return false;
                }
                default:
                    visiting.erase(function);
                    return false;
            }
        }
        block_commits[block] = committed;
    }
    llvm::DenseMap<std::pair<const xir::BasicBlock *, uint8_t>, bool> memo;
    auto path_always_commits = [&](auto &&self, const xir::BasicBlock *bb, bool committed_so_far) noexcept -> bool {
        auto now_committed = committed_so_far || block_commits.lookup(bb);
        auto key = std::pair<const xir::BasicBlock *, uint8_t>{bb, now_committed ? 1u : 0u};
        if (auto it = memo.find(key); it != memo.end()) { return it->second; }
        auto res = false;
        if (bb->is_terminated() && bb->terminator()->derived_instruction_tag() == xir::DerivedInstructionTag::RETURN) {
            res = now_committed;
        } else if (bb->is_terminated() && bb->terminator()->derived_instruction_tag() == xir::DerivedInstructionTag::UNREACHABLE) {
            res = true;
        } else {
            auto has_succ = false;
            auto all_ok = true;
            bb->traverse_successors(false, [&](const xir::BasicBlock *succ) noexcept {
                has_succ = true;
                if (all_ok && !self(self, succ, now_committed)) { all_ok = false; }
            });
            res = has_succ && all_ok;
        }
        memo[key] = res;
        return res;
    };
    auto ok = path_always_commits(path_always_commits, definition->body_block(), false);
    visiting.erase(function);
    return ok;
}

[[nodiscard]] bool ray_query_surface_handler_always_commits_only(
    const xir::Function *function, const xir::Value *query) noexcept {
    llvm::DenseSet<const xir::Function *> visiting;
    return ray_query_surface_handler_always_commits_only_impl(function, query, visiting);
}
}// namespace

void CUDACodegenLLVMImpl::_translate_ray_query_loop_inst(IB &, FunctionContext &, const xir::RayQueryLoopInst *) noexcept {
    LUISA_ERROR_WITH_LOCATION("CUDA LLVM requires lower_ray_query_to_pipeline before emitting ray queries.");
}

void CUDACodegenLLVMImpl::_translate_ray_query_dispatch_inst(IB &, FunctionContext &, const xir::RayQueryDispatchInst *) noexcept {
    LUISA_ERROR_WITH_LOCATION("CUDA LLVM cannot emit an unlowered ray-query dispatch.");
}

llvm::Value *CUDACodegenLLVMImpl::_load_ray_query_field(IB &b, llvm::Value *query, unsigned field) noexcept {
    auto type = _get_llvm_ray_query_type();
    return b.CreateLoad(type->getStructElementType(field), b.CreateStructGEP(type, query, field));
}

void CUDACodegenLLVMImpl::_store_ray_query_field(IB &b, llvm::Value *query, unsigned field, llvm::Value *value) noexcept {
    b.CreateStore(value, b.CreateStructGEP(_get_llvm_ray_query_type(), query, field));
}

llvm::Value *CUDACodegenLLVMImpl::_ray_query_surface_candidate(IB &b) noexcept {
    auto bary = _call_optix_get_triangle_barycentrics(b);
    if (_rt_analysis.curve_basis_set.any()) {
        auto kind = _call_optix_get_hit_kind(b);
        auto triangle = b.CreateOr(b.CreateICmpEQ(kind, b.getInt32(OPTIX_HIT_KIND_TRIANGLE_FRONT_FACE)),
                                   b.CreateICmpEQ(kind, b.getInt32(OPTIX_HIT_KIND_TRIANGLE_BACK_FACE)));
        auto curve_bary = _create_llvm_vector(b, {_call_optix_get_curve_parameter(b),
                                                  llvm::ConstantFP::get(b.getFloatTy(), -1.)});
        bary = b.CreateSelect(triangle, bary, curve_bary);
    }
    auto hit = static_cast<llvm::Value *>(llvm::Constant::getNullValue(_get_llvm_surface_hit_type()));
    hit = b.CreateInsertValue(hit, _call_optix_read_instance_index(b), llvm_surface_hit_type_inst_id_index);
    hit = b.CreateInsertValue(hit, _call_optix_read_primitive_index(b), llvm_surface_hit_type_prim_id_index);
    hit = b.CreateInsertValue(hit, bary, llvm_surface_hit_type_bary_index);
    return b.CreateInsertValue(hit, _call_optix_get_hit_distance(b), llvm_surface_hit_type_t_index);
}

llvm::Value *CUDACodegenLLVMImpl::_translate_ray_query_object_read_inst(IB &b, FunctionContext &func_ctx, const xir::RayQueryObjectReadInst *inst) noexcept {
    LUISA_ASSERT(inst->operand_count() == 1u && inst->operand(0)->is_lvalue(), "Invalid ray-query object read.");
    auto query = _get_llvm_value(b, func_ctx, inst->operand(0));
    switch (inst->op()) {
        case xir::RayQueryObjectReadOp::RAY_QUERY_OBJECT_WORLD_SPACE_RAY:
            return _load_ray_query_field(b, query, llvm_ray_query_type_ray_index);
        case xir::RayQueryObjectReadOp::RAY_QUERY_OBJECT_CANDIDATE_OBJECT_SPACE_RAY: {
            auto ray = _call_optix_get_object_space_ray(b);
            auto world_ray = _load_ray_query_field(b, query, llvm_ray_query_type_ray_index);
            // OptiX reports the candidate distance in AH, not the committed bound.
            return b.CreateInsertValue(ray, b.CreateExtractValue(world_ray, llvm_ray_type_t_max_index), llvm_ray_type_t_max_index);
        }
        case xir::RayQueryObjectReadOp::RAY_QUERY_OBJECT_PROCEDURAL_CANDIDATE_HIT: {
            auto hit = static_cast<llvm::Value *>(llvm::Constant::getNullValue(_get_llvm_procedural_hit_type()));
            hit = b.CreateInsertValue(hit, _call_optix_read_instance_index(b), llvm_procedural_hit_type_inst_id_index);
            return b.CreateInsertValue(hit, _call_optix_read_primitive_index(b), llvm_procedural_hit_type_prim_id_index);
        }
        case xir::RayQueryObjectReadOp::RAY_QUERY_OBJECT_TRIANGLE_CANDIDATE_HIT:
            return _ray_query_surface_candidate(b);
        case xir::RayQueryObjectReadOp::RAY_QUERY_OBJECT_COMMITTED_HIT:
            return _load_ray_query_field(b, query, llvm_ray_query_type_hit_index);
        case xir::RayQueryObjectReadOp::RAY_QUERY_OBJECT_IS_TRIANGLE_CANDIDATE:
            return b.CreateICmpEQ(_load_ray_query_field(b, query, llvm_ray_query_type_state_index), b.getInt8(llvm_ray_query_state_surface_candidate));
        case xir::RayQueryObjectReadOp::RAY_QUERY_OBJECT_IS_PROCEDURAL_CANDIDATE:
            return b.CreateICmpEQ(_load_ray_query_field(b, query, llvm_ray_query_type_state_index), b.getInt8(llvm_ray_query_state_procedural_candidate));
        case xir::RayQueryObjectReadOp::RAY_QUERY_OBJECT_IS_TERMINATED:
            return b.CreateOr(_load_ray_query_field(b, query, llvm_ray_query_type_terminated_index),
                              b.CreateICmpEQ(_load_ray_query_field(b, query, llvm_ray_query_type_state_index), b.getInt8(llvm_ray_query_state_surface_terminated)));
    }
    LUISA_ERROR_WITH_LOCATION("Invalid ray-query read operation.");
}

void CUDACodegenLLVMImpl::_commit_ray_query_hit(IB &b, llvm::Value *query, llvm::Value *t, bool procedural) noexcept {
    auto ray = b.CreateStructGEP(_get_llvm_ray_query_type(), query, llvm_ray_query_type_ray_index);
    auto t_max_ptr = b.CreateStructGEP(_get_llvm_ray_type(), ray, llvm_ray_type_t_max_index);
    auto write_commit = [&]() noexcept {
        auto hit = static_cast<llvm::Value *>(llvm::Constant::getNullValue(_get_llvm_committed_hit_type()));
        hit = b.CreateInsertValue(hit, _call_optix_read_instance_index(b), llvm_committed_hit_type_inst_id_index);
        hit = b.CreateInsertValue(hit, _call_optix_read_primitive_index(b), llvm_committed_hit_type_prim_id_index);
        if (!procedural) {
            auto surface = _ray_query_surface_candidate(b);
            hit = b.CreateInsertValue(hit, b.CreateExtractValue(surface, llvm_surface_hit_type_bary_index), llvm_committed_hit_type_bary_index);
        }
        hit = b.CreateInsertValue(hit, b.getInt32(static_cast<uint32_t>(procedural ? HitType::Procedural : HitType::Surface)), llvm_committed_hit_type_hit_kind_index);
        hit = b.CreateInsertValue(hit, t, llvm_committed_hit_type_t_index);
        _store_ray_query_field(b, query, llvm_ray_query_type_hit_index, hit);
        b.CreateStore(t, t_max_ptr);
        _store_ray_query_field(b, query, llvm_ray_query_type_committed_index, b.getTrue());
    };
    if (!procedural) {
        // OptiX hardware already enforces t_min <= t <= t_max for triangle candidates.
        // Emitting the commit unconditionally lets LLVM DCE dead hit/bound getters in AH.
        write_commit();
        return;
    }
    auto t_min_ptr = b.CreateStructGEP(_get_llvm_ray_type(), ray, llvm_ray_type_t_min_index);
    auto t_min = b.CreateLoad(b.getFloatTy(), t_min_ptr);
    auto t_max = b.CreateLoad(b.getFloatTy(), t_max_ptr);
    auto valid = b.CreateAnd(b.CreateFCmpOGE(t, t_min), b.CreateFCmpOLE(t, t_max));
    auto function = b.GetInsertBlock()->getParent();
    auto accept = llvm::BasicBlock::Create(_llvm_context, "query.commit", function);
    auto merge = llvm::BasicBlock::Create(_llvm_context, "query.commit.end", function);
    // Rejected distances leave the previous hit, bound and acceptance flag
    // untouched, including a hit committed earlier in this callback.
    b.CreateCondBr(valid, accept, merge);
    b.SetInsertPoint(accept);
    write_commit();
    b.CreateBr(merge);
    b.SetInsertPoint(merge);
}

void CUDACodegenLLVMImpl::_translate_ray_query_object_write_inst(IB &b, FunctionContext &func_ctx, const xir::RayQueryObjectWriteInst *inst) noexcept {
    LUISA_ASSERT(inst->operand_count() >= 1u && inst->operand(0)->is_lvalue(), "Invalid ray-query object write.");
    auto query = _get_llvm_value(b, func_ctx, inst->operand(0));
    switch (inst->op()) {
        case xir::RayQueryObjectWriteOp::RAY_QUERY_OBJECT_COMMIT_TRIANGLE:
            _commit_ray_query_hit(b, query, _call_optix_get_hit_distance(b), false);
            break;
        case xir::RayQueryObjectWriteOp::RAY_QUERY_OBJECT_COMMIT_PROCEDURAL:
            LUISA_ASSERT(inst->operand_count() == 2u, "Procedural commit requires a distance.");
            _commit_ray_query_hit(b, query, _get_llvm_value(b, func_ctx, inst->operand(1)), true);
            break;
        case xir::RayQueryObjectWriteOp::RAY_QUERY_OBJECT_TERMINATE:
            _store_ray_query_field(b, query, llvm_ray_query_type_terminated_index, b.getTrue());
            break;
        case xir::RayQueryObjectWriteOp::RAY_QUERY_OBJECT_PROCEED:
            LUISA_ERROR_WITH_LOCATION("CUDA LLVM requires reconstruct_ray_query_loop and lower_ray_query_to_pipeline for query.proceed().");
    }
}

void CUDACodegenLLVMImpl::_translate_ray_query_pipeline_inst(IB &b, FunctionContext &func_ctx, const xir::RayQueryPipelineInst *inst) noexcept {
    llvm::DenseSet<const xir::Function *> visited;
    validate_ray_query_handler(inst->on_surface_function(), visited);
    validate_ray_query_handler(inst->on_procedural_function(), visited);
    auto generic_pointer = [&b](llvm::Value *value) noexcept {
        return value->getType()->isPointerTy() && value->getType()->getPointerAddressSpace() != 0u ?
                   b.CreateAddrSpaceCast(value, b.getPtrTy()) :
                   value;
    };
    auto is_local_ptr_value = [](const xir::Value *xir_val, const llvm::Value *llvm_val) noexcept {
        if (xir_val != nullptr && xir_val->isa<xir::AllocaInst>()) { return true; }
        if (llvm_val == nullptr || !llvm_val->getType()->isPointerTy()) { return false; }
        return llvm::isa<llvm::AllocaInst>(llvm_val) ||
               llvm_val->getType()->getPointerAddressSpace() == nvptx_address_space_local;
    };
    auto raw_query = _get_llvm_value(b, func_ctx, inst->query_object());
    auto query_is_local_ptr = is_local_ptr_value(inst->query_object(), raw_query);
    auto query = generic_pointer(raw_query);
    auto id = static_cast<uint32_t>(_ray_query_pipelines.size());
    auto needs_pipeline_id = _rt_analysis.ray_query_pipeline_count > 1u;
    auto hardware_result = ray_query_pipeline_has_hardware_result(inst);
    auto has_procedural = !ray_query_handler_is_empty(inst->on_procedural_function());
    auto surface_cb = inst->on_surface_function();
    auto surface_query = (surface_cb != nullptr && !surface_cb->arguments().empty()) ? surface_cb->arguments().front() : nullptr;
    auto disable_anyhit = hardware_result && !has_procedural &&
                          ray_query_surface_handler_always_commits_only(surface_cb, surface_query);
    llvm::SmallVector<llvm::Value *> fields{b.getInt32(id), query};
    llvm::SmallVector<RayQueryCapture> captures;
    struct CaptureSource {
        size_t capture_index;
        const xir::Value *xir_value;
        llvm::Value *raw_llvm_value;
    };
    llvm::SmallVector<CaptureSource> capture_sources;
    struct LargeByValueCandidate {
        size_t capture_index;
        llvm::Type *elem_type;
        llvm::Value *raw_llvm_value;
        uint32_t words;
        CaptureUsage usage;
    };
    llvm::SmallVector<LargeByValueCandidate> large_by_value_candidates;
    if (disable_anyhit) {
        for (size_t i = 0u; i < inst->captured_argument_count(); i++) {
            RayQueryCapture cap{};
            cap.unused = true;
            captures.emplace_back(cap);
        }
    } else {
        for (size_t cap_idx = 0u; cap_idx < inst->captured_argument_count(); cap_idx++) {
            auto value = inst->captured_argument(cap_idx);
            auto usage = ray_query_capture_usage(inst, cap_idx);
            if (usage == CaptureUsage::Unused) {
                RayQueryCapture cap{};
                cap.unused = true;
                captures.emplace_back(cap);
                continue;
            }
            if (value->isa<xir::Argument>()) {
                auto arg = static_cast<const xir::Argument *>(value);
                if (auto iter = _ray_query_resource_origins.find(arg); iter != _ray_query_resource_origins.end()) {
                    RayQueryCapture cap{};
                    cap.root_resource = iter->second;
                    captures.emplace_back(cap);
                    continue;
                }
            }
            auto raw_val = _get_llvm_value(b, func_ctx, value);
            auto capture_index = captures.size();
            if (value->isa<xir::AllocaInst>()) {
                auto alloca_inst = static_cast<const xir::AllocaInst *>(value);
                if (alloca_can_stage_inline(alloca_inst, inst)) {
                    auto elem_type = _get_llvm_type(alloca_inst->type())->mem_type;
                    if (elem_type != nullptr && !elem_type->isEmptyTy()) {
                        auto words = ray_query_payload_word_count(elem_type, *_data_layout, 8u);
                        if (words >= 1u && words <= 2u) {
                            RayQueryCapture cap{};
                            cap.context_field = static_cast<unsigned>(fields.size());
                            cap.by_value_ro = (usage == CaptureUsage::ReadOnly);
                            cap.by_value_rw = (usage == CaptureUsage::ReadWrite);
                            cap.value_type = elem_type;
                            captures.emplace_back(cap);
                            fields.emplace_back(b.CreateLoad(elem_type, raw_val));
                            capture_sources.emplace_back(CaptureSource{capture_index, value, raw_val});
                            continue;
                        }
                        if (words >= 3u && words <= 8u) {
                            large_by_value_candidates.emplace_back(LargeByValueCandidate{
                                capture_index, elem_type, raw_val, words, usage});
                        }
                    }
                }
            }
            // Keep references as references, including aliases and subobject captures.
            RayQueryCapture cap{};
            cap.context_field = static_cast<unsigned>(fields.size());
            cap.is_local_ptr = is_local_ptr_value(value, raw_val);
            captures.emplace_back(cap);
            fields.emplace_back(generic_pointer(raw_val));
            capture_sources.emplace_back(CaptureSource{capture_index, value, raw_val});
        }
    }
    llvm::SmallVector<llvm::Type *> field_types;
    field_types.reserve(fields.size());
    for (auto field : fields) { field_types.emplace_back(field->getType()); }
    auto context_type = llvm::StructType::create(_llvm_context, field_types, "luisa.ray.query.context");
    auto capture_offset = hardware_result ? (needs_pipeline_id ? 1u : 0u) : (needs_pipeline_id ? 3u : 2u);
    auto capture_budget = kRayQueryPayloadWordCount - capture_offset;
    auto direct_payload = disable_anyhit || ray_query_uses_direct_payload(context_type, *_data_layout, capture_budget);
    if (!disable_anyhit && direct_payload && !large_by_value_candidates.empty()) {
        constexpr auto kPromotedPayloadWordLimit = 24u;
        auto used_words = capture_offset;
        for (auto &&src : capture_sources) {
            auto field_index = captures[src.capture_index].context_field;
            used_words += ray_query_payload_word_count(field_types[field_index], *_data_layout, kRayQueryPayloadWordCount);
        }
        std::sort(large_by_value_candidates.begin(), large_by_value_candidates.end(),
                  [](const LargeByValueCandidate &lhs, const LargeByValueCandidate &rhs) noexcept {
                      if (lhs.words != rhs.words) { return lhs.words < rhs.words; }
                      return (lhs.usage == CaptureUsage::ReadWrite) < (rhs.usage == CaptureUsage::ReadWrite);
                  });
        auto upgraded = false;
        for (auto &&cand : large_by_value_candidates) {
            auto extra_words = cand.words - 2u;
            if (used_words + extra_words > kPromotedPayloadWordLimit) { continue; }
            auto &cap = captures[cand.capture_index];
            cap.by_value_ro = (cand.usage == CaptureUsage::ReadOnly);
            cap.by_value_rw = (cand.usage == CaptureUsage::ReadWrite);
            cap.value_type = cand.elem_type;
            cap.is_local_ptr = false;
            fields[cap.context_field] = b.CreateLoad(cand.elem_type, cand.raw_llvm_value);
            field_types[cap.context_field] = cand.elem_type;
            used_words += extra_words;
            upgraded = true;
        }
        if (upgraded) {
            context_type = llvm::StructType::create(_llvm_context, field_types, "luisa.ray.query.context");
        }
    }
    struct StagedAlloca {
        unsigned field_index;
        llvm::Type *elem_type;
        llvm::Value *llvm_alloca;
    };
    llvm::SmallVector<StagedAlloca> staged_allocas;
    if (!direct_payload) {
        for (auto &&src : capture_sources) {
            auto &cap = captures[src.capture_index];
            if (cap.by_value_ro || cap.by_value_rw) {
                cap.by_value_ro = false;
                cap.by_value_rw = false;
                cap.staged_inline = true;
                staged_allocas.emplace_back(StagedAlloca{cap.context_field, cap.value_type, src.raw_llvm_value});
                continue;
            }
            if (src.xir_value->isa<xir::AllocaInst>()) {
                auto alloca_inst = static_cast<const xir::AllocaInst *>(src.xir_value);
                if (alloca_can_stage_inline(alloca_inst, inst)) {
                    auto elem_type = _get_llvm_type(alloca_inst->type())->mem_type;
                    if (elem_type != nullptr && !elem_type->isEmptyTy() &&
                        _data_layout->getTypeAllocSize(elem_type).getFixedValue() <= 64u) {
                        cap.staged_inline = true;
                        field_types[cap.context_field] = elem_type;
                        staged_allocas.emplace_back(StagedAlloca{cap.context_field, elem_type, src.raw_llvm_value});
                    }
                }
            }
        }
        if (!staged_allocas.empty()) {
            context_type = llvm::StructType::create(_llvm_context, field_types, "luisa.ray.query.context");
        }
    }
    llvm::SmallVector<llvm::Value *, kRayQueryPayloadWordCount> payload;
    if (!disable_anyhit) {
        if (needs_pipeline_id) { payload.emplace_back(b.getInt32(id)); }
        if (!hardware_result) { pack_ray_query_payload(b, query, payload); }
    }
    llvm::AllocaInst *context = nullptr;
    struct RWCopyback {
        uint32_t word_offset;
        uint32_t word_count;
        llvm::Type *elem_type;
        llvm::Value *llvm_alloca;
    };
    llvm::SmallVector<RWCopyback> rw_copybacks;
    if (!disable_anyhit && direct_payload) {
        for (auto &&src : capture_sources) {
            auto &cap = captures[src.capture_index];
            auto field_index = cap.context_field;
            auto start_word = static_cast<uint32_t>(payload.size());
            pack_ray_query_payload(b, fields[field_index], payload);
            if (cap.by_value_rw) {
                auto end_word = static_cast<uint32_t>(payload.size());
                cap.payload_word_offset = start_word;
                cap.payload_word_count = end_word - start_word;
                for (auto w = start_word; w < end_word; w++) {
                    _ray_query_payload_rw_mask |= (1u << w);
                }
                rw_copybacks.emplace_back(RWCopyback{cap.payload_word_offset, cap.payload_word_count, cap.value_type, src.raw_llvm_value});
            }
        }
    } else if (!disable_anyhit) {
        // Traversal invokes its handlers synchronously, and nested traversal in
        // a handler is rejected above. Only the fallback needs private scratch;
        // query objects and captured references retain their original identities.
        auto context_size = _data_layout->getTypeAllocSize(context_type).getFixedValue();
        auto &scratch = func_ctx.llvm_ray_query_context_scratch;
        if (scratch == nullptr) {
            IB alloca_b{func_ctx.llvm_alloca_block->getTerminator()};
            scratch = alloca_b.CreateAlloca(alloca_b.getInt8Ty(), alloca_b.getInt64(context_size), "ray.query.context.scratch");
            scratch->setAlignment(llvm::Align{16u});
        } else if (context_size > llvm::cast<llvm::ConstantInt>(scratch->getArraySize())->getZExtValue()) {
            scratch->setOperand(0, b.getInt64(context_size));
        }
        // A smaller later context may still require a wider vector alignment.
        scratch->setAlignment(std::max(scratch->getAlign(), _data_layout->getABITypeAlign(context_type)));
        context = scratch;
        for (auto &&src : capture_sources) {
            auto &cap = captures[src.capture_index];
            auto field_index = cap.context_field;
            auto stored_val = cap.staged_inline && !cap.value_type ?
                                  b.CreateLoad(field_types[field_index], src.raw_llvm_value) :
                                  fields[field_index];
            b.CreateStore(stored_val, b.CreateStructGEP(context_type, context, field_index));
        }
        pack_ray_query_payload(b, generic_pointer(context), payload);
    }
    _ray_query_pipelines.emplace_back(RayQueryPipeline{
        inst, context_type, std::move(captures), hardware_result, direct_payload, query_is_local_ptr, disable_anyhit});
    LUISA_ASSERT(payload.size() <= kRayQueryPayloadWordCount, "Ray-query payload exceeds the register budget.");
    auto accel = _load_ray_query_field(b, query, llvm_ray_query_type_accel_index);
    auto ray = _load_ray_query_field(b, query, llvm_ray_query_type_ray_index);
    auto time = _load_ray_query_field(b, query, llvm_ray_query_type_time_index);
    auto mask = _load_ray_query_field(b, query, llvm_ray_query_type_mask_index);
    // General handlers observe software state while traversing and must
    // track opaque hits too. Qualified callbacks use native traversal bounds
    // and recover the final hardware result after traversal instead.
    auto flags = static_cast<uint32_t>(OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT);
    if (disable_anyhit) {
        flags |= OPTIX_RAY_FLAG_DISABLE_ANYHIT;
    } else if (!hardware_result) {
        flags |= OPTIX_RAY_FLAG_ENFORCE_ANYHIT;
    }
    if (inst->query_object()->type() == Type::of<RayQueryAny>()) {
        flags |= OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT;
    }
    auto trace_call = _call_optix_trace(b, OPTIX_PAYLOAD_TYPE_ID_1, 5u, flags, accel, ray, time, mask, payload);
    for (auto &&cb : rw_copybacks) {
        llvm::SmallVector<llvm::Value *, 8u> words;
        words.reserve(cb.word_count);
        for (auto w = 0u; w < cb.word_count; w++) {
            words.emplace_back(b.CreateExtractValue(trace_call, cb.word_offset + w));
        }
        auto pos = 0u;
        auto updated = unpack_ray_query_payload_from_words(b, cb.elem_type, words, pos);
        b.CreateStore(updated, cb.llvm_alloca);
    }
    for (auto &&staged : staged_allocas) {
        auto updated = b.CreateLoad(staged.elem_type, b.CreateStructGEP(context_type, context, staged.field_index));
        b.CreateStore(updated, staged.llvm_alloca);
    }
    if (hardware_result) {
        auto function = b.GetInsertBlock()->getParent();
        auto hit_block = llvm::BasicBlock::Create(_llvm_context, "surface.filter.hit", function);
        auto merge_block = llvm::BasicBlock::Create(_llvm_context, "surface.filter.result", function);
        auto is_hit = _call_optix_hit_object_is_hit(b);
        // Hit-object attributes are only defined for an actual hit of the
        // corresponding primitive kind. A select cannot guard those getters.
        b.CreateCondBr(is_hit, hit_block, merge_block);
        b.SetInsertPoint(hit_block);
        auto has_procedural = !ray_query_handler_is_empty(inst->on_procedural_function());
        auto kind = _rt_analysis.curve_basis_set.any() || has_procedural ? _call_optix_hit_object_hit_kind(b) : nullptr;
        auto emit_surface_bary = [&]() noexcept -> llvm::Value * {
            if (_rt_analysis.curve_basis_set.any()) {
                auto triangle = b.CreateOr(b.CreateICmpEQ(kind, b.getInt32(OPTIX_HIT_KIND_TRIANGLE_FRONT_FACE)),
                                           b.CreateICmpEQ(kind, b.getInt32(OPTIX_HIT_KIND_TRIANGLE_BACK_FACE)));
                auto triangle_block = llvm::BasicBlock::Create(_llvm_context, "surface.filter.triangle", function);
                auto curve_block = llvm::BasicBlock::Create(_llvm_context, "surface.filter.curve", function);
                auto bary_block = llvm::BasicBlock::Create(_llvm_context, "surface.filter.bary", function);
                b.CreateCondBr(triangle, triangle_block, curve_block);
                b.SetInsertPoint(triangle_block);
                auto triangle_bary = _call_optix_hit_object_triangle_barycentrics(b);
                b.CreateBr(bary_block);
                b.SetInsertPoint(curve_block);
                auto curve_bary = _create_llvm_vector(b, {_call_optix_hit_object_curve_parameter(b),
                                                          llvm::ConstantFP::get(b.getFloatTy(), -1.)});
                b.CreateBr(bary_block);
                b.SetInsertPoint(bary_block);
                auto bary_phi = b.CreatePHI(triangle_bary->getType(), 2u);
                bary_phi->addIncoming(triangle_bary, triangle_block);
                bary_phi->addIncoming(curve_bary, curve_block);
                return bary_phi;
            } else {
                return _call_optix_hit_object_triangle_barycentrics(b);
            }
        };
        llvm::Value *bary;
        llvm::Value *hit_type;
        if (!has_procedural) {
            // Preserve the simpler decoder when custom IS cannot report a hit.
            bary = emit_surface_bary();
            hit_type = b.getInt32(static_cast<uint32_t>(HitType::Surface));
        } else {
            auto surface = b.CreateICmpUGT(kind, b.getInt32(127u));
            auto surface_block = llvm::BasicBlock::Create(_llvm_context, "hardware.result.surface", function);
            auto procedural_block = llvm::BasicBlock::Create(_llvm_context, "hardware.result.procedural", function);
            auto bary_block = llvm::BasicBlock::Create(_llvm_context, "hardware.result.bary", function);
            b.CreateCondBr(surface, surface_block, procedural_block);
            b.SetInsertPoint(surface_block);
            auto surface_bary = emit_surface_bary();
            auto surface_exit = b.GetInsertBlock();
            b.CreateBr(bary_block);
            b.SetInsertPoint(procedural_block);
            auto procedural_bary = llvm::Constant::getNullValue(surface_bary->getType());
            b.CreateBr(bary_block);
            b.SetInsertPoint(bary_block);
            auto bary_phi = b.CreatePHI(surface_bary->getType(), 2u);
            bary_phi->addIncoming(surface_bary, surface_exit);
            bary_phi->addIncoming(procedural_bary, procedural_block);
            bary = bary_phi;
            hit_type = b.CreateSelect(surface, b.getInt32(static_cast<uint32_t>(HitType::Surface)),
                                     b.getInt32(static_cast<uint32_t>(HitType::Procedural)));
        }
        auto t = _call_optix_hit_object_ray_t_max(b);
        auto hit = static_cast<llvm::Value *>(llvm::Constant::getNullValue(_get_llvm_committed_hit_type()));
        hit = b.CreateInsertValue(hit, _call_optix_hit_object_instance_index(b), llvm_committed_hit_type_inst_id_index);
        hit = b.CreateInsertValue(hit, _call_optix_hit_object_primitive_index(b), llvm_committed_hit_type_prim_id_index);
        hit = b.CreateInsertValue(hit, bary, llvm_committed_hit_type_bary_index);
        hit = b.CreateInsertValue(hit, hit_type, llvm_committed_hit_type_hit_kind_index);
        hit = b.CreateInsertValue(hit, t, llvm_committed_hit_type_t_index);
        _store_ray_query_field(b, query, llvm_ray_query_type_hit_index, hit);
        _store_ray_query_field(b, query, llvm_ray_query_type_ray_index,
                               b.CreateInsertValue(ray, t, llvm_ray_type_t_max_index));
        b.CreateBr(merge_block);
        b.SetInsertPoint(merge_block);
    }
    // Explicit commits are authoritative. OptiX's terminating intersection may
    // be synthetic when terminate() is used without accepting the candidate.
    _call_optix_hit_object_reset(b);
    _store_ray_query_field(b, query, llvm_ray_query_type_state_index, b.getInt8(llvm_ray_query_state_surface_terminated));
}

void CUDACodegenLLVMImpl::_materialize_ray_query_pipelines() noexcept {
    // A payload type has one capacity for the whole module. Each pipeline
    // uses id, an optional query pointer, then captures; pad only to this capacity.
    // The private OptiX intrinsic still has its fixed 32-word signature.
    auto i32 = llvm::Type::getInt32Ty(_llvm_context);
    for (auto call : _ray_query_trace_calls) {
        constexpr auto payload_count_arg = 16u;
        constexpr auto payload_first_arg = 17u;
        auto count = static_cast<uint32_t>(llvm::cast<llvm::ConstantInt>(call->getArgOperand(payload_count_arg))->getZExtValue());
        LUISA_ASSERT(count <= _ray_query_payload_count && _ray_query_payload_count <= kRayQueryPayloadWordCount,
                     "Invalid ray-query payload capacity.");
        call->setArgOperand(payload_count_arg, llvm::ConstantInt::get(i32, _ray_query_payload_count));
        for (auto i = count; i < _ray_query_payload_count; i++) {
            call->setArgOperand(payload_first_arg + i, llvm::ConstantInt::get(i32, 0u));
        }
    }
    auto needs_pipeline_id = _rt_analysis.ray_query_pipeline_count > 1u;
    auto active_pipeline_count = 0u;
    auto any_procedural_handler = false;
    for (auto &&pipeline : _ray_query_pipelines) {
        if (pipeline.disable_anyhit) { continue; }
        active_pipeline_count++;
        if (!ray_query_handler_is_empty(pipeline.inst->on_procedural_function())) {
            any_procedural_handler = true;
        }
    }
    for (auto procedural : {false, true}) {
        auto name = procedural ? "__intersection__ray_query" : "__anyhit__ray_query";
        auto function = llvm::Function::Create(llvm::FunctionType::get(llvm::Type::getVoidTy(_llvm_context), false),
                                               llvm::Function::ExternalLinkage, name, _llvm_module.get());
        function->setCallingConv(llvm::CallingConv::PTX_Kernel);
        auto entry = llvm::BasicBlock::Create(_llvm_context, "entry", function);
        IB b{entry};
        // Dead queries still require these entry points because host metadata
        // describes the original kernel rather than the optimized XIR module.
        if (active_pipeline_count == 0u || (procedural && !any_procedural_handler)) {
            b.CreateRetVoid();
            continue;
        }
        b.CreateCall(_get_inline_asm("call (), _optix_set_payload_types, ($0);", "r", true), {b.getInt32(OPTIX_PAYLOAD_TYPE_ID_1)});
        auto exit = llvm::BasicBlock::Create(_llvm_context, "exit", function);
        auto terminate = llvm::BasicBlock::Create(_llvm_context, "terminate", function);
        auto dispatch = llvm::BasicBlock::Create(_llvm_context, "dispatch", function);
        auto finish = llvm::BasicBlock::Create(_llvm_context, "candidate.finish", function);
        auto entry_dispatch_block = entry;
        if (!procedural && any_procedural_handler) {
            auto surface = llvm::BasicBlock::Create(_llvm_context, "surface", function);
            auto reported = llvm::BasicBlock::Create(_llvm_context, "reported", function);
            auto hit_kind = _call_optix_get_hit_kind(b);
            b.CreateCondBr(b.CreateICmpUGT(hit_kind, b.getInt32(127u)), surface, reported);
            b.SetInsertPoint(reported);
            // Custom IS reports carry termination in the hit kind. The explicit
            // query hit remains authoritative, including synthetic termination.
            b.CreateCondBr(b.CreateICmpEQ(hit_kind, b.getInt32(ray_query_procedural_terminated_hit_kind)), terminate, exit);
            b.SetInsertPoint(surface);
            entry_dispatch_block = surface;
        }
        auto get_payload = _get_inline_asm("call ($0), _optix_get_payload, ($1);", "=r,r", true);
        auto set_payload = _get_inline_asm("call (), _optix_set_payload, ($0, $1);", "r,r", true);
        llvm::Value *id = nullptr;
        auto load_pipeline_id = [&]() noexcept -> llvm::Value * {
            if (id == nullptr) {
                id = b.CreateCall(get_payload, {b.getInt32(0u)});
            }
            return id;
        };
        auto recover_local_ptr = [&](llvm::Value *ptr_val) noexcept -> llvm::Value * {
            auto local_ptr = b.CreateAddrSpaceCast(ptr_val, b.getPtrTy(nvptx_address_space_local));
            return b.CreateAddrSpaceCast(local_ptr, b.getPtrTy());
        };
        auto emit_callback = [&](const RayQueryPipeline &pipeline, const xir::Function *callback, llvm::Value *query) noexcept {
            if (callback != nullptr) {
                auto callee = _get_or_declare_llvm_function(callback);
                LUISA_ASSERT(callee->arg_size() == pipeline.inst->captured_argument_count() + 3u,
                             "Invalid ray-query callback capture ABI.");
                llvm::SmallVector<llvm::Value *> args{query};
                auto word_index = pipeline.hardware_result ? (needs_pipeline_id ? 1u : 0u) : (needs_pipeline_id ? 3u : 2u);
                auto direct_payload = pipeline.direct_payload;
                llvm::Value *context_local = nullptr;
                if (!direct_payload) {
                    auto context_generic = unpack_ray_query_payload(b, b.getPtrTy(), get_payload, word_index);
                    context_local = b.CreateAddrSpaceCast(context_generic, b.getPtrTy(nvptx_address_space_local));
                }
                struct RWWriteback {
                    llvm::AllocaInst *local_alloca;
                    llvm::Type *elem_type;
                    uint32_t word_offset;
                    uint32_t word_count;
                };
                llvm::SmallVector<RWWriteback> rw_writebacks;
                for (auto &&capture : pipeline.captures) {
                    if (capture.unused) {
                        args.emplace_back(llvm::PoisonValue::get(callee->getArg(args.size())->getType()));
                    } else if (auto root = capture.root_resource) {
                        auto iter = _ray_query_kernel_resources.find(root);
                        LUISA_ASSERT(iter != _ray_query_kernel_resources.end(), "Missing ray-query root resource parameter.");
                        auto type = _get_llvm_type(root->type());
                        LUISA_ASSERT(type->mem_type == type->reg_type, "Invalid resource descriptor ABI.");
                        // Reload the descriptor, never the contents of the resource.
                        // This preserves bound view offsets and sizes in AH and IS.
                        auto value = b.CreateAlignedLoad(type->mem_type, iter->second.pointer,
                                                         llvm::Align{KernelArgumentStruct::argument_alignment}, "query.resource");
                        _assume_kernel_argument_storage(b, value, root, iter->second.argument_index);
                        args.emplace_back(value);
                    } else if (direct_payload && (capture.by_value_ro || capture.by_value_rw)) {
                        auto start_word = word_index;
                        auto unpacked = unpack_ray_query_payload(b, capture.value_type, get_payload, word_index);
                        IB alloca_b{entry, entry->begin()};
                        auto local_slot = alloca_b.CreateAlloca(capture.value_type, nullptr, "rq.capture.slot");
                        b.CreateStore(unpacked, local_slot);
                        auto slot_arg = local_slot->getAddressSpace() == 0u ? static_cast<llvm::Value *>(local_slot) :
                                                                                b.CreateAddrSpaceCast(local_slot, b.getPtrTy());
                        args.emplace_back(slot_arg);
                        if (capture.by_value_rw) {
                            rw_writebacks.emplace_back(RWWriteback{local_slot, capture.value_type, start_word, word_index - start_word});
                        }
                    } else if (!direct_payload && capture.staged_inline) {
                        auto slot = b.CreateStructGEP(pipeline.context_type, context_local, capture.context_field);
                        args.emplace_back(b.CreateAddrSpaceCast(slot, b.getPtrTy()));
                    } else {
                        auto field_index = capture.context_field;
                        auto field_type = pipeline.context_type->getElementType(field_index);
                        auto value = direct_payload ?
                                         unpack_ray_query_payload(b, field_type, get_payload, word_index) :
                                         b.CreateLoad(field_type, b.CreateStructGEP(pipeline.context_type, context_local, field_index));
                        if (capture.is_local_ptr) { value = recover_local_ptr(value); }
                        args.emplace_back(value);
                    }
                }
                // These hidden arguments are immutable for the whole launch,
                // including callbacks and every transitive callable. Reload
                // them here instead of spilling a copy in every query context.
                LUISA_ASSERT(_llvm_ray_tracing_kernel_id_pointer != nullptr,
                             "Missing OptiX launch parameter ABI.");
                args.emplace_back(_read_optix_launch_size(b));
                args.emplace_back(b.CreateLoad(b.getInt32Ty(), _llvm_ray_tracing_kernel_id_pointer));
                auto call = b.CreateCall(callee, args);
                call->setCallingConv(callee->getCallingConv());
                if (!callee->hasFnAttribute(llvm::Attribute::NoInline)) {
                    call->addFnAttr(llvm::Attribute::AlwaysInline);
                }
                for (auto &&wb : rw_writebacks) {
                    auto updated = b.CreateLoad(wb.elem_type, wb.local_alloca);
                    llvm::SmallVector<llvm::Value *, 8u> words;
                    pack_ray_query_payload(b, updated, words);
                    for (auto k = 0u; k < wb.word_count; k++) {
                        b.CreateCall(set_payload, {b.getInt32(wb.word_offset + k), words[k]});
                    }
                }
            }
        };
        auto hardware_result_count = 0u;
        for (auto &&pipeline : _ray_query_pipelines) {
            if (!pipeline.disable_anyhit && pipeline.hardware_result) {
                hardware_result_count++;
            }
        }
        if (hardware_result_count != 0u) {
            auto general = llvm::BasicBlock::Create(_llvm_context, "general.query", function);
            RayQueryCallbackSharing shared_filters{function};
            struct RootResourceTarget {
                const llvm::Function *callee;
                llvm::SmallVector<const xir::Argument *, 4u> roots;
                llvm::BasicBlock *block;
            };
            llvm::SmallVector<RootResourceTarget, 4u> root_resource_targets;
            llvm::SmallVector<std::pair<uint32_t, llvm::BasicBlock *>, 4u> hw_cases;
            hw_cases.reserve(hardware_result_count);
            auto all_same_target = true;
            llvm::BasicBlock *common_target = nullptr;
            for (auto i = 0u; i < _ray_query_pipelines.size(); i++) {
                auto &&pipeline = _ray_query_pipelines[i];
                if (pipeline.disable_anyhit || !pipeline.hardware_result) { continue; }
                auto callback = procedural ? pipeline.inst->on_procedural_function() : pipeline.inst->on_surface_function();
                auto callee = callback == nullptr ? nullptr : _get_or_declare_llvm_function(callback);
                // With no captures the complete dispatch body uses only fresh
                // query state and this launch's implicit OptiX state. Keep the
                // procedural empty-handler shortcut outside the sharing table.
                auto empty_procedural = procedural && ray_query_handler_is_empty(callback);
                if (empty_procedural) {
                    hw_cases.emplace_back(i, exit);
                    common_target = common_target == nullptr ? exit : common_target;
                    all_same_target &= (common_target == exit);
                    continue;
                }
                auto has_root_captures = false;
                for (auto &&capture : pipeline.captures) {
                    if (capture.root_resource != nullptr) {
                        has_root_captures = true;
                        break;
                    }
                }
                auto only_root_captures =
                    has_root_captures &&
                    pipeline.context_type->getNumElements() == context_capture_offset;
                if (only_root_captures) {
                    llvm::SmallVector<const xir::Argument *, 4u> roots;
                    roots.reserve(pipeline.captures.size());
                    for (auto &&capture : pipeline.captures) {
                        roots.emplace_back(capture.root_resource);
                    }
                    llvm::BasicBlock *reused = nullptr;
                    for (auto &&existing : root_resource_targets) {
                        if (existing.callee == callee && existing.roots == roots) {
                            reused = existing.block;
                            break;
                        }
                    }
                    if (reused != nullptr) {
                        hw_cases.emplace_back(i, reused);
                        common_target = common_target == nullptr ? reused : common_target;
                        all_same_target &= (common_target == reused);
                        continue;
                    }
                }
                auto shareable_body = !has_root_captures &&
                                      pipeline.context_type->getNumElements() == context_capture_offset;
                auto target = shared_filters.get_or_create(callee, shareable_body,
                                                            pipeline.inst->captured_argument_count(),
                                                            "surface.filter." + std::to_string(i));
                if (only_root_captures && target.created) {
                    llvm::SmallVector<const xir::Argument *, 4u> roots;
                    roots.reserve(pipeline.captures.size());
                    for (auto &&capture : pipeline.captures) {
                        roots.emplace_back(capture.root_resource);
                    }
                    root_resource_targets.emplace_back(RootResourceTarget{callee, std::move(roots), target.block});
                }
                hw_cases.emplace_back(i, target.block);
                common_target = common_target == nullptr ? target.block : common_target;
                all_same_target &= (common_target == target.block);
                if (!target.created) { continue; }
                b.SetInsertPoint(target.block);
                // Callbacks may also serve general queries. Give this invocation
                // local state; inlining/SROA retains its live bound and acceptance
                // fields. Captured references keep their original identities.
                IB alloca_b{entry, entry->begin()};
                auto local_query = alloca_b.CreateAlloca(_get_llvm_ray_query_type(), nullptr, "surface.filter.query");
                b.CreateStore(llvm::Constant::getNullValue(_get_llvm_ray_query_type()), local_query);
                llvm::DenseSet<const xir::Function *> ray_visited;
                auto callback_query = (callback != nullptr && !callback->arguments().empty()) ?
                                          callback->arguments().front() :
                                          nullptr;
                if (procedural || ray_query_handler_reads_world_ray(callback, callback_query, ray_visited)) {
                    _store_ray_query_field(b, local_query, llvm_ray_query_type_ray_index, _call_optix_get_world_space_ray(b));
                }
                auto query_arg = local_query->getAddressSpace() == 0u ? static_cast<llvm::Value *>(local_query) :
                                                                        b.CreateAddrSpaceCast(local_query, b.getPtrTy());
                emit_callback(pipeline, callback, query_arg);
                auto accepted = _load_ray_query_field(b, local_query, llvm_ray_query_type_committed_index);
                if (procedural) {
                    auto report = llvm::BasicBlock::Create(_llvm_context, "hardware.result.report", function);
                    b.CreateCondBr(accepted, report, exit);
                    b.SetInsertPoint(report);
                    // Invalid commits do not erase an earlier valid local hit.
                    // One report publishes the callback's final committed bound.
                    auto hit = _load_ray_query_field(b, local_query, llvm_ray_query_type_hit_index);
                    auto t = b.CreateExtractValue(hit, llvm_committed_hit_type_t_index);
                    _call_optix_report_intersection(b, b.getInt32(ray_query_procedural_hit_kind), t);
                    b.CreateBr(exit);
                } else {
                    auto accept_filter = llvm::BasicBlock::Create(_llvm_context, "surface.filter.accept", function);
                    auto ignore_filter = llvm::BasicBlock::Create(_llvm_context, "surface.filter.ignore", function);
                    b.CreateCondBr(accepted, accept_filter, ignore_filter);
                    b.SetInsertPoint(accept_filter);
                    auto terminated = _load_ray_query_field(b, local_query, llvm_ray_query_type_terminated_index);
                    b.CreateCondBr(terminated, terminate, exit);
                    b.SetInsertPoint(ignore_filter);
                    _call_optix_ignore_intersection(b);
                    b.CreateRetVoid();
                }
            }
            if (shared_filters.shared_count() != 0u) {
                LUISA_VERBOSE("CUDA LLVM ray query: shared {} zero-capture {} callback dispatch targets.",
                              shared_filters.shared_count(), procedural ? "procedural" : "surface");
            }
            // Emit the entry branch or switch after target blocks exist.
            b.SetInsertPoint(entry_dispatch_block);
            if (hardware_result_count == active_pipeline_count && all_same_target && common_target != nullptr) {
                b.CreateBr(common_target);
                general->eraseFromParent();
                dispatch->eraseFromParent();
                finish->eraseFromParent();
                b.SetInsertPoint(terminate);
                if (procedural) {
                    b.CreateUnreachable();
                } else {
                    _call_optix_terminate_ray(b);
                    b.CreateRetVoid();
                }
                b.SetInsertPoint(exit);
                b.CreateRetVoid();
                continue;
            }
            auto select_filter = b.CreateSwitch(load_pipeline_id(), general, hardware_result_count);
            for (auto &&[case_id, target_block] : hw_cases) {
                select_filter->addCase(b.getInt32(case_id), target_block);
            }
            b.SetInsertPoint(general);
            if (hardware_result_count == active_pipeline_count) {
                b.CreateUnreachable();
                dispatch->eraseFromParent();
                finish->eraseFromParent();
                b.SetInsertPoint(terminate);
                if (procedural) {
                    b.CreateUnreachable();
                } else {
                    _call_optix_terminate_ray(b);
                    b.CreateRetVoid();
                }
                b.SetInsertPoint(exit);
                b.CreateRetVoid();
                continue;
            }
        }
        auto prefix_word_index = needs_pipeline_id ? 1u : 0u;
        auto query = unpack_ray_query_payload(b, b.getPtrTy(), get_payload, prefix_word_index);
        auto all_general_local = true;
        for (auto &&pipeline : _ray_query_pipelines) {
            if (!pipeline.disable_anyhit && !pipeline.hardware_result && !pipeline.query_is_local_ptr) {
                all_general_local = false;
                break;
            }
        }
        if (all_general_local) { query = recover_local_ptr(query); }
        if (!procedural) {
            _store_ray_query_field(b, query, llvm_ray_query_type_committed_index, b.getFalse());
            _store_ray_query_field(b, query, llvm_ray_query_type_state_index, b.getInt8(llvm_ray_query_state_surface_candidate));
            auto accel = _load_ray_query_field(b, query, llvm_ray_query_type_accel_index);
            auto instance = _get_accel_instance_pointer(b, accel, _call_optix_read_instance_index(b));
            auto flags = b.CreateLoad(b.getInt32Ty(), b.CreateStructGEP(_get_llvm_accel_instance_type(), instance, llvm_accel_instance_type_flags_index));
            auto opaque = b.CreateICmpEQ(b.CreateAnd(flags, b.getInt32(OPTIX_INSTANCE_FLAG_ENFORCE_ANYHIT)), b.getInt32(0));
            auto accept_opaque = llvm::BasicBlock::Create(_llvm_context, "opaque", function);
            b.CreateCondBr(opaque, accept_opaque, dispatch);
            b.SetInsertPoint(accept_opaque);
            _commit_ray_query_hit(b, query, _call_optix_get_hit_distance(b), false);
            b.CreateBr(finish);
        } else {
            _store_ray_query_field(b, query, llvm_ray_query_type_committed_index, b.getFalse());
            _store_ray_query_field(b, query, llvm_ray_query_type_state_index, b.getInt8(llvm_ray_query_state_procedural_candidate));
            b.CreateBr(dispatch);
        }
        b.SetInsertPoint(dispatch);
        auto general_count = active_pipeline_count - hardware_result_count;
        if (general_count == 1u) {
            for (auto i = 0u; i < _ray_query_pipelines.size(); i++) {
                auto &&pipeline = _ray_query_pipelines[i];
                if (pipeline.disable_anyhit || pipeline.hardware_result) { continue; }
                auto callback = procedural ? pipeline.inst->on_procedural_function() : pipeline.inst->on_surface_function();
                emit_callback(pipeline, callback, query);
                b.CreateBr(finish);
                break;
            }
        } else {
            auto invalid = llvm::BasicBlock::Create(_llvm_context, "invalid.context", function);
            auto select = b.CreateSwitch(load_pipeline_id(), invalid, static_cast<unsigned>(general_count));
            for (auto i = 0u; i < _ray_query_pipelines.size(); i++) {
                auto &&pipeline = _ray_query_pipelines[i];
                if (pipeline.disable_anyhit || pipeline.hardware_result) { continue; }
                auto block = llvm::BasicBlock::Create(_llvm_context, "pipeline." + std::to_string(i), function);
                select->addCase(b.getInt32(i), block);
                b.SetInsertPoint(block);
                auto callback = procedural ? pipeline.inst->on_procedural_function() : pipeline.inst->on_surface_function();
                emit_callback(pipeline, callback, query);
                b.CreateBr(finish);
            }
            b.SetInsertPoint(invalid);
            b.CreateUnreachable();
        }
        b.SetInsertPoint(finish);
        auto committed = _load_ray_query_field(b, query, llvm_ray_query_type_committed_index);
        auto terminated = _load_ray_query_field(b, query, llvm_ray_query_type_terminated_index);
        if (procedural) {
            auto report = llvm::BasicBlock::Create(_llvm_context, "report", function);
            b.CreateCondBr(b.CreateOr(committed, terminated), report, exit);
            b.SetInsertPoint(report);
            auto hit = _load_ray_query_field(b, query, llvm_ray_query_type_hit_index);
            auto ray = _load_ray_query_field(b, query, llvm_ray_query_type_ray_index);
            // Reporting TMin lets AH terminate an uncommitted procedural query.
            // Its synthetic hit never replaces the explicit committed result.
            auto t = b.CreateSelect(committed, b.CreateExtractValue(hit, llvm_committed_hit_type_t_index),
                                    b.CreateExtractValue(ray, llvm_ray_type_t_min_index));
            auto hit_kind = b.CreateSelect(terminated, b.getInt32(ray_query_procedural_terminated_hit_kind),
                                           b.getInt32(ray_query_procedural_hit_kind));
            _call_optix_report_intersection(b, hit_kind, t);
            b.CreateBr(exit);
            b.SetInsertPoint(terminate);
            b.CreateUnreachable();
        } else {
            auto accept = llvm::BasicBlock::Create(_llvm_context, "accept", function);
            auto ignore = llvm::BasicBlock::Create(_llvm_context, "ignore", function);
            b.CreateCondBr(terminated, terminate, accept);
            b.SetInsertPoint(accept);
            b.CreateCondBr(committed, exit, ignore);
            b.SetInsertPoint(ignore);
            _call_optix_ignore_intersection(b);
            b.CreateRetVoid();
            b.SetInsertPoint(terminate);
            _call_optix_terminate_ray(b);
            b.CreateRetVoid();
        }
        b.SetInsertPoint(exit);
        b.CreateRetVoid();
    }
}

}// namespace luisa::compute::cuda
