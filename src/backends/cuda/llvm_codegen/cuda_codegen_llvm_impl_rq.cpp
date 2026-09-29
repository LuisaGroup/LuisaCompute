#include <luisa/core/logging.h>
#include <luisa/dsl/rtx/ray_query.h>
#include <luisa/runtime/rtx/hit.h>

#include "../optix_api.h"
#include "cuda_codegen_llvm_impl.h"

namespace luisa::compute::cuda {

namespace {
constexpr auto context_capture_offset = 2u;
constexpr auto kRayQueryPayloadWordCount = 32u;
constexpr auto kRayQueryPayloadCaptureOffset = 3u;
constexpr auto kRayQueryPayloadCaptureWordCount = kRayQueryPayloadWordCount - kRayQueryPayloadCaptureOffset;
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
[[nodiscard]] uint32_t ray_query_payload_word_count(llvm::Type *type, const llvm::DataLayout &layout) noexcept {
    constexpr auto overflow = kRayQueryPayloadCaptureWordCount + 1u;
    if (auto structure = llvm::dyn_cast<llvm::StructType>(type)) {
        if (structure->isOpaque()) { return overflow; }
        auto count = 0u;
        for (auto field : structure->elements()) {
            auto words = ray_query_payload_word_count(field, layout);
            if (words > kRayQueryPayloadCaptureWordCount - count) { return overflow; }
            count += words;
        }
        return count;
    }
    auto repeated_count = [&layout](llvm::Type *element, uint64_t count) noexcept {
        if (count == 0u) { return 0u; }
        auto words = ray_query_payload_word_count(element, layout);
        if (words == 0u) { return 0u; }
        return count > kRayQueryPayloadCaptureWordCount / words ? overflow : static_cast<uint32_t>(count) * words;
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

[[nodiscard]] bool ray_query_uses_direct_payload(llvm::StructType *context_type, const llvm::DataLayout &layout) noexcept {
    auto count = 0u;
    for (auto i = context_capture_offset; i < context_type->getNumElements(); i++) {
        auto words = ray_query_payload_word_count(context_type->getElementType(i), layout);
        if (words > kRayQueryPayloadCaptureWordCount - count) { return false; }
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

[[nodiscard]] bool ray_query_handler_is_empty(const xir::Function *function) noexcept {
    if (function == nullptr) { return true; }
    auto definition = function->definition();
    if (definition == nullptr) { return false; }
    auto return_count = 0u;
    for (auto block : definition->basic_blocks()) {
        for (auto inst : block->instructions()) {
            if (inst->derived_instruction_tag() != xir::DerivedInstructionTag::RETURN ||
                static_cast<const xir::ReturnInst *>(inst)->return_value() != nullptr) { return false; }
            return_count++;
        }
    }
    return return_count == 1u;
}

[[nodiscard]] bool ray_query_handler_is_surface_filter(const xir::Function *function, const xir::Value *query,
                                                       llvm::DenseSet<const xir::Function *> &visited) noexcept {
    if (function == nullptr) { return true; }
    if (!visited.insert(function).second) { return false; }
    auto definition = function->definition();
    if (definition == nullptr) { return false; }
    if (query != nullptr) {
        // A cast, address calculation, store, or callable argument can expose
        // query state through an alias. Keep such handlers on the general path.
        for (auto use : query->use_list()) {
            auto user = use->user();
            if (user->derived_value_tag() != xir::DerivedValueTag::INSTRUCTION) { return false; }
            auto inst = static_cast<const xir::Instruction *>(user);
            if (inst->operand_count() != 1u || inst->operand(0) != query ||
                (inst->derived_instruction_tag() != xir::DerivedInstructionTag::RAY_QUERY_OBJECT_READ &&
                 inst->derived_instruction_tag() != xir::DerivedInstructionTag::RAY_QUERY_OBJECT_WRITE)) { return false; }
        }
    }
    for (auto block : definition->basic_blocks()) {
        for (auto inst : block->instructions()) {
            switch (inst->derived_instruction_tag()) {
                case xir::DerivedInstructionTag::RAY_QUERY_OBJECT_READ:
                    if (query == nullptr || inst->operand(0) != query ||
                        static_cast<const xir::RayQueryObjectReadInst *>(inst)->op() != xir::RayQueryObjectReadOp::RAY_QUERY_OBJECT_TRIANGLE_CANDIDATE_HIT) { return false; }
                    break;
                case xir::DerivedInstructionTag::RAY_QUERY_OBJECT_WRITE:
                    if (query == nullptr || inst->operand(0) != query ||
                        static_cast<const xir::RayQueryObjectWriteInst *>(inst)->op() != xir::RayQueryObjectWriteOp::RAY_QUERY_OBJECT_COMMIT_TRIANGLE) { return false; }
                    break;
                case xir::DerivedInstructionTag::CALL:
                    if (!ray_query_handler_is_surface_filter(static_cast<const xir::CallInst *>(inst)->callee(), nullptr, visited)) { return false; }
                    break;
                default: break;
            }
        }
    }
    visited.erase(function);
    return true;
}

[[nodiscard]] bool ray_query_pipeline_is_surface_filter(const xir::RayQueryPipelineInst *pipeline) noexcept {
    if (!ray_query_handler_is_empty(pipeline->on_procedural_function())) { return false; }
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
    auto surface = pipeline->on_surface_function();
    if (surface == nullptr) { return true; }
    if (surface->arguments().empty()) { return false; }
    llvm::DenseSet<const xir::Function *> visited;
    return ray_query_handler_is_surface_filter(surface, surface->arguments().front(), visited);
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
        auto triangle = b.CreateOr(b.CreateICmpEQ(kind, b.getInt32(optix::HIT_KIND_TRIANGLE_FRONT_FACE)),
                                   b.CreateICmpEQ(kind, b.getInt32(optix::HIT_KIND_TRIANGLE_BACK_FACE)));
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
    auto ray = _load_ray_query_field(b, query, llvm_ray_query_type_ray_index);
    auto t_min = b.CreateExtractValue(ray, llvm_ray_type_t_min_index);
    auto t_max = b.CreateExtractValue(ray, llvm_ray_type_t_max_index);
    auto valid = b.CreateAnd(b.CreateFCmpOGE(t, t_min), b.CreateFCmpOLE(t, t_max));
    auto hit = static_cast<llvm::Value *>(llvm::Constant::getNullValue(_get_llvm_committed_hit_type()));
    hit = b.CreateInsertValue(hit, _call_optix_read_instance_index(b), llvm_committed_hit_type_inst_id_index);
    hit = b.CreateInsertValue(hit, _call_optix_read_primitive_index(b), llvm_committed_hit_type_prim_id_index);
    if (!procedural) {
        auto surface = _ray_query_surface_candidate(b);
        hit = b.CreateInsertValue(hit, b.CreateExtractValue(surface, llvm_surface_hit_type_bary_index), llvm_committed_hit_type_bary_index);
    }
    hit = b.CreateInsertValue(hit, b.getInt32(static_cast<uint32_t>(procedural ? HitType::Procedural : HitType::Surface)), llvm_committed_hit_type_hit_kind_index);
    hit = b.CreateInsertValue(hit, t, llvm_committed_hit_type_t_index);
    auto previous = _load_ray_query_field(b, query, llvm_ray_query_type_hit_index);
    auto selected = static_cast<llvm::Value *>(llvm::Constant::getNullValue(_get_llvm_committed_hit_type()));
    for (auto field = 0u; field < 5u; field++) {
        selected = b.CreateInsertValue(selected, b.CreateSelect(valid, b.CreateExtractValue(hit, field), b.CreateExtractValue(previous, field)), field);
    }
    _store_ray_query_field(b, query, llvm_ray_query_type_hit_index, selected);
    _store_ray_query_field(b, query, llvm_ray_query_type_ray_index,
                           b.CreateInsertValue(ray, b.CreateSelect(valid, t, t_max), llvm_ray_type_t_max_index));
    auto committed = _load_ray_query_field(b, query, llvm_ray_query_type_committed_index);
    _store_ray_query_field(b, query, llvm_ray_query_type_committed_index, b.CreateOr(committed, valid));
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
    auto query = generic_pointer(_get_llvm_value(b, func_ctx, inst->query_object()));
    auto id = static_cast<uint32_t>(_ray_query_pipelines.size());
    llvm::SmallVector<llvm::Value *> fields{b.getInt32(id), query};
    for (auto capture : inst->captured_argument_uses()) {
        // Keep references as references, including aliases and subobject captures.
        fields.emplace_back(generic_pointer(_get_llvm_value(b, func_ctx, capture->value())));
    }
    llvm::SmallVector<llvm::Type *> field_types;
    for (auto field : fields) { field_types.emplace_back(field->getType()); }
    auto context_type = llvm::StructType::create(_llvm_context, field_types, "luisa.ray.query.context");
    auto surface_filter = ray_query_pipeline_is_surface_filter(inst);
    _ray_query_pipelines.emplace_back(RayQueryPipeline{inst, context_type, surface_filter});
    // Every pipeline shares the query pointer and id prefix. Captures either
    // occupy the remaining registers or live behind a generic context pointer.
    llvm::SmallVector<llvm::Value *, kRayQueryPayloadWordCount> payload;
    // A surface filter only needs AH-local acceptance state. Do not expose
    // the caller's query storage to traversal in this case.
    pack_ray_query_payload(b, surface_filter ? llvm::ConstantPointerNull::get(b.getPtrTy()) : query, payload);
    payload.emplace_back(b.getInt32(id));
    if (ray_query_uses_direct_payload(context_type, *_data_layout)) {
        for (auto i = context_capture_offset; i < fields.size(); i++) {
            pack_ray_query_payload(b, fields[i], payload);
        }
    } else {
        // Traversal invokes its handlers synchronously, and nested traversal in
        // a handler is rejected above. Only the fallback needs private scratch;
        // query objects and captured references retain their original identities.
        auto context_size = _data_layout->getTypeAllocSize(context_type).getFixedValue();
        auto &context = func_ctx.llvm_ray_query_context_scratch;
        if (context == nullptr) {
            IB alloca_b{func_ctx.llvm_alloca_block->getTerminator()};
            context = alloca_b.CreateAlloca(alloca_b.getInt8Ty(), alloca_b.getInt64(context_size), "ray.query.context.scratch");
            context->setAlignment(llvm::Align{16u});
        } else if (context_size > llvm::cast<llvm::ConstantInt>(context->getArraySize())->getZExtValue()) {
            context->setOperand(0, b.getInt64(context_size));
        }
        // A smaller later context may still require a wider vector alignment.
        context->setAlignment(std::max(context->getAlign(), _data_layout->getABITypeAlign(context_type)));
        for (auto i = context_capture_offset; i < fields.size(); i++) {
            b.CreateStore(fields[i], b.CreateStructGEP(context_type, context, i));
        }
        pack_ray_query_payload(b, generic_pointer(context), payload);
    }
    LUISA_ASSERT(payload.size() <= kRayQueryPayloadWordCount, "Ray-query payload exceeds the register budget.");
    auto accel = _load_ray_query_field(b, query, llvm_ray_query_type_accel_index);
    auto ray = _load_ray_query_field(b, query, llvm_ray_query_type_ray_index);
    auto time = _load_ray_query_field(b, query, llvm_ray_query_type_time_index);
    auto mask = _load_ray_query_field(b, query, llvm_ray_query_type_mask_index);
    // General handlers observe the committed bound while traversing, so they
    // must also track opaque hits. A pure surface filter cannot observe that
    // state and can recover the final hardware result after traversal instead.
    auto flags = static_cast<uint32_t>(optix::RAY_FLAG_DISABLE_CLOSESTHIT);
    if (!surface_filter) { flags |= optix::RAY_FLAG_ENFORCE_ANYHIT; }
    if (inst->query_object()->type() == Type::of<RayQueryAny>()) {
        flags |= optix::RAY_FLAG_TERMINATE_ON_FIRST_HIT;
    }
    _call_optix_trace(b, optix::PAYLOAD_TYPE_ID_1, 5u, flags, accel, ray, time, mask, payload);
    if (surface_filter) {
        auto function = b.GetInsertBlock()->getParent();
        auto hit_block = llvm::BasicBlock::Create(_llvm_context, "surface.filter.hit", function);
        auto merge_block = llvm::BasicBlock::Create(_llvm_context, "surface.filter.result", function);
        auto is_hit = _call_optix_hit_object_is_hit(b);
        // Hit-object attributes are only defined for an actual hit of the
        // corresponding primitive kind. A select cannot guard those getters.
        b.CreateCondBr(is_hit, hit_block, merge_block);
        b.SetInsertPoint(hit_block);
        llvm::Value *bary;
        if (_rt_analysis.curve_basis_set.any()) {
            auto kind = _call_optix_hit_object_hit_kind(b);
            auto triangle = b.CreateOr(b.CreateICmpEQ(kind, b.getInt32(optix::HIT_KIND_TRIANGLE_FRONT_FACE)),
                                       b.CreateICmpEQ(kind, b.getInt32(optix::HIT_KIND_TRIANGLE_BACK_FACE)));
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
            bary = bary_phi;
        } else {
            bary = _call_optix_hit_object_triangle_barycentrics(b);
        }
        auto t = _call_optix_hit_object_ray_t_max(b);
        auto hit = static_cast<llvm::Value *>(llvm::Constant::getNullValue(_get_llvm_committed_hit_type()));
        hit = b.CreateInsertValue(hit, _call_optix_hit_object_instance_index(b), llvm_committed_hit_type_inst_id_index);
        hit = b.CreateInsertValue(hit, _call_optix_hit_object_primitive_index(b), llvm_committed_hit_type_prim_id_index);
        hit = b.CreateInsertValue(hit, bary, llvm_committed_hit_type_bary_index);
        hit = b.CreateInsertValue(hit, b.getInt32(static_cast<uint32_t>(HitType::Surface)), llvm_committed_hit_type_hit_kind_index);
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
    // A payload type has one capacity for the whole module. Keep the stable
    // query-pointer/id/capture layout, but do not reserve unused tail words.
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
    for (auto procedural : {false, true}) {
        auto name = procedural ? "__intersection__ray_query" : "__anyhit__ray_query";
        auto function = llvm::Function::Create(llvm::FunctionType::get(llvm::Type::getVoidTy(_llvm_context), false),
                                               llvm::Function::ExternalLinkage, name, _llvm_module.get());
        function->setCallingConv(llvm::CallingConv::PTX_Kernel);
        auto entry = llvm::BasicBlock::Create(_llvm_context, "entry", function);
        IB b{entry};
        // Dead queries still require these entry points because host metadata
        // describes the original kernel rather than the optimized XIR module.
        if (_ray_query_pipelines.empty()) {
            b.CreateRetVoid();
            continue;
        }
        b.CreateCall(_get_inline_asm("call (), _optix_set_payload_types, ($0);", "r", true), {b.getInt32(optix::PAYLOAD_TYPE_ID_1)});
        auto exit = llvm::BasicBlock::Create(_llvm_context, "exit", function);
        auto terminate = llvm::BasicBlock::Create(_llvm_context, "terminate", function);
        auto dispatch = llvm::BasicBlock::Create(_llvm_context, "dispatch", function);
        auto finish = llvm::BasicBlock::Create(_llvm_context, "candidate.finish", function);
        if (!procedural) {
            auto surface = llvm::BasicBlock::Create(_llvm_context, "surface", function);
            auto reported = llvm::BasicBlock::Create(_llvm_context, "reported", function);
            auto hit_kind = _call_optix_get_hit_kind(b);
            b.CreateCondBr(b.CreateICmpUGT(hit_kind, b.getInt32(127u)), surface, reported);
            b.SetInsertPoint(reported);
            // Custom IS reports carry termination in the hit kind. The explicit
            // query hit remains authoritative, including synthetic termination.
            b.CreateCondBr(b.CreateICmpEQ(hit_kind, b.getInt32(ray_query_procedural_terminated_hit_kind)), terminate, exit);
            b.SetInsertPoint(surface);
        }
        auto get_payload = _get_inline_asm("call ($0), _optix_get_payload, ($1);", "=r,r", true);
        auto id = b.CreateCall(get_payload, {b.getInt32(2u)});
        auto emit_callback = [&](const RayQueryPipeline &pipeline, const xir::Function *callback, llvm::Value *query) noexcept {
            if (callback != nullptr) {
                auto callee = _get_or_declare_llvm_function(callback);
                LUISA_ASSERT(callee->arg_size() == pipeline.inst->captured_argument_count() + 3u,
                             "Invalid ray-query callback capture ABI.");
                llvm::SmallVector<llvm::Value *> args{query};
                auto word_index = kRayQueryPayloadCaptureOffset;
                auto direct_payload = ray_query_uses_direct_payload(pipeline.context_type, *_data_layout);
                auto context = direct_payload ? nullptr : unpack_ray_query_payload(b, b.getPtrTy(), get_payload, word_index);
                for (auto j = 0u; j < pipeline.inst->captured_argument_count(); j++) {
                    auto field_index = context_capture_offset + j;
                    auto field_type = pipeline.context_type->getElementType(field_index);
                    args.emplace_back(direct_payload ?
                                          unpack_ray_query_payload(b, field_type, get_payload, word_index) :
                                          b.CreateLoad(field_type, b.CreateStructGEP(pipeline.context_type, context, field_index)));
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
            }
        };
        auto surface_filter_count = 0u;
        for (auto &&pipeline : _ray_query_pipelines) {
            surface_filter_count += pipeline.surface_filter ? 1u : 0u;
        }
        if (surface_filter_count != 0u) {
            auto general = llvm::BasicBlock::Create(_llvm_context, "general.query", function);
            auto select_filter = b.CreateSwitch(id, general, surface_filter_count);
            for (auto i = 0u; i < _ray_query_pipelines.size(); i++) {
                auto &&pipeline = _ray_query_pipelines[i];
                if (!pipeline.surface_filter) { continue; }
                auto filter = llvm::BasicBlock::Create(_llvm_context, "surface.filter." + std::to_string(i), function);
                select_filter->addCase(b.getInt32(i), filter);
                b.SetInsertPoint(filter);
                if (procedural) {
                    // Qualification proves this outlined callback is empty.
                    b.CreateBr(exit);
                    continue;
                }
                // The ordinary callback can also serve general queries. Give
                // this invocation private state and let inlining/SROA retain
                // only its acceptance flag. Captured references are unchanged.
                IB alloca_b{entry, entry->begin()};
                auto local_query = alloca_b.CreateAlloca(_get_llvm_ray_query_type(), nullptr, "surface.filter.query");
                b.CreateStore(llvm::Constant::getNullValue(_get_llvm_ray_query_type()), local_query);
                _store_ray_query_field(b, local_query, llvm_ray_query_type_ray_index, _call_optix_get_world_space_ray(b));
                auto query_arg = local_query->getAddressSpace() == 0u ? static_cast<llvm::Value *>(local_query) :
                                                                        b.CreateAddrSpaceCast(local_query, b.getPtrTy());
                emit_callback(pipeline, pipeline.inst->on_surface_function(), query_arg);
                auto accepted = _load_ray_query_field(b, local_query, llvm_ray_query_type_committed_index);
                auto ignore_filter = llvm::BasicBlock::Create(_llvm_context, "surface.filter.ignore", function);
                b.CreateCondBr(accepted, exit, ignore_filter);
                b.SetInsertPoint(ignore_filter);
                _call_optix_ignore_intersection(b);
                b.CreateRetVoid();
            }
            b.SetInsertPoint(general);
            if (surface_filter_count == _ray_query_pipelines.size()) {
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
        auto prefix_word_index = 0u;
        auto query = unpack_ray_query_payload(b, b.getPtrTy(), get_payload, prefix_word_index);
        if (!procedural) {
            _store_ray_query_field(b, query, llvm_ray_query_type_committed_index, b.getFalse());
            _store_ray_query_field(b, query, llvm_ray_query_type_state_index, b.getInt8(llvm_ray_query_state_surface_candidate));
            auto accel = _load_ray_query_field(b, query, llvm_ray_query_type_accel_index);
            auto instance = _get_accel_instance_pointer(b, accel, _call_optix_read_instance_index(b));
            auto flags = b.CreateLoad(b.getInt32Ty(), b.CreateStructGEP(_get_llvm_accel_instance_type(), instance, llvm_accel_instance_type_flags_index));
            auto opaque = b.CreateICmpEQ(b.CreateAnd(flags, b.getInt32(optix::INSTANCE_FLAG_ENFORCE_ANYHIT)), b.getInt32(0));
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
        auto invalid = llvm::BasicBlock::Create(_llvm_context, "invalid.context", function);
        auto select = b.CreateSwitch(id, invalid, static_cast<unsigned>(_ray_query_pipelines.size()));
        for (auto i = 0u; i < _ray_query_pipelines.size(); i++) {
            auto &&pipeline = _ray_query_pipelines[i];
            if (pipeline.surface_filter) { continue; }
            auto block = llvm::BasicBlock::Create(_llvm_context, "pipeline." + std::to_string(i), function);
            select->addCase(b.getInt32(i), block);
            b.SetInsertPoint(block);
            auto callback = procedural ? pipeline.inst->on_procedural_function() : pipeline.inst->on_surface_function();
            emit_callback(pipeline, callback, query);
            b.CreateBr(finish);
        }
        b.SetInsertPoint(invalid);
        b.CreateUnreachable();
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
