#include <llvm/IR/Module.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Verifier.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IR/Intrinsics.h>
#include <llvm/IR/MatrixBuilder.h>
#include <llvm/IR/Metadata.h>
#include <llvm/IRReader/IRReader.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Analysis/TargetTransformInfo.h>
#include <llvm/ExecutionEngine/ExecutionEngine.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/Target/TargetOptions.h>
#include <llvm/Transforms/IPO.h>
#include <llvm/MC/TargetRegistry.h>

#include <luisa/core/stl/unordered_map.h>
#include <luisa/core/logging.h>
#include <luisa/runtime/rtx/hit.h>
#include <luisa/dsl/rtx/ray_query.h>
#include <luisa/xir/module.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/special_register.h>
#include <luisa/xir/metadata/name.h>
#include <luisa/xir/metadata/location.h>

#include "fallback_accel.h"
#include "fallback_bindless_array.h"
#include "fallback_buffer.h"
#include "fallback_texture.h"
#include "fallback_device_api.h"
#include "fallback_codegen.h"

namespace luisa::compute::fallback {

class FallbackCodegen {

private:
    struct LLVMStruct {
        llvm::StructType *type = nullptr;
        luisa::vector<uint> padded_field_indices{};
    };

    using IRBuilder = llvm::IRBuilder<>;

    struct CurrentFunction {
        llvm::Function *func = nullptr;
        llvm::BasicBlock *exit_block = nullptr;
        llvm::Value *coro_token = nullptr;
        llvm::Value *coro_handle = nullptr;
        llvm::BasicBlock *coro_suspend_block = nullptr;
        llvm::BasicBlock *coro_cleanup_block = nullptr;
        luisa::unordered_map<const xir::Value *, llvm::Value *> value_map{};
        luisa::unordered_set<const llvm::BasicBlock *> translated_basic_blocks{};
        luisa::unordered_map<llvm::BasicBlock *, llvm::BasicBlock *> phi_incoming_overrides{};
        luisa::vector<const xir::PhiInst *> phi_nodes{};

        // builtin variables
#define LUISA_FALLBACK_BACKEND_DECL_BUILTIN_VARIABLE(NAME, INDEX) \
    static constexpr size_t builtin_variable_index_##NAME = INDEX;
        LUISA_FALLBACK_BACKEND_DECL_BUILTIN_VARIABLE(thread_id, 0)
        LUISA_FALLBACK_BACKEND_DECL_BUILTIN_VARIABLE(block_id, 1)
        LUISA_FALLBACK_BACKEND_DECL_BUILTIN_VARIABLE(dispatch_id, 2)
        LUISA_FALLBACK_BACKEND_DECL_BUILTIN_VARIABLE(block_size, 3)
        LUISA_FALLBACK_BACKEND_DECL_BUILTIN_VARIABLE(dispatch_size, 4)
#undef LUISA_FALLBACK_BACKEND_DECL_BUILTIN_VARIABLE
        static constexpr size_t builtin_variable_count = 5;
        llvm::Value *builtin_variables[builtin_variable_count] = {};
    };

private:
    luisa::unordered_map<const xir::Function *, uint> _special_register_usages{};
    [[nodiscard]] uint _analyze_special_register_usage(const xir::Function *f) noexcept {
        if (auto iter = _special_register_usages.find(f); iter != _special_register_usages.end()) {
            return iter->second;
        }
        auto usage = 0u;
        if (auto def = f->definition()) {
            def->traverse_instructions([&](const xir::Instruction *inst) noexcept {
                for (auto op_use : inst->operand_uses()) {
                    if (auto op = op_use->value()) {
                        switch (op->derived_value_tag()) {
                            case xir::DerivedValueTag::FUNCTION: {
                                auto callee = static_cast<const xir::Function *>(op);
                                usage |= _analyze_special_register_usage(callee);
                                break;
                            }
                            case xir::DerivedValueTag::SPECIAL_REGISTER: {
                                auto sreg = static_cast<const xir::SpecialRegister *>(op);
                                switch (sreg->derived_special_register_tag()) {
                                    case xir::DerivedSpecialRegisterTag::THREAD_ID: usage |= (1u << CurrentFunction::builtin_variable_index_thread_id); break;
                                    case xir::DerivedSpecialRegisterTag::BLOCK_ID: usage |= (1u << CurrentFunction::builtin_variable_index_block_id); break;
                                    case xir::DerivedSpecialRegisterTag::WARP_LANE_ID: break;
                                    case xir::DerivedSpecialRegisterTag::DISPATCH_ID: usage |= (1u << CurrentFunction::builtin_variable_index_dispatch_id); break;
                                    case xir::DerivedSpecialRegisterTag::KERNEL_ID:
                                    case xir::DerivedSpecialRegisterTag::RASTER_BARYCENTRICS:
                                    case xir::DerivedSpecialRegisterTag::RASTER_OBJECT_ID:
                                        LUISA_NOT_IMPLEMENTED();
                                    case xir::DerivedSpecialRegisterTag::BLOCK_SIZE: usage |= (1u << CurrentFunction::builtin_variable_index_block_size); break;
                                    case xir::DerivedSpecialRegisterTag::WARP_SIZE: break;
                                    case xir::DerivedSpecialRegisterTag::DISPATCH_SIZE: usage |= (1u << CurrentFunction::builtin_variable_index_dispatch_size); break;
                                }
                                break;
                            }
                            default: break;
                        }
                    }
                }
            });
        }
        _special_register_usages.emplace(f, usage);
        return usage;
    }

    // check if the kernel requires block synchronization, which would require coroutine
    [[nodiscard]] bool _analyze_requires_sync_block(const xir::KernelFunction *kernel) noexcept {
        auto requires_sync_block = false;
        kernel->traverse_instructions([&](const xir::Instruction *inst) noexcept {
            if (inst->isa<xir::ThreadGroupInst>() &&
                static_cast<const xir::ThreadGroupInst *>(inst)->op() ==
                    xir::ThreadGroupOp::SYNCHRONIZE_BLOCK) {
                requires_sync_block = true;// TODO: we should early break the traversal here
            }
        });
        return requires_sync_block;
    }

private:
    llvm::LLVMContext &_llvm_context;
    llvm::Module *_llvm_module = nullptr;
    luisa::unordered_map<const Type *, luisa::unique_ptr<LLVMStruct>> _llvm_struct_types{};
    luisa::unordered_map<const xir::Constant *, llvm::Constant *> _llvm_constants{};
    luisa::unordered_map<const xir::Function *, llvm::Function *> _llvm_functions{};
    FallbackCodeGenFeedback::PrintInstMap _print_inst_map{};
    FallbackCodeGenFeedback::DebugCallbackMap _debug_callback_map{};
    size_t _tls_offset = 0u;

private:
    void _reset() noexcept {
        _llvm_module = nullptr;
        _llvm_struct_types.clear();
        _llvm_constants.clear();
        _llvm_functions.clear();
        _tls_offset = 0u;
    }

private:
    [[nodiscard]] static llvm::StringRef _get_name_from_metadata(auto something, llvm::StringRef fallback = {}) noexcept {
        auto name_md = something->template find_metadata<xir::NameMD>();
        return name_md ? llvm::StringRef{name_md->name()} : fallback;
    }

    [[nodiscard]] static size_t _get_type_size(const Type *t) noexcept {
        LUISA_ASSERT(t != nullptr, "Type is nullptr.");
        if (!t->is_resource() && !t->is_custom()) {
            return t->size();
        }
        switch (t->tag()) {
            case Type::Tag::BUFFER: return sizeof(FallbackBufferView);
            case Type::Tag::TEXTURE: return sizeof(FallbackTextureView);
            case Type::Tag::BINDLESS_ARRAY: return sizeof(FallbackBindlessArrayView);
            case Type::Tag::ACCEL: return sizeof(FallbackAccelView);
            case Type::Tag::CUSTOM: {
                if (t == Type::of<RayQueryAll>() || t == Type::of<RayQueryAny>()) {
                    return api::luisa_fallback_ray_query_object_size();
                }
                break;
            }
            default: break;
        }
        LUISA_ERROR_WITH_LOCATION("Invalid type: {}.", t->description());
    }

    [[nodiscard]] static size_t _get_type_alignment(const Type *t) noexcept {
        LUISA_ASSERT(t != nullptr, "Type is nullptr.");
        if (!t->is_resource() && !t->is_custom()) {
            return t->alignment();
        }
        switch (t->tag()) {
            case Type::Tag::BUFFER: return alignof(FallbackBufferView);
            case Type::Tag::TEXTURE: return alignof(FallbackTextureView);
            case Type::Tag::BINDLESS_ARRAY: return alignof(FallbackBindlessArrayView);
            case Type::Tag::ACCEL: return alignof(FallbackAccelView);
            case Type::Tag::CUSTOM: {
                if (t == Type::of<RayQueryAll>() || t == Type::of<RayQueryAny>()) {
                    return api::luisa_fallback_ray_query_object_alignment();
                }
                break;
            }
            default: break;
        }
        LUISA_ERROR_WITH_LOCATION("Invalid type: {}.", t->description());
    }

    static void _move_llvm_inst_to_block_begin(llvm::Instruction *inst, llvm::BasicBlock *block) noexcept {
        inst->moveBefore(*block, block->begin());
    }

    [[nodiscard]] LLVMStruct *_translate_struct_type(const Type *t) noexcept {
        auto iter = _llvm_struct_types.try_emplace(t, nullptr).first;
        if (iter->second) { return iter->second.get(); }
        auto struct_type = (iter->second = luisa::make_unique<LLVMStruct>()).get();
        auto member_index = 0u;
        llvm::SmallVector<::llvm::Type *> field_types;
        luisa::vector<uint> field_indices;
        size_t size = 0u;
        for (auto member : t->members()) {
            auto aligned_offset = luisa::align(size, _get_type_alignment(member));
            if (aligned_offset > size) {
                auto byte_type = ::llvm::Type::getInt8Ty(_llvm_context);
                auto padding = ::llvm::ArrayType::get(byte_type, aligned_offset - size);
                field_types.emplace_back(padding);
                member_index++;
            }
            auto member_type = _translate_type(member, false);
            field_types.emplace_back(member_type);
            field_indices.emplace_back(member_index++);
            size = aligned_offset + _get_type_size(member);
        }
        if (_get_type_size(t) > size) {// last padding
            auto byte_type = ::llvm::Type::getInt8Ty(_llvm_context);
            auto padding = ::llvm::ArrayType::get(byte_type, _get_type_size(t) - size);
            field_types.emplace_back(padding);
        }
        struct_type->type = ::llvm::StructType::get(_llvm_context, field_types);
        struct_type->padded_field_indices = std::move(field_indices);
        return struct_type;
    }

    [[nodiscard]] llvm::Type *_translate_type(const Type *t, bool register_use) noexcept {
        if (t == nullptr) { return llvm::Type::getVoidTy(_llvm_context); }
        switch (t->tag()) {
            case Type::Tag::BOOL: return llvm::Type::getInt8Ty(_llvm_context);
            case Type::Tag::FLOAT16: return llvm::Type::getHalfTy(_llvm_context);
            case Type::Tag::FLOAT32: return llvm::Type::getFloatTy(_llvm_context);
            case Type::Tag::FLOAT64: return llvm::Type::getDoubleTy(_llvm_context);
            case Type::Tag::INT8: return llvm::Type::getInt8Ty(_llvm_context);
            case Type::Tag::INT16: return llvm::Type::getInt16Ty(_llvm_context);
            case Type::Tag::INT32: return llvm::Type::getInt32Ty(_llvm_context);
            case Type::Tag::INT64: return llvm::Type::getInt64Ty(_llvm_context);
            case Type::Tag::UINT8: return llvm::Type::getInt8Ty(_llvm_context);
            case Type::Tag::UINT16: return llvm::Type::getInt16Ty(_llvm_context);
            case Type::Tag::UINT32: return llvm::Type::getInt32Ty(_llvm_context);
            case Type::Tag::UINT64: return llvm::Type::getInt64Ty(_llvm_context);
            case Type::Tag::VECTOR: {
                // note: we have to pad 3-element vectors to be 4-element
                auto elem_type = _translate_type(t->element(), false);
                auto dim = t->dimension();
                LUISA_ASSERT(dim == 2 || dim == 3 || dim == 4, "Invalid vector dimension.");
                if (register_use) { return llvm::VectorType::get(elem_type, dim, false); }
                return llvm::ArrayType::get(elem_type, dim == 3 ? 4 : dim);
            }
            case Type::Tag::MATRIX: {
                auto col_type = _translate_type(Type::vector(t->element(), t->dimension()), false);
                return llvm::ArrayType::get(col_type, t->dimension());
            }
            case Type::Tag::ARRAY: {
                auto elem_type = _translate_type(t->element(), false);
                return llvm::ArrayType::get(elem_type, t->dimension());
            }
            case Type::Tag::STRUCTURE: return _translate_struct_type(t)->type;
            case Type::Tag::BUFFER: {
                auto llvm_ptr_type = llvm::PointerType::get(_llvm_context, 0);
                auto llvm_i64_type = llvm::Type::getInt64Ty(_llvm_context);
                return llvm::StructType::get(_llvm_context, {llvm_ptr_type, llvm_i64_type});
            }
            case Type::Tag::TEXTURE: {
                auto llvm_ptr_type = llvm::PointerType::get(_llvm_context, 0);
                auto llvm_i64_type = llvm::Type::getInt64Ty(_llvm_context);
                return llvm::StructType::get(_llvm_context, {llvm_ptr_type, llvm_i64_type});
            }
            case Type::Tag::BINDLESS_ARRAY: {
                auto llvm_ptr_type = llvm::PointerType::get(_llvm_context, 0);
                auto llvm_i64_type = llvm::Type::getInt64Ty(_llvm_context);
                return llvm::StructType::get(_llvm_context, {llvm_ptr_type, llvm_i64_type});
            }
            case Type::Tag::ACCEL: {
                auto llvm_ptr_type = llvm::PointerType::get(_llvm_context, 0);
                return llvm::StructType::get(_llvm_context, {llvm_ptr_type, llvm_ptr_type});
            }
            case Type::Tag::CUSTOM: {
                if (t == Type::of<RayQueryAll>() || t == Type::of<RayQueryAny>()) {
                    auto size = api::luisa_fallback_ray_query_object_size();
                    LUISA_ASSERT(size % 16 == 0, "Ray query object size should be 16-byte aligned.");
                    auto llvm_i32_type = llvm::Type::getInt32Ty(_llvm_context);
                    auto llvm_i32x4_type = llvm::VectorType::get(llvm_i32_type, 4, false);
                    auto llvm_array_type = llvm::ArrayType::get(llvm_i32x4_type, size / 16);
                    return llvm::StructType::get(llvm_array_type);
                }
                break;
            }
            default: break;
        }
        LUISA_ERROR_WITH_LOCATION("Invalid type: {}.", t->description());
    }

    [[nodiscard]] llvm::Constant *_translate_literal(const Type *t, const void *data, bool register_use) noexcept {
        auto llvm_type = _translate_type(t, register_use);
        switch (t->tag()) {
            case Type::Tag::BOOL: return llvm::ConstantInt::get(llvm_type, *static_cast<const uint8_t *>(data));
            case Type::Tag::INT8: return llvm::ConstantInt::get(llvm_type, *static_cast<const int8_t *>(data));
            case Type::Tag::UINT8: return llvm::ConstantInt::get(llvm_type, *static_cast<const uint8_t *>(data));
            case Type::Tag::INT16: return llvm::ConstantInt::get(llvm_type, *static_cast<const int16_t *>(data));
            case Type::Tag::UINT16: return llvm::ConstantInt::get(llvm_type, *static_cast<const uint16_t *>(data));
            case Type::Tag::INT32: return llvm::ConstantInt::get(llvm_type, *static_cast<const int32_t *>(data));
            case Type::Tag::UINT32: return llvm::ConstantInt::get(llvm_type, *static_cast<const uint32_t *>(data));
            case Type::Tag::INT64: return llvm::ConstantInt::get(llvm_type, *static_cast<const int64_t *>(data));
            case Type::Tag::UINT64: return llvm::ConstantInt::get(llvm_type, *static_cast<const uint64_t *>(data));
            case Type::Tag::FLOAT16: return llvm::ConstantFP::get(llvm_type, *static_cast<const half *>(data));
            case Type::Tag::FLOAT32: return llvm::ConstantFP::get(llvm_type, *static_cast<const float *>(data));
            case Type::Tag::FLOAT64: return llvm::ConstantFP::get(llvm_type, *static_cast<const double *>(data));
            case Type::Tag::VECTOR: {
                auto elem_type = t->element();
                auto stride = _get_type_size(elem_type);
                auto dim = t->dimension();
                llvm::SmallVector<llvm::Constant *, 4u> elements;
                for (auto i = 0u; i < dim; i++) {
                    auto elem_data = static_cast<const std::byte *>(data) + i * stride;
                    elements.emplace_back(_translate_literal(elem_type, elem_data, false));
                }
                // for register use, we create an immediate vector
                if (register_use) {
                    return llvm::ConstantVector::get(elements);
                }
                // for memory use we have to pad 3-element vectors to 4-element
                if (dim == 3) {
                    auto llvm_elem_type = _translate_type(t->element(), false);
                    auto llvm_padding = llvm::Constant::getNullValue(llvm_elem_type);
                    elements.emplace_back(llvm_padding);
                }
                return llvm::ConstantArray::get(llvm::cast<llvm::ArrayType>(llvm_type), elements);
            }
            case Type::Tag::MATRIX: {
                LUISA_ASSERT(llvm_type->isArrayTy(), "Matrix type should be an array type.");
                auto dim = t->dimension();
                auto col_type = Type::vector(t->element(), dim);
                auto col_stride = _get_type_size(col_type);
                llvm::SmallVector<llvm::Constant *, 4u> elements;
                for (auto i = 0u; i < dim; i++) {
                    auto col_data = static_cast<const std::byte *>(data) + i * col_stride;
                    elements.emplace_back(_translate_literal(col_type, col_data, false));
                }
                return llvm::ConstantArray::get(llvm::cast<llvm::ArrayType>(llvm_type), elements);
            }
            case Type::Tag::ARRAY: {
                LUISA_ASSERT(llvm_type->isArrayTy(), "Array type should be an array type.");
                auto elem_type = t->element();
                auto stride = _get_type_size(elem_type);
                auto dim = t->dimension();
                llvm::SmallVector<llvm::Constant *> elements;
                elements.reserve(dim);
                for (auto i = 0u; i < dim; i++) {
                    auto elem_data = static_cast<const std::byte *>(data) + i * stride;
                    elements.emplace_back(_translate_literal(elem_type, elem_data, false));
                }
                return llvm::ConstantArray::get(llvm::cast<llvm::ArrayType>(llvm_type), elements);
            }
            case Type::Tag::STRUCTURE: {
                auto struct_type = _translate_struct_type(t);
                LUISA_ASSERT(llvm_type == struct_type->type, "Type mismatch.");
                auto padded_field_types = struct_type->type->elements();
                llvm::SmallVector<llvm::Constant *, 16u> fields;
                fields.resize(padded_field_types.size(), nullptr);
                LUISA_ASSERT(t->members().size() == struct_type->padded_field_indices.size(),
                             "Member count mismatch.");
                // fill in data fields
                size_t data_offset = 0u;
                for (auto i = 0u; i < t->members().size(); i++) {
                    auto field_type = t->members()[i];
                    data_offset = luisa::align(data_offset, _get_type_alignment(field_type));
                    auto field_data = static_cast<const std::byte *>(data) + data_offset;
                    data_offset += _get_type_size(field_type);
                    auto padded_index = struct_type->padded_field_indices[i];
                    fields[padded_index] = _translate_literal(field_type, field_data, false);
                }
                // fill in padding fields
                for (auto i = 0u; i < fields.size(); i++) {
                    if (fields[i] == nullptr) {
                        auto field_type = padded_field_types[i];
                        LUISA_ASSERT(field_type->isArrayTy(), "Padding field should be an array type.");
                        fields[i] = llvm::ConstantAggregateZero::get(field_type);
                    }
                }
                return llvm::ConstantStruct::get(struct_type->type, fields);
            }
            default: break;
        }
        LUISA_ERROR_WITH_LOCATION("Invalid type: {}.", t->description());
    }

    [[nodiscard]] llvm::Constant *_translate_constant(const xir::Constant *c) {
        if (auto iter = _llvm_constants.find(c); iter != _llvm_constants.end()) {
            return iter->second;
        }
        auto type = c->type();
        // we have to create a global variable for non-basic constants
        auto llvm_value = _translate_literal(type, c->data(), type->is_basic());
        // promote non-basic constants to globals
        if (!type->is_basic()) {
            auto name = luisa::format("const{}", _llvm_constants.size());
            auto g = llvm::dyn_cast<llvm::GlobalVariable>(
                _llvm_module->getOrInsertGlobal(llvm::StringRef{name}, llvm_value->getType()));
            g->setConstant(true);
            g->setLinkage(llvm::GlobalValue::LinkageTypes::PrivateLinkage);
            g->setInitializer(llvm_value);
            g->setUnnamedAddr(llvm::GlobalValue::UnnamedAddr::Global);
            llvm_value = g;
        }
        // cache
        _llvm_constants.emplace(c, llvm_value);
        return llvm_value;
    }

    void _translate_module(const xir::Module *module) noexcept {
        for (auto f : module->function_list()) {
            static_cast<void>(_translate_function(f));
        }
    }

    [[nodiscard]] llvm::Value *_translate_special_register(CurrentFunction &current, IRBuilder &b,
                                                           const xir::SpecialRegister *sreg) noexcept {
        switch (sreg->derived_special_register_tag()) {
            case xir::DerivedSpecialRegisterTag::THREAD_ID: return current.builtin_variables[CurrentFunction::builtin_variable_index_thread_id];
            case xir::DerivedSpecialRegisterTag::BLOCK_ID: return current.builtin_variables[CurrentFunction::builtin_variable_index_block_id];
            case xir::DerivedSpecialRegisterTag::WARP_LANE_ID: return llvm::ConstantInt::get(b.getInt32Ty(), 0);// CPU only has one lane
            case xir::DerivedSpecialRegisterTag::DISPATCH_ID: return current.builtin_variables[CurrentFunction::builtin_variable_index_dispatch_id];
            case xir::DerivedSpecialRegisterTag::KERNEL_ID:
            case xir::DerivedSpecialRegisterTag::RASTER_BARYCENTRICS:
            case xir::DerivedSpecialRegisterTag::RASTER_OBJECT_ID:
                LUISA_NOT_IMPLEMENTED();
            case xir::DerivedSpecialRegisterTag::BLOCK_SIZE: return current.builtin_variables[CurrentFunction::builtin_variable_index_block_size];
            case xir::DerivedSpecialRegisterTag::WARP_SIZE: return llvm::ConstantInt::get(b.getInt32Ty(), 1);// CPU only has one lane
            case xir::DerivedSpecialRegisterTag::DISPATCH_SIZE: return current.builtin_variables[CurrentFunction::builtin_variable_index_dispatch_size];
        }
        LUISA_ERROR_WITH_LOCATION("Invalid special register.");
    }

    [[nodiscard]] llvm::Value *_lookup_value(CurrentFunction &current, IRBuilder &b, const xir::Value *v, bool load_global = true) noexcept {
        LUISA_ASSERT(v != nullptr, "Value is null.");
        switch (v->derived_value_tag()) {
            case xir::DerivedValueTag::FUNCTION: {
                return _translate_function(static_cast<const xir::Function *>(v));
            }
            case xir::DerivedValueTag::BASIC_BLOCK: {
                return _find_or_create_basic_block(current, static_cast<const xir::BasicBlock *>(v));
            }
            case xir::DerivedValueTag::CONSTANT: {
                auto c = _translate_constant(static_cast<const xir::Constant *>(v));
                if (load_global && c->getType()->isPointerTy()) {
                    auto llvm_type = _translate_type(v->type(), true);
                    auto alignment = _get_type_alignment(v->type());
                    return b.CreateAlignedLoad(llvm_type, c, llvm::MaybeAlign{alignment});
                }
                return c;
            }
            case xir::DerivedValueTag::INSTRUCTION: [[fallthrough]];
            case xir::DerivedValueTag::ARGUMENT: {
                auto iter = current.value_map.find(v);
                LUISA_ASSERT(iter != current.value_map.end(), "Value not found.");
                return iter->second;
            }
            case xir::DerivedValueTag::SPECIAL_REGISTER: {
                auto sreg = static_cast<const xir::SpecialRegister *>(v);
                return _translate_special_register(current, b, sreg);
            }
            case xir::DerivedValueTag::UNDEFINED: {
                auto llvm_type = _translate_type(v->type(), true);
                return llvm::UndefValue::get(llvm_type);
            }
        }
        LUISA_ERROR_WITH_LOCATION("Invalid value.");
    }

    [[nodiscard]] static llvm::Value *_broadcast_like(IRBuilder &b, llvm::Value *value,
                                                      llvm::Value *shape) noexcept {
        LUISA_ASSERT(value != nullptr && shape != nullptr, "Cannot broadcast null LLVM values.");
        if (value->getType() == shape->getType()) {
            return value;
        }
        auto shape_type = llvm::dyn_cast<llvm::FixedVectorType>(shape->getType());
        LUISA_ASSERT(shape_type != nullptr &&
                         value->getType() == shape_type->getElementType(),
                     "Cannot broadcast an LLVM value to the requested vector shape.");
        return b.CreateVectorSplat(shape_type->getNumElements(), value);
    }

    [[nodiscard]] llvm::Constant *_translate_string_or_null(IRBuilder &b, luisa::string_view s) noexcept {
        if (s.empty()) {
            auto ptr_type = llvm::PointerType::get(_llvm_context, 0);
            return llvm::ConstantPointerNull::get(ptr_type);
        }
        return b.CreateGlobalString(s);
    }

    [[nodiscard]] llvm::Value *_translate_gep(CurrentFunction &current, IRBuilder &b,
                                              const Type *expected_elem_type,
                                              const Type *ptr_type, llvm::Value *llvm_ptr,
                                              luisa::span<const xir::Use *const> indices) noexcept {
        LUISA_ASSERT(llvm_ptr->getType()->isPointerTy(), "Invalid pointer type.");
        while (!indices.empty()) {
            // get the index
            auto index = indices.front()->value();
            indices = indices.subspan(1u);
            // structures need special handling
            if (ptr_type->is_structure()) {
                LUISA_ASSERT(index->derived_value_tag() == xir::DerivedValueTag::CONSTANT,
                             "Structure index should be a constant.");
                auto static_index = [c = static_cast<const xir::Constant *>(index)]() noexcept -> uint64_t {
                    auto t = c->type();
                    switch (t->tag()) {
                        case Type::Tag::INT8: return *static_cast<const int8_t *>(c->data());
                        case Type::Tag::UINT8: return *static_cast<const uint8_t *>(c->data());
                        case Type::Tag::INT16: return *static_cast<const int16_t *>(c->data());
                        case Type::Tag::UINT16: return *static_cast<const uint16_t *>(c->data());
                        case Type::Tag::INT32: return *static_cast<const int32_t *>(c->data());
                        case Type::Tag::UINT32: return *static_cast<const uint32_t *>(c->data());
                        case Type::Tag::INT64: return *static_cast<const int64_t *>(c->data());
                        case Type::Tag::UINT64: return *static_cast<const uint64_t *>(c->data());
                        default: break;
                    }
                    LUISA_ERROR_WITH_LOCATION("Invalid index type: {}.", t->description());
                }();
                // remap the index for padded fields
                LUISA_ASSERT(static_index < ptr_type->members().size(), "Invalid structure index.");
                auto llvm_struct_type = _translate_struct_type(ptr_type);
                auto padded_index = llvm_struct_type->padded_field_indices[static_index];
                // index into the struct and update member type
                llvm_ptr = b.CreateStructGEP(llvm_struct_type->type, llvm_ptr, padded_index);
                ptr_type = ptr_type->members()[static_index];
            } else {
                LUISA_ASSERT(ptr_type->is_array() || ptr_type->is_vector() || ptr_type->is_matrix(),
                             "Invalid pointer type: {}", ptr_type->description());
                auto llvm_index = _lookup_value(current, b, index);
                llvm_index = b.CreateZExtOrTrunc(llvm_index, llvm::Type::getInt64Ty(_llvm_context));
                auto llvm_ptr_type = _translate_type(ptr_type, false);
                auto llvm_zero = b.getInt64(0);
                llvm_ptr = b.CreateInBoundsGEP(llvm_ptr_type, llvm_ptr, {llvm_zero, llvm_index});
                ptr_type = ptr_type->is_matrix() ?
                               Type::vector(ptr_type->element(), ptr_type->dimension()) :
                               ptr_type->element();
            }
        }
        LUISA_ASSERT(ptr_type == expected_elem_type, "Type mismatch.");
        return llvm_ptr;
    }

    [[nodiscard]] llvm::Value *_translate_extract(CurrentFunction &current, IRBuilder &b,
                                                  const xir::ArithmeticInst *inst) noexcept {
        auto base = inst->operand(0u);
        auto indices = inst->operand_uses().subspan(1u);
        auto llvm_base = _lookup_value(current, b, base, false);
        // if we have an immediate vector, we can directly extract elements
        if (base->type()->is_vector() && !llvm_base->getType()->isPointerTy()) {
            LUISA_ASSERT(indices.size() == 1u, "Immediate vector should have only one index.");
            auto llvm_index = _lookup_value(current, b, indices.front()->value());
            return b.CreateExtractElement(llvm_base, llvm_index);
        }
        // otherwise we have to use GEP
        if (!llvm_base->getType()->isPointerTy()) {// create a temporary alloca if base is not a pointer
            auto llvm_base_type = _translate_type(base->type(), false);
            auto llvm_temp = b.CreateAlloca(llvm_base_type);
            auto alignment = _get_type_alignment(base->type());
            if (llvm_temp->getAlign() < alignment) {
                llvm_temp->setAlignment(llvm::Align{alignment});
            }
            b.CreateAlignedStore(llvm_base, llvm_temp, llvm::MaybeAlign{alignment});
            llvm_base = llvm_temp;
        }
        // GEP and load the element
        auto llvm_gep = _translate_gep(current, b, inst->type(), base->type(), llvm_base, indices);
        auto llvm_type = _translate_type(inst->type(), true);
        return b.CreateAlignedLoad(llvm_type, llvm_gep, llvm::MaybeAlign{_get_type_alignment(inst->type())});
    }

    [[nodiscard]] llvm::Value *_translate_insert(CurrentFunction &current, IRBuilder &b,
                                                 const xir::ArithmeticInst *inst) noexcept {
        auto base = inst->operand(0u);
        auto value = inst->operand(1u);
        auto indices = inst->operand_uses().subspan(2u);
        auto llvm_base = _lookup_value(current, b, base, false);
        auto llvm_value = _lookup_value(current, b, value, false);
        // if we have an immediate vector, we can directly insert elements
        if (base->type()->is_vector() && !llvm_base->getType()->isPointerTy()) {
            LUISA_ASSERT(indices.size() == 1u, "Immediate vector should have only one index.");
            auto llvm_index = _lookup_value(current, b, indices.front()->value());
            return b.CreateInsertElement(llvm_base, llvm_value, llvm_index);
        }
        // otherwise we have to make a copy of the base, store the element, and load the modified value
        auto llvm_base_type = _translate_type(base->type(), false);
        auto llvm_temp = b.CreateAlloca(llvm_base_type);
        auto alignment = _get_type_alignment(base->type());
        if (llvm_temp->getAlign() < alignment) {
            llvm_temp->setAlignment(llvm::Align{alignment});
        }
        // load the base if it is a pointer
        if (llvm_base->getType()->isPointerTy()) {
            llvm_base = b.CreateAlignedLoad(llvm_base_type, llvm_base, llvm::MaybeAlign{alignment});
        }
        // store the base
        b.CreateAlignedStore(llvm_base, llvm_temp, llvm::MaybeAlign{alignment});
        // GEP and store the element
        auto llvm_gep = _translate_gep(current, b, value->type(), base->type(), llvm_temp, indices);
        b.CreateAlignedStore(llvm_value, llvm_gep, llvm::MaybeAlign{_get_type_alignment(value->type())});
        // load the modified value
        return b.CreateAlignedLoad(llvm_base_type, llvm_temp, llvm::MaybeAlign{alignment});
    }

    [[nodiscard]] llvm::Value *_translate_unary_minus(CurrentFunction &current, IRBuilder &b, const xir::Value *operand) noexcept {
        auto llvm_operand = _lookup_value(current, b, operand);
        auto operand_type = operand->type();
        LUISA_ASSERT(operand_type != nullptr, "Operand type is null.");
        auto operand_elem_type = operand_type->is_vector() ? operand_type->element() : operand_type;
        switch (operand_elem_type->tag()) {
            case Type::Tag::INT8: [[fallthrough]];
            case Type::Tag::INT16: [[fallthrough]];
            case Type::Tag::INT32: [[fallthrough]];
            case Type::Tag::INT64: return b.CreateNSWNeg(llvm_operand);
            case Type::Tag::UINT8: [[fallthrough]];
            case Type::Tag::UINT16: [[fallthrough]];
            case Type::Tag::UINT32: [[fallthrough]];
            case Type::Tag::UINT64: return b.CreateNeg(llvm_operand);
            case Type::Tag::FLOAT16: [[fallthrough]];
            case Type::Tag::FLOAT32: [[fallthrough]];
            case Type::Tag::FLOAT64: return b.CreateFNeg(llvm_operand);
            default: break;
        }
        LUISA_ERROR_WITH_LOCATION("Invalid unary minus operand type: {}.", operand_type->description());
    }

    [[nodiscard]] static llvm::Value *_cmp_ne_zero(IRBuilder &b, llvm::Value *value) noexcept {
        auto zero = llvm::Constant::getNullValue(value->getType());
        return value->getType()->isFPOrFPVectorTy() ?
                   b.CreateFCmpUNE(value, zero) :
                   b.CreateICmpNE(value, zero);
    }

    [[nodiscard]] static llvm::Value *_cmp_eq_zero(IRBuilder &b, llvm::Value *value) noexcept {
        auto zero = llvm::Constant::getNullValue(value->getType());
        return value->getType()->isFPOrFPVectorTy() ?
                   b.CreateFCmpUEQ(value, zero) :
                   b.CreateICmpEQ(value, zero);
    }

    [[nodiscard]] static llvm::Value *_zext_i1_to_i8(IRBuilder &b, llvm::Value *value) noexcept {
        LUISA_DEBUG_ASSERT(value->getType()->isIntOrIntVectorTy(1), "Invalid value type.");
        auto i8_type = llvm::cast<llvm::Type>(llvm::Type::getInt8Ty(b.getContext()));
        if (value->getType()->isVectorTy()) {
            auto vec_type = llvm::cast<llvm::VectorType>(value->getType());
            i8_type = llvm::VectorType::get(i8_type, vec_type->getElementCount());
        }
        return b.CreateZExt(value, i8_type);
    }

    [[nodiscard]] llvm::Value *_translate_unary_bit_not(CurrentFunction &current, IRBuilder &b, const xir::Value *operand) noexcept {
        auto llvm_operand = _lookup_value(current, b, operand);
        LUISA_ASSERT(llvm_operand->getType()->isIntOrIntVectorTy() &&
                         !llvm_operand->getType()->isIntOrIntVectorTy(1),
                     "Invalid operand type.");
        if (operand->type()->is_bool() || operand->type()->is_bool_vector()) {// !b <=> (b == 0)
            auto i1_operand = _cmp_eq_zero(b, llvm_operand);
            return _zext_i1_to_i8(b, i1_operand);
        }
        return b.CreateNot(llvm_operand);
    }

    [[nodiscard]] llvm::Value *_translate_binary_add(CurrentFunction &current, IRBuilder &b, const xir::Value *lhs, const xir::Value *rhs) noexcept {
        LUISA_ASSERT(lhs->type() == rhs->type(), "Type mismatch.");
        auto llvm_lhs = _lookup_value(current, b, lhs);
        auto llvm_rhs = _lookup_value(current, b, rhs);
        auto elem_type = lhs->type()->is_vector() ? lhs->type()->element() : lhs->type();
        switch (elem_type->tag()) {
            case Type::Tag::INT8: [[fallthrough]];
            case Type::Tag::INT16: [[fallthrough]];
            case Type::Tag::INT32: [[fallthrough]];
            case Type::Tag::INT64: return b.CreateNSWAdd(llvm_lhs, llvm_rhs);
            case Type::Tag::UINT8: [[fallthrough]];
            case Type::Tag::UINT16: [[fallthrough]];
            case Type::Tag::UINT32: [[fallthrough]];
            case Type::Tag::UINT64: return b.CreateAdd(llvm_lhs, llvm_rhs);
            case Type::Tag::FLOAT16: [[fallthrough]];
            case Type::Tag::FLOAT32: [[fallthrough]];
            case Type::Tag::FLOAT64: return b.CreateFAdd(llvm_lhs, llvm_rhs);
            default: break;
        }
        LUISA_ERROR_WITH_LOCATION("Invalid binary add operand type: {}.", elem_type->description());
    }

    //swfly tries to write more binary operations
    [[nodiscard]] llvm::Value *_translate_binary_sub(CurrentFunction &current, IRBuilder &b, const xir::Value *lhs, const xir::Value *rhs) noexcept {
        LUISA_ASSERT(lhs->type() == rhs->type(), "Type mismatch.");
        auto llvm_lhs = _lookup_value(current, b, lhs);
        auto llvm_rhs = _lookup_value(current, b, rhs);
        auto elem_type = lhs->type()->is_vector() ? lhs->type()->element() : lhs->type();
        switch (elem_type->tag()) {
            case Type::Tag::INT8: [[fallthrough]];
            case Type::Tag::INT16: [[fallthrough]];
            case Type::Tag::INT32: [[fallthrough]];
            case Type::Tag::INT64: return b.CreateNSWSub(llvm_lhs, llvm_rhs);
            case Type::Tag::UINT8: [[fallthrough]];
            case Type::Tag::UINT16: [[fallthrough]];
            case Type::Tag::UINT32: [[fallthrough]];
            case Type::Tag::UINT64: return b.CreateSub(llvm_lhs, llvm_rhs);
            case Type::Tag::FLOAT16: [[fallthrough]];
            case Type::Tag::FLOAT32: [[fallthrough]];
            case Type::Tag::FLOAT64: return b.CreateFSub(llvm_lhs, llvm_rhs);
            default: break;
        }
        LUISA_ERROR_WITH_LOCATION("Invalid binary sub operand type: {}.", elem_type->description());
    }

    [[nodiscard]] llvm::Value *_translate_binary_mul(CurrentFunction &current, IRBuilder &b, const xir::Value *lhs, const xir::Value *rhs) noexcept {
        LUISA_ASSERT(lhs->type() == rhs->type(), "Type mismatch.");
        auto llvm_lhs = _lookup_value(current, b, lhs);
        auto llvm_rhs = _lookup_value(current, b, rhs);
        auto elem_type = lhs->type()->is_vector() ? lhs->type()->element() : lhs->type();
        switch (elem_type->tag()) {
            case Type::Tag::INT8: [[fallthrough]];
            case Type::Tag::INT16: [[fallthrough]];
            case Type::Tag::INT32: [[fallthrough]];
            case Type::Tag::INT64: return b.CreateNSWMul(llvm_lhs, llvm_rhs);
            case Type::Tag::UINT8: [[fallthrough]];
            case Type::Tag::UINT16: [[fallthrough]];
            case Type::Tag::UINT32: [[fallthrough]];
            case Type::Tag::UINT64: return b.CreateMul(llvm_lhs, llvm_rhs);
            case Type::Tag::FLOAT16: [[fallthrough]];
            case Type::Tag::FLOAT32: [[fallthrough]];
            case Type::Tag::FLOAT64: return b.CreateFMul(llvm_lhs, llvm_rhs);
            default: break;
        }
        LUISA_ERROR_WITH_LOCATION("Invalid binary mul operand type: {}.", elem_type->description());
    }

    [[nodiscard]] llvm::Value *_translate_binary_div(CurrentFunction &current, IRBuilder &b, const xir::Value *lhs, const xir::Value *rhs) noexcept {
        LUISA_ASSERT(lhs->type() == rhs->type(), "Type mismatch.");
        auto llvm_lhs = _lookup_value(current, b, lhs);
        auto llvm_rhs = _lookup_value(current, b, rhs);
        auto elem_type = lhs->type()->is_vector() ? lhs->type()->element() : lhs->type();
        switch (elem_type->tag()) {
            case Type::Tag::INT8: [[fallthrough]];
            case Type::Tag::INT16: [[fallthrough]];
            case Type::Tag::INT32: [[fallthrough]];
            case Type::Tag::INT64: return b.CreateSDiv(llvm_lhs, llvm_rhs);
            case Type::Tag::UINT8: [[fallthrough]];
            case Type::Tag::UINT16: [[fallthrough]];
            case Type::Tag::UINT32: [[fallthrough]];
            case Type::Tag::UINT64: return b.CreateUDiv(llvm_lhs, llvm_rhs);
            case Type::Tag::FLOAT16: [[fallthrough]];
            case Type::Tag::FLOAT32: [[fallthrough]];
            case Type::Tag::FLOAT64: return b.CreateFDiv(llvm_lhs, llvm_rhs);
            default: break;
        }
        LUISA_ERROR_WITH_LOCATION("Invalid binary add operand type: {}.", elem_type->description());
    }

    [[nodiscard]] llvm::Value *_translate_binary_mod(CurrentFunction &current, IRBuilder &b, const xir::Value *lhs, const xir::Value *rhs) noexcept {
        LUISA_ASSERT(lhs->type() == rhs->type(), "Type mismatch.");
        auto llvm_lhs = _lookup_value(current, b, lhs);
        auto llvm_rhs = _lookup_value(current, b, rhs);
        auto elem_type = lhs->type()->is_vector() ? lhs->type()->element() : lhs->type();
        switch (elem_type->tag()) {
            case Type::Tag::INT8: [[fallthrough]];
            case Type::Tag::INT16: [[fallthrough]];
            case Type::Tag::INT32: [[fallthrough]];
            case Type::Tag::INT64: return b.CreateSRem(llvm_lhs, llvm_rhs);// Signed integer remainder
            case Type::Tag::UINT8: [[fallthrough]];
            case Type::Tag::UINT16: [[fallthrough]];
            case Type::Tag::UINT32: [[fallthrough]];
            case Type::Tag::UINT64: return b.CreateURem(llvm_lhs, llvm_rhs);// Unsigned integer remainder
            default: break;
        }
        LUISA_ERROR_WITH_LOCATION("Invalid binary mod operand type: {}.", elem_type->description());
    }

    [[nodiscard]] llvm::Value *_translate_binary_bit_op(CurrentFunction &current, IRBuilder &b, xir::ArithmeticOp op, const xir::Value *lhs, const xir::Value *rhs) noexcept {
        // Lookup LLVM values for operands
        auto llvm_lhs = _lookup_value(current, b, lhs);
        auto llvm_rhs = _lookup_value(current, b, rhs);
        auto lhs_type = lhs->type();
        auto rhs_type = rhs->type();
        auto elem_type = lhs->type()->is_vector() ? lhs->type()->element() : lhs->type();
        // Type and null checks
        LUISA_ASSERT(lhs_type != nullptr && rhs_type != nullptr, "Operand type is null.");
        LUISA_ASSERT(lhs_type == rhs_type, "Type mismatch for bitwise and.");
        LUISA_ASSERT(lhs_type->is_scalar() || lhs_type->is_vector(), "Invalid operand type.");

        // Perform bitwise AND operation
        switch (elem_type->tag()) {
            case Type::Tag::BOOL: [[fallthrough]];
            case Type::Tag::INT8: [[fallthrough]];
            case Type::Tag::INT16: [[fallthrough]];
            case Type::Tag::INT32: [[fallthrough]];
            case Type::Tag::INT64: [[fallthrough]];
            case Type::Tag::UINT8: [[fallthrough]];
            case Type::Tag::UINT16: [[fallthrough]];
            case Type::Tag::UINT32: [[fallthrough]];
            case Type::Tag::UINT64: {
                auto is_bool = elem_type->is_bool();
                if (is_bool) {
                    llvm_lhs = _cmp_ne_zero(b, llvm_lhs);
                    llvm_rhs = _cmp_ne_zero(b, llvm_rhs);
                }
                auto result = [&] {
                    switch (op) {
                        case xir::ArithmeticOp::BINARY_BIT_AND: return b.CreateAnd(llvm_lhs, llvm_rhs);
                        case xir::ArithmeticOp::BINARY_BIT_OR: return b.CreateOr(llvm_lhs, llvm_rhs);
                        case xir::ArithmeticOp::BINARY_BIT_XOR: return b.CreateXor(llvm_lhs, llvm_rhs);
                        default: break;
                    }
                    LUISA_ERROR_WITH_LOCATION("Invalid binary bit operation: {}.", static_cast<uint32_t>(op));
                }();
                return is_bool ? _zext_i1_to_i8(b, result) : result;
            }
            default: break;
        }
        LUISA_ERROR_WITH_LOCATION("Invalid binary bit and operand type: {}.", elem_type->description());
    }

    [[nodiscard]] llvm::Value *_translate_binary_shift_left(CurrentFunction &current, IRBuilder &b, const xir::Value *lhs, const xir::Value *rhs) noexcept {
        // Lookup LLVM values for operands
        auto llvm_lhs = _lookup_value(current, b, lhs);
        auto llvm_rhs = _lookup_value(current, b, rhs);
        auto lhs_type = lhs->type();
        auto rhs_type = rhs->type();
        auto elem_type = lhs->type()->is_vector() ? lhs->type()->element() : lhs->type();

        // Type and null checks
        LUISA_ASSERT(lhs_type != nullptr && rhs_type != nullptr, "Operand type is null.");
        LUISA_ASSERT(lhs_type == rhs_type, "Type mismatch for shift left.");
        LUISA_ASSERT(lhs_type->is_scalar() || lhs_type->is_vector(), "Invalid operand type.");

        // Perform shift left operation (only valid for integer types)
        switch (elem_type->tag()) {
            case Type::Tag::INT8: [[fallthrough]];
            case Type::Tag::INT16: [[fallthrough]];
            case Type::Tag::INT32: [[fallthrough]];
            case Type::Tag::INT64: [[fallthrough]];
            case Type::Tag::UINT8: [[fallthrough]];
            case Type::Tag::UINT16: [[fallthrough]];
            case Type::Tag::UINT32: [[fallthrough]];
            case Type::Tag::UINT64: return b.CreateShl(llvm_lhs, llvm_rhs);
            default: break;
        }
        LUISA_ERROR_WITH_LOCATION("Invalid operand type for shift left operation: {}.", elem_type->description());
    }

    [[nodiscard]] llvm::Value *_translate_binary_shift_right(CurrentFunction &current, IRBuilder &b, const xir::Value *lhs, const xir::Value *rhs) noexcept {
        // Lookup LLVM values for operands
        auto llvm_lhs = _lookup_value(current, b, lhs);
        auto llvm_rhs = _lookup_value(current, b, rhs);
        auto lhs_type = lhs->type();
        auto rhs_type = rhs->type();
        auto elem_type = lhs->type()->is_vector() ? lhs->type()->element() : lhs->type();

        // Type and null checks
        LUISA_ASSERT(lhs_type != nullptr && rhs_type != nullptr, "Operand type is null.");
        LUISA_ASSERT(lhs_type == rhs_type, "Type mismatch for shift left.");
        LUISA_ASSERT(lhs_type->is_scalar() || lhs_type->is_vector(), "Invalid operand type.");

        // Perform shift left operation (only valid for integer types)
        switch (elem_type->tag()) {
            case Type::Tag::INT8: [[fallthrough]];
            case Type::Tag::INT16: [[fallthrough]];
            case Type::Tag::INT32: [[fallthrough]];
            case Type::Tag::INT64: return b.CreateAShr(llvm_lhs, llvm_rhs);
            case Type::Tag::UINT8: [[fallthrough]];
            case Type::Tag::UINT16: [[fallthrough]];
            case Type::Tag::UINT32: [[fallthrough]];
            case Type::Tag::UINT64: return b.CreateLShr(llvm_lhs, llvm_rhs);
            default: break;
        }
        LUISA_ERROR_WITH_LOCATION("Invalid operand type for shift left operation: {}.", elem_type->description());
    }

    [[nodiscard]] llvm::Value *_translate_binary_rotate_left(CurrentFunction &current, IRBuilder &b, const xir::Value *value, const xir::Value *shift) noexcept {
        LUISA_ASSERT(value->type() == shift->type(), "Type mismatch for rotate left.");
        auto llvm_value = _lookup_value(current, b, value);
        auto llvm_shift = _lookup_value(current, b, shift);
        return b.CreateIntrinsic(llvm_value->getType(), llvm::Intrinsic::fshl, {llvm_value, llvm_value, llvm_shift});
    }

    [[nodiscard]] llvm::Value *_translate_binary_rotate_right(CurrentFunction &current, IRBuilder &b, const xir::Value *value, const xir::Value *shift) noexcept {
        LUISA_ASSERT(value->type() == shift->type(), "Type mismatch for rotate right.");
        auto llvm_value = _lookup_value(current, b, value);
        auto llvm_shift = _lookup_value(current, b, shift);
        return b.CreateIntrinsic(llvm_value->getType(), llvm::Intrinsic::fshr, {llvm_value, llvm_value, llvm_shift});
    }

    [[nodiscard]] llvm::Value *_translate_binary_less(CurrentFunction &current, IRBuilder &b, const xir::Value *lhs, const xir::Value *rhs) noexcept {
        // Lookup LLVM values for operands
        auto llvm_lhs = _lookup_value(current, b, lhs);
        auto llvm_rhs = _lookup_value(current, b, rhs);
        auto lhs_type = lhs->type();
        auto rhs_type = rhs->type();
        auto elem_type = lhs->type()->is_vector() ? lhs->type()->element() : lhs->type();

        // Type and null checks
        LUISA_ASSERT(lhs_type != nullptr && rhs_type != nullptr, "Operand type is null.");
        LUISA_ASSERT(lhs_type == rhs_type, "Type mismatch for binary less.");
        LUISA_ASSERT(lhs_type->is_scalar() || lhs_type->is_vector(), "Invalid operand type.");

        // Perform less-than comparison based on the type
        llvm::Value *result = nullptr;
        switch (elem_type->tag()) {
            case Type::Tag::INT8: [[fallthrough]];
            case Type::Tag::INT16: [[fallthrough]];
            case Type::Tag::INT32: [[fallthrough]];
            case Type::Tag::INT64: result = b.CreateICmpSLT(llvm_lhs, llvm_rhs); break;// Signed integer less-than comparison
            case Type::Tag::UINT8: [[fallthrough]];
            case Type::Tag::UINT16: [[fallthrough]];
            case Type::Tag::UINT32: [[fallthrough]];
            case Type::Tag::UINT64: result = b.CreateICmpULT(llvm_lhs, llvm_rhs); break;// Unsigned integer less-than comparison
            case Type::Tag::FLOAT16: [[fallthrough]];
            case Type::Tag::FLOAT32: [[fallthrough]];
            case Type::Tag::FLOAT64: result = b.CreateFCmpOLT(llvm_lhs, llvm_rhs); break;// Floating-point unordered less-than comparison
            default: LUISA_ERROR_WITH_LOCATION("Invalid operand type for binary less-equal operation: {}.", elem_type->description());
        }
        return _zext_i1_to_i8(b, result);
    }

    [[nodiscard]] llvm::Value *_translate_binary_greater(CurrentFunction &current, IRBuilder &b, const xir::Value *lhs, const xir::Value *rhs) noexcept {
        auto llvm_lhs = _lookup_value(current, b, lhs);
        auto llvm_rhs = _lookup_value(current, b, rhs);
        auto lhs_type = lhs->type();
        auto rhs_type = rhs->type();
        auto elem_type = lhs->type()->is_vector() ? lhs->type()->element() : lhs->type();

        LUISA_ASSERT(lhs_type != nullptr && rhs_type != nullptr, "Operand type is null.");
        LUISA_ASSERT(lhs_type == rhs_type, "Type mismatch for binary greater.");
        LUISA_ASSERT(lhs_type->is_scalar() || lhs_type->is_vector(), "Invalid operand type.");

        llvm::Value *result = nullptr;
        switch (elem_type->tag()) {
            case Type::Tag::INT8: [[fallthrough]];
            case Type::Tag::INT16: [[fallthrough]];
            case Type::Tag::INT32: [[fallthrough]];
            case Type::Tag::INT64: result = b.CreateICmpSGT(llvm_lhs, llvm_rhs); break;// Signed integer greater-than
            case Type::Tag::UINT8: [[fallthrough]];
            case Type::Tag::UINT16: [[fallthrough]];
            case Type::Tag::UINT32: [[fallthrough]];
            case Type::Tag::UINT64: result = b.CreateICmpUGT(llvm_lhs, llvm_rhs); break;// Unsigned integer greater-than
            case Type::Tag::FLOAT16: [[fallthrough]];
            case Type::Tag::FLOAT32: [[fallthrough]];
            case Type::Tag::FLOAT64: result = b.CreateFCmpOGT(llvm_lhs, llvm_rhs); break;// Ordered greater-than
            default: LUISA_ERROR_WITH_LOCATION("Invalid operand type for binary less-equal operation: {}.", elem_type->description());
        }
        return _zext_i1_to_i8(b, result);
    }

    [[nodiscard]] llvm::Value *_translate_binary_less_equal(CurrentFunction &current, IRBuilder &b, const xir::Value *lhs, const xir::Value *rhs) noexcept {
        auto llvm_lhs = _lookup_value(current, b, lhs);
        auto llvm_rhs = _lookup_value(current, b, rhs);
        auto lhs_type = lhs->type();
        auto rhs_type = rhs->type();
        auto elem_type = lhs->type()->is_vector() ? lhs->type()->element() : lhs->type();

        LUISA_ASSERT(lhs_type != nullptr && rhs_type != nullptr, "Operand type is null.");
        LUISA_ASSERT(lhs_type == rhs_type, "Type mismatch for binary less-equal.");
        LUISA_ASSERT(lhs_type->is_scalar() || lhs_type->is_vector(), "Invalid operand type.");

        llvm::Value *result = nullptr;
        switch (elem_type->tag()) {
            case Type::Tag::INT8: [[fallthrough]];
            case Type::Tag::INT16: [[fallthrough]];
            case Type::Tag::INT32: [[fallthrough]];
            case Type::Tag::INT64: result = b.CreateICmpSLE(llvm_lhs, llvm_rhs); break;// Signed integer less-than-or-equal
            case Type::Tag::UINT8: [[fallthrough]];
            case Type::Tag::UINT16: [[fallthrough]];
            case Type::Tag::UINT32: [[fallthrough]];
            case Type::Tag::UINT64: result = b.CreateICmpULE(llvm_lhs, llvm_rhs); break;// Unsigned integer less-than-or-equal
            case Type::Tag::FLOAT16: [[fallthrough]];
            case Type::Tag::FLOAT32: [[fallthrough]];
            case Type::Tag::FLOAT64: result = b.CreateFCmpOLE(llvm_lhs, llvm_rhs); break;// Ordered less-than-or-equal
            default: LUISA_ERROR_WITH_LOCATION("Invalid operand type for binary less-equal operation: {}.", elem_type->description());
        }
        return _zext_i1_to_i8(b, result);
    }

    [[nodiscard]] llvm::Value *_translate_binary_greater_equal(CurrentFunction &current, IRBuilder &b, const xir::Value *lhs, const xir::Value *rhs) noexcept {
        auto llvm_lhs = _lookup_value(current, b, lhs);
        auto llvm_rhs = _lookup_value(current, b, rhs);
        auto lhs_type = lhs->type();
        auto rhs_type = rhs->type();
        auto elem_type = lhs->type()->is_vector() ? lhs->type()->element() : lhs->type();

        LUISA_ASSERT(lhs_type != nullptr && rhs_type != nullptr, "Operand type is null.");
        LUISA_ASSERT(lhs_type == rhs_type, "Type mismatch for binary greater-equal.");
        LUISA_ASSERT(lhs_type->is_scalar() || lhs_type->is_vector(), "Invalid operand type.");

        llvm::Value *result = nullptr;
        switch (elem_type->tag()) {
            case Type::Tag::INT8: [[fallthrough]];
            case Type::Tag::INT16: [[fallthrough]];
            case Type::Tag::INT32: [[fallthrough]];
            case Type::Tag::INT64: result = b.CreateICmpSGE(llvm_lhs, llvm_rhs); break;// Signed integer greater-than-or-equal
            case Type::Tag::UINT8: [[fallthrough]];
            case Type::Tag::UINT16: [[fallthrough]];
            case Type::Tag::UINT32: [[fallthrough]];
            case Type::Tag::UINT64: result = b.CreateICmpUGE(llvm_lhs, llvm_rhs); break;// Unsigned integer greater-than-or-equal
            case Type::Tag::FLOAT16: [[fallthrough]];
            case Type::Tag::FLOAT32: [[fallthrough]];
            case Type::Tag::FLOAT64: result = b.CreateFCmpOGE(llvm_lhs, llvm_rhs); break;// Ordered greater-than-or-equal
            default: LUISA_ERROR_WITH_LOCATION("Invalid operand type for binary less-equal operation: {}.", elem_type->description());
        }
        return _zext_i1_to_i8(b, result);
    }

    [[nodiscard]] llvm::Value *_translate_binary_equal(CurrentFunction &current, IRBuilder &b, const xir::Value *lhs, const xir::Value *rhs) noexcept {
        auto llvm_lhs = _lookup_value(current, b, lhs);
        auto llvm_rhs = _lookup_value(current, b, rhs);
        auto lhs_type = lhs->type();
        auto rhs_type = rhs->type();
        auto elem_type = lhs->type()->is_vector() ? lhs->type()->element() : lhs->type();

        LUISA_ASSERT(lhs_type != nullptr && rhs_type != nullptr, "Operand type is null.");
        LUISA_ASSERT(lhs_type == rhs_type, "Type mismatch for binary equal.");
        LUISA_ASSERT(lhs_type->is_scalar() || lhs_type->is_vector(), "Invalid operand type.");

        llvm::Value *result = nullptr;
        switch (elem_type->tag()) {
            case Type::Tag::INT8: [[fallthrough]];
            case Type::Tag::INT16: [[fallthrough]];
            case Type::Tag::INT32: [[fallthrough]];
            case Type::Tag::INT64: [[fallthrough]];
            case Type::Tag::UINT8: [[fallthrough]];
            case Type::Tag::UINT16: [[fallthrough]];
            case Type::Tag::UINT32: [[fallthrough]];
            case Type::Tag::UINT64: result = b.CreateICmpEQ(llvm_lhs, llvm_rhs); break;// Integer equality comparison
            case Type::Tag::FLOAT16: [[fallthrough]];
            case Type::Tag::FLOAT32: [[fallthrough]];
            case Type::Tag::FLOAT64: result = b.CreateFCmpOEQ(llvm_lhs, llvm_rhs); break;// Ordered equality comparison
            default: LUISA_ERROR_WITH_LOCATION("Invalid operand type for binary less-equal operation: {}.", elem_type->description());
        }
        return _zext_i1_to_i8(b, result);
    }

    [[nodiscard]] llvm::Value *_translate_binary_not_equal(CurrentFunction &current, IRBuilder &b, const xir::Value *lhs, const xir::Value *rhs) noexcept {
        auto llvm_lhs = _lookup_value(current, b, lhs);
        auto llvm_rhs = _lookup_value(current, b, rhs);
        auto lhs_type = lhs->type();
        auto rhs_type = rhs->type();
        auto elem_type = lhs->type()->is_vector() ? lhs->type()->element() : lhs->type();

        LUISA_ASSERT(lhs_type != nullptr && rhs_type != nullptr, "Operand type is null.");
        LUISA_ASSERT(lhs_type == rhs_type, "Type mismatch for binary equal.");
        LUISA_ASSERT(lhs_type->is_scalar() || lhs_type->is_vector(), "Invalid operand type.");

        llvm::Value *result = nullptr;
        switch (elem_type->tag()) {
            case Type::Tag::INT8: [[fallthrough]];
            case Type::Tag::INT16: [[fallthrough]];
            case Type::Tag::INT32: [[fallthrough]];
            case Type::Tag::INT64: [[fallthrough]];
            case Type::Tag::UINT8: [[fallthrough]];
            case Type::Tag::UINT16: [[fallthrough]];
            case Type::Tag::UINT32: [[fallthrough]];
            case Type::Tag::UINT64: result = b.CreateICmpNE(llvm_lhs, llvm_rhs); break;// Integer equality comparison
            case Type::Tag::FLOAT16: [[fallthrough]];
            case Type::Tag::FLOAT32: [[fallthrough]];
            case Type::Tag::FLOAT64: result = b.CreateFCmpONE(llvm_lhs, llvm_rhs); break;// Ordered equality comparison
            default: LUISA_ERROR_WITH_LOCATION("Invalid operand type for binary less-equal operation: {}.", elem_type->description());
        }
        return _zext_i1_to_i8(b, result);
    }

    [[nodiscard]] llvm::Value *_translate_unary_fp_math_operation(CurrentFunction &current, IRBuilder &b, const xir::Value *operand, llvm::Intrinsic::ID intrinsic_id) noexcept {
        // Lookup LLVM value for operand
        auto llvm_operand = _lookup_value(current, b, operand);
        auto operand_type = operand->type();

        // Type and null checks
        LUISA_ASSERT(operand_type != nullptr, "Operand type is null.");
        LUISA_ASSERT(operand_type->is_scalar() || operand_type->is_vector(), "Invalid operand type.");

        // Check if the operand is a valid floating-point type
        auto elem_type = operand_type->is_vector() ? operand_type->element() : operand_type;
        switch (elem_type->tag()) {
            case Type::Tag::FLOAT16: [[fallthrough]];
            case Type::Tag::FLOAT32: [[fallthrough]];
            case Type::Tag::FLOAT64:
                // Use LLVM's intrinsic function based on the provided intrinsic ID
                return b.CreateUnaryIntrinsic(intrinsic_id, llvm_operand);
            default:
                break;
        }
        LUISA_ERROR_WITH_LOCATION("Invalid operand type for unary math operation {}: {}.",
                                  intrinsic_id, operand_type->description());
    }

    [[nodiscard]] llvm::Value *_translate_unary_fp_math_operation(CurrentFunction &current, IRBuilder &b, const xir::Value *operand, const char *func) noexcept {
        // Lookup LLVM value for operand
        auto llvm_operand = _lookup_value(current, b, operand);
        auto operand_type = operand->type();

        // Type and null checks
        LUISA_ASSERT(operand_type != nullptr, "Operand type is null.");
        LUISA_ASSERT(operand_type->is_scalar() || operand_type->is_vector(), "Invalid operand type.");

        // Check if the operand is a valid floating-point type
        auto elem_type = operand_type->is_vector() ? operand_type->element() : operand_type;
        auto func_name = [elem_type, func] {
            switch (elem_type->tag()) {
                case Type::Tag::FLOAT16: return luisa::format("luisa.{}.f16", func);
                case Type::Tag::FLOAT32: return luisa::format("luisa.{}.f32", func);
                case Type::Tag::FLOAT64: return luisa::format("luisa.{}.f64", func);
                default: break;
            }
            LUISA_ERROR_WITH_LOCATION("Invalid operand type for unary math operation {}: {}.",
                                      func, elem_type->description());
        }();

        auto llvm_elem_type = _translate_type(elem_type, true);
        auto func_type = llvm::FunctionType::get(llvm_elem_type, {llvm_elem_type}, false);

        // declare the function and mark it as pure
        auto f = _llvm_module->getOrInsertFunction(llvm::StringRef{func_name}, func_type);
        if (auto decl = llvm::dyn_cast<llvm::Function>(f.getCallee())) {
            // mark that the function is pure: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
            decl->addFnAttr(llvm::Attribute::NoCallback);
            decl->setMustProgress();
            decl->setDoesNotFreeMemory();
            decl->setNoSync();
            decl->setDoesNotThrow();
            decl->setSpeculatable();
            decl->setWillReturn();
            decl->setDoesNotAccessMemory();
        } else {
            LUISA_ERROR_WITH_LOCATION("Invalid function declaration for unary math operation: {}.", func_name);
        }
        // call the function
        if (operand_type->is_scalar()) {
            return b.CreateCall(f, {llvm_operand});
        }
        // vector type
        auto llvm_result = llvm::cast<llvm::Value>(llvm::PoisonValue::get(llvm_operand->getType()));
        for (auto i = 0u; i < operand_type->dimension(); i++) {
            auto elem = b.CreateExtractElement(llvm_operand, i);
            auto result_elem = b.CreateCall(f, {elem});
            llvm_result = b.CreateInsertElement(llvm_result, result_elem, i);
        }
        return llvm_result;
    }

    [[nodiscard]] llvm::Value *_translate_binary_fp_math_operation(CurrentFunction &current, IRBuilder &b,
                                                                   const xir::Value *op0, const xir::Value *op1,
                                                                   llvm::Intrinsic::ID intrinsic_id) noexcept {
        auto llvm_op0 = _lookup_value(current, b, op0);
        auto llvm_op1 = _lookup_value(current, b, op1);
        auto op0_type = op0->type();
        auto op1_type = op1->type();
        LUISA_ASSERT(op0_type == op1_type, "Type mismatch.");
        LUISA_ASSERT(op0_type != nullptr && (op0_type->is_scalar() || op0_type->is_vector()), "Invalid operand type.");
        auto elem_type = op0_type->is_vector() ? op0_type->element() : op0_type;
        switch (elem_type->tag()) {
            case Type::Tag::FLOAT16: [[fallthrough]];
            case Type::Tag::FLOAT32: [[fallthrough]];
            case Type::Tag::FLOAT64:
                return b.CreateBinaryIntrinsic(intrinsic_id, llvm_op0, llvm_op1);
            default:
                break;
        }
        LUISA_ERROR_WITH_LOCATION("Invalid operand type for binary math operation {}: {}.",
                                  intrinsic_id, op0_type->description());
    }

    [[nodiscard]] llvm::Value *_translate_binary_fp_math_operation(CurrentFunction &current, IRBuilder &b,
                                                                   const xir::Value *op0, const xir::Value *op1,
                                                                   const char *func) noexcept {
        auto llvm_op0 = _lookup_value(current, b, op0);
        auto llvm_op1 = _lookup_value(current, b, op1);
        auto op0_type = op0->type();
        auto op1_type = op1->type();
        LUISA_ASSERT(op0_type == op1_type, "Type mismatch.");
        LUISA_ASSERT(op0_type != nullptr && (op0_type->is_scalar() || op0_type->is_vector()), "Invalid operand type.");

        auto elem_type = op0_type->is_vector() ? op0_type->element() : op0_type;
        auto func_name = [elem_type, func] {
            switch (elem_type->tag()) {
                case Type::Tag::FLOAT16: return luisa::format("luisa.{}.f16", func);
                case Type::Tag::FLOAT32: return luisa::format("luisa.{}.f32", func);
                case Type::Tag::FLOAT64: return luisa::format("luisa.{}.f64", func);
                default: break;
            }
            LUISA_ERROR_WITH_LOCATION("Invalid operand type for binary math operation {}: {}.",
                                      func, elem_type->description());
        }();

        auto llvm_elem_type = _translate_type(elem_type, true);
        auto func_type = llvm::FunctionType::get(llvm_elem_type, {llvm_elem_type, llvm_elem_type}, false);

        // declare the function and mark it as pure
        auto f = _llvm_module->getOrInsertFunction(llvm::StringRef{func_name}, func_type);
        if (auto decl = llvm::dyn_cast<llvm::Function>(f.getCallee())) {
            // mark that the function is pure: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
            decl->addFnAttr(llvm::Attribute::NoCallback);
            decl->setMustProgress();
            decl->setDoesNotFreeMemory();
            decl->setNoSync();
            decl->setDoesNotThrow();
            decl->setSpeculatable();
            decl->setWillReturn();
            decl->setDoesNotAccessMemory();
        } else {
            LUISA_ERROR_WITH_LOCATION("Invalid function declaration for binary math operation: {}.", func_name);
        }
        // call the function
        if (op0_type->is_scalar()) {
            return b.CreateCall(f, {llvm_op0, llvm_op1});
        }
        // vector type
        auto llvm_result = llvm::cast<llvm::Value>(llvm::PoisonValue::get(llvm_op0->getType()));
        for (auto i = 0u; i < op0_type->dimension(); i++) {
            auto elem0 = b.CreateExtractElement(llvm_op0, i);
            auto elem1 = b.CreateExtractElement(llvm_op1, i);
            auto result_elem = b.CreateCall(f, {elem0, elem1});
            llvm_result = b.CreateInsertElement(llvm_result, result_elem, i);
        }
        return llvm_result;
    }

    [[nodiscard]] llvm::Value *_translate_vector_reduce(CurrentFunction &current, IRBuilder &b,
                                                        xir::ArithmeticOp op, const xir::Value *operand) noexcept {
        LUISA_ASSERT(operand->type() != nullptr && operand->type()->is_vector(),
                     "Invalid operand type for reduce_sum operation.");
        auto operand_elem_type = operand->type()->element();
        auto llvm_operand = _lookup_value(current, b, operand);
        switch (operand->type()->element()->tag()) {
            case Type::Tag::INT8: [[fallthrough]];
            case Type::Tag::INT16: [[fallthrough]];
            case Type::Tag::INT32: [[fallthrough]];
            case Type::Tag::INT64: {
                switch (op) {
                    case xir::ArithmeticOp::REDUCE_SUM: return b.CreateAddReduce(llvm_operand);
                    case xir::ArithmeticOp::REDUCE_PRODUCT: return b.CreateMulReduce(llvm_operand);
                    case xir::ArithmeticOp::REDUCE_MIN: return b.CreateIntMinReduce(llvm_operand, true);
                    case xir::ArithmeticOp::REDUCE_MAX: return b.CreateIntMaxReduce(llvm_operand, true);
                    default: break;
                }
                LUISA_ERROR_WITH_LOCATION("Invalid reduce operation: {}.", xir::to_string(op));
            }
            case Type::Tag::UINT8: [[fallthrough]];
            case Type::Tag::UINT16: [[fallthrough]];
            case Type::Tag::UINT32: [[fallthrough]];
            case Type::Tag::UINT64: {
                switch (op) {
                    case xir::ArithmeticOp::REDUCE_SUM: return b.CreateAddReduce(llvm_operand);
                    case xir::ArithmeticOp::REDUCE_PRODUCT: return b.CreateMulReduce(llvm_operand);
                    case xir::ArithmeticOp::REDUCE_MIN: return b.CreateIntMinReduce(llvm_operand, false);
                    case xir::ArithmeticOp::REDUCE_MAX: return b.CreateIntMaxReduce(llvm_operand, false);
                    default: break;
                }
                LUISA_ERROR_WITH_LOCATION("Invalid reduce operation: {}.", xir::to_string(op));
            }
            case Type::Tag::FLOAT16: [[fallthrough]];
            case Type::Tag::FLOAT32: [[fallthrough]];
            case Type::Tag::FLOAT64: {
                auto llvm_elem_type = _translate_type(operand_elem_type, false);
                auto llvm_zero = llvm::ConstantFP::getNegativeZero(llvm_elem_type);
                auto llvm_one = llvm::ConstantFP::get(llvm_elem_type, 1.);
                switch (op) {
                    case xir::ArithmeticOp::REDUCE_SUM: return b.CreateFAddReduce(llvm_zero, llvm_operand);
                    case xir::ArithmeticOp::REDUCE_PRODUCT: return b.CreateFMulReduce(llvm_one, llvm_operand);
                    case xir::ArithmeticOp::REDUCE_MIN: return b.CreateFPMinReduce(llvm_operand);
                    case xir::ArithmeticOp::REDUCE_MAX: return b.CreateFPMaxReduce(llvm_operand);
                    default: break;
                }
                LUISA_ERROR_WITH_LOCATION("Invalid reduce operation: {}.", xir::to_string(op));
            }
            default: break;
        }
        LUISA_ERROR_WITH_LOCATION("Invalid operand type for reduce_sum operation: {}.",
                                  operand_elem_type->description());
    }

    [[nodiscard]] llvm::Value *_translate_vector_dot(CurrentFunction &current, IRBuilder &b,
                                                     const xir::Value *lhs, const xir::Value *rhs) noexcept {
        auto llvm_mul = _translate_binary_mul(current, b, lhs, rhs);
        if (llvm_mul->getType()->isFPOrFPVectorTy()) {
            auto llvm_elem_type = llvm_mul->getType()->isVectorTy() ?
                                      llvm::cast<llvm::VectorType>(llvm_mul->getType())->getElementType() :
                                      llvm_mul->getType();
            auto llvm_zero = llvm::ConstantFP::getNegativeZero(llvm_elem_type);
            return b.CreateFAddReduce(llvm_zero, llvm_mul);
        }
        return b.CreateAddReduce(llvm_mul);
    }

    [[nodiscard]] llvm::Value *_translate_isinf_isnan(CurrentFunction &current, IRBuilder &b,
                                                      xir::ArithmeticOp op, const xir::Value *x) noexcept {
        auto type = x->type();
        auto elem_type = type->is_vector() ? type->element() : type;
        auto llvm_x = _lookup_value(current, b, x);
        auto [llvm_mask, llvm_test] = [&] {
            switch (elem_type->tag()) {
                case Type::Tag::FLOAT16: return std::make_pair(
                    llvm::cast<llvm::Constant>(b.getInt16(0x7fffu)),
                    llvm::cast<llvm::Constant>(b.getInt16(0x7c00u)));
                case Type::Tag::FLOAT32: return std::make_pair(
                    llvm::cast<llvm::Constant>(b.getInt32(0x7fffffffu)),
                    llvm::cast<llvm::Constant>(b.getInt32(0x7f800000u)));
                case Type::Tag::FLOAT64: return std::make_pair(
                    llvm::cast<llvm::Constant>(b.getInt64(0x7fffffffffffffffull)),
                    llvm::cast<llvm::Constant>(b.getInt64(0x7ff0000000000000ull)));
                default: break;
            }
            LUISA_ERROR_WITH_LOCATION("Invalid operand type for isinf/isnan operation: {}.", type->description());
        }();
        if (type->is_vector()) {
            auto dim = llvm::ElementCount::getFixed(type->dimension());
            llvm_mask = llvm::ConstantVector::getSplat(dim, llvm_mask);
            llvm_test = llvm::ConstantVector::getSplat(dim, llvm_test);
        }
        auto llvm_int_type = llvm_mask->getType();
        auto llvm_bits = b.CreateBitCast(llvm_x, llvm_int_type);
        auto llvm_and = b.CreateAnd(llvm_bits, llvm_mask);
        LUISA_DEBUG_ASSERT(op == xir::ArithmeticOp::ISINF || op == xir::ArithmeticOp::ISNAN,
                           "Unexpected intrinsic operation.");
        auto llvm_cmp = op == xir::ArithmeticOp::ISINF ?
                            b.CreateICmpEQ(llvm_and, llvm_test) :
                            b.CreateICmpUGT(llvm_and, llvm_test);
        return _zext_i1_to_i8(b, llvm_cmp);
    }

    [[nodiscard]] llvm::Value *_get_buffer_element_ptr(CurrentFunction &current, IRBuilder &b,
                                                       const xir::Value *buffer, const xir::Value *slot,
                                                       bool byte_address) noexcept {
        auto llvm_buffer = _lookup_value(current, b, buffer);          // Get the buffer view
        auto llvm_buffer_addr = b.CreateExtractValue(llvm_buffer, {0});// Get the buffer address
        auto llvm_slot = _lookup_value(current, b, slot);              // Get the slot index
        llvm_slot = b.CreateZExtOrTrunc(llvm_slot, b.getInt64Ty());
        auto slot_type = byte_address ? b.getInt8Ty() : _translate_type(buffer->type()->element(), false);
        return b.CreateInBoundsGEP(
            slot_type,       // Element type
            llvm_buffer_addr,// Base pointer
            llvm_slot        // Index
        );
    }

    void _validate_static_byte_buffer_alignment(
        CurrentFunction &current, IRBuilder &b, const xir::Value *offset,
        const Type *access_type) noexcept {
        auto llvm_offset = _lookup_value(current, b, offset);
        if (auto constant = llvm::dyn_cast<llvm::ConstantInt>(llvm_offset)) {
            auto alignment = _get_type_alignment(access_type);
            auto byte_offset = constant->getZExtValue();
            if (byte_offset % alignment != 0u) {
                LUISA_ERROR_WITH_LOCATION(
                    "Misaligned fallback byte-buffer access: type {} at byte "
                    "offset {} requires {}-byte alignment. Use an aligned "
                    "offset or a packed scalar array type (for example "
                    "std::array<float, 3> instead of float3).",
                    access_type->description(), byte_offset, alignment);
            }
        }
    }

    [[nodiscard]] llvm::Value *_translate_buffer_write(CurrentFunction &current, IRBuilder &b,
                                                       const xir::ResourceWriteInst *inst,
                                                       bool is_volatile, bool byte_address = false) noexcept {
        auto buffer = inst->operand(0u);
        auto slot = inst->operand(1u);
        auto value = inst->operand(2u);
        if (byte_address) {
            _validate_static_byte_buffer_alignment(
                current, b, slot, value->type());
        }
        auto llvm_elem_ptr = _get_buffer_element_ptr(current, b, buffer, slot, byte_address);
        auto llvm_value = _lookup_value(current, b, value);// Get the value to write
        auto alignment = _get_type_alignment(value->type());
        return b.CreateAlignedStore(llvm_value, llvm_elem_ptr, llvm::MaybeAlign{alignment}, is_volatile);
    }

    [[nodiscard]] llvm::Value *_translate_buffer_read(CurrentFunction &current, IRBuilder &b,
                                                      const xir::ResourceReadInst *inst,
                                                      bool is_volatile, bool byte_address = false) noexcept {
        auto buffer = inst->operand(0u);
        auto slot = inst->operand(1u);
        if (byte_address) {
            _validate_static_byte_buffer_alignment(
                current, b, slot, inst->type());
        }
        auto llvm_elem_ptr = _get_buffer_element_ptr(current, b, buffer, slot, byte_address);
        auto alignment = _get_type_alignment(inst->type());
        auto llvm_element_type = _translate_type(inst->type(), true);// Type of the value being read
        return b.CreateAlignedLoad(llvm_element_type, llvm_elem_ptr, llvm::MaybeAlign{alignment}, is_volatile);
    }

    [[nodiscard]] llvm::Value *_translate_buffer_size(CurrentFunction &current, IRBuilder &b,
                                                      const xir::ResourceQueryInst *inst,
                                                      bool byte_address = false) noexcept {
        auto buffer = inst->operand(0u);
        LUISA_ASSERT(buffer->type()->is_buffer(), "Invalid buffer type.");
        auto llvm_buffer = _lookup_value(current, b, buffer);
        auto llvm_byte_size = b.CreateExtractValue(llvm_buffer, {1});
        auto llvm_result_type = _translate_type(inst->type(), true);
        llvm_byte_size = b.CreateZExtOrTrunc(llvm_byte_size, llvm_result_type);
        if (!byte_address) {
            auto elem_size = _get_type_size(buffer->type()->element());
            auto llvm_elem_size = llvm::ConstantInt::get(llvm_result_type, elem_size);
            llvm_byte_size = b.CreateUDiv(llvm_byte_size, llvm_elem_size);
        }
        return llvm_byte_size;
    }

    [[nodiscard]] llvm::Value *_translate_buffer_device_address(CurrentFunction &current, IRBuilder &b,
                                                                const xir::ResourceQueryInst *inst) noexcept {
        auto buffer = inst->operand(0u);
        LUISA_ASSERT(buffer->type()->is_buffer(), "Invalid buffer type.");
        auto llvm_buffer = _lookup_value(current, b, buffer);
        auto llvm_buffer_ptr = b.CreateExtractValue(llvm_buffer, {0});
        auto llvm_int_type = _translate_type(inst->type(), false);
        return b.CreatePtrToInt(llvm_buffer_ptr, llvm_int_type);
    }

    [[nodiscard]] llvm::Value *_translate_device_address_read(CurrentFunction &current, IRBuilder &b,
                                                              const xir::ResourceReadInst *inst) noexcept {
        auto llvm_device_address = _lookup_value(current, b, inst->operand(0u));
        auto llvm_ptr_type = llvm::PointerType::get(_llvm_context, 0);
        auto llvm_ptr = b.CreateIntToPtr(llvm_device_address, llvm_ptr_type);
        auto alignment = _get_type_alignment(inst->type());
        auto llvm_result_type = _translate_type(inst->type(), true);
        return b.CreateAlignedLoad(llvm_result_type, llvm_ptr, llvm::MaybeAlign{alignment});
    }

    [[nodiscard]] llvm::Value *_translate_device_address_write(CurrentFunction &current, IRBuilder &b,
                                                               const xir::ResourceWriteInst *inst) noexcept {
        auto llvm_device_address = _lookup_value(current, b, inst->operand(0u));
        auto llvm_value = _lookup_value(current, b, inst->operand(1u));
        auto llvm_ptr_type = llvm::PointerType::get(_llvm_context, 0);
        auto llvm_ptr = b.CreateIntToPtr(llvm_device_address, llvm_ptr_type);
        auto alignment = _get_type_alignment(inst->operand(1u)->type());
        return b.CreateAlignedStore(llvm_value, llvm_ptr, llvm::MaybeAlign{alignment});
    }

    [[nodiscard]] llvm::Value *_translate_aggregate(CurrentFunction &current, IRBuilder &b,
                                                    const xir::ArithmeticInst *inst) noexcept {
        auto type = inst->type();
        auto elem_type = type->element();
        auto dim = type->dimension();
        if (type->is_vector()) {
            auto llvm_elem_type = _translate_type(elem_type, true);

            auto llvm_return_type = llvm::VectorType::get(llvm_elem_type, dim, false);

            auto vector = llvm::cast<llvm::Value>(llvm::PoisonValue::get(llvm_return_type));
            for (int i = 0; i < dim; i++) {
                auto operand = inst->operand(i);
                LUISA_ASSERT(operand->type() == elem_type, "element type doesn't match");
                auto llvm_operand = _lookup_value(current, b, operand);
                LUISA_ASSERT(llvm_operand->getType() == llvm_elem_type, "element type doesn't match");

                vector = b.CreateInsertElement(vector, llvm_operand, static_cast<uint64_t>(i));
            }
            return vector;
        }

        if (type->is_matrix()) {
            auto llvm_matrix_type = _translate_type(type, true);
            auto llvm_matrix = llvm::cast<llvm::Value>(llvm::PoisonValue::get(llvm_matrix_type));
            for (auto i = 0u; i < dim; i++) {
                auto col = inst->operand(i);
                LUISA_ASSERT(col->type()->is_vector() &&
                                 col->type()->dimension() == dim &&
                                 col->type()->element() == elem_type,
                             "element type doesn't match");
                auto llvm_col = _lookup_value(current, b, col);
                for (auto j = 0u; j < dim; j++) {
                    auto llvm_elem = b.CreateExtractElement(llvm_col, j);
                    llvm_matrix = b.CreateInsertValue(llvm_matrix, llvm_elem, {i, j});
                }
            }
            return llvm_matrix;
        }

        LUISA_NOT_IMPLEMENTED();
    }

    [[nodiscard]] llvm::Value *_translate_shuffle(CurrentFunction &current, IRBuilder &b,
                                                  const xir::ArithmeticInst *inst) noexcept {

        auto result_type = inst->type();
        LUISA_ASSERT(result_type->is_vector(), "shuffle target must be a vector");

        auto result_elem_type = result_type->element();
        auto result_dim = result_type->dimension();
        LUISA_ASSERT(inst->operand_count() == result_dim + 1u, "shuffle operand count doesn't match");

        auto src = inst->operand(0);
        LUISA_ASSERT(src != nullptr, "shuffle target is null");

        auto src_type = src->type();
        LUISA_ASSERT(src_type != nullptr && src_type->element() == result_elem_type,
                     "shuffle element type doesn't match");

        auto llvm_src = _lookup_value(current, b, src);

        auto indices = inst->operand_uses().subspan(1);
        auto statically_shuffled = std::all_of(indices.begin(), indices.end(), [](const xir::Use *v) {
            LUISA_ASSERT(v->value() != nullptr, "shuffle index is null");
            return v->value()->derived_value_tag() == xir::DerivedValueTag::CONSTANT;
        });

        // we may use llvm's shufflevector intrinsic if all indices are constant
        if (statically_shuffled) {
            llvm::SmallVector<int, 4u> static_indices;
            for (auto i : indices) {
                auto static_index = [src_dim = src_type->dimension(), index = static_cast<const xir::Constant *>(i->value())] {
                    auto safe_index = [src_dim](auto x) noexcept {
                        LUISA_ASSERT(x >= 0u && x < src_dim, "shuffle index out of range");
                        return static_cast<int>(x);
                    };
                    switch (index->type()->tag()) {
                        case Type::Tag::INT8: return safe_index(index->as<int8_t>());
                        case Type::Tag::UINT8: return safe_index(index->as<uint8_t>());
                        case Type::Tag::INT16: return safe_index(index->as<int16_t>());
                        case Type::Tag::UINT16: return safe_index(index->as<uint16_t>());
                        case Type::Tag::INT32: return safe_index(index->as<int32_t>());
                        case Type::Tag::UINT32: return safe_index(index->as<uint32_t>());
                        case Type::Tag::INT64: return safe_index(index->as<int64_t>());
                        case Type::Tag::UINT64: return safe_index(index->as<uint64_t>());
                        default: break;
                    }
                    LUISA_ERROR_WITH_LOCATION("invalid shuffle index type: {}.", index->type()->description());
                }();
                static_indices.push_back(static_index);
            }
            auto llvm_poison_value = llvm::PoisonValue::get(llvm_src->getType());
            return b.CreateShuffleVector(llvm_src, llvm_poison_value, static_indices);
        }

        // otherwise, we extract elements and create the target vector
        auto llvm_result_type = _translate_type(result_type, true);
        auto llvm_result = llvm::cast<llvm::Value>(llvm::PoisonValue::get(llvm_result_type));
        for (auto i = 0u; i < result_dim; i++) {
            auto llvm_index = _lookup_value(current, b, indices[i]->value());
            auto llvm_src_elem = b.CreateExtractElement(llvm_src, llvm_index);
            llvm_result = b.CreateInsertElement(llvm_result, llvm_src_elem, i);
        }
        return llvm_result;
    }

    [[nodiscard]] llvm::Type *_get_llvm_bindless_slot_type() noexcept {
        // struct Slot {
        //   ptr buffer_ptr;
        //   i64 buffer_size[63:8] sampler2d[7:4] sampler3d[3:0];
        //   ptr tex2d_ptr;
        //   ptr tex3d_ptr;
        // };
        auto llvm_ptr_type = llvm::PointerType::get(_llvm_context, 0);
        auto llvm_i64_type = llvm::Type::getInt64Ty(_llvm_context);
        return llvm::StructType::get(llvm_ptr_type, llvm_i64_type, llvm_ptr_type, llvm_ptr_type);
    }

    [[nodiscard]] llvm::Value *_load_bindless_array_slot(CurrentFunction &current, IRBuilder &b,
                                                         llvm::Type *llvm_slot_type,
                                                         const xir::Value *bindless,
                                                         const xir::Value *slot_index) noexcept {
        auto llvm_bindless = _lookup_value(current, b, bindless);
        auto llvm_bindless_slots = b.CreateExtractValue(llvm_bindless, {0});
        auto llvm_slot_index = _lookup_value(current, b, slot_index);
        llvm_slot_index = b.CreateZExtOrTrunc(llvm_slot_index, b.getInt64Ty());
        auto llvm_slot_ptr = b.CreateInBoundsGEP(llvm_slot_type, llvm_bindless_slots, {llvm_slot_index});
        return b.CreateAlignedLoad(llvm_slot_type, llvm_slot_ptr, llvm::MaybeAlign{16}, "bindless.slot");
    }

    [[nodiscard]] llvm::Value *_translate_bindless_buffer_read(CurrentFunction &current, IRBuilder &b,
                                                               const xir::ResourceReadInst *inst,
                                                               bool byte_address = false) noexcept {
        auto llvm_slot_type = _get_llvm_bindless_slot_type();
        auto result_type = inst->type();
        auto bindless = inst->operand(0);
        auto slot_index = inst->operand(1);
        auto index_or_offset = inst->operand(2);
        if (byte_address) {
            _validate_static_byte_buffer_alignment(
                current, b, index_or_offset, result_type);
        }
        auto llvm_slot = _load_bindless_array_slot(current, b, llvm_slot_type, bindless, slot_index);
        auto llvm_buffer_ptr = b.CreateExtractValue(llvm_slot, {0});
        auto llvm_offset = _lookup_value(current, b, index_or_offset);
        llvm_offset = b.CreateZExtOrTrunc(llvm_offset, b.getInt64Ty());
        auto llvm_addressing_type = byte_address ? b.getInt8Ty() : _translate_type(result_type, false);
        auto llvm_target_address = b.CreateInBoundsGEP(llvm_addressing_type, llvm_buffer_ptr, {llvm_offset});
        auto alignment = _get_type_alignment(result_type);
        auto llvm_result_type = _translate_type(result_type, true);
        return b.CreateAlignedLoad(llvm_result_type, llvm_target_address, llvm::MaybeAlign{alignment});
    }

    [[nodiscard]] llvm::Value *_translate_bindless_buffer_write(CurrentFunction &current, IRBuilder &b,
                                                                const xir::ResourceWriteInst *inst,
                                                                bool byte_address = false) noexcept {
        auto llvm_slot_type = _get_llvm_bindless_slot_type();
        auto bindless = inst->operand(0);
        auto slot_index = inst->operand(1);
        auto index_or_offset = inst->operand(2);
        auto value = inst->operand(3);
        if (byte_address) {
            _validate_static_byte_buffer_alignment(
                current, b, index_or_offset, value->type());
        }
        auto llvm_slot = _load_bindless_array_slot(current, b, llvm_slot_type, bindless, slot_index);
        auto llvm_buffer_ptr = b.CreateExtractValue(llvm_slot, {0});
        auto llvm_offset = _lookup_value(current, b, index_or_offset);
        llvm_offset = b.CreateZExtOrTrunc(llvm_offset, b.getInt64Ty());
        auto llvm_value = _lookup_value(current, b, value);
        auto llvm_addressing_type = byte_address ? b.getInt8Ty() : _translate_type(value->type(), false);
        auto llvm_target_address = b.CreateInBoundsGEP(llvm_addressing_type, llvm_buffer_ptr, {llvm_offset});
        auto alignment = _get_type_alignment(value->type());
        return b.CreateAlignedStore(llvm_value, llvm_target_address, llvm::MaybeAlign{alignment});
    }

    [[nodiscard]] llvm::Value *_translate_bindless_buffer_size(CurrentFunction &current, IRBuilder &b,
                                                               const xir::ResourceQueryInst *inst,
                                                               bool byte_address = false) noexcept {
        auto llvm_slot_type = _get_llvm_bindless_slot_type();
        auto bindless = inst->operand(0);
        auto slot_index = inst->operand(1);
        auto llvm_slot = _load_bindless_array_slot(current, b, llvm_slot_type, bindless, slot_index);
        auto llvm_byte_size = b.CreateExtractValue(llvm_slot, {1});
        llvm_byte_size = b.CreateLShr(llvm_byte_size, 8);
        if (!byte_address) {
            auto stride = inst->operand(2);
            auto llvm_stride = _lookup_value(current, b, stride);
            llvm_stride = b.CreateZExtOrTrunc(llvm_stride, llvm_byte_size->getType());
            llvm_byte_size = b.CreateUDiv(llvm_byte_size, llvm_stride);
        }
        auto llvm_result_type = _translate_type(inst->type(), true);
        return b.CreateZExtOrTrunc(llvm_byte_size, llvm_result_type);
    }

    [[nodiscard]] llvm::Value *_translate_bindless_buffer_device_address(CurrentFunction &current, IRBuilder &b,
                                                                         const xir::ResourceQueryInst *inst) noexcept {
        auto llvm_slot_type = _get_llvm_bindless_slot_type();
        auto bindless = inst->operand(0);
        auto slot_index = inst->operand(1);
        auto llvm_slot = _load_bindless_array_slot(current, b, llvm_slot_type, bindless, slot_index);
        auto llvm_buffer_ptr = b.CreateExtractValue(llvm_slot, {0});
        auto llvm_int_type = _translate_type(inst->type(), false);
        return b.CreatePtrToInt(llvm_buffer_ptr, llvm_int_type);
    }

    [[nodiscard]] llvm::Value *_translate_matrix_transpose(CurrentFunction &current, IRBuilder &b,
                                                           const xir::ArithmeticInst *inst) noexcept {
        auto matrix = inst->operand(0u);
        LUISA_ASSERT(matrix->type() != nullptr &&
                         matrix->type() == inst->type() &&
                         matrix->type()->is_matrix(),
                     "Invalid matrix type for transpose operation.");
        auto dimension = inst->type()->dimension();
        auto llvm_matrix = _lookup_value(current, b, matrix);
        auto llvm_matrix_alloca = b.CreateAlloca(llvm_matrix->getType());
        b.CreateStore(llvm_matrix, llvm_matrix_alloca);
        auto llvm_result_type = _translate_type(inst->type(), true);
        auto llvm_result_alloca = b.CreateAlloca(llvm_result_type);
        auto llvm_func_name = luisa::format("luisa.matrix{}d.transpose", dimension);
        auto llvm_func = _llvm_module->getFunction(llvm::StringRef{llvm_func_name});
        LUISA_ASSERT(llvm_func != nullptr, "Function not found.");
        b.CreateCall(llvm_func, {llvm_matrix_alloca, llvm_result_alloca});
        return b.CreateLoad(llvm_result_type, llvm_result_alloca);
    }

    [[nodiscard]] llvm::Value *_translate_matrix_comp_op(CurrentFunction &current, IRBuilder &b,
                                                         const xir::Value *lhs, const xir::Value *rhs,
                                                         llvm::Instruction::BinaryOps llvm_op) noexcept {
        auto lhs_is_matrix = lhs->type()->is_matrix();
        auto rhs_is_matrix = rhs->type()->is_matrix();
        LUISA_ASSERT((lhs_is_matrix && rhs_is_matrix) ||
                         (lhs_is_matrix && rhs->type()->is_scalar()) ||
                         (lhs->type()->is_scalar() && rhs_is_matrix),
                     "Matrix or vector expected.");
        auto matrix_type = lhs_is_matrix ? lhs->type() : rhs->type();
        auto llvm_matrix_type = _translate_type(matrix_type, true);
        auto llvm_lhs = _lookup_value(current, b, lhs);
        auto llvm_rhs = _lookup_value(current, b, rhs);
        auto llvm_result = llvm::cast<llvm::Value>(llvm::PoisonValue::get(llvm_matrix_type));
        auto dim = matrix_type->dimension();
        for (auto i = 0u; i < dim; i++) {
            for (auto j = 0u; j < dim; j++) {
                auto llvm_lhs_elem = lhs_is_matrix ? b.CreateExtractValue(llvm_lhs, {i, j}) : llvm_lhs;
                auto llvm_rhs_elem = rhs_is_matrix ? b.CreateExtractValue(llvm_rhs, {i, j}) : llvm_rhs;
                auto llvm_elem = b.CreateBinOp(llvm_op, llvm_lhs_elem, llvm_rhs_elem);
                llvm_result = b.CreateInsertValue(llvm_result, llvm_elem, {i, j});
            }
        }
        return llvm_result;
    }

    [[nodiscard]] llvm::Value *_translate_matrix_linalg_multiply(CurrentFunction &current, IRBuilder &b,
                                                                 const xir::ArithmeticInst *inst) noexcept {
        auto lhs = inst->operand(0u);
        auto rhs = inst->operand(1u);
        auto lhs_type = lhs->type();
        auto rhs_type = rhs->type();
        LUISA_ASSERT(lhs_type != nullptr && rhs_type != nullptr &&
                         (lhs_type->is_matrix() || lhs_type->is_float_vector()) &&
                         (rhs_type->is_matrix() || rhs_type->is_float_vector()) &&
                         (lhs_type->is_matrix() || rhs_type->is_matrix()) &&
                         lhs_type->dimension() == rhs_type->dimension() &&
                         lhs_type->element() == rhs_type->element(),
                     "Invalid matrix multiplication operands.");
        auto llvm_lhs = _lookup_value(current, b, lhs);
        auto llvm_rhs = _lookup_value(current, b, rhs);
        if (lhs_type->is_float_vector()) {
            LUISA_ASSERT(rhs_type->is_matrix() && inst->type() == lhs_type,
                         "Invalid vector-matrix multiplication result.");
            auto dim = lhs_type->dimension();
            auto llvm_result = static_cast<llvm::Value *>(
                llvm::PoisonValue::get(llvm_lhs->getType()));
            auto llvm_zero = llvm::ConstantFP::getNegativeZero(
                llvm_lhs->getType()->getScalarType());
            for (auto i = 0u; i < dim; i++) {
                auto llvm_rhs_storage_column =
                    b.CreateExtractValue(llvm_rhs, {i});
                auto llvm_rhs_column = static_cast<llvm::Value *>(
                    llvm::PoisonValue::get(llvm_lhs->getType()));
                for (auto j = 0u; j < dim; j++) {
                    auto llvm_element =
                        b.CreateExtractValue(llvm_rhs_storage_column, {j});
                    llvm_rhs_column = b.CreateInsertElement(
                        llvm_rhs_column, llvm_element, j);
                }
                auto llvm_product = b.CreateFMul(llvm_lhs, llvm_rhs_column);
                auto llvm_component = b.CreateFAddReduce(llvm_zero, llvm_product);
                llvm_result = b.CreateInsertElement(llvm_result, llvm_component, i);
            }
            return llvm_result;
        }
        auto llvm_func_name = [&] {
            if (rhs_type->is_float_vector()) {
                LUISA_ASSERT(inst->type() == rhs_type,
                             "Invalid matrix-vector multiplication result.");
                return luisa::format("luisa.matrix{}d.mul.vector", lhs_type->dimension());
            }
            LUISA_ASSERT(rhs_type->is_matrix() && inst->type() == lhs_type,
                         "Invalid matrix-matrix multiplication result.");
            return luisa::format("luisa.matrix{}d.mul.matrix", lhs_type->dimension());
        }();
        auto llvm_lhs_alloca = b.CreateAlloca(llvm_lhs->getType());
        auto llvm_rhs_alloca = b.CreateAlloca(llvm_rhs->getType());
        b.CreateStore(llvm_lhs, llvm_lhs_alloca);
        b.CreateStore(llvm_rhs, llvm_rhs_alloca);
        auto llvm_result_type = _translate_type(inst->type(), true);
        auto llvm_result_alloca = b.CreateAlloca(llvm_result_type);
        auto llvm_func = _llvm_module->getFunction(llvm::StringRef{llvm_func_name});
        LUISA_ASSERT(llvm_func != nullptr, "Function not found.");
        b.CreateCall(llvm_func, {llvm_lhs_alloca, llvm_rhs_alloca, llvm_result_alloca});
        return b.CreateLoad(llvm_result_type, llvm_result_alloca);
    }

    [[nodiscard]] llvm::Value *_translate_outer_product(CurrentFunction &current, IRBuilder &b,
                                                        const xir::ArithmeticInst *inst) noexcept {
        auto result_type = inst->type();
        LUISA_ASSERT(result_type != nullptr && result_type->is_matrix(),
                     "Invalid outer product operands.");
        // outer_product(lhs, rhs) = lhs * transpose(rhs)
        auto lhs = inst->operand(0u);
        auto rhs = inst->operand(1u);
        auto llvm_func_name = [lhs_type = lhs->type(), rhs_type = rhs->type(), result_type] {
            LUISA_ASSERT(lhs_type != nullptr && rhs_type != nullptr,
                         "Invalid outer product operands.");
            LUISA_ASSERT((lhs_type->is_matrix() && rhs_type->is_matrix()) ||
                             (lhs_type->is_vector() && rhs_type->is_vector()),
                         "Invalid outer product operands: ({}, {}) -> {}.",
                         lhs_type->description(), rhs_type->description(), result_type->description());
            LUISA_ASSERT(lhs_type->dimension() == result_type->dimension() &&
                             rhs_type->dimension() == result_type->dimension() &&
                             lhs_type->element() == result_type->element() &&
                             rhs_type->element() == result_type->element(),
                         "Dimension and/or element mismatch.");
            return lhs_type->is_matrix() && rhs_type->is_matrix() ?
                       luisa::format("luisa.matrix{}d.outer.product", result_type->dimension()) :
                       luisa::format("luisa.vector{}d.outer.product", result_type->dimension());
        }();
        auto llvm_lhs = _lookup_value(current, b, lhs);
        auto llvm_rhs = _lookup_value(current, b, rhs);
        auto llvm_lhs_alloca = b.CreateAlloca(llvm_lhs->getType());
        auto llvm_rhs_alloca = b.CreateAlloca(llvm_rhs->getType());
        b.CreateStore(llvm_lhs, llvm_lhs_alloca);
        b.CreateStore(llvm_rhs, llvm_rhs_alloca);
        auto llvm_result_type = _translate_type(result_type, true);
        auto llvm_result_alloca = b.CreateAlloca(llvm_result_type);
        auto llvm_func = _llvm_module->getFunction(llvm::StringRef{llvm_func_name});
        LUISA_ASSERT(llvm_func != nullptr, "Function not found.");
        b.CreateCall(llvm_func, {llvm_lhs_alloca, llvm_rhs_alloca, llvm_result_alloca});
        return b.CreateLoad(llvm_result_type, llvm_result_alloca);
    }

    [[nodiscard]] llvm::Value *_translate_texture_read(CurrentFunction &current, IRBuilder &b,
                                                       const xir::ResourceReadInst *inst) noexcept {
        auto view = inst->operand(0u);
        LUISA_ASSERT(view->type()->is_texture(), "Invalid texture view type.");
        auto coord = inst->operand(1u);
        auto llvm_func_name = [t = view->type()] {
            auto dim = t->dimension();
            switch (t->element()->tag()) {
                case Type::Tag::INT32: return luisa::format("luisa.texture{}d.read.int", dim);
                case Type::Tag::UINT32: return luisa::format("luisa.texture{}d.read.uint", dim);
                case Type::Tag::FLOAT32: return luisa::format("luisa.texture{}d.read.float", dim);
                default: break;
            }
            LUISA_ERROR_WITH_LOCATION("Unsupported texture element type: {}.", t->element()->description());
        }();
        auto llvm_func = _llvm_module->getFunction(llvm::StringRef{llvm_func_name});
        LUISA_ASSERT(llvm_func != nullptr, "Function not found.");
        auto llvm_view = _lookup_value(current, b, view);
        auto llvm_coord = _lookup_value(current, b, coord);
        auto llvm_view_alloca = b.CreateAlloca(llvm_view->getType());
        auto llvm_coord_alloca = b.CreateAlloca(llvm_coord->getType());
        b.CreateStore(llvm_view, llvm_view_alloca);
        b.CreateStore(llvm_coord, llvm_coord_alloca);
        auto llvm_value_type = _translate_type(inst->type(), true);
        auto llvm_value_alloca = b.CreateAlloca(llvm_value_type);
        b.CreateCall(llvm_func, {llvm_view_alloca, llvm_coord_alloca, llvm_value_alloca});
        return b.CreateLoad(llvm_value_type, llvm_value_alloca);
    }

    [[nodiscard]] llvm::Value *_translate_texture_write(CurrentFunction &current, IRBuilder &b,
                                                        const xir::ResourceWriteInst *inst) noexcept {
        auto view = inst->operand(0u);
        LUISA_ASSERT(view->type()->is_texture(), "Invalid texture view type.");
        auto coord = inst->operand(1u);
        auto value = inst->operand(2u);
        auto llvm_func_name = [t = view->type()] {
            auto dim = t->dimension();
            switch (t->element()->tag()) {
                case Type::Tag::INT32: return luisa::format("luisa.texture{}d.write.int", dim);
                case Type::Tag::UINT32: return luisa::format("luisa.texture{}d.write.uint", dim);
                case Type::Tag::FLOAT32: return luisa::format("luisa.texture{}d.write.float", dim);
                default: break;
            }
            LUISA_ERROR_WITH_LOCATION("Unsupported texture element type: {}.", t->element()->description());
        }();
        auto llvm_func = _llvm_module->getFunction(llvm::StringRef{llvm_func_name});
        LUISA_ASSERT(llvm_func != nullptr, "Function not found.");
        auto llvm_view = _lookup_value(current, b, view);
        auto llvm_coord = _lookup_value(current, b, coord);
        auto llvm_value = _lookup_value(current, b, value);
        auto llvm_view_alloca = b.CreateAlloca(llvm_view->getType());
        auto llvm_coord_alloca = b.CreateAlloca(llvm_coord->getType());
        auto llvm_value_alloca = b.CreateAlloca(llvm_value->getType());
        b.CreateStore(llvm_view, llvm_view_alloca);
        b.CreateStore(llvm_coord, llvm_coord_alloca);
        b.CreateStore(llvm_value, llvm_value_alloca);
        return b.CreateCall(llvm_func, {llvm_view_alloca, llvm_coord_alloca, llvm_value_alloca});
    }

    [[nodiscard]] llvm::Value *_translate_texture_size(CurrentFunction &current, IRBuilder &b, const xir::ResourceQueryInst *inst) noexcept {
        auto view = inst->operand(0u);
        LUISA_ASSERT(view->type()->is_texture(), "Invalid texture view type.");
        auto llvm_func_name = luisa::format("luisa.texture{}d.size", view->type()->dimension());
        auto llvm_func = _llvm_module->getFunction(llvm::StringRef{llvm_func_name});
        LUISA_ASSERT(llvm_func != nullptr, "Function not found.");
        auto llvm_view = _lookup_value(current, b, view);
        auto llvm_view_alloca = b.CreateAlloca(llvm_view->getType());
        b.CreateStore(llvm_view, llvm_view_alloca);
        auto llvm_size_type = _translate_type(inst->type(), true);
        auto llvm_size_alloca = b.CreateAlloca(llvm_size_type);
        b.CreateCall(llvm_func, {llvm_view_alloca, llvm_size_alloca});
        return b.CreateLoad(llvm_size_type, llvm_size_alloca);
    }

    [[nodiscard]] llvm::Value *_translate_matrix_inverse(CurrentFunction &current, IRBuilder &b, const xir::ArithmeticInst *inst) noexcept {
        auto m = inst->operand(0u);
        LUISA_ASSERT(m->type()->is_matrix() && m->type() == inst->type(), "Invalid matrix type.");
        auto llvm_m = _lookup_value(current, b, m);
        auto llvm_func_name = luisa::format("luisa.matrix{}d.inverse", m->type()->dimension());
        auto llvm_func = _llvm_module->getFunction(llvm::StringRef{llvm_func_name});
        LUISA_ASSERT(llvm_func != nullptr, "Function not found.");
        auto llvm_m_alloca = b.CreateAlloca(llvm_m->getType());
        b.CreateStore(llvm_m, llvm_m_alloca);
        auto llvm_result_type = _translate_type(inst->type(), true);
        auto llvm_result_alloca = b.CreateAlloca(llvm_result_type);
        b.CreateCall(llvm_func, {llvm_m_alloca, llvm_result_alloca});
        return b.CreateLoad(llvm_result_type, llvm_result_alloca);
    }

    [[nodiscard]] llvm::Value *_translate_matrix_determinant(CurrentFunction &current, IRBuilder &b, const xir::ArithmeticInst *inst) noexcept {
        auto m = inst->operand(0u);
        LUISA_ASSERT(m->type()->is_matrix() && m->type()->element() == inst->type(), "Invalid matrix type.");
        auto llvm_m = _lookup_value(current, b, m);
        auto llvm_func_name = luisa::format("luisa.matrix{}d.determinant", m->type()->dimension());
        auto llvm_func = _llvm_module->getFunction(llvm::StringRef{llvm_func_name});
        LUISA_ASSERT(llvm_func != nullptr, "Function not found.");
        auto llvm_m_alloca = b.CreateAlloca(llvm_m->getType());
        b.CreateStore(llvm_m, llvm_m_alloca);
        return b.CreateCall(llvm_func, {llvm_m_alloca});
    }

    [[nodiscard]] llvm::Value *_translate_bindless_texture_access(CurrentFunction &current, IRBuilder &b,
                                                                  llvm::StringRef llvm_func_name,
                                                                  const xir::Instruction *inst) noexcept {
        auto llvm_bindless = _lookup_value(current, b, inst->operand(0u));
        auto llvm_bindless_alloca = b.CreateAlloca(llvm_bindless->getType());
        b.CreateStore(llvm_bindless, llvm_bindless_alloca);
        auto llvm_slot_index = _lookup_value(current, b, inst->operand(1u));
        llvm_slot_index = b.CreateZExtOrTrunc(llvm_slot_index, b.getInt32Ty());
        llvm::SmallVector<llvm::Value *, 8u> llvm_args{llvm_bindless_alloca, llvm_slot_index};
        for (auto arg_use : inst->operand_uses().subspan(2)) {
            auto arg = arg_use->value();
            auto llvm_arg = _lookup_value(current, b, arg);
            if (arg->type()->is_scalar()) {
                llvm_args.emplace_back(llvm_arg);
            } else {
                auto llvm_arg_alloca = b.CreateAlloca(llvm_arg->getType());
                b.CreateStore(llvm_arg, llvm_arg_alloca);
                llvm_args.emplace_back(llvm_arg_alloca);
            }
        }
        auto llvm_result_type = _translate_type(inst->type(), true);
        auto llvm_result_alloca = b.CreateAlloca(llvm_result_type);
        llvm_args.emplace_back(llvm_result_alloca);
        auto llvm_func = _llvm_module->getFunction(llvm_func_name);
        LUISA_ASSERT(llvm_func != nullptr, "Function not found.");
        b.CreateCall(llvm_func, llvm_args);
        return b.CreateLoad(llvm_result_type, llvm_result_alloca);
    }

    [[nodiscard]] llvm::Value *_translate_accel_access(CurrentFunction &current, IRBuilder &b,
                                                       llvm::StringRef llvm_func_name,
                                                       const xir::Instruction *inst) noexcept {
        auto llvm_func = _llvm_module->getFunction(llvm_func_name);
        LUISA_ASSERT(llvm_func != nullptr, "Function not found.");
        auto llvm_accel = _lookup_value(current, b, inst->operand(0u));
        auto llvm_accel_alloca = b.CreateAlloca(llvm_accel->getType());
        b.CreateStore(llvm_accel, llvm_accel_alloca);
        llvm::SmallVector<llvm::Value *, 8u> llvm_args{llvm_accel_alloca};
        for (auto arg_use : inst->operand_uses().subspan(1)) {
            auto arg = arg_use->value();
            auto llvm_arg = _lookup_value(current, b, arg);
            if (arg->type()->is_scalar()) {
                // Scalar ABI types in the embedded device library are
                // target-dependent (notably C++ bool may be i1 while XIR
                // bool is represented as i8). Match the declared callee
                // parameter instead of assuming an i32 integer ABI.
                auto llvm_parameter_type =
                    llvm_func->getFunctionType()->getParamType(
                        llvm_args.size());
                if (llvm_arg->getType()->isIntegerTy() &&
                    llvm_parameter_type->isIntegerTy()) {
                    llvm_arg = b.CreateZExtOrTrunc(
                        llvm_arg, llvm_parameter_type);
                }
                llvm_args.emplace_back(llvm_arg);
            } else {
                auto llvm_arg_alloca = b.CreateAlloca(llvm_arg->getType());
                auto alignment = std::max<size_t>(_get_type_alignment(arg->type()), llvm_arg_alloca->getAlign().value());
                llvm_arg_alloca->setAlignment(llvm::Align{alignment});
                b.CreateStore(llvm_arg, llvm_arg_alloca);
                llvm_args.emplace_back(llvm_arg_alloca);
            }
        }
        if (auto result_type = inst->type()) {
            auto llvm_result_type = _translate_type(result_type, true);
            auto llvm_result_alloca = b.CreateAlloca(llvm_result_type);
            llvm_args.emplace_back(llvm_result_alloca);
            b.CreateCall(llvm_func, llvm_args);
            return b.CreateLoad(llvm_result_type, llvm_result_alloca);
        }
        // void return
        return b.CreateCall(llvm_func, llvm_args);
    }

    [[nodiscard]] llvm::Value *_translate_ray_query_object_read_inst(CurrentFunction &current, IRBuilder &b, const xir::RayQueryObjectReadInst *inst) noexcept {
        LUISA_DEBUG_ASSERT(inst->operand_count() == 1u, "Invalid ray query object read instruction.");
        auto llvm_func_name = [op = inst->op()] {
            using namespace std::string_view_literals;
            switch (op) {
                case xir::RayQueryObjectReadOp::RAY_QUERY_OBJECT_WORLD_SPACE_RAY: return "luisa.ray.query.object.world.space.ray"sv;
                case xir::RayQueryObjectReadOp::RAY_QUERY_OBJECT_PROCEDURAL_CANDIDATE_HIT: return "luisa.ray.query.object.procedural.candidate.hit"sv;
                case xir::RayQueryObjectReadOp::RAY_QUERY_OBJECT_TRIANGLE_CANDIDATE_HIT: return "luisa.ray.query.object.surface.candidate.hit"sv;
                case xir::RayQueryObjectReadOp::RAY_QUERY_OBJECT_COMMITTED_HIT: return "luisa.ray.query.object.committed.hit"sv;
                case xir::RayQueryObjectReadOp::RAY_QUERY_OBJECT_IS_TRIANGLE_CANDIDATE: LUISA_NOT_IMPLEMENTED();
                case xir::RayQueryObjectReadOp::RAY_QUERY_OBJECT_IS_PROCEDURAL_CANDIDATE: LUISA_NOT_IMPLEMENTED();
                case xir::RayQueryObjectReadOp::RAY_QUERY_OBJECT_IS_TERMINATED: LUISA_NOT_IMPLEMENTED();
                default: break;
            }
            LUISA_ERROR_WITH_LOCATION("Invalid ray query object read operation.");
        }();
        auto llvm_func = _llvm_module->getFunction(llvm_func_name);
        LUISA_ASSERT(llvm_func != nullptr, "Function not found.");
        auto llvm_object = _lookup_value(current, b, inst->operand(0));
        auto llvm_out_type = _translate_type(inst->type(), true);
        auto llvm_out_alloca = b.CreateAlloca(llvm_out_type);
        b.CreateCall(llvm_func, {llvm_object, llvm_out_alloca});
        return b.CreateLoad(llvm_out_type, llvm_out_alloca);
    }

    [[nodiscard]] llvm::Value *_translate_ray_query_object_write_inst(CurrentFunction &current, IRBuilder &b, const xir::RayQueryObjectWriteInst *inst) noexcept {
        LUISA_DEBUG_ASSERT(inst->operand_count() >= 1u, "Invalid ray query object write instruction.");
        auto llvm_func_name = [op = inst->op()] {
            using namespace std::string_view_literals;
            switch (op) {
                case xir::RayQueryObjectWriteOp::RAY_QUERY_OBJECT_COMMIT_TRIANGLE: return "luisa.ray.query.object.commit.surface.hit"sv;
                case xir::RayQueryObjectWriteOp::RAY_QUERY_OBJECT_COMMIT_PROCEDURAL: return "luisa.ray.query.object.commit.procedural.hit"sv;
                case xir::RayQueryObjectWriteOp::RAY_QUERY_OBJECT_TERMINATE: return "luisa.ray.query.object.terminate"sv;
                case xir::RayQueryObjectWriteOp::RAY_QUERY_OBJECT_PROCEED: LUISA_NOT_IMPLEMENTED();
                default: break;
            }
            LUISA_ERROR_WITH_LOCATION("Invalid ray query object write operation.");
        }();
        auto llvm_func = _llvm_module->getFunction(llvm_func_name);
        LUISA_ASSERT(llvm_func != nullptr, "Function not found.");
        llvm::SmallVector<llvm::Value *, 4u> llvm_args;
        llvm_args.reserve(inst->operand_count());
        for (auto op_use : inst->operand_uses()) {
            llvm_args.emplace_back(_lookup_value(current, b, op_use->value()));
        }
        return b.CreateCall(llvm_func, llvm_args);
    }

    [[nodiscard]] llvm::Constant *_generate_ray_query_pipeline_function_wrapper(llvm::StringRef name, size_t capture_count,
                                                                                llvm::Function *llvm_func, uint sreg_usage,
                                                                                llvm::StructType *llvm_capture_struct,
                                                                                llvm::ArrayRef<uint> member_to_arg) noexcept {
        // void wrapper(ptr query_object, ptr captured_args) {
        //   impl(query_object, captured_args.member(inv(member_to_arg)[0])...);
        // }
        auto llvm_ptr_type = llvm::PointerType::get(_llvm_context, 0);
        if (llvm_func == nullptr) { return llvm::Constant::getNullValue(llvm_ptr_type); }
        auto llvm_void_type = llvm::Type::getVoidTy(_llvm_context);
        auto llvm_wrapper_func_type = llvm::FunctionType::get(llvm_void_type, {llvm_ptr_type, llvm_ptr_type}, false);
        auto llvm_wrapper_func = llvm::Function::Create(llvm_wrapper_func_type, llvm::Function::PrivateLinkage, name, _llvm_module);
        auto entry = llvm::BasicBlock::Create(_llvm_context, "entry", llvm_wrapper_func);
        IRBuilder b{entry};
        llvm::SmallVector<llvm::Value *> llvm_args;
        llvm_args.resize(llvm_func->arg_size());
        auto llvm_wrapper_query_object = llvm_wrapper_func->getArg(0);
        llvm_args[0] = llvm_wrapper_query_object;
        for (auto i = 0u; i < CurrentFunction::builtin_variable_count; i++) {
            if ((sreg_usage & (1u << i)) == 0u) {
                auto arg_index = 1u + capture_count + i;
                llvm_args[arg_index] = llvm::Constant::getNullValue(llvm_func->getArg(arg_index)->getType());
            }
        }
        if (!member_to_arg.empty()) {
            auto llvm_wrapper_captured_struct = b.CreateLoad(llvm_capture_struct, llvm_wrapper_func->getArg(1));
            for (auto member_index = 0; member_index < member_to_arg.size(); member_index++) {
                auto arg_index = member_to_arg[member_index];
                auto llvm_member = b.CreateExtractValue(llvm_wrapper_captured_struct, member_index);
                llvm_args[1u + arg_index] = llvm_member;
            }
#ifndef NDEBUG
            for (auto llvm_arg : llvm_args) {
                LUISA_ASSERT(llvm_arg != nullptr, "Invalid argument.");
            }
#endif
        }
        LUISA_DEBUG_ASSERT(llvm_func != nullptr, "Invalid ray query pipeline function.");
        auto llvm_call = b.CreateCall(llvm_func, llvm_args);
        llvm_call->setCallingConv(llvm::CallingConv::Fast);
        b.CreateRetVoid();
        return llvm_wrapper_func;
    }

    [[nodiscard]] llvm::Value *_translate_ray_query_pipeline_inst(CurrentFunction &current, IRBuilder &b, const xir::RayQueryPipelineInst *inst) noexcept {
        // find the built-in function
        auto query_object = inst->query_object();
        LUISA_DEBUG_ASSERT(query_object != nullptr &&
                               (query_object->type() == Type::of<RayQueryAll>() ||
                                query_object->type() == Type::of<RayQueryAny>()),
                           "Invalid ray query object type.");
        auto llvm_query_object = _lookup_value(current, b, query_object);
        using namespace std::string_view_literals;
        auto llvm_func_name = query_object->type() == Type::of<RayQueryAll>() ?
                                  "luisa.ray.query.pipeline.all"sv :
                                  "luisa.ray.query.pipeline.any"sv;
        auto llvm_func = _llvm_module->getFunction(llvm_func_name);
        LUISA_ASSERT(llvm_func != nullptr, "Function not found.");
        // on_surface and on_procedural callbacks
        auto non_empty_or_null_with_sreg_usage = [&](const xir::Function *f) noexcept -> std::pair<llvm::Function *, uint> {
            if (!f) { return std::make_pair(nullptr, 0u); }
            if (auto def = f->definition()) {
                if (auto t = def->body_block()->terminator();
                    t == def->body_block()->instructions().front() &&
                    (t->derived_instruction_tag() == xir::DerivedInstructionTag::RETURN ||
                     t->derived_instruction_tag() == xir::DerivedInstructionTag::UNREACHABLE)) {
                    return std::make_pair(nullptr, 0u);
                }
            }
            auto llvm_f = _lookup_value(current, b, f);
            LUISA_DEBUG_ASSERT(llvm_f == nullptr || llvm::isa<llvm::Function>(llvm_f), "Invalid function.");
            auto usage = _analyze_special_register_usage(f);
            return std::make_pair(llvm::cast<llvm::Function>(llvm_f), usage);
        };
        auto [llvm_on_surface_func, on_surface_func_sreg_usage] = non_empty_or_null_with_sreg_usage(inst->on_surface_function());
        auto [llvm_on_procedural_func, on_procedural_func_sreg_usage] = non_empty_or_null_with_sreg_usage(inst->on_procedural_function());
        auto sreg_usage = on_surface_func_sreg_usage | on_procedural_func_sreg_usage;
        // create capture argument struct
        auto captured_arg_uses = inst->captured_argument_uses();
        llvm::SmallVector<uint> member_to_captured_arg;
        member_to_captured_arg.reserve(captured_arg_uses.size() + CurrentFunction::builtin_variable_count);
        auto llvm_capture = [&]() noexcept -> std::pair<llvm::StructType *, llvm::Value *> {
            auto llvm_ptr_type = llvm::PointerType::get(_llvm_context, 0);
            // degenerate case: no capture needed
            if (captured_arg_uses.empty() && sreg_usage == 0u) {
                return std::make_pair(nullptr, llvm::Constant::getNullValue(llvm_ptr_type));
            }
            // general case
            llvm::SmallVector<size_t> captured_arg_ranks;
            captured_arg_ranks.reserve(captured_arg_uses.size() + CurrentFunction::builtin_variable_count);
            for (auto arg_use : captured_arg_uses) {
                auto arg = arg_use->value();
                LUISA_DEBUG_ASSERT(arg != nullptr && arg->type() != nullptr, "Invalid null captured arg.");
                auto arg_rank = [&] {
                    auto size = arg->is_lvalue() ? 8u : _get_type_size(arg->type());
                    auto alignment = arg->is_lvalue() ? 8u : _get_type_alignment(arg->type());
                    return (alignment << 32u) | size;
                }();
                auto arg_index = captured_arg_ranks.size();
                captured_arg_ranks.emplace_back(arg_rank);
                member_to_captured_arg.emplace_back(arg_index);
            }
            // built-in variables
            for (auto i = 0u; i < CurrentFunction::builtin_variable_count; i++) {
                if (sreg_usage & (1u << i)) {
                    auto arg_rank = ~0u;
                    auto arg_index = captured_arg_uses.size() + i;
                    captured_arg_ranks.emplace_back(arg_rank);
                    member_to_captured_arg.emplace_back(arg_index);
                }
            }
            // sort the captured args in descending order of ranks
            std::stable_sort(member_to_captured_arg.begin(), member_to_captured_arg.end(), [&](auto lhs, auto rhs) noexcept {
                return captured_arg_ranks[lhs] > captured_arg_ranks[rhs];
            });
            llvm::SmallVector<llvm::Type *> llvm_member_types;
            llvm_member_types.reserve(member_to_captured_arg.size());
            for (auto arg_index : member_to_captured_arg) {
                if (arg_index < captured_arg_uses.size()) {
                    auto arg = captured_arg_uses[arg_index]->value();
                    llvm_member_types.emplace_back(arg->is_lvalue() ? llvm_ptr_type : _translate_type(arg->type(), true));
                } else {
                    auto builtin_index = arg_index - captured_arg_uses.size();
                    llvm_member_types.emplace_back(current.builtin_variables[builtin_index]->getType());
                }
            }
            // create the struct type
            auto llvm_struct_type = llvm::StructType::get(_llvm_context, llvm_member_types);
            // alloca
            auto llvm_struct_alloca = b.CreateAlloca(llvm_struct_type);
            llvm_struct_alloca->setAlignment(llvm::Align{16});
            // store captured args
            for (auto member_index = 0u; member_index < member_to_captured_arg.size(); member_index++) {
                auto arg_index = member_to_captured_arg[member_index];
                auto llvm_arg = arg_index < captured_arg_uses.size() ?
                                    _lookup_value(current, b, captured_arg_uses[arg_index]->value()) :
                                    current.builtin_variables[arg_index - captured_arg_uses.size()];
                auto llvm_member = b.CreateStructGEP(llvm_struct_type, llvm_struct_alloca, member_index);
                b.CreateStore(llvm_arg, llvm_member);
            }
            return std::make_pair(llvm_struct_type, llvm_struct_alloca);
        }();
        // create the wrapper functions
        auto llvm_on_surface_func_wrapper = _generate_ray_query_pipeline_function_wrapper(
            "ray.query.pipeline.on.surface.wrapper", captured_arg_uses.size(),
            llvm_on_surface_func, on_surface_func_sreg_usage,
            llvm_capture.first, member_to_captured_arg);
        auto llvm_on_procedural_func_wrapper = _generate_ray_query_pipeline_function_wrapper(
            "ray.query.pipeline.on.procedural.wrapper", captured_arg_uses.size(),
            llvm_on_procedural_func, on_procedural_func_sreg_usage,
            llvm_capture.first, member_to_captured_arg);
        return b.CreateCall(llvm_func, {llvm_query_object, llvm_capture.second, llvm_on_surface_func_wrapper, llvm_on_procedural_func_wrapper});
    }

    [[nodiscard]] llvm::Value *_translate_atomic_op(CurrentFunction &current, IRBuilder &b,
                                                    const char *op_name, const xir::AtomicInst *inst,
                                                    bool byte_address = false) noexcept {
        auto llvm_func_name = [&] {
            switch (inst->type()->tag()) {
                case Type::Tag::INT32: return luisa::format("luisa.atomic.{}.int", op_name);
                case Type::Tag::UINT32: return luisa::format("luisa.atomic.{}.uint", op_name);
                case Type::Tag::INT64: return luisa::format("luisa.atomic.{}.long", op_name);
                case Type::Tag::UINT64: return luisa::format("luisa.atomic.{}.ulong", op_name);
                case Type::Tag::FLOAT32: return luisa::format("luisa.atomic.{}.float", op_name);
                default: break;
            }
            LUISA_ERROR_WITH_LOCATION("Unsupported atomic operation value type: {}.", inst->type()->description());
        }();
        auto value_count = inst->value_count();
        auto indices = inst->index_uses();
        auto llvm_elem_ptr = [&] {
            auto base = inst->base();
            LUISA_ASSERT(base != nullptr && base->type() != nullptr, "Invalid atomic operation base.");
            if (base->type()->is_buffer()) {
                LUISA_ASSERT(base->type()->is_buffer(), "Invalid buffer type.");
                auto buffer_elem_type = base->type()->element();
                LUISA_ASSERT(buffer_elem_type != nullptr, "Invalid buffer element type.");
                auto slot = indices.front()->value();
                auto llvm_elem = _get_buffer_element_ptr(current, b, base, slot, byte_address);
                indices = indices.subspan(1);
                if (byte_address) {
                    LUISA_ASSERT(2 + value_count == inst->operand_count(), "Invalid atomic operation.");
                }
                if (!indices.empty()) {
                    llvm_elem = _translate_gep(current, b, inst->type(),
                                               buffer_elem_type, llvm_elem, indices);
                }
                return llvm_elem;
            }
            // otherwise, it's a shared memory atomic operation
            LUISA_ASSERT(base->isa<xir::AllocaInst>() && static_cast<const xir::AllocaInst *>(base)->is_shared(),
                         "Invalid shared memory atomic operation base.");
            llvm_func_name.append(".smem");
            auto llvm_smem = _lookup_value(current, b, base);
            return _translate_gep(current, b, inst->type(),
                                  base->type(), llvm_smem, indices);
        }();
        auto llvm_func = _llvm_module->getFunction(llvm::StringRef{llvm_func_name});
        LUISA_ASSERT(llvm_func != nullptr && llvm_func->arg_size() == 1 + value_count, "Invalid atomic operation function.");
        llvm::SmallVector<llvm::Value *, 4u> llvm_args{llvm_elem_ptr};
        for (auto value_uses : inst->value_uses()) {
            auto value = value_uses->value();
            LUISA_ASSERT(value->type() == inst->type(), "Atomic operation value type mismatch.");
            auto llvm_value = _lookup_value(current, b, value);
            llvm_args.emplace_back(llvm_value);
        }
        return b.CreateCall(llvm_func, llvm_args);
    }

    [[nodiscard]] llvm::Value *_translate_arithmetic_inst(CurrentFunction &current, IRBuilder &b, const xir::ArithmeticInst *inst) noexcept {
        switch (inst->op()) {
            case xir::ArithmeticOp::UNARY_MINUS: return _translate_unary_minus(current, b, inst->operand(0u));
            case xir::ArithmeticOp::UNARY_BIT_NOT: return _translate_unary_bit_not(current, b, inst->operand(0u));
            case xir::ArithmeticOp::BINARY_ADD: return _translate_binary_add(current, b, inst->operand(0u), inst->operand(1u));
            case xir::ArithmeticOp::BINARY_SUB: return _translate_binary_sub(current, b, inst->operand(0u), inst->operand(1u));
            case xir::ArithmeticOp::BINARY_MUL: return _translate_binary_mul(current, b, inst->operand(0u), inst->operand(1u));
            case xir::ArithmeticOp::BINARY_DIV: return _translate_binary_div(current, b, inst->operand(0u), inst->operand(1u));
            case xir::ArithmeticOp::BINARY_MOD: return _translate_binary_mod(current, b, inst->operand(0u), inst->operand(1u));
            case xir::ArithmeticOp::BINARY_BIT_AND: return _translate_binary_bit_op(current, b, inst->op(), inst->operand(0u), inst->operand(1u));
            case xir::ArithmeticOp::BINARY_BIT_OR: return _translate_binary_bit_op(current, b, inst->op(), inst->operand(0u), inst->operand(1u));
            case xir::ArithmeticOp::BINARY_BIT_XOR: return _translate_binary_bit_op(current, b, inst->op(), inst->operand(0u), inst->operand(1u));
            case xir::ArithmeticOp::BINARY_SHIFT_LEFT: return _translate_binary_shift_left(current, b, inst->operand(0u), inst->operand(1u));
            case xir::ArithmeticOp::BINARY_SHIFT_RIGHT: return _translate_binary_shift_right(current, b, inst->operand(0u), inst->operand(1u));
            case xir::ArithmeticOp::BINARY_ROTATE_LEFT: return _translate_binary_rotate_left(current, b, inst->operand(0u), inst->operand(1u));
            case xir::ArithmeticOp::BINARY_ROTATE_RIGHT: return _translate_binary_rotate_right(current, b, inst->operand(0u), inst->operand(1u));
            case xir::ArithmeticOp::BINARY_LESS: return _translate_binary_less(current, b, inst->operand(0u), inst->operand(1u));
            case xir::ArithmeticOp::BINARY_GREATER: return _translate_binary_greater(current, b, inst->operand(0u), inst->operand(1u));
            case xir::ArithmeticOp::BINARY_LESS_EQUAL: return _translate_binary_less_equal(current, b, inst->operand(0u), inst->operand(1u));
            case xir::ArithmeticOp::BINARY_GREATER_EQUAL: return _translate_binary_greater_equal(current, b, inst->operand(0u), inst->operand(1u));
            case xir::ArithmeticOp::BINARY_EQUAL: return _translate_binary_equal(current, b, inst->operand(0u), inst->operand(1u));
            case xir::ArithmeticOp::BINARY_NOT_EQUAL: return _translate_binary_not_equal(current, b, inst->operand(0u), inst->operand(1u));
            case xir::ArithmeticOp::ALL: {
                auto llvm_operand = _lookup_value(current, b, inst->operand(0u));
                return b.CreateAndReduce(llvm_operand);
            }
            case xir::ArithmeticOp::ANY: {
                auto llvm_operand = _lookup_value(current, b, inst->operand(0u));
                return b.CreateOrReduce(llvm_operand);
            }
            case xir::ArithmeticOp::SELECT: {
                // note that the order of operands is reversed
                auto llvm_false_value = _lookup_value(current, b, inst->operand(0u));
                auto llvm_true_value = _lookup_value(current, b, inst->operand(1u));
                auto llvm_condition = _lookup_value(current, b, inst->operand(2u));
                auto llvm_zero = llvm::Constant::getNullValue(llvm_condition->getType());
                auto llvm_cmp = b.CreateICmpNE(llvm_condition, llvm_zero);
                return b.CreateSelect(llvm_cmp, llvm_true_value, llvm_false_value);
            }
            case xir::ArithmeticOp::CLAMP: {
                // clamp(x, lb, ub) = min(max(x, lb), ub)
                auto x_type = inst->operand(0u)->type();
                auto lb_type = inst->operand(1u)->type();
                auto ub_type = inst->operand(2u)->type();
                LUISA_ASSERT(x_type != nullptr && x_type == lb_type && x_type == ub_type, "Type mismatch for clamp.");
                auto llvm_x = _lookup_value(current, b, inst->operand(0u));
                auto llvm_lb = _lookup_value(current, b, inst->operand(1u));
                auto llvm_ub = _lookup_value(current, b, inst->operand(2u));
                auto elem_type = x_type->is_vector() ? x_type->element() : x_type;
                switch (elem_type->tag()) {
                    case Type::Tag::INT8: [[fallthrough]];
                    case Type::Tag::INT16: [[fallthrough]];
                    case Type::Tag::INT32: [[fallthrough]];
                    case Type::Tag::INT64: {
                        auto llvm_max = b.CreateBinaryIntrinsic(llvm::Intrinsic::smax, llvm_x, llvm_lb);
                        return b.CreateBinaryIntrinsic(llvm::Intrinsic::smin, llvm_max, llvm_ub);
                    }
                    case Type::Tag::UINT8: [[fallthrough]];
                    case Type::Tag::UINT16: [[fallthrough]];
                    case Type::Tag::UINT32: [[fallthrough]];
                    case Type::Tag::UINT64: {
                        auto llvm_max = b.CreateBinaryIntrinsic(llvm::Intrinsic::umax, llvm_x, llvm_lb);
                        return b.CreateBinaryIntrinsic(llvm::Intrinsic::umin, llvm_max, llvm_ub);
                    }
                    case Type::Tag::FLOAT16: [[fallthrough]];
                    case Type::Tag::FLOAT32: [[fallthrough]];
                    case Type::Tag::FLOAT64: {
                        auto llvm_max = b.CreateBinaryIntrinsic(llvm::Intrinsic::maxnum, llvm_x, llvm_lb);
                        return b.CreateBinaryIntrinsic(llvm::Intrinsic::minnum, llvm_max, llvm_ub);
                    }
                    default: break;
                }
                LUISA_ERROR_WITH_LOCATION("Invalid operand type for clamp operation: {}.", x_type->description());
            }
            case xir::ArithmeticOp::SATURATE: {
                // saturate(x) = min(max(x, 0), 1)
                auto llvm_x = _lookup_value(current, b, inst->operand(0u));
                LUISA_ASSERT(llvm_x->getType()->isFPOrFPVectorTy(), "Invalid operand type for saturate operation.");
                auto llvm_zero = llvm::ConstantFP::get(llvm_x->getType(), 0.);
                auto llvm_one = llvm::ConstantFP::get(llvm_x->getType(), 1.);
                auto llvm_max = b.CreateBinaryIntrinsic(llvm::Intrinsic::maxnum, llvm_x, llvm_zero);
                return b.CreateBinaryIntrinsic(llvm::Intrinsic::minnum, llvm_max, llvm_one);
            }
            case xir::ArithmeticOp::LERP: {
                // lerp(a, b, t) = (b - a) * t + a = fma(t, b - a, a)
                auto type = inst->type();
                auto va_type = inst->operand(0u)->type();
                auto vb_type = inst->operand(1u)->type();
                auto t_type = inst->operand(2u)->type();
                LUISA_ASSERT(type != nullptr && va_type == type && vb_type == type &&
                                 (t_type == type ||
                                  (type->is_vector() && t_type == type->element())),
                             "Type mismatch for lerp.");
                auto llvm_va = _lookup_value(current, b, inst->operand(0u));
                auto llvm_vb = _lookup_value(current, b, inst->operand(1u));
                auto llvm_t = _broadcast_like(
                    b, _lookup_value(current, b, inst->operand(2u)), llvm_va);
                LUISA_ASSERT(llvm_va->getType()->isFPOrFPVectorTy(), "Invalid operand type for lerp operation.");
                auto llvm_diff = b.CreateFSub(llvm_vb, llvm_va);
                return b.CreateIntrinsic(llvm_va->getType(), llvm::Intrinsic::fma, {llvm_t, llvm_diff, llvm_va});
            }
            case xir::ArithmeticOp::SMOOTHSTEP: {
                // smoothstep(edge0, edge1, x) = t * t * (3.0 - 2.0 * t), where t = clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0)
                auto type = inst->type();
                LUISA_ASSERT(type != nullptr &&
                                 type == inst->operand(2u)->type() &&
                                 (type == inst->operand(0u)->type() ||
                                  (type->is_vector() &&
                                   type->element() == inst->operand(0u)->type())) &&
                                 (type == inst->operand(1u)->type() ||
                                  (type->is_vector() &&
                                   type->element() == inst->operand(1u)->type())),
                             "Type mismatch for smoothstep.");
                auto elem_type = type->is_vector() ? type->element() : type;
                LUISA_ASSERT(elem_type->is_float16() || elem_type->is_float32() || elem_type->is_float64(),
                             "Invalid operand type for smoothstep operation.");
                auto llvm_x = _lookup_value(current, b, inst->operand(2u));
                auto llvm_edge0 = _broadcast_like(
                    b, _lookup_value(current, b, inst->operand(0u)), llvm_x);
                auto llvm_edge1 = _broadcast_like(
                    b, _lookup_value(current, b, inst->operand(1u)), llvm_x);
                // constant 0, 1, -2, and 3
                auto llvm_elem_type = _translate_type(elem_type, false);
                auto llvm_zero = llvm::ConstantFP::get(llvm_elem_type, 0.);
                auto llvm_one = llvm::ConstantFP::get(llvm_elem_type, 1.);
                auto llvm_minus_two = llvm::ConstantFP::get(llvm_elem_type, -2.);
                auto llvm_three = llvm::ConstantFP::get(llvm_elem_type, 3.);
                if (type->is_vector()) {
                    auto dim = llvm::ElementCount::getFixed(type->dimension());
                    llvm_zero = llvm::ConstantVector::getSplat(dim, llvm_zero);
                    llvm_one = llvm::ConstantVector::getSplat(dim, llvm_one);
                    llvm_minus_two = llvm::ConstantVector::getSplat(dim, llvm_minus_two);
                    llvm_three = llvm::ConstantVector::getSplat(dim, llvm_three);
                }
                // t = clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0)
                auto llvm_x_minus_edge0 = b.CreateFSub(llvm_x, llvm_edge0);
                auto llvm_edge1_minus_edge0 = b.CreateFSub(llvm_edge1, llvm_edge0);
                auto llvm_t = b.CreateFDiv(llvm_x_minus_edge0, llvm_edge1_minus_edge0);
                // clamp t
                llvm_t = b.CreateBinaryIntrinsic(llvm::Intrinsic::maxnum, llvm_zero, llvm_t);
                llvm_t = b.CreateBinaryIntrinsic(llvm::Intrinsic::minnum, llvm_one, llvm_t);
                // smoothstep
                auto llvm_tt = b.CreateFMul(llvm_t, llvm_t);
                auto llvm_three_minus_two_t = b.CreateIntrinsic(llvm_t->getType(), llvm::Intrinsic::fma, {llvm_minus_two, llvm_t, llvm_three});
                return b.CreateFMul(llvm_tt, llvm_three_minus_two_t);
            }
            case xir::ArithmeticOp::STEP: {
                // step(edge, x) = x < edge ? 0 : 1
                auto type = inst->type();
                auto edge_type = inst->operand(0u)->type();
                LUISA_ASSERT(type != nullptr && inst->operand(1u)->type() == type &&
                                 (edge_type == type ||
                                  (type->is_vector() &&
                                   edge_type == type->element())),
                             "Type mismatch for step.");
                auto llvm_x = _lookup_value(current, b, inst->operand(1u));
                auto llvm_edge = _broadcast_like(
                    b, _lookup_value(current, b, inst->operand(0u)), llvm_x);
                auto llvm_cmp = b.CreateFCmpOLT(llvm_x, llvm_edge);
                auto llvm_zero = llvm::Constant::getNullValue(llvm_x->getType());
                auto llvm_one = llvm::ConstantFP::get(llvm_x->getType(), 1.);
                return b.CreateSelect(llvm_cmp, llvm_zero, llvm_one);
            }
            case xir::ArithmeticOp::ABS: {
                auto llvm_x = _lookup_value(current, b, inst->operand(0u));
                if (llvm_x->getType()->isIntOrIntVectorTy()) {
                    auto llvm_false = llvm::ConstantInt::getFalse(_llvm_context);
                    return b.CreateBinaryIntrinsic(llvm::Intrinsic::abs, llvm_x, llvm_false);
                }
                return b.CreateUnaryIntrinsic(llvm::Intrinsic::fabs, llvm_x);
            }
            case xir::ArithmeticOp::MIN: {
                auto llvm_x = _lookup_value(current, b, inst->operand(0u));
                auto llvm_y = _lookup_value(current, b, inst->operand(1u));
                auto elem_type = inst->type()->is_vector() ? inst->type()->element() : inst->type();
                switch (elem_type->tag()) {
                    case Type::Tag::INT8: [[fallthrough]];
                    case Type::Tag::INT16: [[fallthrough]];
                    case Type::Tag::INT32: [[fallthrough]];
                    case Type::Tag::INT64: return b.CreateBinaryIntrinsic(llvm::Intrinsic::smin, llvm_x, llvm_y);
                    case Type::Tag::UINT8: [[fallthrough]];
                    case Type::Tag::UINT16: [[fallthrough]];
                    case Type::Tag::UINT32: [[fallthrough]];
                    case Type::Tag::UINT64: return b.CreateBinaryIntrinsic(llvm::Intrinsic::umin, llvm_x, llvm_y);
                    case Type::Tag::FLOAT16: [[fallthrough]];
                    case Type::Tag::FLOAT32: [[fallthrough]];
                    case Type::Tag::FLOAT64: return b.CreateBinaryIntrinsic(llvm::Intrinsic::minnum, llvm_x, llvm_y);
                    default: break;
                }
                LUISA_ERROR_WITH_LOCATION("Invalid operand type for min operation: {}.",
                                          inst->type()->description());
            }
            case xir::ArithmeticOp::MAX: {
                auto llvm_x = _lookup_value(current, b, inst->operand(0u));
                auto llvm_y = _lookup_value(current, b, inst->operand(1u));
                auto elem_type = inst->type()->is_vector() ? inst->type()->element() : inst->type();
                switch (elem_type->tag()) {
                    case Type::Tag::INT8: [[fallthrough]];
                    case Type::Tag::INT16: [[fallthrough]];
                    case Type::Tag::INT32: [[fallthrough]];
                    case Type::Tag::INT64: return b.CreateBinaryIntrinsic(llvm::Intrinsic::smax, llvm_x, llvm_y);
                    case Type::Tag::UINT8: [[fallthrough]];
                    case Type::Tag::UINT16: [[fallthrough]];
                    case Type::Tag::UINT32: [[fallthrough]];
                    case Type::Tag::UINT64: return b.CreateBinaryIntrinsic(llvm::Intrinsic::umax, llvm_x, llvm_y);
                    case Type::Tag::FLOAT16: [[fallthrough]];
                    case Type::Tag::FLOAT32: [[fallthrough]];
                    case Type::Tag::FLOAT64: return b.CreateBinaryIntrinsic(llvm::Intrinsic::maxnum, llvm_x, llvm_y);
                    default: break;
                }
                LUISA_ERROR_WITH_LOCATION("Invalid operand type for max operation: {}.",
                                          inst->type()->description());
            }
            case xir::ArithmeticOp::CLZ: {
                auto llvm_x = _lookup_value(current, b, inst->operand(0u));
                return b.CreateBinaryIntrinsic(llvm::Intrinsic::ctlz, llvm_x, b.getInt1(false));
            }
            case xir::ArithmeticOp::CTZ: {
                auto llvm_x = _lookup_value(current, b, inst->operand(0u));
                return b.CreateBinaryIntrinsic(llvm::Intrinsic::cttz, llvm_x, b.getInt1(false));
            }
            case xir::ArithmeticOp::POPCOUNT: {
                auto llvm_x = _lookup_value(current, b, inst->operand(0u));
                return b.CreateUnaryIntrinsic(llvm::Intrinsic::ctpop, llvm_x);
            }
            case xir::ArithmeticOp::REVERSE: {
                auto llvm_x = _lookup_value(current, b, inst->operand(0u));
                return b.CreateUnaryIntrinsic(llvm::Intrinsic::bitreverse, llvm_x);
            }
            case xir::ArithmeticOp::ISINF: return _translate_isinf_isnan(current, b, inst->op(), inst->operand(0u));
            case xir::ArithmeticOp::ISNAN: return _translate_isinf_isnan(current, b, inst->op(), inst->operand(0u));
            case xir::ArithmeticOp::ACOS: {
#if LLVM_VERSION_MAJOR >= 19
                return _translate_unary_fp_math_operation(current, b, inst->operand(0u), llvm::Intrinsic::acos);
#else
                return _translate_unary_fp_math_operation(current, b, inst->operand(0u), "acos");
#endif
            }
            case xir::ArithmeticOp::ACOSH: {
                // acosh(x) = log(x + sqrt(x^2 - 1))
                auto llvm_x = _lookup_value(current, b, inst->operand(0u));
                auto llvm_x2 = b.CreateFMul(llvm_x, llvm_x);
                auto llvm_one = llvm::ConstantFP::get(llvm_x->getType(), 1.f);
                auto llvm_x2_minus_one = b.CreateFSub(llvm_x2, llvm_one);
                auto llvm_sqrt = b.CreateUnaryIntrinsic(llvm::Intrinsic::sqrt, llvm_x2_minus_one);
                auto llvm_x_plus_sqrt = b.CreateFAdd(llvm_x, llvm_sqrt);
                return b.CreateUnaryIntrinsic(llvm::Intrinsic::log, llvm_x_plus_sqrt);
            }
            case xir::ArithmeticOp::ASIN: {
#if LLVM_VERSION_MAJOR >= 19
                return _translate_unary_fp_math_operation(current, b, inst->operand(0u), llvm::Intrinsic::asin);
#else
                return _translate_unary_fp_math_operation(current, b, inst->operand(0u), "asin");
#endif
            }
            case xir::ArithmeticOp::ASINH: {
                // asinh(x) = log(x + sqrt(x^2 + 1))
                auto llvm_x = _lookup_value(current, b, inst->operand(0u));
                auto llvm_x2 = b.CreateFMul(llvm_x, llvm_x);
                auto llvm_one = llvm::ConstantFP::get(llvm_x->getType(), 1.f);
                auto llvm_x2_plus_one = b.CreateFAdd(llvm_x2, llvm_one);
                auto llvm_sqrt = b.CreateUnaryIntrinsic(llvm::Intrinsic::sqrt, llvm_x2_plus_one);
                auto llvm_x_plus_sqrt = b.CreateFAdd(llvm_x, llvm_sqrt);
                return b.CreateUnaryIntrinsic(llvm::Intrinsic::log, llvm_x_plus_sqrt);
            }
            case xir::ArithmeticOp::ATAN: {
#if LLVM_VERSION_MAJOR >= 19
                return _translate_unary_fp_math_operation(current, b, inst->operand(0u), llvm::Intrinsic::atan);
#else
                return _translate_unary_fp_math_operation(current, b, inst->operand(0u), "atan");
#endif
            }
            case xir::ArithmeticOp::ATAN2: {
#if LLVM_VERSION_MAJOR >= 20
                return _translate_binary_fp_math_operation(current, b, inst->operand(0), inst->operand(1), llvm::Intrinsic::atan2);
#else
                return _translate_binary_fp_math_operation(current, b, inst->operand(0), inst->operand(1), "atan2");
#endif
            }
            case xir::ArithmeticOp::ATANH: {
                // atanh(x) = 0.5 * log((1 + x) / (1 - x))
                auto llvm_x = _lookup_value(current, b, inst->operand(0u));
                auto llvm_one = llvm::ConstantFP::get(llvm_x->getType(), 1.f);
                auto llvm_one_plus_x = b.CreateFAdd(llvm_one, llvm_x);
                auto llvm_one_minus_x = b.CreateFSub(llvm_one, llvm_x);
                auto llvm_div = b.CreateFDiv(llvm_one_plus_x, llvm_one_minus_x);
                auto llvm_log = b.CreateUnaryIntrinsic(llvm::Intrinsic::log, llvm_div);
                auto llvm_half = llvm::ConstantFP::get(llvm_x->getType(), .5f);
                return b.CreateFMul(llvm_half, llvm_log);
            }
            case xir::ArithmeticOp::COS: return _translate_unary_fp_math_operation(current, b, inst->operand(0u), llvm::Intrinsic::cos);
            case xir::ArithmeticOp::COSH: {
#if LLVM_VERSION_MAJOR >= 19
                return _translate_unary_fp_math_operation(current, b, inst->operand(0u), llvm::Intrinsic::cosh);
#else
                // cosh(x) = 0.5 * (exp(x) + exp(-x))
                auto llvm_x = _lookup_value(current, b, inst->operand(0u));
                auto llvm_exp_x = b.CreateUnaryIntrinsic(llvm::Intrinsic::exp, llvm_x);
                auto llvm_neg_x = b.CreateFNeg(llvm_x);
                auto llvm_exp_neg_x = b.CreateUnaryIntrinsic(llvm::Intrinsic::exp, llvm_neg_x);
                auto llvm_exp_sum = b.CreateFAdd(llvm_exp_x, llvm_exp_neg_x);
                auto llvm_half = llvm::ConstantFP::get(llvm_x->getType(), .5f);
                return b.CreateFMul(llvm_half, llvm_exp_sum);
#endif
            }
            case xir::ArithmeticOp::SIN: return _translate_unary_fp_math_operation(current, b, inst->operand(0u), llvm::Intrinsic::sin);
            case xir::ArithmeticOp::SINH: {
#if LLVM_VERSION_MAJOR >= 19
                return _translate_unary_fp_math_operation(current, b, inst->operand(0u), llvm::Intrinsic::sinh);
#else
                // sinh(x) = 0.5 * (exp(x) - exp(-x))
                auto llvm_x = _lookup_value(current, b, inst->operand(0u));
                auto llvm_exp_x = b.CreateUnaryIntrinsic(llvm::Intrinsic::exp, llvm_x);
                auto llvm_neg_x = b.CreateFNeg(llvm_x);
                auto llvm_exp_neg_x = b.CreateUnaryIntrinsic(llvm::Intrinsic::exp, llvm_neg_x);
                auto llvm_exp_diff = b.CreateFSub(llvm_exp_x, llvm_exp_neg_x);
                auto llvm_half = llvm::ConstantFP::get(llvm_x->getType(), .5f);
                return b.CreateFMul(llvm_half, llvm_exp_diff);
#endif
            }
            case xir::ArithmeticOp::TAN: {
#if LLVM_VERSION_MAJOR >= 19
                return _translate_unary_fp_math_operation(current, b, inst->operand(0u), llvm::Intrinsic::tan);
#else
                auto llvm_sin = _translate_unary_fp_math_operation(current, b, inst->operand(0u), llvm::Intrinsic::sin);
                auto llvm_cos = _translate_unary_fp_math_operation(current, b, inst->operand(0u), llvm::Intrinsic::cos);
                return b.CreateFDiv(llvm_sin, llvm_cos);
#endif
            }
            case xir::ArithmeticOp::TANH: {
#if LLVM_VERSION_MAJOR >= 19
                return _translate_unary_fp_math_operation(current, b, inst->operand(0u), llvm::Intrinsic::tanh);
#else
                // tanh(x) = sinh(x) / cosh(x) = (exp(2x) - 1) / (exp(2x) + 1)
                auto llvm_x = _lookup_value(current, b, inst->operand(0u));
                auto llvm_two_x = b.CreateFMul(llvm_x, llvm::ConstantFP::get(llvm_x->getType(), 2.f));
                auto llvm_exp_2x = b.CreateUnaryIntrinsic(llvm::Intrinsic::exp, llvm_two_x);
                auto llvm_exp_2x_minus_one = b.CreateFSub(llvm_exp_2x, llvm::ConstantFP::get(llvm_x->getType(), 1.f));
                auto llvm_exp_2x_plus_one = b.CreateFAdd(llvm_exp_2x, llvm::ConstantFP::get(llvm_x->getType(), 1.f));
                return b.CreateFDiv(llvm_exp_2x_minus_one, llvm_exp_2x_plus_one);
#endif
            }
            case xir::ArithmeticOp::EXP: return _translate_unary_fp_math_operation(current, b, inst->operand(0u), llvm::Intrinsic::exp);
            case xir::ArithmeticOp::EXP2: return _translate_unary_fp_math_operation(current, b, inst->operand(0u), llvm::Intrinsic::exp2);
            case xir::ArithmeticOp::EXP10: return _translate_unary_fp_math_operation(current, b, inst->operand(0u), llvm::Intrinsic::exp10);
            case xir::ArithmeticOp::LOG: return _translate_unary_fp_math_operation(current, b, inst->operand(0u), llvm::Intrinsic::log);
            case xir::ArithmeticOp::LOG2: return _translate_unary_fp_math_operation(current, b, inst->operand(0u), llvm::Intrinsic::log2);
            case xir::ArithmeticOp::LOG10: return _translate_unary_fp_math_operation(current, b, inst->operand(0u), llvm::Intrinsic::log10);
            case xir::ArithmeticOp::POW: return _translate_binary_fp_math_operation(current, b, inst->operand(0u), inst->operand(1u), llvm::Intrinsic::pow);
            case xir::ArithmeticOp::POW_INT: {
                auto base = inst->operand(0u);
                auto exponent = inst->operand(1u);
                auto llvm_base = _lookup_value(current, b, base);
                auto llvm_exponent = _lookup_value(current, b, exponent);
                LUISA_ASSERT(llvm_base->getType()->isFPOrFPVectorTy() &&
                                 llvm_exponent->getType()->isIntOrIntVectorTy() &&
                                 !llvm_exponent->getType()->isIntOrIntVectorTy(1),
                             "Invalid operand type for pow_int operation.");
                // llvm.powi interprets its integer operand as signed, while
                // XIR POW_INT preserves the exponent's declared signedness and
                // full bit width. It also permits a scalar exponent to be
                // broadcast over a vector base. Implement the XIR contract
                // explicitly instead of relying on the target-dependent
                // intrinsic lowering.
                auto call_scalar_power_magnitude = [&](llvm::Value *scalar_base,
                                                       llvm::Value *magnitude,
                                                       llvm::Value *negative) noexcept {
                    auto base_type = scalar_base->getType();
                    auto exponent_type = llvm::cast<llvm::IntegerType>(
                        magnitude->getType());
                    auto base_code = base_type->isHalfTy() ? "f16" :
                                         base_type->isFloatTy() ? "f32" :
                                                                 "f64";
                    auto name = fmt::format(
                        "luisa.fallback.pow_int.{}.u{}",
                        base_code, exponent_type->getBitWidth());
                    auto helper = _llvm_module->getFunction(name);
                    if (helper == nullptr) {
                        auto helper_type = llvm::FunctionType::get(
                            base_type,
                            {base_type, exponent_type, b.getInt1Ty()}, false);
                        helper = llvm::Function::Create(
                            helper_type, llvm::Function::PrivateLinkage,
                            name, _llvm_module);
                        helper->addFnAttr(llvm::Attribute::AlwaysInline);

                        auto entry = llvm::BasicBlock::Create(
                            _llvm_context, "entry", helper);
                        auto positive = llvm::BasicBlock::Create(
                            _llvm_context, "positive", helper);
                        auto negative_block = llvm::BasicBlock::Create(
                            _llvm_context, "negative", helper);
                        auto loop = llvm::BasicBlock::Create(
                            _llvm_context, "loop", helper);
                        auto body = llvm::BasicBlock::Create(
                            _llvm_context, "body", helper);
                        auto exit = llvm::BasicBlock::Create(
                            _llvm_context, "exit", helper);
                        IRBuilder helper_b{entry};
                        helper_b.setFastMathFlags(b.getFastMathFlags());
                        helper_b.CreateCondBr(
                            helper->getArg(2), negative_block, positive);

                        auto one = llvm::ConstantFP::get(base_type, 1.0);
                        helper_b.SetInsertPoint(positive);
                        helper_b.CreateBr(loop);

                        helper_b.SetInsertPoint(negative_block);
                        auto reciprocal_base = helper_b.CreateFDiv(
                            one, helper->getArg(0));
                        helper_b.CreateBr(loop);

                        helper_b.SetInsertPoint(loop);
                        auto current_magnitude = helper_b.CreatePHI(
                            exponent_type, 3u, "magnitude");
                        auto current_result = helper_b.CreatePHI(
                            base_type, 3u, "result");
                        auto current_factor = helper_b.CreatePHI(
                            base_type, 3u, "factor");
                        auto zero = llvm::ConstantInt::get(
                            exponent_type, 0u);
                        current_magnitude->addIncoming(
                            helper->getArg(1), positive);
                        current_magnitude->addIncoming(
                            helper->getArg(1), negative_block);
                        current_result->addIncoming(one, positive);
                        current_result->addIncoming(one, negative_block);
                        current_factor->addIncoming(
                            helper->getArg(0), positive);
                        current_factor->addIncoming(
                            reciprocal_base, negative_block);
                        auto done = helper_b.CreateICmpEQ(
                            current_magnitude, zero);
                        helper_b.CreateCondBr(done, exit, body);

                        helper_b.SetInsertPoint(body);
                        auto odd = helper_b.CreateTrunc(
                            current_magnitude, helper_b.getInt1Ty());
                        auto multiplied = helper_b.CreateFMul(
                            current_result, current_factor);
                        auto next_result = helper_b.CreateSelect(
                            odd, multiplied, current_result);
                        auto next_magnitude = helper_b.CreateLShr(
                            current_magnitude, 1u);
                        auto next_factor = helper_b.CreateFMul(
                            current_factor, current_factor);
                        helper_b.CreateBr(loop);
                        current_magnitude->addIncoming(
                            next_magnitude, body);
                        current_result->addIncoming(next_result, body);
                        current_factor->addIncoming(next_factor, body);

                        helper_b.SetInsertPoint(exit);
                        helper_b.CreateRet(current_result);
                    }
                    return b.CreateCall(
                        helper, {scalar_base, magnitude, negative});
                };
                auto exponent_element_type = exponent->type()->is_vector() ?
                                                 exponent->type()->element() :
                                                 exponent->type();
                auto exponent_is_signed = exponent_element_type->is_int();
                auto call_scalar_power = [&](llvm::Value *scalar_base,
                                             llvm::Value *scalar_exponent) noexcept {
                    if (!exponent_is_signed) {
                        return call_scalar_power_magnitude(
                            scalar_base, scalar_exponent, b.getFalse());
                    }
                    auto exponent_type = llvm::cast<llvm::IntegerType>(
                        scalar_exponent->getType());
                    auto zero = llvm::ConstantInt::get(
                        exponent_type, 0u);
                    auto negative = b.CreateICmpSLT(
                        scalar_exponent, zero);
                    auto magnitude = b.CreateSelect(
                        negative,
                        b.CreateSub(zero, scalar_exponent),
                        scalar_exponent);
                    return call_scalar_power_magnitude(
                        scalar_base, magnitude, negative);
                };
                if (auto base_type = llvm::dyn_cast<llvm::FixedVectorType>(
                        llvm_base->getType())) {
                    auto result = static_cast<llvm::Value *>(
                        llvm::PoisonValue::get(base_type));
                    auto exponent_is_vector =
                        llvm_exponent->getType()->isVectorTy();
                    for (auto i = 0u;
                         i < base_type->getNumElements(); i++) {
                        auto base_element = b.CreateExtractElement(
                            llvm_base, i);
                        auto exponent_element = exponent_is_vector ?
                                                    b.CreateExtractElement(
                                                        llvm_exponent, i) :
                                                    llvm_exponent;
                        auto element_result = call_scalar_power(
                            base_element, exponent_element);
                        result = b.CreateInsertElement(
                            result, element_result, i);
                    }
                    return result;
                }
                return call_scalar_power(llvm_base, llvm_exponent);
            }
            case xir::ArithmeticOp::SQRT: return _translate_unary_fp_math_operation(current, b, inst->operand(0u), llvm::Intrinsic::sqrt);
            case xir::ArithmeticOp::RSQRT: {
                auto llvm_operand = _lookup_value(current, b, inst->operand(0u));
                auto llvm_sqrt = b.CreateUnaryIntrinsic(llvm::Intrinsic::sqrt, llvm_operand);
                auto operand_type = inst->operand(0u)->type();
                auto operand_elem_type = operand_type->is_vector() ? operand_type->element() : operand_type;
                auto llvm_elem_type = _translate_type(operand_elem_type, false);
                auto llvm_one = llvm::ConstantFP::get(llvm_elem_type, 1.f);
                if (operand_type->is_vector()) {
                    auto dim = operand_type->dimension();
                    llvm_one = llvm::ConstantVector::getSplat(llvm::ElementCount::getFixed(dim), llvm_one);
                }
                return b.CreateFDiv(llvm_one, llvm_sqrt);
            }
            case xir::ArithmeticOp::CEIL: return _translate_unary_fp_math_operation(current, b, inst->operand(0u), llvm::Intrinsic::ceil);
            case xir::ArithmeticOp::FLOOR: return _translate_unary_fp_math_operation(current, b, inst->operand(0u), llvm::Intrinsic::floor);
            case xir::ArithmeticOp::FRACT: {
                // fract(x) = x - floor(x)
                auto llvm_operand = _lookup_value(current, b, inst->operand(0u));
                auto llvm_floor = _translate_unary_fp_math_operation(current, b, inst->operand(0u), llvm::Intrinsic::floor);
                return b.CreateFSub(llvm_operand, llvm_floor);
            }
            case xir::ArithmeticOp::TRUNC: return _translate_unary_fp_math_operation(current, b, inst->operand(0u), llvm::Intrinsic::trunc);
            case xir::ArithmeticOp::ROUND: return _translate_unary_fp_math_operation(current, b, inst->operand(0u), llvm::Intrinsic::round);
            case xir::ArithmeticOp::RINT: return _translate_unary_fp_math_operation(current, b, inst->operand(0u), llvm::Intrinsic::rint);
            case xir::ArithmeticOp::FMA: {
                // fma(a, b, c) = a * b + c, or we can use llvm intrinsic for fp
                auto va = inst->operand(0u);
                auto vb = inst->operand(1u);
                auto vc = inst->operand(2u);
                LUISA_ASSERT(va->type() != nullptr && va->type() == vb->type() && va->type() == vc->type(), "Type mismatch for fma.");
                auto llvm_va = _lookup_value(current, b, va);
                auto llvm_vb = _lookup_value(current, b, vb);
                auto llvm_vc = _lookup_value(current, b, vc);
                auto elem_type = va->type()->is_vector() ? va->type()->element() : va->type();
                switch (elem_type->tag()) {
                    case Type::Tag::INT8: [[fallthrough]];
                    case Type::Tag::INT16: [[fallthrough]];
                    case Type::Tag::INT32: [[fallthrough]];
                    case Type::Tag::INT64: {
                        auto llvm_mul = b.CreateNSWMul(llvm_va, llvm_vb);
                        return b.CreateNSWAdd(llvm_mul, llvm_vc);
                    }
                    case Type::Tag::UINT8: [[fallthrough]];
                    case Type::Tag::UINT16: [[fallthrough]];
                    case Type::Tag::UINT32: [[fallthrough]];
                    case Type::Tag::UINT64: {
                        auto llvm_mul = b.CreateMul(llvm_va, llvm_vb);
                        return b.CreateAdd(llvm_mul, llvm_vc);
                    }
                    case Type::Tag::FLOAT16: [[fallthrough]];
                    case Type::Tag::FLOAT32: [[fallthrough]];
                    case Type::Tag::FLOAT64: {
                        return b.CreateIntrinsic(llvm_va->getType(), llvm::Intrinsic::fma, {llvm_va, llvm_vb, llvm_vc});
                    }
                    default: break;
                }
                LUISA_ERROR_WITH_LOCATION("Invalid operand type for fma operation: {}.", va->type()->description());
            }
            case xir::ArithmeticOp::COPYSIGN: return _translate_binary_fp_math_operation(current, b, inst->operand(0u), inst->operand(1u), llvm::Intrinsic::copysign);
            case xir::ArithmeticOp::CROSS: {
                // cross(u, v) = (u.y * v.z - v.y * u.z, u.z * v.x - v.z * u.x, u.x * v.y - v.x * u.y)
                auto u = _lookup_value(current, b, inst->operand(0u));
                auto v = _lookup_value(current, b, inst->operand(1u));
                auto poison = llvm::PoisonValue::get(u->getType());
                auto s0 = b.CreateShuffleVector(u, poison, {1, 2, 0});
                auto s1 = b.CreateShuffleVector(v, poison, {2, 0, 1});
                auto s2 = b.CreateShuffleVector(v, poison, {1, 2, 0});
                auto s3 = b.CreateShuffleVector(u, poison, {2, 0, 1});
                auto s01 = b.CreateFMul(s0, s1);
                auto s23 = b.CreateFMul(s2, s3);
                return b.CreateFSub(s01, s23);
            }
            case xir::ArithmeticOp::DOT: return _translate_vector_dot(current, b, inst->operand(0u), inst->operand(1u));
            case xir::ArithmeticOp::LENGTH: {
                auto v = inst->operand(0u);
                auto llvm_length_squared = _translate_vector_dot(current, b, v, v);
                return b.CreateUnaryIntrinsic(llvm::Intrinsic::sqrt, llvm_length_squared);
            }
            case xir::ArithmeticOp::LENGTH_SQUARED: {
                auto v = inst->operand(0u);
                return _translate_vector_dot(current, b, v, v);
            }
            case xir::ArithmeticOp::NORMALIZE: {
                auto v = inst->operand(0u);
                auto llvm_length_squared = _translate_vector_dot(current, b, v, v);
                auto llvm_length = b.CreateUnaryIntrinsic(llvm::Intrinsic::sqrt, llvm_length_squared);
                auto llvm_v = _lookup_value(current, b, v);
                auto llvm_length_splat = b.CreateVectorSplat(v->type()->dimension(), llvm_length);
                return b.CreateFDiv(llvm_v, llvm_length_splat);
            }
            case xir::ArithmeticOp::FACEFORWARD: {
                // faceforward(n, i, n_ref) = dot(i, n_ref) < 0 ? n : -n
                auto n = inst->operand(0u);
                auto i = inst->operand(1u);
                auto n_ref = inst->operand(2u);
                auto llvm_dot = _translate_vector_dot(current, b, i, n_ref);
                auto llvm_zero = llvm::ConstantFP::get(llvm_dot->getType(), 0.);
                auto llvm_cmp = b.CreateFCmpOLT(llvm_dot, llvm_zero);
                auto llvm_pos_n = _lookup_value(current, b, n);
                auto llvm_neg_n = b.CreateFNeg(llvm_pos_n);
                return b.CreateSelect(llvm_cmp, llvm_pos_n, llvm_neg_n);
            }
            case xir::ArithmeticOp::REFLECT: {
                // reflect(I, N) = I - 2.0 * dot(N, I) * N = fma(splat(-2.0 * dot(N, I)), N, I)
                auto i = inst->operand(0u);
                auto n = inst->operand(1u);
                auto llvm_dot = _translate_vector_dot(current, b, n, i);
                auto llvm_i = _lookup_value(current, b, i);
                auto llvm_n = _lookup_value(current, b, n);
                auto dim = llvm::cast<llvm::VectorType>(llvm_n->getType())->getElementCount();
                auto llvm_minus_two = llvm::ConstantFP::get(llvm_dot->getType(), -2.);
                auto llvm_minus_two_dot = b.CreateFMul(llvm_minus_two, llvm_dot);
                llvm_minus_two_dot = b.CreateVectorSplat(dim, llvm_minus_two_dot);
                return b.CreateIntrinsic(llvm_n->getType(), llvm::Intrinsic::fma,
                                         {llvm_minus_two_dot, llvm_n, llvm_i});
            }
            case xir::ArithmeticOp::REDUCE_SUM: return _translate_vector_reduce(current, b, xir::ArithmeticOp::REDUCE_SUM, inst->operand(0u));
            case xir::ArithmeticOp::REDUCE_PRODUCT: return _translate_vector_reduce(current, b, xir::ArithmeticOp::REDUCE_PRODUCT, inst->operand(0u));
            case xir::ArithmeticOp::REDUCE_MIN: return _translate_vector_reduce(current, b, xir::ArithmeticOp::REDUCE_MIN, inst->operand(0u));
            case xir::ArithmeticOp::REDUCE_MAX: return _translate_vector_reduce(current, b, xir::ArithmeticOp::REDUCE_MAX, inst->operand(0u));
            case xir::ArithmeticOp::OUTER_PRODUCT: return _translate_outer_product(current, b, inst);
            case xir::ArithmeticOp::MATRIX_COMP_NEG: {
                auto m = inst->operand(0u);
                LUISA_ASSERT(m->type() != nullptr && m->type()->is_matrix(),
                             "Invalid operand type for matrix component-wise negation.");
                auto dim = m->type()->dimension();
                auto llvm_m = _lookup_value(current, b, m);
                auto llvm_result = llvm::cast<llvm::Value>(llvm::PoisonValue::get(llvm_m->getType()));
                for (auto i = 0u; i < dim; i++) {
                    for (auto j = 0u; j < dim; j++) {
                        auto llvm_elem = b.CreateExtractValue(llvm_m, {i, j});
                        llvm_elem = b.CreateFNeg(llvm_elem);
                        llvm_result = b.CreateInsertValue(llvm_result, llvm_elem, {i, j});
                    }
                }
                return llvm_result;
            }
            case xir::ArithmeticOp::MATRIX_COMP_ADD: return _translate_matrix_comp_op(current, b, inst->operand(0), inst->operand(1), llvm::Instruction::FAdd);
            case xir::ArithmeticOp::MATRIX_COMP_SUB: return _translate_matrix_comp_op(current, b, inst->operand(0), inst->operand(1), llvm::Instruction::FSub);
            case xir::ArithmeticOp::MATRIX_COMP_MUL: return _translate_matrix_comp_op(current, b, inst->operand(0), inst->operand(1), llvm::Instruction::FMul);
            case xir::ArithmeticOp::MATRIX_COMP_DIV: return _translate_matrix_comp_op(current, b, inst->operand(0), inst->operand(1), llvm::Instruction::FDiv);
            case xir::ArithmeticOp::MATRIX_LINALG_MUL: return _translate_matrix_linalg_multiply(current, b, inst);
            case xir::ArithmeticOp::MATRIX_DETERMINANT: return _translate_matrix_determinant(current, b, inst);
            case xir::ArithmeticOp::MATRIX_TRANSPOSE: return _translate_matrix_transpose(current, b, inst);
            case xir::ArithmeticOp::MATRIX_INVERSE: return _translate_matrix_inverse(current, b, inst);
            case xir::ArithmeticOp::AGGREGATE: return _translate_aggregate(current, b, inst);
            case xir::ArithmeticOp::SHUFFLE: return _translate_shuffle(current, b, inst);
            case xir::ArithmeticOp::INSERT: return _translate_insert(current, b, inst);
            case xir::ArithmeticOp::EXTRACT: return _translate_extract(current, b, inst);
        }
        LUISA_ERROR_WITH_LOCATION("Unexpected arithmetic operation: {}.", xir::to_string(inst->op()));
    }

    [[nodiscard]] llvm::Value *_translate_resource_query_inst(CurrentFunction &current, IRBuilder &b,
                                                              const xir::ResourceQueryInst *inst) noexcept {
        switch (inst->op()) {
            case xir::ResourceQueryOp::BUFFER_SIZE: return _translate_buffer_size(current, b, inst);
            case xir::ResourceQueryOp::BYTE_BUFFER_SIZE: return _translate_buffer_size(current, b, inst, true);
            case xir::ResourceQueryOp::TEXTURE2D_SIZE: return _translate_texture_size(current, b, inst);
            case xir::ResourceQueryOp::TEXTURE3D_SIZE: return _translate_texture_size(current, b, inst);
            case xir::ResourceQueryOp::TEXTURE2D_SAMPLE: break;
            case xir::ResourceQueryOp::TEXTURE2D_SAMPLE_LEVEL: break;
            case xir::ResourceQueryOp::TEXTURE2D_SAMPLE_GRAD: break;
            case xir::ResourceQueryOp::TEXTURE2D_SAMPLE_GRAD_LEVEL: break;
            case xir::ResourceQueryOp::TEXTURE3D_SAMPLE: break;
            case xir::ResourceQueryOp::TEXTURE3D_SAMPLE_LEVEL: break;
            case xir::ResourceQueryOp::TEXTURE3D_SAMPLE_GRAD: break;
            case xir::ResourceQueryOp::TEXTURE3D_SAMPLE_GRAD_LEVEL: break;
            case xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE: return _translate_bindless_texture_access(current, b, "luisa.bindless.texture2d.sample", inst);
            case xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_LEVEL: return _translate_bindless_texture_access(current, b, "luisa.bindless.texture2d.sample.level", inst);
            case xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD: return _translate_bindless_texture_access(current, b, "luisa.bindless.texture2d.sample.grad", inst);
            case xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD_LEVEL: return _translate_bindless_texture_access(current, b, "luisa.bindless.texture2d.sample.grad.level", inst);
            case xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE: return _translate_bindless_texture_access(current, b, "luisa.bindless.texture3d.sample", inst);
            case xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_LEVEL: return _translate_bindless_texture_access(current, b, "luisa.bindless.texture3d.sample.level", inst);
            case xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_GRAD: return _translate_bindless_texture_access(current, b, "luisa.bindless.texture3d.sample.grad", inst);
            case xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_GRAD_LEVEL: return _translate_bindless_texture_access(current, b, "luisa.bindless.texture3d.sample.grad.level", inst);
            case xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_SAMPLER: break;
            case xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_LEVEL_SAMPLER: break;
            case xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD_SAMPLER: break;
            case xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD_LEVEL_SAMPLER: break;
            case xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_SAMPLER: break;
            case xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_LEVEL_SAMPLER: break;
            case xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_GRAD_SAMPLER: break;
            case xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_GRAD_LEVEL_SAMPLER: break;
            case xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SIZE: return _translate_bindless_texture_access(current, b, "luisa.bindless.texture2d.size", inst);
            case xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SIZE: return _translate_bindless_texture_access(current, b, "luisa.bindless.texture3d.size", inst);
            case xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SIZE_LEVEL: return _translate_bindless_texture_access(current, b, "luisa.bindless.texture2d.size.level", inst);
            case xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SIZE_LEVEL: return _translate_bindless_texture_access(current, b, "luisa.bindless.texture3d.size.level", inst);
            case xir::ResourceQueryOp::BINDLESS_BUFFER_SIZE: return _translate_bindless_buffer_size(current, b, inst);
            case xir::ResourceQueryOp::BINDLESS_BYTE_BUFFER_SIZE: return _translate_bindless_buffer_size(current, b, inst, true);
            case xir::ResourceQueryOp::BUFFER_DEVICE_ADDRESS: return _translate_buffer_device_address(current, b, inst);
            case xir::ResourceQueryOp::BINDLESS_BUFFER_DEVICE_ADDRESS: return _translate_bindless_buffer_device_address(current, b, inst);
            case xir::ResourceQueryOp::RAY_TRACING_INSTANCE_TRANSFORM: return _translate_accel_access(current, b, "luisa.accel.instance.transform", inst);
            case xir::ResourceQueryOp::RAY_TRACING_INSTANCE_USER_ID: return _translate_accel_access(current, b, "luisa.accel.instance.user.id", inst);
            case xir::ResourceQueryOp::RAY_TRACING_INSTANCE_VISIBILITY_MASK: return _translate_accel_access(current, b, "luisa.accel.instance.visibility.mask", inst);
            case xir::ResourceQueryOp::RAY_TRACING_TRACE_CLOSEST: return _translate_accel_access(current, b, "luisa.accel.trace.closest", inst);
            case xir::ResourceQueryOp::RAY_TRACING_TRACE_ANY: return _translate_accel_access(current, b, "luisa.accel.trace.any", inst);
            case xir::ResourceQueryOp::RAY_TRACING_INSTANCE_MOTION_MATRIX: return _translate_accel_access(current, b, "luisa.accel.instance.motion.matrix", inst);
            case xir::ResourceQueryOp::RAY_TRACING_INSTANCE_MOTION_SRT: return _translate_accel_access(current, b, "luisa.accel.instance.motion.srt", inst);
            case xir::ResourceQueryOp::RAY_TRACING_TRACE_CLOSEST_MOTION_BLUR: return _translate_accel_access(current, b, "luisa.accel.trace.closest.motion", inst);
            case xir::ResourceQueryOp::RAY_TRACING_TRACE_ANY_MOTION_BLUR: return _translate_accel_access(current, b, "luisa.accel.trace.any.motion", inst);
            case xir::ResourceQueryOp::RAY_TRACING_QUERY_ALL: return _translate_accel_access(current, b, "luisa.accel.traverse", inst);
            case xir::ResourceQueryOp::RAY_TRACING_QUERY_ANY: return _translate_accel_access(current, b, "luisa.accel.traverse", inst);
            case xir::ResourceQueryOp::RAY_TRACING_QUERY_ALL_MOTION_BLUR: return _translate_accel_access(current, b, "luisa.accel.traverse.motion", inst);
            case xir::ResourceQueryOp::RAY_TRACING_QUERY_ANY_MOTION_BLUR: return _translate_accel_access(current, b, "luisa.accel.traverse.motion", inst);
        }
        LUISA_ERROR_WITH_LOCATION("Unexpected resource query operation: {}.", xir::to_string(inst->op()));
    }

    [[nodiscard]] llvm::Value *_translate_resource_read_inst(CurrentFunction &current, IRBuilder &b,
                                                             const xir::ResourceReadInst *inst) noexcept {
        switch (inst->op()) {
            case xir::ResourceReadOp::BUFFER_VOLATILE_READ: return _translate_buffer_read(current, b, inst, true);
            case xir::ResourceReadOp::BUFFER_READ: return _translate_buffer_read(current, b, inst, false);
            case xir::ResourceReadOp::BYTE_BUFFER_VOLATILE_READ: return _translate_buffer_read(current, b, inst, true, true);
            case xir::ResourceReadOp::BYTE_BUFFER_READ: return _translate_buffer_read(current, b, inst, false, true);
            case xir::ResourceReadOp::TEXTURE2D_READ: return _translate_texture_read(current, b, inst);
            case xir::ResourceReadOp::TEXTURE3D_READ: return _translate_texture_read(current, b, inst);
            case xir::ResourceReadOp::BINDLESS_TEXTURE2D_READ: return _translate_bindless_texture_access(current, b, "luisa.bindless.texture2d.read", inst);
            case xir::ResourceReadOp::BINDLESS_TEXTURE3D_READ: return _translate_bindless_texture_access(current, b, "luisa.bindless.texture3d.read", inst);
            case xir::ResourceReadOp::BINDLESS_TEXTURE2D_READ_LEVEL: return _translate_bindless_texture_access(current, b, "luisa.bindless.texture2d.read.level", inst);
            case xir::ResourceReadOp::BINDLESS_TEXTURE3D_READ_LEVEL: return _translate_bindless_texture_access(current, b, "luisa.bindless.texture3d.read.level", inst);
            case xir::ResourceReadOp::BINDLESS_BUFFER_READ: return _translate_bindless_buffer_read(current, b, inst);
            case xir::ResourceReadOp::BINDLESS_BYTE_BUFFER_READ: return _translate_bindless_buffer_read(current, b, inst, true);
            case xir::ResourceReadOp::DEVICE_ADDRESS_READ: return _translate_device_address_read(current, b, inst);
        }
        LUISA_ERROR_WITH_LOCATION("Unexpected resource read operation: {}.", xir::to_string(inst->op()));
    }

    [[nodiscard]] llvm::Value *_translate_resource_write_inst(CurrentFunction &current, IRBuilder &b,
                                                              const xir::ResourceWriteInst *inst) noexcept {
        switch (inst->op()) {
            case xir::ResourceWriteOp::BUFFER_VOLATILE_WRITE: return _translate_buffer_write(current, b, inst, true);
            case xir::ResourceWriteOp::BUFFER_WRITE: return _translate_buffer_write(current, b, inst, false);
            case xir::ResourceWriteOp::BYTE_BUFFER_VOLATILE_WRITE: return _translate_buffer_write(current, b, inst, true, true);
            case xir::ResourceWriteOp::BYTE_BUFFER_WRITE: return _translate_buffer_write(current, b, inst, false, true);
            case xir::ResourceWriteOp::TEXTURE2D_WRITE: return _translate_texture_write(current, b, inst);
            case xir::ResourceWriteOp::TEXTURE3D_WRITE: return _translate_texture_write(current, b, inst);
            case xir::ResourceWriteOp::BINDLESS_BUFFER_WRITE: return _translate_bindless_buffer_write(current, b, inst);
            case xir::ResourceWriteOp::BINDLESS_BYTE_BUFFER_WRITE: return _translate_bindless_buffer_write(current, b, inst, true);
            case xir::ResourceWriteOp::DEVICE_ADDRESS_WRITE: return _translate_device_address_write(current, b, inst);
            case xir::ResourceWriteOp::RAY_TRACING_SET_INSTANCE_TRANSFORM: return _translate_accel_access(current, b, "luisa.accel.set.instance.transform", inst);
            case xir::ResourceWriteOp::RAY_TRACING_SET_INSTANCE_VISIBILITY_MASK: return _translate_accel_access(current, b, "luisa.accel.set.instance.visibility.mask", inst);
            case xir::ResourceWriteOp::RAY_TRACING_SET_INSTANCE_OPACITY: return _translate_accel_access(current, b, "luisa.accel.set.instance.opacity", inst);
            case xir::ResourceWriteOp::RAY_TRACING_SET_INSTANCE_USER_ID: return _translate_accel_access(current, b, "luisa.accel.set.instance.user.id", inst);
            case xir::ResourceWriteOp::RAY_TRACING_SET_INSTANCE_MOTION_MATRIX: return _translate_accel_access(current, b, "luisa.accel.set.instance.motion.matrix", inst);
            case xir::ResourceWriteOp::RAY_TRACING_SET_INSTANCE_MOTION_SRT: return _translate_accel_access(current, b, "luisa.accel.set.instance.motion.srt", inst);
            case xir::ResourceWriteOp::INDIRECT_DISPATCH_SET_KERNEL: break;
            case xir::ResourceWriteOp::INDIRECT_DISPATCH_SET_COUNT: break;
        }
        LUISA_ERROR_WITH_LOCATION("Unexpected resource write operation: {}.", xir::to_string(inst->op()));
    }

    [[nodiscard]] llvm::Value *_translate_cast_inst(CurrentFunction &current, IRBuilder &b,
                                                    const Type *dst_type, xir::CastOp op,
                                                    const xir::Value *src_value) noexcept {
        auto llvm_value = _lookup_value(current, b, src_value);
        auto src_type = src_value->type();
        switch (op) {
            case xir::CastOp::STATIC_CAST: {
                if (dst_type == src_type) { return llvm_value; }
                LUISA_ASSERT((dst_type->is_scalar() && src_type->is_scalar()) ||
                                 (dst_type->is_vector() && src_type->is_vector() &&
                                  dst_type->dimension() == src_type->dimension()),
                             "Invalid static cast.");
                auto dst_is_scalar = dst_type->is_scalar();
                auto src_is_scalar = src_type->is_scalar();
                auto dst_elem_type = dst_is_scalar ? dst_type : dst_type->element();
                auto src_elem_type = src_is_scalar ? src_type : src_type->element();
                // typeN -> boolN
                if (dst_elem_type->is_bool()) {
                    auto cmp = _cmp_ne_zero(b, llvm_value);
                    return _zext_i1_to_i8(b, cmp);
                }
                // general case
                auto classify = [](const Type *t) noexcept {
                    return std::make_tuple(
                        t->is_float16() || t->is_float32() || t->is_float64(),
                        t->is_int8() || t->is_int16() || t->is_int32() || t->is_int64(),
                        t->is_uint8() || t->is_uint16() || t->is_uint32() || t->is_uint64() || t->is_bool());
                };
                auto llvm_dst_type = _translate_type(dst_type, true);
                auto [dst_float, dst_int, dst_uint] = classify(dst_elem_type);
                auto [src_float, src_int, src_uint] = classify(src_elem_type);
                if (dst_float) {
                    if (src_float) { return b.CreateFPCast(llvm_value, llvm_dst_type); }
                    if (src_int) { return b.CreateSIToFP(llvm_value, llvm_dst_type); }
                    if (src_uint) { return b.CreateUIToFP(llvm_value, llvm_dst_type); }
                }
                if (dst_int) {
                    if (src_float) { return b.CreateFPToSI(llvm_value, llvm_dst_type); }
                    if (src_int) { return b.CreateIntCast(llvm_value, llvm_dst_type, true); }
                    if (src_uint) { return b.CreateIntCast(llvm_value, llvm_dst_type, false); }
                }
                if (dst_uint) {
                    if (src_float) { return b.CreateFPToUI(llvm_value, llvm_dst_type); }
                    if (src_int) { return b.CreateIntCast(llvm_value, llvm_dst_type, true); }
                    if (src_uint) { return b.CreateIntCast(llvm_value, llvm_dst_type, false); }
                }
                LUISA_ERROR_WITH_LOCATION("Invalid static cast.");
            }
            case xir::CastOp::BITWISE_CAST: {
                // create a temporary alloca
                auto alignment = std::max(_get_type_alignment(src_type),
                                          _get_type_alignment(dst_type));
                LUISA_ASSERT(_get_type_size(src_type) == _get_type_size(dst_type), "Bitwise cast size mismatch.");
                auto llvm_src_type = _translate_type(src_type, true);
                auto llvm_dst_type = _translate_type(dst_type, true);
                auto llvm_temp = b.CreateAlloca(llvm_src_type);
                if (llvm_temp->getAlign() < alignment) {
                    llvm_temp->setAlignment(llvm::Align{alignment});
                }
                // store src value
                b.CreateAlignedStore(llvm_value, llvm_temp, llvm::MaybeAlign{alignment});
                // load dst value
                return b.CreateAlignedLoad(llvm_dst_type, llvm_temp, llvm::MaybeAlign{alignment});
            }
        }
        LUISA_ERROR_WITH_LOCATION("Invalid cast operation.");
    }

    [[nodiscard]] llvm::Value *_translate_debug_break_inst(CurrentFunction &current, IRBuilder &b,
                                                           const xir::DebugBreakInst *inst) noexcept {
        llvm::SmallVector<llvm::Value *, 8> llvm_args;
        llvm::SmallVector<llvm::Type *, 8> llvm_arg_types;
        llvm_args.reserve(inst->operand_count());
        llvm_arg_types.reserve(inst->operand_count());
        for (auto use : inst->operand_uses()) {
            LUISA_ASSERT(use->value() != nullptr && use->value()->type() != nullptr,
                         "Invalid operand for debug break.");
            llvm_args.emplace_back(_lookup_value(current, b, use->value()));
            llvm_arg_types.emplace_back(llvm_args.back()->getType());
        }
        auto llvm_ptr_type = llvm::PointerType::get(_llvm_context, 0);
        auto llvm_i64_type = llvm::IntegerType::get(_llvm_context, 64);
        auto llvm_struct_type = llvm::StructType::get(_llvm_context, llvm_arg_types);
        // declare a callback function: (ptr struct, ptr eval_func) -> void
        auto llvm_callback_type = llvm::FunctionType::get(
            b.getVoidTy(), {llvm_ptr_type, llvm_ptr_type}, false);
        auto llvm_callback = llvm::Function::Create(
            llvm_callback_type, llvm::Function::ExternalLinkage, "luisa.debug.break.callback", _llvm_module);
        {
            llvm_callback->setDoesNotThrow();
            llvm_callback->setDoesNotRecurse();
            llvm_callback->setMustProgress();
            llvm_callback->setWillReturn();
            _debug_callback_map.emplace_back(inst->callback(), llvm_callback->getName().str());
        }
        // create a evaluation function: (ptr struct, i64 index) -> GEP(struct, index)
        auto llvm_eval_func_type = llvm::FunctionType::get(
            llvm_ptr_type, {llvm_ptr_type, llvm_i64_type}, false);
        auto llvm_eval_func = llvm::Function::Create(
            llvm_eval_func_type, llvm::Function::PrivateLinkage, "luisa.debug.break.eval", _llvm_module);
        {
            auto llvm_body = llvm::BasicBlock::Create(_llvm_context, "body", llvm_eval_func);
            IRBuilder b{llvm_body};
            auto llvm_struct_ptr = llvm_eval_func->getArg(0);
            auto llvm_arg_index = llvm_eval_func->getArg(1);
            auto llvm_unreachable_block = llvm::BasicBlock::Create(_llvm_context, "unreachable", llvm_eval_func);
            auto llvm_switch = b.CreateSwitch(llvm_arg_index, llvm_unreachable_block, llvm_args.size());
            for (auto i = 0u; i < llvm_args.size(); i++) {
                auto llvm_case_value = b.getInt64(i);
                auto llvm_case_block = llvm::BasicBlock::Create(_llvm_context, "", llvm_eval_func);
                llvm_switch->addCase(llvm_case_value, llvm_case_block);
                b.SetInsertPoint(llvm_case_block);
                auto llvm_arg_ptr = b.CreateStructGEP(llvm_struct_type, llvm_struct_ptr, i);
                b.CreateRet(llvm_arg_ptr);
            }
            b.SetInsertPoint(llvm_unreachable_block);
            b.CreateUnreachable();
        }

        // store the arguments in a struct
        auto llvm_struct_alloca = b.CreateAlloca(llvm_struct_type);
        b.CreateLifetimeStart(llvm_struct_alloca);
        for (auto i = 0u; i < llvm_args.size(); i++) {
            auto llvm_arg_ptr = b.CreateStructGEP(llvm_struct_type, llvm_struct_alloca, i);
            b.CreateStore(llvm_args[i], llvm_arg_ptr);
        }
        // invoke the callback
        auto llvm_call = b.CreateCall(llvm_callback, {llvm_struct_alloca, llvm_eval_func});
        b.CreateLifetimeEnd(llvm_struct_alloca);
        return llvm_call;
    }

    [[nodiscard]] llvm::Value *_translate_print_inst(CurrentFunction &current, IRBuilder &b,
                                                     const xir::PrintInst *inst) noexcept {
        // create argument struct
        llvm::SmallVector<llvm::Type *, 8> llvm_field_types;
        llvm::SmallVector<size_t, 8> llvm_field_offsets;
        size_t accum_size = 0u;
        auto llvm_i8_type = b.getInt8Ty();
        for (auto o : inst->operand_uses()) {
            auto t = o->value()->type();
            auto alignment = _get_type_alignment(t);
            auto align_size = luisa::align(accum_size, alignment);
            if (align_size > accum_size) {
                auto llvm_pad_type = llvm::ArrayType::get(llvm_i8_type, align_size - accum_size);
                llvm_field_types.emplace_back(llvm_pad_type);
            }
            llvm_field_offsets.emplace_back(llvm_field_types.size());
            auto llvm_t = _translate_type(t, true);
            llvm_field_types.emplace_back(llvm_t);
            accum_size = align_size + _get_type_size(t);
        }
        auto total_size = luisa::align(accum_size, 16u);
        if (total_size > accum_size) {
            auto llvm_pad_type = llvm::ArrayType::get(llvm_i8_type, total_size - accum_size);
            llvm_field_types.emplace_back(llvm_pad_type);
        }
        auto llvm_struct_type = llvm::StructType::get(_llvm_context, llvm_field_types);
        // fill argument struct
        auto llvm_struct_alloca = b.CreateAlloca(llvm_struct_type);
        auto llvm_total_size = b.getInt64(total_size);
#if LLVM_VERSION_MAJOR >= 22
        b.CreateLifetimeStart(llvm_struct_alloca);
#else
        b.CreateLifetimeStart(llvm_struct_alloca, llvm_total_size);
#endif
        llvm_struct_alloca->setAlignment(llvm::Align{16});
        for (auto i = 0u; i < inst->operand_count(); i++) {
            auto llvm_field_offset = llvm_field_offsets[i];
            auto llvm_field_ptr = b.CreateStructGEP(llvm_struct_type, llvm_struct_alloca, llvm_field_offset);
            auto field = inst->operand(i);
            auto llvm_field = _lookup_value(current, b, field);
            auto alignment = _get_type_alignment(field->type());
            b.CreateAlignedStore(llvm_field, llvm_field_ptr, llvm::MaybeAlign{alignment});
        }
        // declare print function
        auto llvm_i64_type = b.getInt64Ty();
        auto llvm_ptr_type = llvm::PointerType::get(_llvm_context, 0);
        // void print(const void *ctx, size_t fmt_id, const void *args);
        auto llvm_print_func_type = llvm::FunctionType::get(b.getVoidTy(), {llvm_ptr_type, llvm_i64_type, llvm_ptr_type}, false);
        auto llvm_print_func = llvm::Function::Create(llvm_print_func_type, llvm::Function::ExternalLinkage, "luisa.print", _llvm_module);
        auto llvm_print_context = _llvm_module->getOrInsertGlobal("luisa.print.context", llvm::StructType::get(_llvm_context));
        auto fmt_id = static_cast<uint64_t>(_print_inst_map.size());
        _print_inst_map.emplace_back(inst, llvm_print_func->getName());
        llvm_print_func->setCallingConv(llvm::CallingConv::C);
        llvm_print_func->addFnAttr(llvm::Attribute::NoCallback);
        llvm_print_func->setNoSync();
        llvm_print_func->setMustProgress();
        llvm_print_func->setWillReturn();
        llvm_print_func->setDoesNotThrow();
        llvm_print_func->setOnlyAccessesInaccessibleMemOrArgMem();
        llvm_print_func->setDoesNotFreeMemory();
        for (auto &&llvm_print_arg : llvm_print_func->args()) {
            if (llvm_print_arg.getType()->isPointerTy()) {
#if LLVM_VERSION_MAJOR < 21
                llvm_print_arg.addAttr(llvm::Attribute::NoCapture);
#endif
                llvm_print_arg.addAttr(llvm::Attribute::NoAlias);
                llvm_print_arg.addAttr(llvm::Attribute::ReadOnly);
                llvm_print_arg.addAttr(llvm::Attribute::NoUndef);
                llvm_print_arg.addAttr(llvm::Attribute::NonNull);
            }
        }
        // call print function
        auto llvm_fmt_id = b.getInt64(fmt_id);
        auto llvm_call = b.CreateCall(llvm_print_func, {llvm_print_context, llvm_fmt_id, llvm_struct_alloca});
#if LLVM_VERSION_MAJOR >= 22
        b.CreateLifetimeEnd(llvm_struct_alloca);
#else
        b.CreateLifetimeEnd(llvm_struct_alloca, llvm_total_size);
#endif
        return llvm_call;
    }

    [[nodiscard]] llvm::Value *_translate_thread_group_inst(CurrentFunction &current, IRBuilder &b,
                                                            const xir::ThreadGroupInst *inst) noexcept {
        // SER translates to no-op
        if (inst->op() == xir::ThreadGroupOp::SHADER_EXECUTION_REORDER) {
            return nullptr;
        }
        // sync block is only allowed in kernel
        if (inst->op() == xir::ThreadGroupOp::SYNCHRONIZE_BLOCK) {
            LUISA_ASSERT(inst->parent_function()->isa<xir::KernelFunction>(),
                         "Synchronize block is only allowed in kernel.");
            // %signal = call i8 @llvm.coro.suspend(token none, i1 false)
            auto llvm_i8_type = b.getInt8Ty();
            auto llvm_i8_one = b.getInt8(1);
            auto llvm_i8_zero = b.getInt8(0);
            auto llvm_i1_false = b.getInt1(false);
            auto llvm_token_none = llvm::ConstantTokenNone::get(_llvm_context);
            auto llvm_signal = b.CreateIntrinsic(llvm_i8_type, llvm::Intrinsic::coro_suspend, {llvm_token_none, llvm_i1_false});
            // switch %signal, label %llvm_coro_suspend_block, [i8 0, label %llvm_coro_resume_block,
            //                                                  i8 1, label %llvm_coro_cleanup_block]
            auto llvm_coro_resume_block = llvm::BasicBlock::Create(_llvm_context, "coro.resume", current.func);
            auto llvm_coro_switch = b.CreateSwitch(llvm_signal, current.coro_suspend_block, 2);
            llvm_coro_switch->addCase(llvm_i8_zero, llvm_coro_resume_block);
            llvm_coro_switch->addCase(llvm_i8_one, current.coro_cleanup_block);
            // record the override to redirect phi_incoming
            auto [_, success] = current.phi_incoming_overrides.emplace(b.GetInsertBlock(), llvm_coro_resume_block);
            LUISA_ASSERT(success, "Failed to override phi incoming block which is already set.");
            // llvm_coro_suspend_block:
            b.SetInsertPoint(llvm_coro_resume_block);
            return nullptr;
        }
        LUISA_NOT_IMPLEMENTED();
    }

    [[nodiscard]] llvm::Value *_translate_instruction(CurrentFunction &current, IRBuilder &b,
                                                      const xir::Instruction *inst) noexcept {
        switch (inst->derived_instruction_tag()) {
            case xir::DerivedInstructionTag::IF: {
                auto if_inst = static_cast<const xir::IfInst *>(inst);
                auto llvm_condition = _lookup_value(current, b, if_inst->condition());
                auto llvm_false = b.getInt8(0);
                llvm_condition = b.CreateICmpNE(llvm_condition, llvm_false);
                auto llvm_true_block = _find_or_create_basic_block(current, if_inst->true_block());
                auto llvm_false_block = _find_or_create_basic_block(current, if_inst->false_block());
                auto llvm_merge_block = _find_or_create_basic_block(current, if_inst->merge_block());
                auto llvm_inst = b.CreateCondBr(llvm_condition, llvm_true_block, llvm_false_block);
                _translate_instructions_in_basic_block(current, llvm_true_block, if_inst->true_block());
                _translate_instructions_in_basic_block(current, llvm_false_block, if_inst->false_block());
                _translate_instructions_in_basic_block(current, llvm_merge_block, if_inst->merge_block());
                return llvm_inst;
            }
            case xir::DerivedInstructionTag::SWITCH: {
                auto switch_inst = static_cast<const xir::SwitchInst *>(inst);
                auto llvm_condition = _lookup_value(current, b, switch_inst->value());
                auto llvm_merge_block = _find_or_create_basic_block(current, switch_inst->merge_block());
                auto llvm_default_block = _find_or_create_basic_block(current, switch_inst->default_block());
                auto llvm_inst = b.CreateSwitch(llvm_condition, llvm_default_block, switch_inst->case_count());
                for (auto i = 0u; i < switch_inst->case_count(); i++) {
                    auto case_value = switch_inst->case_value(i);
                    auto llvm_case_value = b.getIntN(
                        llvm_condition->getType()->getIntegerBitWidth(), case_value);
                    auto llvm_case_block = _find_or_create_basic_block(current, switch_inst->case_block(i));
                    llvm_inst->addCase(llvm_case_value, llvm_case_block);
                }
                _translate_instructions_in_basic_block(current, llvm_default_block, switch_inst->default_block());
                for (auto &case_block_use : switch_inst->case_block_uses()) {
                    auto case_block = static_cast<const xir::BasicBlock *>(case_block_use->value());
                    auto llvm_case_block = _find_or_create_basic_block(current, case_block);
                    _translate_instructions_in_basic_block(current, llvm_case_block, case_block);
                }
                _translate_instructions_in_basic_block(current, llvm_merge_block, switch_inst->merge_block());
                return llvm_inst;
            }
            case xir::DerivedInstructionTag::INDEXED_BRANCH:
                LUISA_ERROR_WITH_LOCATION(
                    "Fallback XIR codegen requires structured control flow; "
                    "run restructure_cfg before codegen.");
            case xir::DerivedInstructionTag::LOOP: {
                auto loop_inst = static_cast<const xir::LoopInst *>(inst);
                auto llvm_merge_block = _find_or_create_basic_block(current, loop_inst->merge_block());
                auto llvm_prepare_block = _find_or_create_basic_block(current, loop_inst->prepare_block());
                auto llvm_body_block = _find_or_create_basic_block(current, loop_inst->body_block());
                auto llvm_update_block = _find_or_create_basic_block(current, loop_inst->update_block());
                auto llvm_inst = b.CreateBr(llvm_prepare_block);
                _translate_instructions_in_basic_block(current, llvm_prepare_block, loop_inst->prepare_block());
                _translate_instructions_in_basic_block(current, llvm_body_block, loop_inst->body_block());
                _translate_instructions_in_basic_block(current, llvm_update_block, loop_inst->update_block());
                _translate_instructions_in_basic_block(current, llvm_merge_block, loop_inst->merge_block());
                return llvm_inst;
            }
            case xir::DerivedInstructionTag::SIMPLE_LOOP: {
                auto loop_inst = static_cast<const xir::SimpleLoopInst *>(inst);
                auto llvm_merge_block = _find_or_create_basic_block(current, loop_inst->merge_block());
                auto llvm_body_block = _find_or_create_basic_block(current, loop_inst->body_block());
                auto llvm_inst = b.CreateBr(llvm_body_block);
                _translate_instructions_in_basic_block(current, llvm_body_block, loop_inst->body_block());
                _translate_instructions_in_basic_block(current, llvm_merge_block, loop_inst->merge_block());
                return llvm_inst;
            }
            case xir::DerivedInstructionTag::CONDITIONAL_BRANCH: {
                auto cond_br_inst = static_cast<const xir::ConditionalBranchInst *>(inst);
                auto llvm_condition = _lookup_value(current, b, cond_br_inst->condition());
                auto llvm_false = b.getInt8(0);
                llvm_condition = b.CreateICmpNE(llvm_condition, llvm_false);
                auto llvm_true_block = _find_or_create_basic_block(current, cond_br_inst->true_block());
                auto llvm_false_block = _find_or_create_basic_block(current, cond_br_inst->false_block());
                auto llvm_inst = b.CreateCondBr(llvm_condition, llvm_true_block, llvm_false_block);
                _translate_instructions_in_basic_block(current, llvm_true_block, cond_br_inst->true_block());
                _translate_instructions_in_basic_block(current, llvm_false_block, cond_br_inst->false_block());
                return llvm_inst;
            }
            case xir::DerivedInstructionTag::UNREACHABLE: {
                LUISA_ASSERT(inst->type() == nullptr, "Unreachable instruction should not have a type.");
                return b.CreateUnreachable();
            }
            case xir::DerivedInstructionTag::BRANCH: [[fallthrough]];
            case xir::DerivedInstructionTag::BREAK: [[fallthrough]];
            case xir::DerivedInstructionTag::CONTINUE: {
                auto br_inst = static_cast<const xir::BranchTerminatorInstruction *>(inst);
                auto llvm_target_block = _find_or_create_basic_block(current, br_inst->target_block());
                auto llvm_inst = b.CreateBr(llvm_target_block);
                _translate_instructions_in_basic_block(current, llvm_target_block, br_inst->target_block());
                return llvm_inst;
            }
            case xir::DerivedInstructionTag::RETURN: {
                auto return_inst = static_cast<const xir::ReturnInst *>(inst);
                // a kernel has a merged exit block for easier handling of block synchronization (simulated with coroutines)
                if (return_inst->parent_function()->isa<xir::KernelFunction>()) {
                    LUISA_DEBUG_ASSERT(return_inst->return_value() == nullptr,
                                       "Kernel function should not have a return value.");
                    if (current.exit_block == nullptr) {
                        current.exit_block = llvm::BasicBlock::Create(_llvm_context, "exit", current.func);
                        IRBuilder exit_builder{current.exit_block};
                        exit_builder.CreateRetVoid();
                    }
                    return b.CreateBr(current.exit_block);
                }
                // other functions just return in place
                if (auto ret_val = return_inst->return_value()) {
                    auto llvm_ret_val = _lookup_value(current, b, ret_val);
                    return b.CreateRet(llvm_ret_val);
                }
                return b.CreateRetVoid();
            }
            case xir::DerivedInstructionTag::RASTER_DISCARD: LUISA_NOT_IMPLEMENTED();
            case xir::DerivedInstructionTag::PHI: {
                // Phi nodes might reference values that are not translated yet. So we just create an empty
                // phi and record it, which will be filled later when we are ready.
                auto phi_inst = static_cast<const xir::PhiInst *>(inst);
                current.phi_nodes.emplace_back(phi_inst);
                auto llvm_type = _translate_type(phi_inst->type(), true);
                return b.CreatePHI(llvm_type, phi_inst->incoming_count());
            }
            case xir::DerivedInstructionTag::ALLOCA: {
                // shared memory should be hoisted to global thread-local variables
                if (auto a = static_cast<const xir::AllocaInst *>(inst); a->op() == xir::AllocaOp::SHARED) {
                    // call luisa.shared.memory() to get the shared memory base address
                    auto llvm_ptr_type = llvm::PointerType::get(_llvm_context, 0);
                    auto llvm_func = _llvm_module->getOrInsertFunction(
                        "luisa.shared.memory", llvm::FunctionType::get(llvm_ptr_type, {}, false));
                    // mark that this function is pure and speculatable
                    {
                        auto llvm_func_signature = llvm::cast<llvm::Function>(llvm_func.getCallee());
                        llvm_func_signature->setDoesNotAccessMemory();
                        llvm_func_signature->setDoesNotThrow();
                        llvm_func_signature->setDoesNotFreeMemory();
                        llvm_func_signature->setWillReturn();
                        llvm_func_signature->setSpeculatable();
                        llvm_func_signature->setDoesNotRecurse();
                        llvm_func_signature->setNoSync();
                        llvm_func_signature->setMustProgress();
                        llvm_func_signature->addFnAttr(llvm::Attribute::NoCallback);
                    }
                    auto llvm_shared_memory = b.CreateCall(llvm_func);
                    // find offset into shared memory
                    auto align = std::max<size_t>(_get_type_alignment(inst->type()), 16u);
                    auto size = luisa::align(_get_type_size(inst->type()), align);
                    auto offset = luisa::align(_tls_offset, align);
                    _tls_offset += size;
                    LUISA_ASSERT(_tls_offset <= max_shared_memory_size, "Shared memory size exceeds limit.");
                    return b.CreateInBoundsPtrAdd(llvm_shared_memory, b.getInt64(offset));
                }
                // otherwise we may directly allocate the variable in the stack
                auto llvm_type = _translate_type(inst->type(), false);
                auto llvm_inst = b.CreateAlloca(llvm_type);
                auto alignment = _get_type_alignment(inst->type());
                if (llvm_inst->getAlign() < alignment) {
                    llvm_inst->setAlignment(llvm::Align{alignment});
                }
                return llvm_inst;
            }
            case xir::DerivedInstructionTag::LOAD: {
                auto load_inst = static_cast<const xir::LoadInst *>(inst);
                auto alignment = _get_type_alignment(load_inst->variable()->type());
                auto llvm_type = _translate_type(load_inst->type(), true);
                auto llvm_ptr = _lookup_value(current, b, load_inst->variable());
                return b.CreateAlignedLoad(llvm_type, llvm_ptr, llvm::MaybeAlign{alignment});
            }
            case xir::DerivedInstructionTag::STORE: {
                auto store_inst = static_cast<const xir::StoreInst *>(inst);
                auto alignment = _get_type_alignment(store_inst->variable()->type());
                auto ptr = store_inst->variable();
                auto value = store_inst->value();
                auto llvm_ptr = _lookup_value(current, b, ptr);
                auto llvm_value = _lookup_value(current, b, value);
                return b.CreateAlignedStore(llvm_value, llvm_ptr, llvm::MaybeAlign{alignment});
            }
            case xir::DerivedInstructionTag::GEP: {
                auto gep_inst = static_cast<const xir::GEPInst *>(inst);
                auto ptr = gep_inst->base();
                auto llvm_ptr = _lookup_value(current, b, ptr, false);
                return _translate_gep(current, b, gep_inst->type(), ptr->type(), llvm_ptr, gep_inst->index_uses());
            }
            case xir::DerivedInstructionTag::CALL: {
                auto call_inst = static_cast<const xir::CallInst *>(inst);
                auto llvm_func = llvm::cast<llvm::Function>(_lookup_value(current, b, call_inst->callee()));
                llvm::SmallVector<llvm::Value *> llvm_args;
                llvm_args.reserve(call_inst->argument_count());
                for (auto i = 0u; i < call_inst->argument_count(); i++) {
                    auto llvm_arg = _lookup_value(current, b, call_inst->argument(i));
                    llvm_args.emplace_back(llvm_arg);
                }
                // built-in variables
                for (auto &builtin_variable : current.builtin_variables) {
                    llvm_args.emplace_back(builtin_variable);
                }
                auto llvm_call = b.CreateCall(llvm_func, llvm_args);
                llvm_call->setCallingConv(llvm::CallingConv::Fast);
                return llvm_call;
            }
            case xir::DerivedInstructionTag::ARITHMETIC: {
                auto arithmetic_inst = static_cast<const xir::ArithmeticInst *>(inst);
                return _translate_arithmetic_inst(current, b, arithmetic_inst);
            }
            case xir::DerivedInstructionTag::CAST: {
                auto cast_inst = static_cast<const xir::CastInst *>(inst);
                return _translate_cast_inst(current, b, cast_inst->type(), cast_inst->op(), cast_inst->value());
            }
            case xir::DerivedInstructionTag::PRINT: {
                auto print_inst = static_cast<const xir::PrintInst *>(inst);
                return _translate_print_inst(current, b, print_inst);
            }
            case xir::DerivedInstructionTag::DEBUG_BREAK: {
                auto debug_break_inst = static_cast<const xir::DebugBreakInst *>(inst);
                return _translate_debug_break_inst(current, b, debug_break_inst);
            }
            case xir::DerivedInstructionTag::ASSERT: {
                auto assert_inst = static_cast<const xir::AssertInst *>(inst);
                auto llvm_condition = _lookup_value(current, b, assert_inst->condition());
                auto llvm_message = _translate_string_or_null(b, assert_inst->message());
                auto llvm_void_type = llvm::Type::getVoidTy(_llvm_context);
                auto assert_func_type = llvm::FunctionType::get(llvm_void_type, {llvm_condition->getType(), llvm_message->getType()}, false);
                auto external_assert = _llvm_module->getOrInsertFunction("luisa.assert", assert_func_type);
                auto external_assert_signature = llvm::cast<llvm::Function>(external_assert.getCallee());
                external_assert_signature->setOnlyReadsMemory();
                return b.CreateCall(external_assert, {llvm_condition, llvm_message});
            }
            case xir::DerivedInstructionTag::ASSUME: {
                auto assume_inst = static_cast<const xir::AssumeInst *>(inst);
                auto llvm_condition = _lookup_value(current, b, assume_inst->condition());
                return b.CreateAssumption(llvm_condition);// TODO: we ignore assumption message for now
            }
            case xir::DerivedInstructionTag::OUTLINE: {
                auto outline_inst = static_cast<const xir::OutlineInst *>(inst);
                auto llvm_target_block = _find_or_create_basic_block(current, outline_inst->target_block());
                auto llvm_merge_block = _find_or_create_basic_block(current, outline_inst->merge_block());
                auto llvm_inst = b.CreateBr(llvm_target_block);
                _translate_instructions_in_basic_block(current, llvm_target_block, outline_inst->target_block());
                _translate_instructions_in_basic_block(current, llvm_merge_block, outline_inst->merge_block());
                return llvm_inst;
            }
            case xir::DerivedInstructionTag::AUTODIFF_SCOPE: LUISA_ERROR_WITH_LOCATION("Unexpected autodiff_scope instruction. Run autodiff pass first.");
            case xir::DerivedInstructionTag::AUTODIFF_INTRINSIC: LUISA_ERROR_WITH_LOCATION("Unexpected autodiff_intrinsic instruction. Run autodiff pass first.");
            case xir::DerivedInstructionTag::CLOCK: {
                auto call = b.CreateIntrinsic(llvm::Intrinsic::readcyclecounter, {}, {}, {}, "clock");
                auto llvm_result_type = _translate_type(inst->type(), true);
                return b.CreateZExtOrTrunc(call, llvm_result_type);
            }
            case xir::DerivedInstructionTag::RESOURCE_QUERY: {
                auto query_inst = static_cast<const xir::ResourceQueryInst *>(inst);
                return _translate_resource_query_inst(current, b, query_inst);
            }
            case xir::DerivedInstructionTag::RESOURCE_READ: {
                auto read_inst = static_cast<const xir::ResourceReadInst *>(inst);
                return _translate_resource_read_inst(current, b, read_inst);
            }
            case xir::DerivedInstructionTag::RESOURCE_WRITE: {
                auto write_inst = static_cast<const xir::ResourceWriteInst *>(inst);
                return _translate_resource_write_inst(current, b, write_inst);
            }
            case xir::DerivedInstructionTag::THREAD_GROUP: {
                auto cta_inst = static_cast<const xir::ThreadGroupInst *>(inst);
                return _translate_thread_group_inst(current, b, cta_inst);
            }
            case xir::DerivedInstructionTag::ATOMIC: {
                auto atomic_inst = static_cast<const xir::AtomicInst *>(inst);
                switch (atomic_inst->op()) {
                    case xir::AtomicOp::EXCHANGE: return _translate_atomic_op(current, b, "exchange", atomic_inst);
                    case xir::AtomicOp::COMPARE_EXCHANGE: return _translate_atomic_op(current, b, "compare.exchange", atomic_inst);
                    case xir::AtomicOp::FETCH_ADD: return _translate_atomic_op(current, b, "fetch.add", atomic_inst);
                    case xir::AtomicOp::FETCH_SUB: return _translate_atomic_op(current, b, "fetch.sub", atomic_inst);
                    case xir::AtomicOp::FETCH_AND: return _translate_atomic_op(current, b, "fetch.and", atomic_inst);
                    case xir::AtomicOp::FETCH_OR: return _translate_atomic_op(current, b, "fetch.or", atomic_inst);
                    case xir::AtomicOp::FETCH_XOR: return _translate_atomic_op(current, b, "fetch.xor", atomic_inst);
                    case xir::AtomicOp::FETCH_MIN: return _translate_atomic_op(current, b, "fetch.min", atomic_inst);
                    case xir::AtomicOp::FETCH_MAX: return _translate_atomic_op(current, b, "fetch.max", atomic_inst);
                }
                LUISA_ERROR_WITH_LOCATION("Invalid atomic operation.");
            }
            case xir::DerivedInstructionTag::RAY_QUERY_LOOP: LUISA_ERROR_WITH_LOCATION("Unexpected RayQueryLoop instruction. Run lower_ray_query_loop pass first.");
            case xir::DerivedInstructionTag::RAY_QUERY_DISPATCH: LUISA_ERROR_WITH_LOCATION("Unexpected RayQueryDispatch instruction. Run lower_ray_query_loop pass first.");
            case xir::DerivedInstructionTag::RAY_QUERY_OBJECT_READ: {
                auto object_read_inst = static_cast<const xir::RayQueryObjectReadInst *>(inst);
                return _translate_ray_query_object_read_inst(current, b, object_read_inst);
            }
            case xir::DerivedInstructionTag::RAY_QUERY_OBJECT_WRITE: {
                auto object_write_inst = static_cast<const xir::RayQueryObjectWriteInst *>(inst);
                return _translate_ray_query_object_write_inst(current, b, object_write_inst);
            }
            case xir::DerivedInstructionTag::RAY_QUERY_PIPELINE: {
                auto pipeline_inst = static_cast<const xir::RayQueryPipelineInst *>(inst);
                return _translate_ray_query_pipeline_inst(current, b, pipeline_inst);
            }
        }
        LUISA_ERROR_WITH_LOCATION("Invalid instruction.");
    }

    [[nodiscard]] llvm::BasicBlock *_find_or_create_basic_block(CurrentFunction &current, const xir::BasicBlock *bb) noexcept {
        if (bb == nullptr) { return nullptr; }
        auto iter = current.value_map.try_emplace(bb, nullptr).first;
        if (iter->second) { return llvm::cast<llvm::BasicBlock>(iter->second); }
        auto llvm_bb = llvm::BasicBlock::Create(_llvm_context, _get_name_from_metadata(bb), current.func);
        iter->second = llvm_bb;
        return llvm_bb;
    }

    void _translate_instructions_in_basic_block(CurrentFunction &current, llvm::BasicBlock *llvm_bb, const xir::BasicBlock *bb) noexcept {
        if (bb == nullptr) { return; }
        if (current.translated_basic_blocks.emplace(llvm_bb).second) {
            IRBuilder b{llvm_bb};
            for (auto inst : bb->instructions()) {
                auto llvm_value = _translate_instruction(current, b, inst);
                auto [_, success] = current.value_map.emplace(inst, llvm_value);
                LUISA_ASSERT(success, "Instruction already translated.");
            }
        }
    }

    [[nodiscard]] llvm::BasicBlock *_translate_basic_block(CurrentFunction &current, const xir::BasicBlock *bb) noexcept {
        auto llvm_bb = _find_or_create_basic_block(current, bb);
        _translate_instructions_in_basic_block(current, llvm_bb, bb);
        return llvm_bb;
    }

    struct KernelWrapper {
        llvm::Function *func{};
        llvm::BasicBlock *entry_block{};
        llvm::Value *block_id{};
        llvm::Value *block_size{};
        llvm::Value *block_size_x{};
        llvm::Value *block_size_y{};
        llvm::Value *thread_count{};
        llvm::Value *dispatch_size{};
        llvm::SmallVector<llvm::Value *, 32u> args{};
    };

    [[nodiscard]] KernelWrapper _create_kernel_wrapper(const xir::KernelFunction *f) noexcept {
        // create a wrapper function for the kernel with the following template:
        // struct Params { params... };
        // struct LaunchConfig {
        //   uint3 block_id;
        //   uint3 dispatch_size;
        //   uint3 block_size;
        // };
        // void kernel_wrapper(Params *params, LaunchConfig *config) {
        // entry:
        //   block_id = config->block_id;
        //   block_size = config->block_size;
        //   /* assume(block_size == f.block_size) */
        //   dispatch_size = config->dispatch_size;
        //   thread_count = block_size.x * block_size.y * block_size.z;
        //   ...
        // }

        auto llvm_void_type = llvm::Type::getVoidTy(_llvm_context);
        auto llvm_ptr_type = llvm::PointerType::get(_llvm_context, 0);
        auto function_name = luisa::format("kernel.main");
        auto llvm_wrapper_type = llvm::FunctionType::get(llvm_void_type, {llvm_ptr_type, llvm_ptr_type}, false);
        auto llvm_wrapper_function = llvm::Function::Create(llvm_wrapper_type, llvm::Function::ExternalLinkage, llvm::Twine{function_name}, _llvm_module);
        auto llvm_kernel_params = llvm_wrapper_function->getArg(0);
        auto llvm_kernel_config = llvm_wrapper_function->getArg(1);
        llvm_kernel_params->setName("launch.params");
        llvm_kernel_config->setName("launch.config");
        _llvm_functions.emplace(f, llvm_wrapper_function);

        // create the entry block for the wrapper function
        auto llvm_entry_block = llvm::BasicBlock::Create(_llvm_context, "entry", llvm_wrapper_function);
        IRBuilder b{llvm_entry_block};

        // create the params struct type
        auto llvm_i8_type = llvm::Type::getInt8Ty(_llvm_context);
        llvm::SmallVector<size_t, 32u> padded_param_indices;
        llvm::SmallVector<llvm::Type *, 64u> llvm_param_types;
        constexpr auto param_alignment = 16u;
        auto param_size_accum = static_cast<size_t>(0);
        {
            auto i = 0u;
            for (auto arg : f->arguments()) {
                auto arg_type = arg->type();
                LUISA_ASSERT(_get_type_alignment(arg_type) <= param_alignment, "Invalid argument alignment.");
                auto param_offset = luisa::align(param_size_accum, param_alignment);
                // pad if necessary
                if (param_offset > param_size_accum) {
                    auto llvm_padding_type = llvm::ArrayType::get(llvm_i8_type, param_offset - param_size_accum);
                    llvm_param_types.emplace_back(llvm_padding_type);
                }
                padded_param_indices.emplace_back(llvm_param_types.size());
                auto param_type = _translate_type(arg_type, false);
                llvm_param_types.emplace_back(param_type);
                param_size_accum = param_offset + _get_type_size(arg_type);
            }
        }
        // pad the last argument
        if (auto total_size = luisa::align(param_size_accum, param_alignment);
            total_size > param_size_accum) {
            auto llvm_padding_type = llvm::ArrayType::get(llvm_i8_type, total_size - param_size_accum);
            llvm_param_types.emplace_back(llvm_padding_type);
        }
        auto llvm_params_struct_type = llvm::StructType::create(_llvm_context, llvm_param_types, "LaunchParams");
        // load the params
        auto llvm_param_ptr = llvm_wrapper_function->getArg(0);
        llvm::SmallVector<llvm::Value *, 32u> llvm_args;
        {
            auto i = 0u;
            for (auto arg : f->arguments()) {
                auto arg_type = arg->type();
                auto padded_index = padded_param_indices[i];
                auto arg_name = luisa::format("arg{}", i);
                auto llvm_gep = b.CreateStructGEP(llvm_params_struct_type, llvm_param_ptr, padded_index,
                                                  llvm::Twine{arg_name}.concat(".ptr"));
                auto llvm_arg_type = _translate_type(arg_type, true);
                auto llvm_arg = b.CreateAlignedLoad(llvm_arg_type, llvm_gep,
                                                    llvm::MaybeAlign{_get_type_alignment(arg_type)},
                                                    llvm::Twine{arg_name});
                llvm_args.emplace_back(llvm_arg);
                i++;
            }
        }
        // load the launch config
        auto llvm_builtin_storage_type = _translate_type(Type::of<uint3>(), false);
        auto llvm_launch_config_struct_type = llvm::StructType::create(
            _llvm_context,
            {
                llvm_builtin_storage_type,// block_id
                llvm_builtin_storage_type,// dispatch_size
                llvm_builtin_storage_type,// block_size (optionally read)
            },
            "LaunchConfig");
        auto llvm_i32_type = llvm::Type::getInt32Ty(_llvm_context);
        auto llvm_builtin_type = llvm::VectorType::get(llvm_i32_type, 3, false);
        auto llvm_config_ptr = llvm_wrapper_function->getArg(1);
        auto load_builtin = [&](const char *name, size_t index) noexcept {
            auto llvm_gep = b.CreateStructGEP(llvm_launch_config_struct_type, llvm_config_ptr, index, llvm::Twine{name}.concat(".ptr"));
            return b.CreateAlignedLoad(llvm_builtin_type, llvm_gep, llvm::MaybeAlign{alignof(uint3)}, llvm::Twine{name});
        };
        auto llvm_block_id = load_builtin("block_id", 0);
        auto llvm_dispatch_size = load_builtin("dispatch_size", 1);
        auto static_block_size = f->block_size();
        auto llvm_block_size = !all(static_block_size == 0u) ?
                                   llvm::cast<llvm::Value>(_translate_literal(Type::of<uint3>(), &static_block_size, true)) :
                                   llvm::cast<llvm::Value>(load_builtin("block_size", 2));
        // compute number of threads per block
        auto llvm_block_size_x = b.CreateExtractElement(llvm_block_size, static_cast<uint64_t>(0));
        auto llvm_block_size_y = b.CreateExtractElement(llvm_block_size, static_cast<uint64_t>(1));
        auto llvm_block_size_z = b.CreateExtractElement(llvm_block_size, static_cast<uint64_t>(2));
        // hint that block size is power of two if we are using dynamic block size
        if (all(static_block_size == 0u)) {
            auto assume_power_of_two = [&](llvm::Value *v) noexcept {
                // power of two check: (v & (v - 1)) == 0
                auto v_minus_one = b.CreateNUWSub(v, b.getInt32(1));
                auto v_and_v_minus_one = b.CreateAnd(v, v_minus_one);
                auto is_zero = b.CreateICmpEQ(v_and_v_minus_one, b.getInt32(0));
                return b.CreateAssumption(is_zero);
            };
            assume_power_of_two(llvm_block_size_x);
            assume_power_of_two(llvm_block_size_y);
            assume_power_of_two(llvm_block_size_z);
        }
        auto llvm_thread_count = b.CreateNUWMul(llvm_block_size_x, llvm_block_size_y, "thread_count");
        llvm_thread_count = b.CreateNUWMul(llvm_thread_count, llvm_block_size_z);

        return {
            .func = llvm_wrapper_function,
            .entry_block = llvm_entry_block,
            .block_id = llvm_block_id,
            .block_size = llvm_block_size,
            .block_size_x = llvm_block_size_x,
            .block_size_y = llvm_block_size_y,
            .thread_count = llvm_thread_count,
            .dispatch_size = llvm_dispatch_size,
            .args = std::move(llvm_args),
        };
    }

    struct KernelInvokeIndex {
        llvm::Value *thread_id{};
        llvm::Value *dispatch_id{};
        llvm::Value *in_range{};
    };

    [[nodiscard]] KernelInvokeIndex _compute_kernel_invoke_index(const KernelWrapper &llvm_wrapper, IRBuilder &b, llvm::Value *llvm_i) const noexcept {
        auto llvm_i32_type = llvm::Type::getInt32Ty(_llvm_context);
        auto llvm_builtin_type = llvm::VectorType::get(llvm_i32_type, 3, false);
        auto llvm_thread_id_x = b.CreateURem(llvm_i, llvm_wrapper.block_size_x, "thread_id.x");
        auto llvm_thread_id_yz = b.CreateUDiv(llvm_i, llvm_wrapper.block_size_x);
        auto llvm_thread_id_y = b.CreateURem(llvm_thread_id_yz, llvm_wrapper.block_size_y, "thread_id.y");
        auto llvm_thread_id_z = b.CreateUDiv(llvm_thread_id_yz, llvm_wrapper.block_size_y, "thread_id.z");
        auto llvm_thread_id = llvm::cast<llvm::Value>(llvm::PoisonValue::get(llvm_builtin_type));
        llvm_thread_id = b.CreateInsertElement(llvm_thread_id, llvm_thread_id_x, static_cast<uint64_t>(0));
        llvm_thread_id = b.CreateInsertElement(llvm_thread_id, llvm_thread_id_y, static_cast<uint64_t>(1));
        llvm_thread_id = b.CreateInsertElement(llvm_thread_id, llvm_thread_id_z, static_cast<uint64_t>(2));
        llvm_thread_id->setName("thread_id");
        // compute dispatch id
        auto llvm_dispatch_id = b.CreateNUWMul(llvm_wrapper.block_id, llvm_wrapper.block_size);
        llvm_dispatch_id = b.CreateNUWAdd(llvm_dispatch_id, llvm_thread_id, "dispatch_id");
        auto llvm_in_range = b.CreateICmpULT(llvm_dispatch_id, llvm_wrapper.dispatch_size);
        llvm_in_range = b.CreateAndReduce(llvm_in_range);
        llvm_in_range->setName("thread_id.in.range");
        return {
            .thread_id = llvm_thread_id,
            .dispatch_id = llvm_dispatch_id,
            .in_range = llvm_in_range,
        };
    }

    static llvm::Value *_invoke_kernel_function(const KernelWrapper &llvm_wrapper, llvm::Function *llvm_kernel,
                                                IRBuilder &b, const KernelInvokeIndex &llvm_invoke_index) noexcept {
        auto call_args = llvm_wrapper.args;
        for (auto i = 0u; i < CurrentFunction::builtin_variable_count; i++) {
            switch (i) {
                case CurrentFunction::builtin_variable_index_thread_id: call_args.emplace_back(llvm_invoke_index.thread_id); break;
                case CurrentFunction::builtin_variable_index_block_id: call_args.emplace_back(llvm_wrapper.block_id); break;
                case CurrentFunction::builtin_variable_index_dispatch_id: call_args.emplace_back(llvm_invoke_index.dispatch_id); break;
                case CurrentFunction::builtin_variable_index_block_size: call_args.emplace_back(llvm_wrapper.block_size); break;
                case CurrentFunction::builtin_variable_index_dispatch_size: call_args.emplace_back(llvm_wrapper.dispatch_size); break;
                default: LUISA_ERROR_WITH_LOCATION("Invalid builtin variable index.");
            }
        }
        auto llvm_call = b.CreateCall(llvm_kernel, call_args);
        llvm_call->setCallingConv(llvm::CallingConv::Fast);
        return llvm_call;
    }

    [[nodiscard]] llvm::Function *_translate_kernel_function(const xir::KernelFunction *f) noexcept {
        // void kernel_wrapper(Params *params, LaunchConfig *config) {
        // entry:
        //   ...
        //   br loop;
        //   i = alloca i32
        //   store 0, i
        // loop:
        //   load i
        //   thread_id_x = i % block_size.x;
        //   thread_id_y = (i / block_size.x) % block_size.y;
        //   thread_id_z = i / (block_size.x * block_size.y);
        //   thread_id = (thread_id_x, thread_id_y, thread_id_z);
        //   dispatch_id = block_id * block_size + thread_id;
        //   in_range = reduce_and(dispatch_id < dispatch_size);
        //   br in_range, body, update;
        // body:
        //   call f(params, thread_id, block_id, dispatch_id, block_size, dispatch_size);
        //   br update;
        // update:
        //   next_i = i + 1;
        //   store next_i, i
        //   br next_i < thread_count, loop, merge;
        // merge:
        //   ret;
        // }

        // create the kernel function
        auto llvm_kernel = _translate_function_definition(f, llvm::Function::PrivateLinkage, "kernel", false);

        // create the wrapper function
        auto llvm_wrapper = _create_kernel_wrapper(f);

        // thread-in-block loop
        IRBuilder b{llvm_wrapper.entry_block};
        auto llvm_i_alloca = b.CreateAlloca(b.getInt32Ty(), nullptr, "loop.i.alloca");
        b.CreateStore(b.getInt32(0), llvm_i_alloca);

        // loop head
        auto llvm_loop_block = llvm::BasicBlock::Create(_llvm_context, "loop.head", llvm_wrapper.func);
        b.CreateBr(llvm_loop_block);
        b.SetInsertPoint(llvm_loop_block);

        // compute thread id
        auto llvm_i = b.CreateLoad(b.getInt32Ty(), llvm_i_alloca, "loop.i");
        auto llvm_invoke_index = _compute_kernel_invoke_index(llvm_wrapper, b, llvm_i);

        // branch
        auto llvm_loop_body_block = llvm::BasicBlock::Create(_llvm_context, "loop.body", llvm_wrapper.func);
        auto llvm_loop_update_block = llvm::BasicBlock::Create(_llvm_context, "loop.update", llvm_wrapper.func);
        b.CreateCondBr(llvm_invoke_index.in_range, llvm_loop_body_block, llvm_loop_update_block);

        // loop body
        b.SetInsertPoint(llvm_loop_body_block);
        _invoke_kernel_function(llvm_wrapper, llvm_kernel, b, llvm_invoke_index);
        b.CreateBr(llvm_loop_update_block);

        // loop update
        b.SetInsertPoint(llvm_loop_update_block);
        auto llvm_next_i = b.CreateNUWAdd(llvm_i, b.getInt32(1), "loop.i.next");
        b.CreateStore(llvm_next_i, llvm_i_alloca);
        auto llvm_loop_cond = b.CreateICmpULT(llvm_next_i, llvm_wrapper.thread_count, "loop.cond");
        auto llvm_loop_merge_block = llvm::BasicBlock::Create(_llvm_context, "loop.merge", llvm_wrapper.func);
        b.CreateCondBr(llvm_loop_cond, llvm_loop_block, llvm_loop_merge_block);

        // loop merge
        b.SetInsertPoint(llvm_loop_merge_block);
        b.CreateRetVoid();

        // hoist the alloca to the entry block and return the kernel wrapper
        _move_llvm_inst_to_block_begin(llvm_i_alloca, llvm_wrapper.entry_block);
        return llvm_wrapper.func;
    }

    [[nodiscard]] llvm::Function *_translate_kernel_function_coro(const xir::KernelFunction *f) noexcept {

        // entry:
        //   ...
        //   %coro.frames = alloca [ ptr x thread_count ]
        //   store zeroinitializer, %coro.frames
        // init.head:
        //   i = phi (0, next_i)
        //   invoke_index = ...
        //   cond_br (invoke_index.in_range, init.body, init.update)
        // init.body:
        //   coro.frams[i] = call kernel
        //   br init.update
        // init.update:
        //   next_i = i + 1
        //   cond_br i < thread_count, init.head, init.done
        // init.done:

        auto llvm_kernel = _translate_function_definition(f, llvm::Function::PrivateLinkage, "kernel", true);
        auto llvm_wrapper = _create_kernel_wrapper(f);

        IRBuilder b{llvm_wrapper.entry_block};

        // useful types and constants
        auto llvm_ptr_type = llvm::PointerType::get(_llvm_context, 0);
        auto llvm_ptr_null = llvm::ConstantPointerNull::get(llvm_ptr_type);
        auto llvm_void_type = llvm::Type::getVoidTy(_llvm_context);
        auto llvm_i1_type = llvm::Type::getInt1Ty(_llvm_context);
        auto llvm_i8_type = llvm::IntegerType::get(_llvm_context, 8);
        auto llvm_i32_type = llvm::IntegerType::get(_llvm_context, 32);

        // create the handle array
        auto llvm_any_alive_alloca = b.CreateAlloca(llvm_i8_type);
        auto llvm_handle_array = b.CreateAlloca(llvm_ptr_type, llvm_wrapper.thread_count, "coro.handle.array");

        // initialize loop
        auto init_loop_i_alloca = b.CreateAlloca(llvm_i32_type, nullptr, "init.loop.i");
        b.CreateStore(b.getInt32(0), init_loop_i_alloca);
        auto init_loop_head = llvm::BasicBlock::Create(_llvm_context, "init.loop.head", llvm_wrapper.func);
        auto init_loop_body = llvm::BasicBlock::Create(_llvm_context, "init.loop.body", llvm_wrapper.func);
        auto init_loop_update = llvm::BasicBlock::Create(_llvm_context, "init.loop.update", llvm_wrapper.func);
        auto init_loop_exit = llvm::BasicBlock::Create(_llvm_context, "init.loop.exit", llvm_wrapper.func);
        b.CreateBr(init_loop_head);
        {
            // init loop head
            b.SetInsertPoint(init_loop_head);
            auto llvm_i = b.CreateLoad(llvm_i32_type, init_loop_i_alloca, "init.loop.i");
            auto llvm_handle_ptr = b.CreateInBoundsGEP(llvm_ptr_type, llvm_handle_array, llvm_i);
            b.CreateStore(llvm_ptr_null, llvm_handle_ptr);
            auto llvm_index = _compute_kernel_invoke_index(llvm_wrapper, b, llvm_i);
            b.CreateCondBr(llvm_index.in_range, init_loop_body, init_loop_update);

            // init loop body
            b.SetInsertPoint(init_loop_body);
            auto llvm_handle = _invoke_kernel_function(llvm_wrapper, llvm_kernel, b, llvm_index);
            b.CreateStore(llvm_handle, llvm_handle_ptr);
            b.CreateBr(init_loop_update);

            // init loop update
            b.SetInsertPoint(init_loop_update);
            auto llvm_next_i = b.CreateAdd(llvm_i, b.getInt32(1), "init.loop.i.next", true, true);
            b.CreateStore(llvm_next_i, init_loop_i_alloca);
            auto llvm_init_loop_cond = b.CreateICmpULT(llvm_next_i, llvm_wrapper.thread_count, "init.loop.cond");
            b.CreateCondBr(llvm_init_loop_cond, init_loop_head, init_loop_exit);
        }

        // resume loop
        auto resume_loop_pre = llvm::BasicBlock::Create(_llvm_context, "resume.loop.pre", llvm_wrapper.func);
        auto resume_loop_head = llvm::BasicBlock::Create(_llvm_context, "resume.loop.head", llvm_wrapper.func);
        auto resume_loop_body = llvm::BasicBlock::Create(_llvm_context, "resume.loop.body", llvm_wrapper.func);
        auto resume_loop_invoke = llvm::BasicBlock::Create(_llvm_context, "resume.loop.invoke", llvm_wrapper.func);
        auto resume_loop_destroy = llvm::BasicBlock::Create(_llvm_context, "resume.loop.destroy", llvm_wrapper.func);
        auto resume_loop_update = llvm::BasicBlock::Create(_llvm_context, "resume.loop.update", llvm_wrapper.func);
        auto resume_loop_post = llvm::BasicBlock::Create(_llvm_context, "resume.loop.post", llvm_wrapper.func);

        // main loop that checks all coroutine instances until they all done
        b.SetInsertPoint(init_loop_exit);
        b.CreateBr(resume_loop_pre);

        b.SetInsertPoint(resume_loop_pre);
        auto resume_loop_i_alloca = b.CreateAlloca(llvm_i32_type, nullptr, "resume.loop.i");
        b.CreateStore(b.getInt32(0), resume_loop_i_alloca);
        b.CreateStore(b.getInt8(0), llvm_any_alive_alloca);
        b.CreateBr(resume_loop_head);
        {
            // resume loop head
            b.SetInsertPoint(resume_loop_head);
            auto llvm_i = b.CreateLoad(llvm_i32_type, resume_loop_i_alloca, "resume.loop.i");
            auto llvm_handle_ptr = b.CreateInBoundsGEP(llvm_ptr_type, llvm_handle_array, llvm_i);
            auto llvm_handle = b.CreateLoad(llvm_ptr_type, llvm_handle_ptr);
            auto llvm_handle_is_null = b.CreateICmpEQ(llvm_handle, llvm_ptr_null);
            b.CreateCondBr(llvm_handle_is_null, resume_loop_update, resume_loop_body);

            b.SetInsertPoint(resume_loop_body);
            auto llvm_handle_is_done = b.CreateIntrinsic(llvm_i1_type, llvm::Intrinsic::coro_done, {llvm_handle});
            b.CreateCondBr(llvm_handle_is_done, resume_loop_destroy, resume_loop_invoke);

            b.SetInsertPoint(resume_loop_invoke);
            b.CreateIntrinsic(llvm_void_type, llvm::Intrinsic::coro_resume, {llvm_handle});
            b.CreateStore(b.getInt8(1), llvm_any_alive_alloca);
            b.CreateBr(resume_loop_update);

            b.SetInsertPoint(resume_loop_destroy);
            b.CreateIntrinsic(llvm_void_type, llvm::Intrinsic::coro_destroy, {llvm_handle});
            b.CreateStore(llvm_ptr_null, llvm_handle_ptr);
            b.CreateBr(resume_loop_update);

            // resume loop update
            b.SetInsertPoint(resume_loop_update);
            auto llvm_next_i = b.CreateAdd(llvm_i, b.getInt32(1), "resume.loop.i.next", true, true);
            b.CreateStore(llvm_next_i, resume_loop_i_alloca);
            auto llvm_resume_loop_cond = b.CreateICmpULT(llvm_next_i, llvm_wrapper.thread_count, "resume.loop.cond");
            b.CreateCondBr(llvm_resume_loop_cond, resume_loop_head, resume_loop_post);
        }
        auto exit_block = llvm::BasicBlock::Create(_llvm_context, "exit", llvm_wrapper.func);
        b.SetInsertPoint(resume_loop_post);
        auto llvm_any_alive = b.CreateLoad(llvm_i8_type, llvm_any_alive_alloca);
        auto llvm_any_alive_is_zero = b.CreateICmpEQ(llvm_any_alive, b.getInt8(0));
        b.CreateCondBr(llvm_any_alive_is_zero, exit_block, resume_loop_pre);

        // hoist allocated variables to the entry block
        _move_llvm_inst_to_block_begin(llvm_any_alive_alloca, llvm_wrapper.entry_block);
        _move_llvm_inst_to_block_begin(llvm_handle_array, llvm_wrapper.entry_block);
        _move_llvm_inst_to_block_begin(init_loop_i_alloca, llvm_wrapper.entry_block);
        _move_llvm_inst_to_block_begin(resume_loop_i_alloca, llvm_wrapper.entry_block);

        // return
        b.SetInsertPoint(exit_block);
        b.CreateRetVoid();
        return llvm_wrapper.func;
    }

    [[nodiscard]] llvm::Function *_translate_function_definition(const xir::FunctionDefinition *f,
                                                                 llvm::Function::LinkageTypes linkage,
                                                                 llvm::StringRef default_name,
                                                                 bool requires_block_sync) noexcept {
        auto llvm_ret_type = _translate_type(f->type(), true);
        auto llvm_ptr_type = llvm::PointerType::get(_llvm_context, 0);
        if (requires_block_sync) {// block synchronization requires a coroutine, which always returns a pointer to the coroutine frame
            LUISA_ASSERT(llvm_ret_type->isVoidTy(), "Block synchronization requires void return type.");
            llvm_ret_type = llvm_ptr_type;
        }
        llvm::SmallVector<llvm::Type *, 64u> llvm_arg_types;
        for (auto arg : f->arguments()) {
            if (arg->is_reference()) {
                // reference arguments are passed by pointer
                llvm_arg_types.emplace_back(llvm::PointerType::get(_llvm_context, 0));
            } else {
                // value and resource arguments are passed by value
                llvm_arg_types.emplace_back(_translate_type(arg->type(), true));
            }
        }
        // built-in variables
        auto llvm_i32_type = llvm::Type::getInt32Ty(_llvm_context);
        auto llvm_i32x3_type = llvm::VectorType::get(llvm_i32_type, 3, false);
        for (auto builtin = 0u; builtin < CurrentFunction::builtin_variable_count; builtin++) {
            llvm_arg_types.emplace_back(llvm_i32x3_type);
        }

        // create function
        auto llvm_func_type = llvm::FunctionType::get(llvm_ret_type, llvm_arg_types, false);
        auto func_name = _get_name_from_metadata(f, default_name);
        auto llvm_func = llvm::Function::Create(llvm_func_type, linkage, llvm::Twine{func_name}, _llvm_module);
        _llvm_functions.emplace(f, llvm_func);

        // use fastcc
        llvm_func->setCallingConv(llvm::CallingConv::Fast);

        // inline functions that have too many arguments
        // static constexpr auto max_argument_count = 16u;
        // if (f->derived_function_tag() != xir::DerivedFunctionTag::KERNEL &&
        //     llvm_func->arg_size() > max_argument_count) {
        //     llvm_func->addFnAttr(llvm::Attribute::AlwaysInline);
        // }
        // create current translation context
        CurrentFunction current{.func = llvm_func};
        // map arguments
        {
            auto i = 0u;
            // non-built-in arguments
            for (auto arg : f->arguments()) {
                current.value_map.emplace(arg, llvm_func->getArg(i));
                i++;
            }
            // built-in variables
            for (auto builtin = 0u; builtin < CurrentFunction::builtin_variable_count; builtin++) {
                auto llvm_arg = llvm_func->getArg(i);
                switch (builtin) {
                    case CurrentFunction::builtin_variable_index_thread_id: llvm_arg->setName("thread_id"); break;
                    case CurrentFunction::builtin_variable_index_block_id: llvm_arg->setName("block_id"); break;
                    case CurrentFunction::builtin_variable_index_dispatch_id: llvm_arg->setName("dispatch_id"); break;
                    case CurrentFunction::builtin_variable_index_block_size: llvm_arg->setName("block_size"); break;
                    case CurrentFunction::builtin_variable_index_dispatch_size: llvm_arg->setName("dispatch_size"); break;
                    default: LUISA_ERROR_WITH_LOCATION("Invalid builtin variable index.");
                }
                current.builtin_variables[builtin] = llvm_arg;
                i++;
            }
        }
        // if block synchronization is required, we need to create a coroutine
        if (requires_block_sync) {

            // add `presplitcoroutine` attribute to the function
            llvm_func->setPresplitCoroutine();
            llvm_func->setCoroDestroyOnlyWhenComplete();

            // some llvm types and constants for convenience
            auto llvm_token_type = llvm::Type::getTokenTy(_llvm_context);
            auto llvm_void_type = llvm::Type::getVoidTy(_llvm_context);
            auto llvm_i1_type = llvm::Type::getInt1Ty(_llvm_context);
            auto llvm_i8_type = llvm::Type::getInt8Ty(_llvm_context);
            auto llvm_i64_type = llvm::Type::getInt64Ty(_llvm_context);
            auto llvm_i32_zero = llvm::ConstantInt::get(llvm_i32_type, 0);
            auto llvm_token_none = llvm::ConstantTokenNone::get(_llvm_context);
            auto llvm_ptr_null = llvm::ConstantPointerNull::get(llvm_ptr_type);
            auto llvm_i1_true = llvm::ConstantInt::get(llvm_i1_type, 1);
            auto llvm_i1_false = llvm::ConstantInt::get(llvm_i1_type, 0);

            // coro.entry:
            auto entry_block = llvm::BasicBlock::Create(_llvm_context, "coro.entry", llvm_func);
            IRBuilder coro_builder{entry_block};
            //   %id = call token @llvm.coro.id(i32 0, ptr null, ptr null, ptr null)
            current.coro_token = coro_builder.CreateIntrinsic(
                llvm_token_type, llvm::Intrinsic::coro_id,
                {llvm_i32_zero, llvm_ptr_null, llvm_ptr_null, llvm_ptr_null});
            //   if (i1 @llvm.coro.alloc(token %id)) { br coro.dyn.alloc } else { br coro.begin }
            auto llvm_requires_dyn_alloc = coro_builder.CreateIntrinsic(
                llvm_i1_type, llvm::Intrinsic::coro_alloc, {current.coro_token});
            auto llvm_coro_dyn_alloc_block = llvm::BasicBlock::Create(_llvm_context, "coro.dyn.alloc", llvm_func);
            auto llvm_coro_begin_block = llvm::BasicBlock::Create(_llvm_context, "coro.begin", llvm_func);
            coro_builder.CreateCondBr(llvm_requires_dyn_alloc, llvm_coro_dyn_alloc_block, llvm_coro_begin_block);

            // coro.dyn.alloc:
            coro_builder.SetInsertPoint(llvm_coro_dyn_alloc_block);
            //   %size = call i32 @llvm.coro.size.i32()
            auto llvm_coro_size = coro_builder.CreateIntrinsic(
                llvm_i32_type, llvm::Intrinsic::coro_size, {});
            //   %alloc = call ptr @luisa.coro.alloc(i64 %size)
            auto llvm_coro_size_i64 = coro_builder.CreateZExt(llvm_coro_size, llvm_i64_type);
            auto llvm_luisa_coro_alloc = _llvm_module->getOrInsertFunction(
                "luisa.coro.alloc", llvm::FunctionType::get(llvm_ptr_type, {llvm_i64_type}, false));
            {
                auto llvm_luisa_coro_alloc_func = llvm::cast<llvm::Function>(llvm_luisa_coro_alloc.getCallee());
                llvm_luisa_coro_alloc_func->addFnAttr(llvm::Attribute::NoCallback);
                llvm_luisa_coro_alloc_func->setNoSync();
                llvm_luisa_coro_alloc_func->setDoesNotThrow();
                llvm_luisa_coro_alloc_func->setDoesNotRecurse();
                llvm_luisa_coro_alloc_func->setDoesNotFreeMemory();
                llvm_luisa_coro_alloc_func->setOnlyAccessesInaccessibleMemory();
                llvm_luisa_coro_alloc_func->setWillReturn();
                llvm_luisa_coro_alloc_func->setMustProgress();
                llvm_luisa_coro_alloc_func->setReturnDoesNotAlias();
            }
            auto llvm_dyn_coro_handle = coro_builder.CreateCall(llvm_luisa_coro_alloc, {llvm_coro_size_i64});
            //   br coro.begin
            coro_builder.CreateBr(llvm_coro_begin_block);

            // coro.begin:
            coro_builder.SetInsertPoint(llvm_coro_begin_block);
            //   %frame = phi ptr [ %alloc, coro.dyn.alloc ], [ null, entry ]
            auto llvm_coro_frame = coro_builder.CreatePHI(llvm_ptr_type, 2);
            llvm_coro_frame->addIncoming(llvm_dyn_coro_handle, llvm_coro_dyn_alloc_block);
            llvm_coro_frame->addIncoming(llvm_ptr_null, entry_block);
            //   %handle = call ptr @llvm.coro.begin(token %id, ptr %frame)
            current.coro_handle = coro_builder.CreateIntrinsic(
                llvm_ptr_type, llvm::Intrinsic::coro_begin, {current.coro_token, llvm_coro_frame});
            // will br func.body later

            // now we create the coroutine exit block (i.e., final suspend)
            current.exit_block = llvm::BasicBlock::Create(_llvm_context, "coro.exit", llvm_func);

            // coro.exit:
            coro_builder.SetInsertPoint(current.exit_block);
            //   %signal = call i8 @llvm.coro.suspend(token none, i1 true)
            auto llvm_coro_signal = coro_builder.CreateIntrinsic(
                llvm_i8_type, llvm::Intrinsic::coro_suspend, {llvm_token_none, llvm_i1_true});
            // switch i8 %signal, label %coro.suspend [i8 0, label %coro.unreachable
            //                                         i8 1, label %coro.cleanup]
            current.coro_suspend_block = llvm::BasicBlock::Create(_llvm_context, "coro.suspend", llvm_func);
            current.coro_cleanup_block = llvm::BasicBlock::Create(_llvm_context, "coro.cleanup", llvm_func);
            auto llvm_coro_unreachable_block = llvm::BasicBlock::Create(_llvm_context, "coro.unreachable", llvm_func);
            auto llvm_coro_switch = coro_builder.CreateSwitch(llvm_coro_signal, current.coro_suspend_block, 2);
            llvm_coro_switch->addCase(llvm::ConstantInt::get(llvm_i8_type, 0), llvm_coro_unreachable_block);
            llvm_coro_switch->addCase(llvm::ConstantInt::get(llvm_i8_type, 1), current.coro_cleanup_block);

            // coro.unreachable:
            coro_builder.SetInsertPoint(llvm_coro_unreachable_block);
            //   unreachable
            coro_builder.CreateUnreachable();

            // coro.cleanup:
            coro_builder.SetInsertPoint(current.coro_cleanup_block);
            //   %mem = call ptr @llvm.coro.free(token %id, ptr %hdl)
            auto llvm_coro_mem = coro_builder.CreateIntrinsic(
                llvm_ptr_type, llvm::Intrinsic::coro_free,
                {current.coro_token, current.coro_handle});
            //   if (%mem != null) { br coro.dyn.free } else { br coro.suspend }
            auto llvm_coro_dyn_free_block = llvm::BasicBlock::Create(_llvm_context, "coro.dyn.free", llvm_func);
            auto llvm_coro_free_cond = coro_builder.CreateICmpNE(llvm_coro_mem, llvm_ptr_null);
            coro_builder.CreateCondBr(llvm_coro_free_cond, llvm_coro_dyn_free_block, current.coro_suspend_block);

            // coro.dyn.free:
            coro_builder.SetInsertPoint(llvm_coro_dyn_free_block);
            //   call void @luisa.coro.free(ptr %mem)
            // luisa.coro.free does nothing so we can just ignore it
            // auto llvm_luisa_coro_free = _llvm_module->getOrInsertFunction(
            //     "luisa.coro.free", llvm::FunctionType::get(llvm_void_type, {llvm_ptr_type}, false));
            // coro_builder.CreateCall(llvm_luisa_coro_free, {llvm_coro_mem});
            //   br coro.suspend
            coro_builder.CreateBr(current.coro_suspend_block);

            // coro.suspend:
            coro_builder.SetInsertPoint(current.coro_suspend_block);
            //   %unused = call i1 @llvm.coro.end(ptr %hdl, i1 false, token none)
            auto llvm_coro_end = coro_builder.CreateIntrinsic(
                llvm_i1_type, llvm::Intrinsic::coro_end,
                {current.coro_handle, llvm_i1_false, llvm_token_none});
            // ret ptr %hdl
            coro_builder.CreateRet(current.coro_handle);

            // finally we can create the function body and br to it
            auto llvm_func_body_block = _translate_basic_block(current, f->body_block());
            coro_builder.SetInsertPoint(llvm_coro_begin_block);
            coro_builder.CreateBr(llvm_func_body_block);

        } else {// otherwise, we may translate the body directly
            static_cast<void>(_translate_basic_block(current, f->body_block()));
        }

        // fill the phi nodes
        {
            IRBuilder b{_llvm_context};
            for (auto phi : current.phi_nodes) {
                auto llvm_phi = llvm::cast<llvm::PHINode>(current.value_map.at(phi));
                b.SetInsertPoint(llvm_phi);
                for (auto i = 0u; i < phi->incoming_count(); i++) {
                    auto incoming = phi->incoming(i);
                    auto llvm_incoming_value = _lookup_value(current, b, incoming.value);
                    auto llvm_incoming_block = _find_or_create_basic_block(current, incoming.block);
                    // handle incoming block redirections due to block synchronization
                    for (;;) {
                        auto iter = current.phi_incoming_overrides.find(llvm_incoming_block);
                        if (iter == current.phi_incoming_overrides.end()) { break; }
                        llvm_incoming_block = iter->second;
                    }
                    llvm_phi->addIncoming(llvm_incoming_value, llvm_incoming_block);
                }
            }
        }

        // we should hoist all alloca instructions to the beginning of the function
        {
            luisa::vector<llvm::AllocaInst *> alloca_insts;
            for (auto &&llvm_bb : *llvm_func) {
                for (auto &&llvm_inst : llvm_bb) {
                    if (auto alloca_inst = llvm::dyn_cast<llvm::AllocaInst>(&llvm_inst)) {
                        alloca_insts.emplace_back(alloca_inst);
                    }
                }
            }
            // reverse the order of alloca instructions for better readability
            std::reverse(alloca_insts.begin(), alloca_insts.end());
            // move alloca instructions to the beginning of the function
            auto &llvm_entry = llvm_func->getEntryBlock();
            for (auto inst : alloca_insts) {
                _move_llvm_inst_to_block_begin(inst, &llvm_entry);
            }
        }
        // return
        return llvm_func;
    }

    [[nodiscard]] llvm::Function *_translate_callable_function(const xir::CallableFunction *f) noexcept {
        return _translate_function_definition(f, llvm::Function::PrivateLinkage, "callable", false);
    }

    [[nodiscard]] llvm::Function *_translate_function(const xir::Function *f) noexcept {
        if (auto iter = _llvm_functions.find(f); iter != _llvm_functions.end()) {
            return iter->second;
        }
        auto llvm_func = [&] {
            switch (f->derived_function_tag()) {
                case xir::DerivedFunctionTag::KERNEL: {
                    auto k = static_cast<const xir::KernelFunction *>(f);
                    return _analyze_requires_sync_block(k) ?
                               _translate_kernel_function_coro(k) :
                               _translate_kernel_function(k);
                }
                case xir::DerivedFunctionTag::CALLABLE:
                    return _translate_callable_function(static_cast<const xir::CallableFunction *>(f));
                case xir::DerivedFunctionTag::EXTERNAL: LUISA_NOT_IMPLEMENTED();
            }
            LUISA_ERROR_WITH_LOCATION("Invalid function type.");
        }();
        _llvm_functions.emplace(f, llvm_func);
        return llvm_func;
    }

public:
    explicit FallbackCodegen(llvm::LLVMContext &ctx) noexcept
        : _llvm_context{ctx} {}

    FallbackCodeGenFeedback emit(llvm::Module *llvm_module, const xir::Module *module) noexcept {
        auto location_md = module->find_metadata<xir::LocationMD>();
        auto module_location = location_md ? location_md->file().string() : "unknown";
        llvm_module->setSourceFileName(location_md ? location_md->file().string() : "unknown");
        if (auto module_name = _get_name_from_metadata(module); !module_name.empty()) {
            llvm_module->setModuleIdentifier(module_name);
        }
        _llvm_module = llvm_module;
        _translate_module(module);
        _reset();
        return {
            .print_inst_map = std::exchange(_print_inst_map, {}),
            .debug_callback_map = std::exchange(_debug_callback_map, {}),
        };
    }
};

FallbackCodeGenFeedback
luisa_fallback_backend_codegen(llvm::LLVMContext &llvm_ctx, llvm::Module *llvm_module, const xir::Module *module) noexcept {
    FallbackCodegen codegen{llvm_ctx};
    return codegen.emit(llvm_module, module);
}

}// namespace luisa::compute::fallback
