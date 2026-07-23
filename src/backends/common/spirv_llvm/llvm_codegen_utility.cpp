#include "llvm_codegen_utility.h"
#include "llvm_codegen_stack_data.h"
#include "llvm_compat.h"
#include "llvm_state_visitor.h"
#include "vulkan_binding_properties.h"

#include <llvm/IR/Constants.h>
#include <llvm/IR/GlobalVariable.h>
#include <llvm/IR/Verifier.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Bitcode/BitcodeWriter.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Transforms/Utils/Cloning.h>

#include <spirv-tools/libspirv.hpp>
#include <spirv-tools/optimizer.hpp>

// SPIR-V specific LLVM intrinsics (llvm.spv.thread.id, etc.)
#include <llvm/IR/IntrinsicsSPIRV.h>

#include "../hlsl/shader_property.h"

#include <luisa/ast/variable.h>
#include <luisa/ast/usage.h>
#include <luisa/runtime/rhi/resource.h>
#ifdef alloca
#undef alloca
#endif

namespace lc::llvm_codegen {

// ============================================================================
// LLVMCodegenUtility
// ============================================================================

LLVMCodegenUtility::LLVMCodegenUtility()
    : _context(new llvm::LLVMContext()),
      _module(new llvm::Module("luisa_llvm_module", *_context)),
      _builder(new llvm::IRBuilder<>(*_context)) {
    opt = LLVMCodegenStackData::Allocate(this);
}

LLVMCodegenUtility::~LLVMCodegenUtility() {
    LLVMCodegenStackData::DeAllocate(std::move(opt));
}

void LLVMCodegenUtility::ResetModule() {
    _module.reset(new llvm::Module("luisa_llvm_module", *_context));
    _builder.reset(new llvm::IRBuilder<>(*_context));
    _current_function = nullptr;
    LLVMCodegenStackData::DeAllocate(std::move(opt));
    opt = LLVMCodegenStackData::Allocate(this);
}

// ============================================================================
// Type Mapping
// ============================================================================

llvm::Type *LLVMCodegenUtility::ToLLVMType(Type const &type) {
    switch (type.tag()) {
        // --- Scalar types ---
        case Type::Tag::BOOL:
            return _builder->getInt1Ty();
        case Type::Tag::INT8:
            return _builder->getInt8Ty();
        case Type::Tag::UINT8:
            return _builder->getInt8Ty();
        case Type::Tag::INT16:
            return _builder->getInt16Ty();
        case Type::Tag::UINT16:
            return _builder->getInt16Ty();
        case Type::Tag::INT32:
            return _builder->getInt32Ty();
        case Type::Tag::UINT32:
            return _builder->getInt32Ty();
        case Type::Tag::INT64:
            return _builder->getInt64Ty();
        case Type::Tag::UINT64:
            return _builder->getInt64Ty();
        case Type::Tag::FLOAT16:
            // NOTE: LLVM half is supported; fallback to i16 if issues arise
            return _builder->getHalfTy();
        case Type::Tag::FLOAT32:
            return _builder->getFloatTy();
        case Type::Tag::FLOAT64:
            return _builder->getDoubleTy();
        case Type::Tag::FLOAT8_E4M3:
        case Type::Tag::FLOAT8_E5M2:
            // No native LLVM type for 8-bit float; use i8
            return _builder->getInt8Ty();

        // --- Vector types ---
        case Type::Tag::VECTOR: {
            auto *elem_type = ToLLVMType(*type.element());
            auto dim = type.dimension();
            return llvm::FixedVectorType::get(elem_type, dim);
        }

        // --- Matrix types (array of vectors) ---
        case Type::Tag::MATRIX: {
            auto dim = type.dimension();
            auto *row_type = llvm::FixedVectorType::get(_builder->getFloatTy(), dim);
            return llvm::ArrayType::get(row_type, dim);
        }

        // --- Array types ---
        case Type::Tag::ARRAY: {
            auto *elem_type = ToLLVMType(*type.element());
            auto dim = type.dimension();
            return llvm::ArrayType::get(elem_type, dim);
        }

        // --- Structure types ---
        case Type::Tag::STRUCTURE: {
            return RegistStructType(&type);
        }

        // --- Resource types (opaque pointers) ---
        case Type::Tag::BUFFER:
        case Type::Tag::TEXTURE:
        case Type::Tag::BINDLESS_ARRAY:
        case Type::Tag::ACCEL:
            return _builder->getPtrTy(0);// opaque pointer in addr space 0

        // --- Cooperative types ---
        case Type::Tag::COOPERATIVE_VECTOR:
        case Type::Tag::COOPERATIVE_VECTOR_REF:
        case Type::Tag::COOPERATIVE_MATRIX_REF:
            // Represent as i32 metadata
            return _builder->getInt32Ty();

        // --- Custom types ---
        case Type::Tag::CUSTOM:
            // Fallback to opaque pointer
            return _builder->getPtrTy(0);

        default:
            LUISA_ERROR_WITH_LOCATION("Unsupported type tag: {}",
                                      static_cast<uint32_t>(type.tag()));
            return _builder->getInt32Ty();// unreachable
    }
}

llvm::StructType *LLVMCodegenUtility::RegistStructType(Type const *type) {
    if (type->tag() != Type::Tag::STRUCTURE) {
        LUISA_ERROR_WITH_LOCATION("RegistStructType called on non-struct type.");
    }

    auto it = opt->struct_types.find(type);
    if (it != opt->struct_types.end()) {
        return it->second;
    }

    // Create a named struct type
    vstd::StringBuilder name_builder;
    GetTypeName(*type, name_builder);

    auto *struct_type = llvm::StructType::create(*_context, name_builder.data());

    // Set body
    luisa::vector<llvm::Type *> member_types;
    auto members = type->members();
    member_types.reserve(members.size());
    for (auto *member : members) {
        member_types.push_back(ToLLVMType(*member));
    }
    struct_type->setBody(member_types);

    opt->struct_types[type] = struct_type;
    return struct_type;
}

void LLVMCodegenUtility::GetTypeName(Type const &type, vstd::StringBuilder &str) {
    switch (type.tag()) {
        case Type::Tag::BOOL: str << "bool"; break;
        case Type::Tag::INT8: str << "int8_t"; break;
        case Type::Tag::UINT8: str << "uint8_t"; break;
        case Type::Tag::INT16: str << "int16_t"; break;
        case Type::Tag::UINT16: str << "uint16_t"; break;
        case Type::Tag::INT32: str << "int"; break;
        case Type::Tag::UINT32: str << "uint"; break;
        case Type::Tag::INT64: str << "int64_t"; break;
        case Type::Tag::UINT64: str << "uint64_t"; break;
        case Type::Tag::FLOAT16: str << "half"; break;
        case Type::Tag::FLOAT32: str << "float"; break;
        case Type::Tag::FLOAT64: str << "double"; break;
        case Type::Tag::VECTOR: {
            GetTypeName(*type.element(), str);
            str << vstd::to_string(type.dimension());
            break;
        }
        case Type::Tag::MATRIX: {
            str << "float" << vstd::to_string(type.dimension())
                << "x" << vstd::to_string(type.dimension());
            break;
        }
        case Type::Tag::ARRAY: {
            str << "array_";
            GetTypeName(*type.element(), str);
            str << "_" << vstd::to_string(type.dimension());
            break;
        }
        case Type::Tag::STRUCTURE: {
            str << "struct." << type.hash();
            break;
        }
        case Type::Tag::BUFFER:
            str << "buffer";
            break;
        case Type::Tag::TEXTURE:
            str << "texture";
            break;
        case Type::Tag::BINDLESS_ARRAY:
            str << "bindless_array";
            break;
        case Type::Tag::ACCEL:
            str << "accel";
            break;
        default:
            str << "type_" << vstd::to_string(type.hash());
            break;
    }
}

// ============================================================================
// Variable Naming
// ============================================================================

void LLVMCodegenUtility::GetVariableName(Function func, Variable const &v, vstd::StringBuilder &str) {
    GetVariableName(func, v.tag(), v.uid(), str);
}

void LLVMCodegenUtility::GetVariableName(Function func, Variable::Tag tag, uint32_t id, vstd::StringBuilder &str) {
    switch (tag) {
        case Variable::Tag::LOCAL:
            str << "_V" << vstd::to_string(id);
            break;
        case Variable::Tag::SHARED:
            str << "_S" << vstd::to_string(id);
            break;
        case Variable::Tag::REFERENCE:
            str << "_R" << vstd::to_string(id);
            break;
        case Variable::Tag::BUFFER:
            str << "_B" << vstd::to_string(id);
            break;
        case Variable::Tag::TEXTURE:
            str << "_T" << vstd::to_string(id);
            break;
        case Variable::Tag::BINDLESS_ARRAY:
            str << "_BA" << vstd::to_string(id);
            break;
        case Variable::Tag::ACCEL:
            str << "_A" << vstd::to_string(id);
            break;
        case Variable::Tag::THREAD_ID:
            str << "_thread_id";
            break;
        case Variable::Tag::BLOCK_ID:
            str << "_block_id";
            break;
        case Variable::Tag::DISPATCH_ID:
            str << "_dispatch_id";
            break;
        case Variable::Tag::DISPATCH_SIZE:
            str << "_dispatch_size";
            break;
        case Variable::Tag::KERNEL_ID:
            str << "_kernel_id";
            break;
        case Variable::Tag::WARP_LANE_COUNT:
            str << "_warp_lane_count";
            break;
        case Variable::Tag::WARP_LANE_ID:
            str << "_warp_lane_id";
            break;
        default:
            str << "_var_" << vstd::to_string(id);
            break;
    }
}

// ============================================================================
// Function Naming
// ============================================================================

void LLVMCodegenUtility::GetFunctionName(Function callable, vstd::StringBuilder &result) {
    auto name = callable.name();
    if (!name.empty()) {
        result << name;
    } else {
        result << "_func_" << vstd::to_string(callable.hash());
    }
}

void LLVMCodegenUtility::GetFunctionName(CallExpr const *expr, vstd::StringBuilder &result, LLVMStateVisitor &visitor) {
    // For built-in calls: use the op name
    // For custom calls: use the function name
    if (expr->is_builtin()) {
        result << "_builtin_" << vstd::to_string(static_cast<uint32_t>(expr->op()));
    } else if (expr->is_custom()) {
        GetFunctionName(expr->custom(), result);
    } else {
        LUISA_ERROR_WITH_LOCATION("Unknown call type.");
    }
}

// ============================================================================
// Constant Data
// ============================================================================

static llvm::Constant *CreateConstantImpl(ConstantData const &data, llvm::Type *type, llvm::LLVMContext &ctx) {
    auto *const_data = data.raw();
    auto size = data.type()->size();

    if (type->isIntegerTy()) {
        auto bit_width = type->getIntegerBitWidth();
        uint64_t val = 0;
        if (bit_width <= 8) {
            val = *reinterpret_cast<uint8_t const *>(const_data);
        } else if (bit_width <= 16) {
            val = *reinterpret_cast<uint16_t const *>(const_data);
        } else if (bit_width <= 32) {
            val = *reinterpret_cast<uint32_t const *>(const_data);
        } else if (bit_width <= 64) {
            val = *reinterpret_cast<uint64_t const *>(const_data);
        }
        return llvm::ConstantInt::get(type, val);
    }

    if (type->isHalfTy()) {
        // half is 16 bits
        uint16_t raw = *reinterpret_cast<uint16_t const *>(const_data);
        return llvm::ConstantFP::get(type, llvm::APFloat(llvm::APFloat::IEEEhalf(), llvm::APInt(16, raw)));
    }

    if (type->isFloatTy()) {
        float val = *reinterpret_cast<float const *>(const_data);
        return llvm::ConstantFP::get(type, val);
    }

    if (type->isDoubleTy()) {
        double val = *reinterpret_cast<double const *>(const_data);
        return llvm::ConstantFP::get(type, val);
    }

    if (auto *vec_type = llvm::dyn_cast<llvm::FixedVectorType>(type)) {
        auto elem_count = vec_type->getNumElements();
        auto *elem_type = vec_type->getElementType();
        luisa::vector<llvm::Constant *> elems;
        elems.reserve(elem_count);
        auto elem_size = data.type()->size() / elem_count;
        for (unsigned i = 0; i < elem_count; ++i) {
            // Create sub-constant for each element
            // This is approximate — proper handling needs actual element offsets
            auto *sub_data = reinterpret_cast<char const *>(const_data) + i * elem_size;
            // For simplicity, handle scalar elements
            if (elem_type->isFloatTy()) {
                float f = *reinterpret_cast<float const *>(sub_data);
                elems.push_back(llvm::ConstantFP::get(elem_type, f));
            } else if (elem_type->isIntegerTy()) {
                uint64_t v = 0;
                auto bw = elem_type->getIntegerBitWidth();
                if (bw == 32)
                    v = *reinterpret_cast<uint32_t const *>(sub_data);
                else if (bw == 64)
                    v = *reinterpret_cast<uint64_t const *>(sub_data);
                elems.push_back(llvm::ConstantInt::get(elem_type, v));
            } else {
                elems.push_back(llvm::Constant::getNullValue(elem_type));
            }
        }
        return llvm::ConstantVector::get(elems);
    }

    if (auto *arr_type = llvm::dyn_cast<llvm::ArrayType>(type)) {
        // Build array constant from raw bytes
        luisa::vector<llvm::Constant *> elems;
        auto elem_count = arr_type->getNumElements();
        auto *elem_type = arr_type->getElementType();
        auto elem_size = data.type()->size() / elem_count;
        for (unsigned i = 0; i < elem_count; ++i) {
            auto *sub_data = reinterpret_cast<char const *>(const_data) + i * elem_size;
            // For float elements
            if (elem_type->isFloatTy()) {
                float f = *reinterpret_cast<float const *>(sub_data);
                elems.push_back(llvm::ConstantFP::get(elem_type, f));
            } else {
                elems.push_back(llvm::Constant::getNullValue(elem_type));
            }
        }
        return llvm::ConstantArray::get(arr_type, elems);
    }

    // Fallback: zero initialize
    return llvm::Constant::getNullValue(type);
}

llvm::Constant *LLVMCodegenUtility::CreateConstant(ConstantData const &data, llvm::Type *type) {
    return CreateConstantImpl(data, type, *_context);
}

llvm::GlobalVariable *LLVMCodegenUtility::CreateConstantGlobal(ConstantData const &data, llvm::Type *type) {
    auto *init = CreateConstant(data, type);
    auto *gv = new llvm::GlobalVariable(
        *_module, type, true,// isConstant
        llvm::GlobalValue::InternalLinkage, init, "const");
    return gv;
}

// ============================================================================
// Function Code Generation
// ============================================================================

llvm::Function *LLVMCodegenUtility::CodegenFunction(Function func) {
    // Check if already codegen'd
    auto hash = func.hash();
    auto it = opt->func_types.find(hash);
    if (it != opt->func_types.end()) {
        return it->second;
    }

    // Save current IRBuilder insertion point and variable map so we can restore
    // them after recursively codegen'ing a callee callable.
    auto saved_ip = _builder->saveIP();
    auto *saved_function = _current_function;
    auto saved_variables = opt->variables;

    // Build function type
    auto *ret_type = func.return_type() ? ToLLVMType(*func.return_type()) : _builder->getVoidTy();
    luisa::vector<llvm::Type *> param_types;
    // Vulkan SPIR-V requires entry point functions to have zero parameters
    // (VUID-StandaloneSpirv-None-04633). For kernel entry points:
    //   - Resource arguments  → addrspace(1)/addrspace(2) GlobalVariable
    //   - Non-resource args   → _Global cbuffer in addrspace(1)
    //   - Builtin variables   → @llvm.spv.thread.id / .thread.id.in.group / .group.id intrinsics
    // For callables, all arguments remain as function parameters.
    if (func.tag() != Function::Tag::KERNEL) {
        for (auto &arg : func.arguments()) {
            param_types.push_back(ToLLVMType(*arg.type()));
        }
    }

    auto *func_type = llvm::FunctionType::get(ret_type, param_types, false);

    // Create function name — use "main" for kernel entry points
    // to match the hardcoded entry point name in ComputeShader.
    // Resource arguments are global variables, arithmetic arguments are
    // packed into a cbuffer-like global structured buffer _Global
    // (mirroring src/backends/common/hlsl/codegen_utils/cbuffer.cpp).
    vstd::StringBuilder name_builder;
    if (func.tag() == Function::Tag::KERNEL) {
        name_builder << "main";
    } else {
        GetFunctionName(func, name_builder);
    }

    // Use ExternalLinkage for kernel entry points, InternalLinkage for callables
    auto linkage = (func.tag() == Function::Tag::KERNEL) ? llvm::Function::ExternalLinkage : llvm::Function::InternalLinkage;
    auto *llvm_func = llvm::Function::Create(
        func_type, linkage,
        llvm::StringRef(name_builder.data(), name_builder.size()), _module.get());

    // Mark kernel functions as Vulkan compute entry points for the SPIR-V backend
    if (func.tag() == Function::Tag::KERNEL) {
        llvm_func->addFnAttr("hlsl.shader", "compute");
        // Set workgroup size via hlsl.numthreads
        auto block_size = func.block_size();
        auto numthreads_str = luisa::format("{},{},{}", block_size.x, block_size.y, block_size.z);
        std::string numthreads_std(numthreads_str.data(), numthreads_str.size());
        llvm_func->addFnAttr("hlsl.numthreads", numthreads_std);
    }

    opt->func_types[hash] = llvm_func;

    // Set argument names (only for callables with params)
    size_t arg_idx = 0;
    if (func.tag() != Function::Tag::KERNEL) {
        for (auto &arg : func.arguments()) {
            vstd::StringBuilder arg_name;
            GetVariableName(func, arg, arg_name);
            llvm_func->getArg(arg_idx)->setName(arg_name.data());
            arg_idx++;
        }
    }

    // Create entry basic block
    auto *entry_bb = llvm::BasicBlock::Create(*_context, "entry", llvm_func);
    _builder->SetInsertPoint(entry_bb);
    _current_function = llvm_func;

    // Allocate local variables at entry
    for (auto &v : func.local_variables()) {
        vstd::StringBuilder var_name;
        GetVariableName(func, v, var_name);
        auto *alloca = _builder->CreateAlloca(
            ToLLVMType(*v.type()), nullptr,
            var_name.data());
        opt->variables[v.uid()] = alloca;
    }

    // Allocate shared variables
    for (auto &v : func.shared_variables()) {
        vstd::StringBuilder var_name;
        GetVariableName(func, v, var_name);
        // Shared variables use addr space 3 (shared/local) — for now use addr space 0
        auto *alloca = _builder->CreateAlloca(
            ToLLVMType(*v.type()), nullptr,
            var_name.data());
        opt->variables[v.uid()] = alloca;
        opt->shared_variable_uids.insert(v.uid());
    }

    // For kernel entry points, create global variables for all arguments
    // (resources + cbuffer) and emit intrinsics for builtins.
    if (func.tag() == Function::Tag::KERNEL) {
        // --- Resource arguments: create GlobalVariable in addrspace(1) ---
        // The LLVM SPIR-V backend converts these to OpVariable with
        // CrossWorkgroup storage class and adds them to the entry point interface.
        for (auto &arg : func.arguments()) {
            if (arg.is_resource()) {
                vstd::StringBuilder var_name;
                GetVariableName(func, arg, var_name);
                auto *global = new llvm::GlobalVariable(
                    *_module, ToLLVMType(*arg.type()), false,
                    llvm::GlobalValue::ExternalLinkage, nullptr,
                    llvm::StringRef(var_name.data(), var_name.size()),
                    nullptr, llvm::GlobalVariable::NotThreadLocal,
                    1u);// addrspace(1) = CrossWorkgroup/StorageBuffer
                opt->variables[arg.uid()] = global;
            } else if (arg.tag() == Variable::Tag::REFERENCE) {
                // Reference arguments: create GlobalVariable in addrspace(1)
                vstd::StringBuilder var_name;
                GetVariableName(func, arg, var_name);
                auto *global = new llvm::GlobalVariable(
                    *_module, ToLLVMType(*arg.type()), false,
                    llvm::GlobalValue::ExternalLinkage, nullptr,
                    llvm::StringRef(var_name.data(), var_name.size()),
                    nullptr, llvm::GlobalVariable::NotThreadLocal,
                    1u);
                opt->variables[arg.uid()] = global;
            }
        }

        // --- Non-resource (arithmetic) arguments: cbuffer _Global in addrspace(1) ---
        luisa::vector<const Variable *> value_args;
        for (auto &arg : func.arguments()) {
            if (!arg.is_resource() && arg.tag() != Variable::Tag::REFERENCE) {
                value_args.push_back(&arg);
            }
        }
        if (!value_args.empty()) {
            luisa::vector<llvm::Type *> arg_llvm_types;
            for (auto *arg : value_args) {
                arg_llvm_types.push_back(ToLLVMType(*arg->type()));
            }
            auto *args_struct_type = llvm::StructType::create(
                *_context, arg_llvm_types, "_Args");
            auto *args_global = new llvm::GlobalVariable(
                *_module, args_struct_type, false,
                llvm::GlobalValue::ExternalLinkage, nullptr,
                "_Global", nullptr,
                llvm::GlobalVariable::NotThreadLocal,
                1u);// addrspace(1) = CrossWorkgroup/StorageBuffer

            for (size_t i = 0; i < value_args.size(); i++) {
                auto *gep = _builder->CreateStructGEP(
                    args_struct_type, args_global,
                    static_cast<unsigned>(i));
                auto *loaded = _builder->CreateLoad(
                    arg_llvm_types[i], gep);
                auto *alloca = _builder->CreateAlloca(
                    arg_llvm_types[i], nullptr);
                _builder->CreateStore(loaded, alloca);
                opt->variables[value_args[i]->uid()] = alloca;
            }
        }

        // --- Builtin variables: use LLVM SPIR-V intrinsics ---
        // @llvm.spv.thread.id         → GlobalInvocationId (SV_DispatchThreadID)
        // @llvm.spv.thread.id.in.group → LocalInvocationId (SV_GroupThreadId)
        // @llvm.spv.group.id          → WorkgroupId (SV_GroupId)
        for (auto &bv : func.builtin_variables()) {
            llvm::Intrinsic::ID intrinsic_id;
            switch (bv.tag()) {
                case Variable::Tag::DISPATCH_ID:
                    intrinsic_id = llvm::Intrinsic::spv_thread_id;
                    break;
                case Variable::Tag::THREAD_ID:
                    intrinsic_id = llvm::Intrinsic::spv_thread_id_in_group;
                    break;
                case Variable::Tag::BLOCK_ID:
                    intrinsic_id = llvm::Intrinsic::spv_group_id;
                    break;
                default:
                    // Other builtins (DISPATCH_SIZE, KERNEL_ID, etc.)
                    // fall back to an alloca with undef
                    {
                        auto *alloca = _builder->CreateAlloca(
                            ToLLVMType(*bv.type()), nullptr);
                        _builder->CreateStore(
                            llvm::UndefValue::get(ToLLVMType(*bv.type())), alloca);
                        opt->variables[bv.uid()] = alloca;
                    }
                    continue;
            }
            // Declare and call the intrinsic
            auto *intr_type = llvm::FunctionType::get(
                ToLLVMType(*bv.type()), {_builder->getInt32Ty()}, false);
            auto *intr = llvm::Intrinsic::getDeclarationIfExists(
                _module.get(), intrinsic_id, {ToLLVMType(*bv.type())});
            if (intr) {
                auto *call = _builder->CreateCall(intr, {_builder->getInt32(0)});
                auto *alloca = _builder->CreateAlloca(
                    ToLLVMType(*bv.type()), nullptr);
                _builder->CreateStore(call, alloca);
                opt->variables[bv.uid()] = alloca;
            } else {
                // Intrinsic not available, use undef
                auto *alloca = _builder->CreateAlloca(
                    ToLLVMType(*bv.type()), nullptr);
                _builder->CreateStore(
                    llvm::UndefValue::get(ToLLVMType(*bv.type())), alloca);
                opt->variables[bv.uid()] = alloca;
            }
        }
    }

    // Visit the function body
    auto *body = func.body();
    if (body) {
        LLVMStateVisitor visitor(func, *this);
        visitor.VisitFunction(func);
    }

    // If no terminator, add ret void or unreachable
    if (!terminator_or_null(_builder->GetInsertBlock())) {
        if (ret_type->isVoidTy()) {
            _builder->CreateRetVoid();
        } else {
            _builder->CreateRet(llvm::UndefValue::get(ret_type));
        }
    }

    _current_function = saved_function;

    // Verify the function
    if (llvm::verifyFunction(*llvm_func, &llvm::errs())) {
        LUISA_WARNING("LLVM function verification failed for: {}", name_builder.data());
    }

    // Restore IRBuilder insertion point and variable map for the caller
    _builder->restoreIP(saved_ip);
    opt->variables = std::move(saved_variables);

    return llvm_func;
}

llvm::Function *LLVMCodegenUtility::GetOrDeclareFunction(Function func) {
    auto hash = func.hash();
    auto it = opt->func_types.find(hash);
    if (it != opt->func_types.end()) {
        return it->second;
    }
    return CodegenFunction(func);
}

llvm::Function *LLVMCodegenUtility::CodegenKernelEntry(Function kernel) {
    // For a kernel, we wrap the generated function with entry-point handling.
    // Currently this delegates to CodegenFunction; future work adds
    // builtin-variable loading and dispatch metadata.
    return CodegenFunction(kernel);
}

vstd::StringBuilder LLVMCodegenUtility::GetNewTempVarName() {
    vstd::StringBuilder sb;
    sb << "_tmp" << vstd::to_string(opt->temp_count++);
    return sb;
}

luisa::string LLVMCodegenUtility::ToString() const {
    std::string result;
    llvm::raw_string_ostream os(result);
    _module->print(os, nullptr);
    return luisa::string(result.data(), result.size());
}

void LLVMCodegenUtility::WriteBitcodeToFile(luisa::string_view path) const {
    std::error_code ec;
    std::string path_str(path.data(), path.size());
    llvm::raw_fd_ostream os(path_str, ec);
    llvm::WriteBitcodeToFile(*_module, os);
}

// ============================================================================
// SPIR-V Emission
// ============================================================================

void LLVMCodegenUtility::InitializeSPIRVModule() {
    // Ensure LLVM SPIR-V target is registered (prevents dead-stripping on Windows)
    InitializeLLVMSPIRVTarget();

    // Set target triple for SPIR-V. We use spirv64 (physical 64-bit):
    // - spirv64: 64-bit pointers (spirv32 crashes in SPIRVLegalizePointerCast)
    // - Physical addressing emits OpCapability Addresses and OpPtrAccessChain
    //   which are disallowed by Vulkan; fixed in post-processing via
    //   strip_addresses_capability().
    // - Entry point functions are emitted with parameters; also fixed in
    //   post-processing via fix_entry_point_parameters().
    _module->setTargetTriple(llvm::Triple("spirv64-unknown-vulkan1.2"));

    // Look up the SPIR-V target
    std::string error;
    auto *target = llvm::TargetRegistry::lookupTarget(llvm::Triple("spirv64"), error);
    if (!target) {
        LUISA_ERROR_WITH_LOCATION("LLVM SPIRV target not found: {}", error);
    }

    llvm::TargetOptions opt;
    _target_machine.reset(target->createTargetMachine(
        llvm::Triple("spirv64-unknown-vulkan1.2"), "generic",
        "", opt, std::optional<llvm::Reloc::Model>(llvm::Reloc::PIC_),
        std::optional<llvm::CodeModel::Model>(llvm::CodeModel::Small),
        llvm::CodeGenOptLevel::Default, false));

    if (!_target_machine) {
        LUISA_ERROR_WITH_LOCATION(
            "Failed to create LLVM SPIRV target machine.");
    }

    // Use the target machine's data layout to avoid mismatches
    _module->setDataLayout(_target_machine->createDataLayout());
}

/// Recursively scalarize an aggregate load: produce element-wise loads
/// and assemble them back into an aggregate value via insertvalue.
static llvm::Value *BuildAggregateLoad(llvm::IRBuilder<> &B, llvm::Type *Ty, llvm::Value *Ptr) {
    if (auto *ST = llvm::dyn_cast<llvm::StructType>(Ty)) {
        llvm::Value *Agg = llvm::UndefValue::get(ST);
        for (unsigned i = 0; i < ST->getNumElements(); ++i) {
            auto *ElemPtr = B.CreateStructGEP(ST, Ptr, i);
            auto *ElemVal = BuildAggregateLoad(B, ST->getElementType(i), ElemPtr);
            Agg = B.CreateInsertValue(Agg, ElemVal, i);
        }
        return Agg;
    }
    if (auto *AT = llvm::dyn_cast<llvm::ArrayType>(Ty)) {
        llvm::Value *Agg = llvm::UndefValue::get(AT);
        auto *ElemTy = AT->getElementType();
        for (unsigned i = 0; i < AT->getNumElements(); ++i) {
            auto *ElemPtr = B.CreateInBoundsGEP(AT, Ptr, {B.getInt32(0), B.getInt32(i)});
            auto *ElemVal = BuildAggregateLoad(B, ElemTy, ElemPtr);
            Agg = B.CreateInsertValue(Agg, ElemVal, i);
        }
        return Agg;
    }
    // Scalar or vector – direct load.
    return B.CreateLoad(Ty, Ptr);
}

/// Recursively scalarize an aggregate store: extract each element and store
/// it individually via GEP.
static void StoreAggregateValue(llvm::IRBuilder<> &B, llvm::Type *Ty, llvm::Value *Ptr, llvm::Value *Val) {
    if (auto *ST = llvm::dyn_cast<llvm::StructType>(Ty)) {
        for (unsigned i = 0; i < ST->getNumElements(); ++i) {
            auto *ElemPtr = B.CreateStructGEP(ST, Ptr, i);
            auto *ElemVal = B.CreateExtractValue(Val, i);
            StoreAggregateValue(B, ST->getElementType(i), ElemPtr, ElemVal);
        }
        return;
    }
    if (auto *AT = llvm::dyn_cast<llvm::ArrayType>(Ty)) {
        auto *ElemTy = AT->getElementType();
        for (unsigned i = 0; i < AT->getNumElements(); ++i) {
            auto *ElemPtr = B.CreateInBoundsGEP(AT, Ptr, {B.getInt32(0), B.getInt32(i)});
            auto *ElemVal = B.CreateExtractValue(Val, i);
            StoreAggregateValue(B, ElemTy, ElemPtr, ElemVal);
        }
        return;
    }
    // Scalar or vector – direct store.
    B.CreateStore(Val, Ptr);
}

/// Replace all aggregate loads/stores with per-element operations.
/// The LLVM SPIR-V Vulkan backend cannot legalize aggregate memory ops
/// (structs, arrays) and asserts or crashes in SPIRVLegalizePointerCast.
static void ScalarizeAggregateMemOps(llvm::Module &M) {
    llvm::SmallVector<llvm::LoadInst *, 16> loads;
    llvm::SmallVector<llvm::StoreInst *, 16> stores;

    for (auto &F : M) {
        for (auto &BB : F) {
            for (auto &I : BB) {
                if (auto *LI = llvm::dyn_cast<llvm::LoadInst>(&I)) {
                    if (LI->getType()->isAggregateType())
                        loads.push_back(LI);
                } else if (auto *SI = llvm::dyn_cast<llvm::StoreInst>(&I)) {
                    if (SI->getValueOperand()->getType()->isAggregateType())
                        stores.push_back(SI);
                }
            }
        }
    }

    for (auto *LI : loads) {
        llvm::IRBuilder<> B(LI);
        auto *Val = BuildAggregateLoad(B, LI->getType(), LI->getPointerOperand());
        LI->replaceAllUsesWith(Val);
        LI->eraseFromParent();
    }

    for (auto *SI : stores) {
        llvm::IRBuilder<> B(SI);
        StoreAggregateValue(B, SI->getValueOperand()->getType(),
                            SI->getPointerOperand(), SI->getValueOperand());
        SI->eraseFromParent();
    }
}

/// Lower functions with aggregate return types to use void + out-parameter.
/// This must run before LLVM's SPIRVPrepareFunctions pass, which would
/// otherwise mutate the return type to i32 and leave broken IR (mismatched
/// stores) that crashes SPIRVLegalizePointerCast.
static void LowerAggregateReturns(llvm::Module &M) {
    llvm::DenseMap<llvm::Function *, llvm::Function *> func_map;
    llvm::SmallVector<llvm::Function *, 4> funcs_to_lower;

    for (auto &F : M) {
        if (F.isDeclaration() || F.isIntrinsic())
            continue;
        if (F.getReturnType()->isAggregateType())
            funcs_to_lower.push_back(&F);
    }

    if (funcs_to_lower.empty())
        return;

    // First pass: create new function shells
    for (auto *F : funcs_to_lower) {
        llvm::IRBuilder<> B(F->getContext());
        llvm::SmallVector<llvm::Type *, 8> new_arg_types;
        for (auto &Arg : F->args())
            new_arg_types.push_back(Arg.getType());
        new_arg_types.push_back(B.getPtrTy());

        auto *new_func_type = llvm::FunctionType::get(B.getVoidTy(), new_arg_types, false);
        auto *new_func = llvm::Function::Create(
            new_func_type, F->getLinkage(), F->getAddressSpace(),
            F->getName(), &M);
        new_func->setCallingConv(F->getCallingConv());
        new_func->copyAttributesFrom(F);
        new_func->copyMetadata(F, 0);
        func_map[F] = new_func;
    }

    // Second pass: clone bodies and replace returns
    for (auto *F : funcs_to_lower) {
        auto *new_func = func_map[F];
        llvm::IRBuilder<> B(F->getContext());

        llvm::ValueToValueMapTy vmap;
        auto new_arg_it = new_func->arg_begin();
        for (auto &arg : F->args()) {
            new_arg_it->setName(arg.getName());
            vmap[&arg] = &(*new_arg_it);
            ++new_arg_it;
        }
        new_arg_it->setName("ret_ptr");
        auto *ret_ptr_arg = &(*new_arg_it);

        llvm::SmallVector<llvm::ReturnInst *, 4> returns;
        llvm::CloneFunctionInto(new_func, F, vmap,
                                llvm::CloneFunctionChangeType::LocalChangesOnly,
                                returns);

        // Update calls inside cloned body to already-lowered callees
        for (auto &BB : *new_func) {
            for (auto &I : BB) {
                if (auto *CI = llvm::dyn_cast<llvm::CallInst>(&I)) {
                    auto *called = CI->getCalledFunction();
                    if (called && func_map.count(called))
                        CI->setCalledFunction(func_map[called]);
                }
            }
        }

        // Replace ret instructions with store + ret void
        for (auto *RI : returns) {
            B.SetInsertPoint(RI);
            llvm::Value *ret_val = RI->getReturnValue();
            if (ret_val)
                B.CreateStore(ret_val, ret_ptr_arg);
            B.CreateRetVoid();
            RI->eraseFromParent();
        }
    }

    // Third pass: update call sites
    for (auto *F : funcs_to_lower) {
        auto *new_func = func_map[F];
        llvm::IRBuilder<> B(F->getContext());
        llvm::SmallVector<llvm::CallInst *, 4> calls;

        for (auto *U : F->users()) {
            if (auto *CI = llvm::dyn_cast<llvm::CallInst>(U)) {
                if (CI->getCalledFunction() == F)
                    calls.push_back(CI);
            }
        }

        for (auto *CI : calls) {
            B.SetInsertPoint(CI);
            llvm::SmallVector<llvm::Value *, 8> new_args;
            for (auto &arg : CI->args())
                new_args.push_back(arg);

            llvm::Value *store_dst = nullptr;
            if (CI->hasOneUse()) {
                auto *SI = llvm::dyn_cast<llvm::StoreInst>(*CI->user_begin());
                if (SI) {
                    store_dst = SI->getPointerOperand();
                    if (!llvm::isa<llvm::AllocaInst>(store_dst))
                        store_dst = nullptr;
                }
            }

            if (store_dst) {
                // Pass the alloca destination directly as the out-param
                new_args.push_back(store_dst);
                auto *new_call = B.CreateCall(new_func, new_args);
                new_call->setCallingConv(CI->getCallingConv());
                auto *SI = llvm::cast<llvm::StoreInst>(*CI->user_begin());
                SI->eraseFromParent();
                CI->eraseFromParent();
            } else {
                // Allocate a temporary, pass it, and load from it
                auto *tmp = B.CreateAlloca(F->getReturnType(), nullptr, "ret_tmp");
                new_args.push_back(tmp);
                auto *new_call = B.CreateCall(new_func, new_args);
                new_call->setCallingConv(CI->getCallingConv());
                auto *load = B.CreateLoad(F->getReturnType(), tmp);
                CI->replaceAllUsesWith(load);
                CI->eraseFromParent();
            }
        }

        F->eraseFromParent();
    }
}

luisa::vector<uint32_t> LLVMCodegenUtility::EmitSPIRV() {
    auto &module = *_module;
    auto &target = *_target_machine;

    // Scalarize aggregate memory operations before the LLVM SPIR-V backend
    // sees them. The backend cannot legalize aggregate loads/stores.
    ScalarizeAggregateMemOps(module);
    // Lower aggregate returns before the LLVM SPIR-V backend sees them.
    // SPIRVPrepareFunctions would otherwise break the IR.
    LowerAggregateReturns(module);
    // LowerAggregateReturns may introduce new aggregate loads/stores
    // (e.g., load from ret_tmp, store to ret_ptr). Scalarize again.
    ScalarizeAggregateMemOps(module);

    // Use legacy pass manager to emit SPIR-V object (binary)
    llvm::legacy::PassManager pm;

    // Try to emit as object file (ELF-wrapped SPIR-V binary).
    // If that fails, emit as assembly text and convert.
    llvm::SmallVector<char, 0> buffer;
    llvm::raw_svector_ostream os(buffer);

    if (target.addPassesToEmitFile(
            pm, os, nullptr, llvm::CodeGenFileType::ObjectFile)) {
        LUISA_ERROR_WITH_LOCATION(
            "LLVM SPIRV target cannot emit object file.");
    }

    // Dump LLVM IR for debugging
    std::error_code ec;
    llvm::raw_fd_ostream ir_file("llvm_ir_debug.ll", ec);
    if (!ec) {
        module.print(ir_file, nullptr);
    }

    // Verify module before running passes
    std::string verify_err;
    llvm::raw_string_ostream verify_os(verify_err);
    if (llvm::verifyModule(module, &verify_os)) {
        LUISA_ERROR_WITH_LOCATION(
            "LLVM module verification failed: {}", verify_err);
    }

    pm.run(module);

    // The buffer should contain ELF-wrapped SPIR-V.
    // For now, copy the raw bytes as uint32_t words.
    // A proper implementation would extract the .spv section from the ELF.
    auto data = reinterpret_cast<const uint32_t *>(buffer.data());
    auto size = buffer.size() / sizeof(uint32_t);

    // Check for ELF magic: 0x7f 'E' 'L' 'F'
    luisa::vector<uint32_t> spv_bin;
    if (buffer.size() >= 4 &&
        static_cast<unsigned char>(buffer[0]) == 0x7f &&
        buffer[1] == 'E' && buffer[2] == 'L' && buffer[3] == 'F') {
        // ELF-wrapped: skip the ELF header and find the .spv section.
        // Minimal ELF parsing for SPIR-V extraction.
        // ELF header: e_shoff at offset 0x28 (64-bit) or 0x20 (32-bit).
        // For now, fall back to using spirv-tools assembly.
        LUISA_WARNING(
            "LLVM SPIRV backend emitted ELF-wrapped output; "
            "textual assembly fallback not yet implemented. "
            "Returning raw ELF buffer as fallback.");
        spv_bin.assign(data, data + size);
    } else {
        spv_bin.assign(data, data + size);
    }

    return spv_bin;
}

void LLVMCodegenUtility::GenerateProperties(
    Function kernel,
    LLVMCodegenResult::Properties &properties) {
    // Detect bindless usage from propagated builtin callables
    const auto &builtins = kernel.propagated_builtin_callables();
    bool use_buffer_bindless = [&]() noexcept -> bool {
        static constexpr CallOp ops[] = {
            CallOp::BINDLESS_BUFFER_READ,
            CallOp::BINDLESS_BUFFER_WRITE,
            CallOp::BINDLESS_BYTE_BUFFER_READ,
            CallOp::UNIFORM_BINDLESS_BUFFER_READ,
            CallOp::UNIFORM_BINDLESS_BUFFER_WRITE,
            CallOp::UNIFORM_BINDLESS_BYTE_BUFFER_READ,
            CallOp::TYPED_BINDLESS_BUFFER_READ,
            CallOp::TYPED_BINDLESS_BUFFER_WRITE,
            CallOp::TYPED_UNIFORM_BINDLESS_BUFFER_READ,
            CallOp::TYPED_UNIFORM_BINDLESS_BUFFER_WRITE,
        };
        for (auto op : ops) {
            if (builtins.test(op)) return true;
        }
        return false;
    }();
    bool use_tex2d_bindless = [&]() noexcept -> bool {
        static constexpr CallOp ops[] = {
            CallOp::BINDLESS_TEXTURE2D_SAMPLE,
            CallOp::BINDLESS_TEXTURE2D_SAMPLE_LEVEL,
            CallOp::BINDLESS_TEXTURE2D_READ,
            CallOp::BINDLESS_TEXTURE2D_SIZE,
            CallOp::UNIFORM_BINDLESS_TEXTURE2D_SAMPLE,
            CallOp::UNIFORM_BINDLESS_TEXTURE2D_SAMPLE_LEVEL,
            CallOp::UNIFORM_BINDLESS_TEXTURE2D_READ,
            CallOp::UNIFORM_BINDLESS_TEXTURE2D_SIZE,
            CallOp::TYPED_BINDLESS_TEXTURE2D_SAMPLE_SAMPLER,
            CallOp::TYPED_UNIFORM_BINDLESS_TEXTURE2D_SAMPLE_SAMPLER,
        };
        for (auto op : ops) {
            if (builtins.test(op)) return true;
        }
        return false;
    }();
    bool use_tex3d_bindless = [&]() noexcept -> bool {
        static constexpr CallOp ops[] = {
            CallOp::BINDLESS_TEXTURE3D_SAMPLE,
            CallOp::BINDLESS_TEXTURE3D_SAMPLE_LEVEL,
            CallOp::BINDLESS_TEXTURE3D_READ,
            CallOp::BINDLESS_TEXTURE3D_SIZE,
            CallOp::UNIFORM_BINDLESS_TEXTURE3D_SAMPLE,
            CallOp::UNIFORM_BINDLESS_TEXTURE3D_SAMPLE_LEVEL,
            CallOp::UNIFORM_BINDLESS_TEXTURE3D_READ,
            CallOp::UNIFORM_BINDLESS_TEXTURE3D_SIZE,
            CallOp::TYPED_BINDLESS_TEXTURE3D_SAMPLE_SAMPLER,
            CallOp::TYPED_UNIFORM_BINDLESS_TEXTURE3D_SAMPLE_SAMPLER,
        };
        for (auto op : ops) {
            if (builtins.test(op)) return true;
        }
        return false;
    }();

    // Write bindless flags to the stack data for later collection in CompileSPIRV
    opt->useBufferBindless = use_buffer_bindless;
    opt->useTex2DBindless = use_tex2d_bindless;
    opt->useTex3DBindless = use_tex3d_bindless;

    luisa::vector<LLVMVulkanBindingArgument> arguments;
    arguments.reserve(kernel.arguments().size());
    for (auto &&argument : kernel.arguments()) {
        auto type = argument.type();
        arguments.emplace_back(LLVMVulkanBindingArgument{
            .tag = type->tag(),
            .usage = kernel.variable_usage(argument.uid()),
            .indirect_dispatch_buffer =
                type->is_custom() &&
                type->description() == "LC_IndirectDispatchBuffer"sv});
    }
    auto request = LLVMVulkanBindingPropertyRequest{
        .arguments = arguments,
        .use_buffer_bindless = use_buffer_bindless,
        .use_tex2d_bindless = use_tex2d_bindless,
        .use_tex3d_bindless = use_tex3d_bindless,
        .printer_count =
            static_cast<uint32_t>(kernel.requires_printing())};
    auto support = validate_llvm_vulkan_resource_model(request);
    LUISA_ASSERT(
        support,
        "AST-to-LLVM Vulkan SPIR-V cannot lower this shader's runtime "
        "resource interface: {}. Use the XIR-to-SPIR-V or HLSL-to-SPIR-V "
        "path for this shader.",
        llvm_vulkan_resource_model_error_name(support.error));
    auto plan = plan_llvm_vulkan_binding_properties(request);
    properties = std::move(plan.properties);
}

// ============================================================================
// SPIR-V Post-Processing (fix LLVM physical addressing issues)
// ============================================================================

/// Strip OpCapability Addresses and convert Op*PtrAccessChain to Op*AccessChain.
/// The LLVM spirv64 target emits physical addressing instructions that require
/// the Addresses capability. We remove the capability and convert the
/// instructions to their logical equivalents.
static void strip_addresses_capability(luisa::vector<uint32_t> &spv_bin) {
    if (spv_bin.size() < 5) {
        LUISA_WARNING("SPIR-V binary too small ({} words), stripping skipped.", spv_bin.size());
        return;
    }

    // Verify SPIR-V magic: 0x07230203
    if (spv_bin[0] != 0x07230203u) {
        LUISA_WARNING("SPIR-V magic mismatch (0x{:08x}), stripping skipped.", spv_bin[0]);
        return;
    }

    // Log SPIR-V version from header word 1
    uint32_t version = spv_bin[1];
    LUISA_INFO("SPIR-V version: {}.{}.{}", (version >> 16) & 0xFF,
               (version >> 8) & 0xFF, version & 0xFF);

    constexpr uint32_t kOpCapability = 17;
    constexpr uint32_t kAddressesCap = 4;// SpvCapabilityAddresses = 4
    constexpr uint32_t kLinkageCap = 5;  // SpvCapabilityLinkage = 5
    constexpr uint32_t kOpDecorate = 71;
    constexpr uint32_t kLinkageAttributesDec = 69;// SpvDecorationLinkageAttributes = 69
    constexpr uint32_t kOpPtrAccessChain = 67;
    constexpr uint32_t kOpAccessChain = 65;
    constexpr uint32_t kOpInBoundsPtrAccessChain = 70;
    constexpr uint32_t kOpInBoundsAccessChain = 66;

    // Build new binary (cannot resize in-place due to shifting)
    luisa::vector<uint32_t> out;
    out.reserve(spv_bin.size());

    // Copy the 5-word header
    for (size_t h = 0; h < 5; ++h) out.push_back(spv_bin[h]);

    bool found_addresses = false;
    size_t i = 5;// Start after header
    while (i < spv_bin.size()) {
        uint32_t word = spv_bin[i];
        uint32_t word_count = word >> 16;
        uint32_t opcode = word & 0xFFFF;

        if (word_count == 0 || i + word_count > spv_bin.size()) {
            while (i < spv_bin.size()) out.push_back(spv_bin[i++]);
            break;
        }
        if (opcode == kOpCapability && word_count == 2 &&
            (spv_bin[i + 1] == kAddressesCap ||
             spv_bin[i + 1] == kLinkageCap)) {
            // Skip OpCapability Addresses/Linkage
            found_addresses = true;
            i += word_count;
        } else if (opcode == kOpDecorate && word_count >= 3 &&
                   spv_bin[i + 2] == kLinkageAttributesDec) {
            // Skip OpDecorate LinkageAttributes (requires Linkage capability)
            i += word_count;
            found_addresses = true;
        } else if ((opcode == kOpInBoundsPtrAccessChain ||
                    opcode == kOpPtrAccessChain) &&
                   word_count >= 4) {
            // Convert PtrAccessChain to AccessChain by removing the Element
            // operand (at index 4). The Element operand is the extra pointer
            // that distinguishes PtrAccessChain from AccessChain.
            uint32_t new_opcode = (opcode == kOpInBoundsPtrAccessChain) ? kOpInBoundsAccessChain : kOpAccessChain;
            uint32_t new_wc = word_count - 1;
            out.push_back((new_wc << 16) | new_opcode);
            // Copy words 1-3 (ResultType, Result, Base)
            out.push_back(spv_bin[i + 1]);
            out.push_back(spv_bin[i + 2]);
            out.push_back(spv_bin[i + 3]);
            // Skip word 4 (Element), copy remaining indexes
            for (size_t j = 5; j < word_count; ++j)
                out.push_back(spv_bin[i + j]);
            i += word_count;
        } else {
            // Copy as-is
            for (size_t j = 0; j < word_count; ++j)
                out.push_back(spv_bin[i + j]);
            i += word_count;
        }
    }

    if (found_addresses) {
        LUISA_INFO("Stripped OpCapability Addresses from SPIR-V binary.");
    } else {
        LUISA_INFO("OpCapability Addresses not found; no stripping needed.");
    }

    spv_bin = std::move(out);
}

/// Validate SPIR-V binary using spirv-tools with Vulkan 1.2 environment.
/// Mirrors lc::spirv::luisa_spirv_validate from the XIR path.
static void luisa_spirv_validate_post_llvm(luisa::span<const uint32_t> words, luisa::string_view stage) {
    spvtools::SpirvTools tools(SPV_ENV_VULKAN_1_2);
    luisa::string message;
    tools.SetMessageConsumer(
        [&message](spv_message_level_t level, const char *source,
                   const spv_position_t &position, const char *text) {
            auto level_name = [level]() noexcept {
                switch (level) {
                    case SPV_MSG_FATAL: return "fatal";
                    case SPV_MSG_INTERNAL_ERROR: return "internal";
                    case SPV_MSG_ERROR: return "error";
                    case SPV_MSG_WARNING: return "warning";
                    case SPV_MSG_INFO: return "info";
                    case SPV_MSG_DEBUG: return "debug";
                }
                return "unknown";
            }();
            message.append(luisa::format("{} [{}:{}:{}]: {}\n",
                                         level_name,
                                         source == nullptr ? "" : source,
                                         position.line,
                                         position.column,
                                         text == nullptr ? "" : text));
        });
    spvtools::ValidatorOptions options;
    if (!tools.Validate(words.data(), words.size(), options)) {
        LUISA_ERROR("LLVM SPIR-V validation failed at {} stage:\n{}", stage, message);
    }
}

LLVMCodegenResult LLVMCodegenUtility::CompileSPIRV(
    Function kernel,
    const ShaderOption &option) {

    LLVMCodegenResult result;
    // 1. Create utility and initialize SPIR-V module
    LLVMCodegenUtility util;
    util.InitializeSPIRVModule();

    // 2. Validate and plan the runtime interface before AST lowering. The
    // experimental LLVM path deliberately rejects resource families whose
    // descriptor-handle lowering is not implemented yet; doing this first
    // keeps those kernels away from partially implemented visitor stubs.
    util.GenerateProperties(kernel, result.properties);

    // 3. Codegen the kernel function into LLVM IR
    util.CodegenFunction(kernel);

    // 4. Collect bindless usage flags from stack data
    result.useTex2DBindless = util.opt->useTex2DBindless;
    result.useTex3DBindless = util.opt->useTex3DBindless;
    result.useBufferBindless = util.opt->useBufferBindless;

    // 5. Collect printer info
    result.printers = std::move(util.opt->printers);

    // 6. Emit SPIR-V binary via LLVM SPIRV target
    result.spv_bin = util.EmitSPIRV();

    // 7. Strip Addresses/Linkage capabilities and convert PtrAccessChain
    strip_addresses_capability(result.spv_bin);

    // 8. Validate the Vulkan SPIR-V binary before returning it
    luisa_spirv_validate_post_llvm(result.spv_bin, "post-llvm");

    return result;
}

}// namespace lc::llvm_codegen
