#include "llvm_codegen_utility.h"
#include "llvm_codegen_stack_data.h"
#include "llvm_state_visitor.h"

#include <llvm/IR/Constants.h>
#include <llvm/IR/GlobalVariable.h>
#include <llvm/IR/Verifier.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Bitcode/BitcodeWriter.h>
#include <llvm/ADT/SmallVector.h>

#include "../hlsl/shader_property.h"
#include "../hlsl/hlsl_codegen.h"

#include <luisa/ast/variable.h>
#include <luisa/ast/usage.h>
#include <luisa/runtime/rhi/resource.h>

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
            return _builder->getPtrTy(0); // opaque pointer in addr space 0

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
            return _builder->getInt32Ty(); // unreachable
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
            str << "buffer"; break;
        case Type::Tag::TEXTURE:
            str << "texture"; break;
        case Type::Tag::BINDLESS_ARRAY:
            str << "bindless_array"; break;
        case Type::Tag::ACCEL:
            str << "accel"; break;
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
                if (bw == 32) v = *reinterpret_cast<uint32_t const *>(sub_data);
                else if (bw == 64) v = *reinterpret_cast<uint64_t const *>(sub_data);
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
        *_module, type, true, // isConstant
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

    // Build function type
    auto *ret_type = ToLLVMType(*func.return_type());
    luisa::vector<llvm::Type *> param_types;
    for (auto &arg : func.arguments()) {
        param_types.push_back(ToLLVMType(*arg.type()));
    }
    // Add builtin variable parameters for kernel
    if (func.tag() == Function::Tag::KERNEL) {
        for (auto &bv : func.builtin_variables()) {
            param_types.push_back(ToLLVMType(*bv.type()));
        }
    }

    auto *func_type = llvm::FunctionType::get(ret_type, param_types, false);

    // Create function name
    vstd::StringBuilder name_builder;
    GetFunctionName(func, name_builder);

    auto *llvm_func = llvm::Function::Create(
        func_type, llvm::Function::ExternalLinkage,
        llvm::StringRef(name_builder.data(), name_builder.size()), _module.get());

    opt->func_types[hash] = llvm_func;

    // Set argument names
    size_t arg_idx = 0;
    for (auto &arg : func.arguments()) {
        vstd::StringBuilder arg_name;
        GetVariableName(func, arg, arg_name);
        llvm_func->getArg(arg_idx)->setName(arg_name.data());
        arg_idx++;
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

    // Map arguments
    arg_idx = 0;
    for (auto &arg : func.arguments()) {
        auto *param = llvm_func->getArg(static_cast<unsigned>(arg_idx));
        if (arg.tag() == Variable::Tag::REFERENCE) {
            // Store reference parameter into alloca
            std::string ref_name = "_R" + std::to_string(arg.uid()) + "_ptr";
            auto *alloca = _builder->CreateAlloca(
                param->getType(), nullptr, ref_name);
            _builder->CreateStore(param, alloca);
            opt->variables[arg.uid()] = alloca;
        } else if (arg.is_resource()) {
            // Resources are passed as opaque pointers
            opt->variables[arg.uid()] = param;
        } else {
            // Regular arguments: store into alloca
            auto *alloca = _builder->CreateAlloca(
                param->getType(), nullptr);
            _builder->CreateStore(param, alloca);
            opt->variables[arg.uid()] = alloca;
        }
        arg_idx++;
    }

    // Handle builtin variables for kernels
    if (func.tag() == Function::Tag::KERNEL) {
        for (auto &bv : func.builtin_variables()) {
            auto *param = llvm_func->getArg(static_cast<unsigned>(arg_idx));
            auto *alloca = _builder->CreateAlloca(
                param->getType(), nullptr);
            _builder->CreateStore(param, alloca);
            opt->variables[bv.uid()] = alloca;
            arg_idx++;
        }
    }

    // Visit the function body
    auto *body = func.body();
    if (body) {
        LLVMStateVisitor visitor(func, *this);
        visitor.VisitFunction(func);
    }

    // If no terminator, add ret void or unreachable
    if (!_builder->GetInsertBlock()->getTerminator()) {
        if (ret_type->isVoidTy()) {
            _builder->CreateRetVoid();
        } else {
            _builder->CreateRet(llvm::UndefValue::get(ret_type));
        }
    }

    _current_function = nullptr;

    // Verify the function
    if (llvm::verifyFunction(*llvm_func, &llvm::errs())) {
        LUISA_WARNING("LLVM function verification failed for: {}", name_builder.data());
    }

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
    // Set target triple and data layout for SPIR-V
    _module->setTargetTriple(llvm::Triple("spirv64-unknown-unknown"));
    _module->setDataLayout(
        "e-i64:64-v16:16-v24:32-v32:32-v48:64-"
        "v96:128-v192:256-v256:256-v512:512-v1024:1024");

    // Look up the SPIR-V target
    std::string error;
    auto *target = llvm::TargetRegistry::lookupTarget("spirv64", error);
    if (!target) {
        LUISA_ERROR_WITH_LOCATION("LLVM SPIRV target not found: {}", error);
    }

    llvm::TargetOptions opt;
    _target_machine.reset(target->createTargetMachine(
        "spirv64-unknown-unknown", "generic",
        "", opt, llvm::Reloc::PIC_));

    if (!_target_machine) {
        LUISA_ERROR_WITH_LOCATION(
            "Failed to create LLVM SPIRV target machine.");
    }
}

luisa::vector<uint32_t> LLVMCodegenUtility::EmitSPIRV() {
    auto &module = *_module;
    auto &target = *_target_machine;

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

    // Mirror the XIR SpirvCodegenEntry::generate_binding() logic.
    // Walk kernel arguments (Variables) and generate property entries.

    // Detect writable usage from the kernel's variable usage map
    auto is_writable = [&](const Variable &v) {
        return (static_cast<uint>(kernel.variable_usage(v.uid())) &
                static_cast<uint>(Usage::WRITE)) != 0;
    };

    // Detect cbuffer non-empty: any argument that is not a resource or builtin
    bool cbuffer_non_empty = false;
    for (auto &&arg : kernel.arguments()) {
        auto tag = arg.tag();
        switch (tag) {
            case Variable::Tag::BUFFER:
            case Variable::Tag::TEXTURE:
            case Variable::Tag::BINDLESS_ARRAY:
            case Variable::Tag::ACCEL:
            case Variable::Tag::THREAD_ID:
            case Variable::Tag::BLOCK_ID:
            case Variable::Tag::DISPATCH_ID:
            case Variable::Tag::DISPATCH_SIZE:
            case Variable::Tag::KERNEL_ID:
            case Variable::Tag::WARP_LANE_COUNT:
            case Variable::Tag::WARP_LANE_ID:
                break;
            default:
                cbuffer_non_empty = true;
                break;
        }
    }

    // Detect bindless usage from propagated builtin callables
    const auto &builtins = kernel.propagated_builtin_callables();
    bool use_buffer_bindless = [&]() noexcept -> bool {
        static constexpr CallOp ops[] = {
            CallOp::BINDLESS_BUFFER_READ, CallOp::BINDLESS_BUFFER_WRITE,
            CallOp::BINDLESS_BYTE_BUFFER_READ,
            CallOp::UNIFORM_BINDLESS_BUFFER_READ, CallOp::UNIFORM_BINDLESS_BUFFER_WRITE,
            CallOp::UNIFORM_BINDLESS_BYTE_BUFFER_READ,
            CallOp::TYPED_BINDLESS_BUFFER_READ, CallOp::TYPED_BINDLESS_BUFFER_WRITE,
            CallOp::TYPED_UNIFORM_BINDLESS_BUFFER_READ, CallOp::TYPED_UNIFORM_BINDLESS_BUFFER_WRITE,
        };
        for (auto op : ops) { if (builtins.test(op)) return true; }
        return false;
    }();
    bool use_tex2d_bindless = [&]() noexcept -> bool {
        static constexpr CallOp ops[] = {
            CallOp::BINDLESS_TEXTURE2D_SAMPLE, CallOp::BINDLESS_TEXTURE2D_SAMPLE_LEVEL,
            CallOp::BINDLESS_TEXTURE2D_READ, CallOp::BINDLESS_TEXTURE2D_SIZE,
            CallOp::UNIFORM_BINDLESS_TEXTURE2D_SAMPLE, CallOp::UNIFORM_BINDLESS_TEXTURE2D_SAMPLE_LEVEL,
            CallOp::UNIFORM_BINDLESS_TEXTURE2D_READ, CallOp::UNIFORM_BINDLESS_TEXTURE2D_SIZE,
            CallOp::TYPED_BINDLESS_TEXTURE2D_SAMPLE_SAMPLER,
            CallOp::TYPED_UNIFORM_BINDLESS_TEXTURE2D_SAMPLE_SAMPLER,
        };
        for (auto op : ops) { if (builtins.test(op)) return true; }
        return false;
    }();
    bool use_tex3d_bindless = [&]() noexcept -> bool {
        static constexpr CallOp ops[] = {
            CallOp::BINDLESS_TEXTURE3D_SAMPLE, CallOp::BINDLESS_TEXTURE3D_SAMPLE_LEVEL,
            CallOp::BINDLESS_TEXTURE3D_READ, CallOp::BINDLESS_TEXTURE3D_SIZE,
            CallOp::UNIFORM_BINDLESS_TEXTURE3D_SAMPLE, CallOp::UNIFORM_BINDLESS_TEXTURE3D_SAMPLE_LEVEL,
            CallOp::UNIFORM_BINDLESS_TEXTURE3D_READ, CallOp::UNIFORM_BINDLESS_TEXTURE3D_SIZE,
            CallOp::TYPED_BINDLESS_TEXTURE3D_SAMPLE_SAMPLER,
            CallOp::TYPED_UNIFORM_BINDLESS_TEXTURE3D_SAMPLE_SAMPLER,
        };
        for (auto op : ops) { if (builtins.test(op)) return true; }
        return false;
    }();

    // Write bindless flags to the stack data for later collection in CompileSPIRV
    opt->useBufferBindless = use_buffer_bindless;
    opt->useTex2DBindless = use_tex2d_bindless;
    opt->useTex3DBindless = use_tex3d_bindless;

    // CBuffer (global argument buffer) — fixed at reg=0 if non-empty
    uint reg_count = cbuffer_non_empty ? 1u : 0u;
    if (cbuffer_non_empty) {
        properties.emplace_back(hlsl::Property{
            hlsl::ShaderVariableType::StructuredBuffer, 0u, 0u, 1u});
    }

    // Kernel resource arguments
    for (auto &&arg : kernel.arguments()) {
        hlsl::Property prop;
        switch (arg.type()->tag()) {
            case Type::Tag::BUFFER:
                if (is_writable(arg)) {
                    prop.type = hlsl::ShaderVariableType::RWStructuredBuffer;
                } else {
                    prop.type = hlsl::ShaderVariableType::StructuredBuffer;
                }
                break;
            case Type::Tag::TEXTURE:
                if (is_writable(arg)) {
                    prop.type = hlsl::ShaderVariableType::UAVTextureHeap;
                } else {
                    prop.type = hlsl::ShaderVariableType::SRVTextureHeap;
                }
                break;
            case Type::Tag::BINDLESS_ARRAY:
                prop.type = hlsl::ShaderVariableType::StructuredBuffer;
                break;
            case Type::Tag::ACCEL:
                prop.type = hlsl::ShaderVariableType::SPIRVAccel;
                break;
            default:
                continue; // skip non-resource bindings
        }
        prop.space_index = 0u;
        prop.register_index = reg_count++;
        prop.array_size = 1u;
        properties.push_back(prop);
    }
}

LLVMCodegenResult LLVMCodegenUtility::CompileSPIRV(
    Function kernel,
    const ShaderOption &option) {

    LLVMCodegenResult result;

    // 1. Create utility and initialize SPIR-V module
    LLVMCodegenUtility util;
    util.InitializeSPIRVModule();

    // 2. Codegen the kernel function into LLVM IR
    util.CodegenFunction(kernel);

    // 3. Generate binding properties from kernel arguments
    util.GenerateProperties(kernel, result.properties);

    // 4. Collect bindless usage flags from stack data
    result.useTex2DBindless = util.opt->useTex2DBindless;
    result.useTex3DBindless = util.opt->useTex3DBindless;
    result.useBufferBindless = util.opt->useBufferBindless;

    // 5. Collect printer info
    result.printers = std::move(util.opt->printers);

    // 6. Collect constant UBO data
    result.constant_ubo_data = std::move(util.opt->constant_ubo_data);

    // 7. Emit SPIR-V binary via LLVM SPIRV target
    result.spv_bin = util.EmitSPIRV();

    // 8. Compute type MD5 for caching
    result.typeMD5 = hlsl::CodegenUtility::GetTypeMD5(kernel);

    return result;
}

} // namespace lc::llvm_codegen
