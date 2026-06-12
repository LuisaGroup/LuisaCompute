#pragma once

#include <luisa/vstl/common.h>
#include <luisa/vstl/functional.h>
#include <luisa/vstl/string_builder.h>
#include <luisa/ast/function.h>
#include <luisa/ast/expression.h>
#include <luisa/ast/statement.h>
#include <luisa/core/stl/unordered_map.h>
#include <luisa/core/stl/vector.h>
#include <luisa/core/logging.h>
#include <luisa/runtime/rhi/resource.h>

#include "llvm_codegen_result.h"

#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Type.h>
#include <llvm/IR/Value.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/MC/TargetRegistry.h>
#include <llvm/Support/TargetSelect.h>

#include <memory>
#include <string>

namespace lc::llvm_codegen {

/// Explicitly initialize LLVM's SPIR-V target (prevents linker dead-stripping).
void InitializeLLVMSPIRVTarget();

using namespace luisa;
using namespace luisa::compute;

struct LLVMCodegenStackData;
class LLVMStateVisitor;

/**
 * @brief LLVM IR code generation utility (mirrors hlsl::CodegenUtility).
 *
 * Owns the LLVM context, module, and IR builder. Provides type mapping,
 * variable/function naming, constant management, and code generation
 * entry points.
 */
class LLVMCodegenUtility {
public:
    vstd::unique_ptr<LLVMCodegenStackData> opt{};

    /// Main entry point for VK backend: codegen Function → SPIR-V result.
    /// Mirrors hlsl::CodegenUtility::Codegen() and
    /// lc::spirv::SpirvCodegenEntry::compile_spirv().
    [[nodiscard]] static LLVMCodegenResult CompileSPIRV(
        Function kernel,
        const ShaderOption &option);

private:
    std::unique_ptr<llvm::LLVMContext> _context;
    std::unique_ptr<llvm::Module> _module;
    std::unique_ptr<llvm::IRBuilder<>> _builder;

    // Current function being codegen'd (set during CodegenFunction)
    llvm::Function *_current_function{nullptr};

public:
    LLVMCodegenUtility();
    ~LLVMCodegenUtility();

    // --- Accessors ---
    llvm::LLVMContext &context() { return *_context; }
    llvm::Module &module() { return *_module; }
    llvm::IRBuilder<> &builder() { return *_builder; }
    llvm::Function *current_function() { return _current_function; }
    void set_current_function(llvm::Function *f) { _current_function = f; }

    // --- Type mapping ---
    /// Convert a Luisa Type* to an LLVM Type*
    [[nodiscard]] llvm::Type *ToLLVMType(Type const &type);

    /// Register a struct type (creates named LLVM struct type)
    [[nodiscard]] llvm::StructType *RegistStructType(Type const *type);

    /// Get the LLVM type name for a Luisa type
    void GetTypeName(Type const &type, vstd::StringBuilder &str);

    // --- Variable naming ---
    /// Generate HLSL-style variable name for a Function-scoped variable
    void GetVariableName(Function func, Variable const &v, vstd::StringBuilder &str);
    void GetVariableName(Function func, Variable::Tag tag, uint32_t id, vstd::StringBuilder &str);

    // --- Function naming ---
    void GetFunctionName(Function callable, vstd::StringBuilder &result);
    void GetFunctionName(CallExpr const *expr, vstd::StringBuilder &result, LLVMStateVisitor &visitor);

    // --- Constant data ---
    /// Create an LLVM Constant from a Luisa ConstantData
    [[nodiscard]] llvm::Constant *CreateConstant(ConstantData const &data, llvm::Type *type);

    /// Create a global variable for constant data
    [[nodiscard]] llvm::GlobalVariable *CreateConstantGlobal(ConstantData const &data, llvm::Type *type);

    // --- Function code generation ---
    /// Main entry point: codegen a Function → llvm::Function*
    [[nodiscard]] llvm::Function *CodegenFunction(Function func);

    /// Declare (or get previously declared) an external function
    [[nodiscard]] llvm::Function *GetOrDeclareFunction(Function func);

    /// Codegen a kernel entry point
    [[nodiscard]] llvm::Function *CodegenKernelEntry(Function kernel);

    // --- Temp variable name ---
    vstd::StringBuilder GetNewTempVarName();

    // --- Module output ---
    /// Serialize the LLVM module to a string (LLVM IR text format)
    [[nodiscard]] luisa::string ToString() const;

    /// Write the LLVM module to a file (bitcode format)
    void WriteBitcodeToFile(luisa::string_view path) const;

    /// Reset the module (for fresh codegen session)
    void ResetModule();

    // --- SPIR-V emission ---
    /// Generate binding properties (bindings) from the kernel arguments.
    /// Mirrors SpirvCodegenEntry::generate_binding().
    void GenerateProperties(Function kernel,
                           LLVMCodegenResult::Properties &properties);

    /// Convert the current llvm::Module to SPIR-V binary via LLVM SPIRV target.
    [[nodiscard]] luisa::vector<uint32_t> EmitSPIRV();

    /// Reset and reinitialize the module with proper SPIR-V target triple/layout.
    void InitializeSPIRVModule();

private:
    // LLVM SPIRV target machine
    std::unique_ptr<llvm::TargetMachine> _target_machine;
};

} // namespace lc::llvm_codegen
