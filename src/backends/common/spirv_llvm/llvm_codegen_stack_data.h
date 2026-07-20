#pragma once

#include <luisa/vstl/common.h>
#include <luisa/vstl/functional.h>
#include <luisa/ast/function.h>
#include <luisa/ast/expression.h>
#include <luisa/core/stl/unordered_map.h>
#include <luisa/core/stl/vector.h>

#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Type.h>
#include <llvm/IR/Value.h>
#include <llvm/IR/BasicBlock.h>

namespace lc::llvm_codegen {

using namespace luisa;
using namespace luisa::compute;

class LLVMCodegenUtility;

/**
 * @brief Per-codegen mutable state (mirrors hlsl::CodegenStackData).
 *
 * Owns counters, type maps, constant maps, function maps, and the loop
 * break/continue stack shared across the visitor and utility.
 */
struct LLVMCodegenStackData : public vstd::IOperatorNewBase {

    LLVMCodegenUtility *util{nullptr};

    // --- Counters ---
    uint64_t struct_count{0};
    uint64_t const_count{0};
    uint64_t func_count{0};
    uint64_t temp_count{0};

    // --- Type maps ---
    /// Luisa Type* → count/index
    luisa::unordered_map<Type const *, uint64_t> type_counts;

    /// Luisa Type* → LLVM struct type (populated by RegistStructType)
    luisa::unordered_map<Type const *, llvm::StructType *> struct_types;

    /// Constant data hash → index into constants array
    luisa::unordered_map<uint64_t, uint64_t> const_types;

    /// Function hash → {index, LLVM Function*}
    luisa::unordered_map<uint64_t, llvm::Function *> func_types;

    /// Variable uid → llvm::Value* (alloca)
    luisa::unordered_map<uint32_t, llvm::Value *> variables;

    /// Argument uid → llvm::Value* (parameter or alloca)
    luisa::unordered_map<uint32_t, llvm::Value *> arguments;

    // --- Shared variable tracking ---
    /// Set of shared variable UIDs
    luisa::unordered_set<uint32_t> shared_variable_uids;

    // --- Loop stack for break/continue ---
    struct LoopContext {
        llvm::BasicBlock *break_target;
        llvm::BasicBlock *continue_target;
    };
    luisa::vector<LoopContext> loop_stack;

    // --- Break target stack for break statements (loops + switches) ---
    luisa::vector<llvm::BasicBlock *> break_stack;

    // --- Switch state ---
    llvm::SwitchInst *current_switch{nullptr};
    llvm::BasicBlock *switch_merge_block{nullptr};
    size_t switch_case_counter{0};

    // --- Atomic function cache ---
    luisa::unordered_map<uint64_t, llvm::Function *> atomic_funcs;

    // --- Helper state ---
    uint arg_offset{0};
    int64_t scope_count{-1};

    // --- Bindless tracking ---
    bool useTex2DBindless{false};
    bool useTex3DBindless{false};
    bool useBufferBindless{false};

    // --- Printer tracking ---
    vstd::vector<std::pair<vstd::string, Type const *>> printers;

    LLVMCodegenStackData();
    ~LLVMCodegenStackData();

    void Clear();

    /// Allocate from global pool
    static vstd::unique_ptr<LLVMCodegenStackData> Allocate(LLVMCodegenUtility *util);
    /// Return to global pool
    static void DeAllocate(vstd::unique_ptr<LLVMCodegenStackData> &&v);

    /// Get or assign struct type count
    uint64_t GetTypeCount(Type const *t);
    /// Get or assign const count
    std::pair<uint64_t, bool> GetConstCount(uint64_t data_hash);
    /// Get or assign function count and LLVM function
    llvm::Function *GetFunc(llvm::Function *f, uint64_t hash);
};

} // namespace lc::llvm_codegen
