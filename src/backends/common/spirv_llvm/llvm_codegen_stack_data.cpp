#include "llvm_codegen_stack_data.h"

#include <mutex>

namespace lc::llvm_codegen {

LLVMCodegenStackData::LLVMCodegenStackData() = default;

LLVMCodegenStackData::~LLVMCodegenStackData() = default;

void LLVMCodegenStackData::Clear() {
    struct_count = 0;
    const_count = 0;
    func_count = 0;
    temp_count = 0;
    type_counts.clear();
    struct_types.clear();
    const_types.clear();
    func_types.clear();
    variables.clear();
    arguments.clear();
    shared_variable_uids.clear();
    loop_stack.clear();
    current_switch = nullptr;
    switch_merge_block = nullptr;
    switch_case_counter = 0;
    atomic_funcs.clear();
    arg_offset = 0;
    scope_count = -1;
    useTex2DBindless = false;
    useTex3DBindless = false;
    useBufferBindless = false;
    printers.clear();
}

uint64_t LLVMCodegenStackData::GetTypeCount(Type const *t) {
    auto ite = type_counts.try_emplace(
        t,
        vstd::lazy_eval([this] { return struct_count++; }));
    return ite.first->second;
}

std::pair<uint64_t, bool> LLVMCodegenStackData::GetConstCount(uint64_t data_hash) {
    auto ite = const_types.try_emplace(
        data_hash,
        vstd::lazy_eval([this] { return const_count++; }));
    return {ite.first->second, ite.second};
}

llvm::Function *LLVMCodegenStackData::GetFunc(llvm::Function *f, uint64_t hash) {
    auto ite = func_types.try_emplace(
        hash, f);
    return ite.first->second;
}

namespace detail {

struct LLVMCodegenGlobalPool {
    std::mutex mtx;
    vstd::vector<vstd::unique_ptr<LLVMCodegenStackData>> all_codegen;

    vstd::unique_ptr<LLVMCodegenStackData> Allocate() {
        std::lock_guard lck(mtx);
        if (!all_codegen.empty()) {
            auto item = std::move(all_codegen.back());
            all_codegen.pop_back();
            return item;
        }
        return vstd::unique_ptr<LLVMCodegenStackData>(new LLVMCodegenStackData());
    }

    void DeAllocate(vstd::unique_ptr<LLVMCodegenStackData> &&v) {
        std::lock_guard lck(mtx);
        v->Clear();
        all_codegen.emplace_back(std::move(v));
    }
};

static LLVMCodegenGlobalPool global_pool;

} // namespace detail

vstd::unique_ptr<LLVMCodegenStackData> LLVMCodegenStackData::Allocate(LLVMCodegenUtility *util) {
    auto ptr = detail::global_pool.Allocate();
    ptr->util = util;
    return ptr;
}

void LLVMCodegenStackData::DeAllocate(vstd::unique_ptr<LLVMCodegenStackData> &&v) {
    detail::global_pool.DeAllocate(std::move(v));
}

} // namespace lc::llvm_codegen
