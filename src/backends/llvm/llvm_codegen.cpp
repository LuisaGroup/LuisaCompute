//
// Created by Mike Smith on 2022/5/23.
//

#include <backends/llvm/llvm_codegen.h>

namespace luisa::compute::llvm {

LLVMCodegen::LLVMCodegen(::llvm::LLVMContext &ctx) noexcept
    : _context{ctx} {}

std::unique_ptr<::llvm::Module> LLVMCodegen::emit(Function f) noexcept {
    auto module_name = luisa::format("module_{:016x}", f.hash());
    auto module = std::make_unique<::llvm::Module>(
        ::llvm::StringRef{module_name.data(), module_name.size()}, _context);
    _module = module.get();
    static_cast<void>(_create_function(f));
    for (auto &&func : _module->functions()) {
        for (auto &&bb : func) {
            for (auto &&inst : bb) {
                if (::llvm::isa<::llvm::FPMathOperator>(&inst)) {
                    inst.setFast(true);
                }
            }
        }
    }
    _module = nullptr;
    _constants.clear();
    _struct_types.clear();
    return module;
}

LLVMCodegen::FunctionContext *LLVMCodegen::_current_context() noexcept {
    LUISA_ASSERT(!_function_stack.empty(), "Empty function context stack.");
    return _function_stack.back().get();
}

}// namespace luisa::compute::llvm
