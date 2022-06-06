//
// Created by Mike Smith on 2022/5/23.
//

#include <backends/llvm/llvm_codegen.h>

namespace luisa::compute::llvm {

LLVMCodegen::LLVMCodegen(::llvm::LLVMContext &ctx) noexcept
    : _context{ctx} {}

luisa::unique_ptr<::llvm::Module> LLVMCodegen::emit(Function f) noexcept {
    auto module_name = luisa::format("module_{:016x}", f.hash());
    auto module = luisa::make_unique<::llvm::Module>(
        ::llvm::StringRef{module_name.data(), module_name.size()}, _context);
    _module = module.get();
    auto _ = _create_function(f);
    _module = nullptr;
    _constants.clear();
    LUISA_ASSERT(
        _function_stack.empty(),
        "Function stack is not empty after emitting function.");
    ::llvm::verifyModule(*module, &::llvm::errs());
    return module;
}

LLVMCodegen::FunctionContext *LLVMCodegen::_current_context() noexcept {
    LUISA_ASSERT(!_function_stack.empty(), "Empty function context stack.");
    return _function_stack.back().get();
}

}// namespace luisa::compute::llvm
