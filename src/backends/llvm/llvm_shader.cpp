//
// Created by Mike Smith on 2022/2/11.
//

#include <cmath>
#include <core/mathematics.h>
#include <backends/llvm/llvm_shader.h>
#include <backends/llvm/llvm_device.h>
#include <backends/llvm/llvm_codegen.h>

namespace luisa::compute::llvm {

LLVMShader::LLVMShader(LLVMDevice *device, Function func) noexcept {
    // compute argument offsets
    _argument_offsets.reserve(func.arguments().size());
    for (auto &&arg : func.arguments()) {
        auto aligned_offset = luisa::align(_argument_buffer_size, 16u);
        _argument_offsets.emplace(arg.uid(), aligned_offset);
        if (arg.type()->is_buffer()) {
            _argument_buffer_size = aligned_offset + LLVMCodegen::buffer_handle_size;
        } else if (arg.type()->is_texture()) {
            _argument_buffer_size = aligned_offset + LLVMCodegen::texture_handle_size;
        } else if (arg.type()->is_accel()) {
            _argument_buffer_size = aligned_offset + LLVMCodegen::accel_handle_size;
        } else if (arg.type()->is_bindless_array()) {
            _argument_buffer_size = aligned_offset + LLVMCodegen::bindless_array_handle_size;
        } else {
            _argument_buffer_size = aligned_offset + arg.type()->size();
        }
    }
    _argument_buffer_size = luisa::align(_argument_buffer_size, 16u);

    // codegen
    std::error_code ec;
    _context = luisa::make_unique<::llvm::LLVMContext>();
    Clock clk;
    LLVMCodegen codegen{*_context};
    auto module = codegen.emit(func);
    LUISA_INFO("Codegen: {} ms.", clk.toc());
    ::llvm::raw_fd_ostream file{"kernel.ll", ec};
    if (ec) {
        LUISA_ERROR_WITH_LOCATION(
            "Failed to create file 'kernel.ll': {}.",
            ec.message());
    }
    module->print(file, nullptr);
    if (::llvm::verifyModule(*module, &::llvm::errs())) {
        LUISA_ERROR_WITH_LOCATION("Failed to verify module.");
    }

    // optimize
    ::llvm::PassManagerBuilder pass_manager_builder;
    pass_manager_builder.OptLevel = ::llvm::CodeGenOpt::Aggressive;
    pass_manager_builder.Inliner = ::llvm::createFunctionInliningPass(
        pass_manager_builder.OptLevel, 0, false);
    pass_manager_builder.LoopsInterleaved = true;
    pass_manager_builder.LoopVectorize = true;
    pass_manager_builder.SLPVectorize = true;
    pass_manager_builder.MergeFunctions = true;
    pass_manager_builder.PerformThinLTO = true;
    pass_manager_builder.NewGVN = true;
    auto machine = device->target_machine();
    machine->adjustPassManager(pass_manager_builder);
    module->setDataLayout(machine->createDataLayout());
    module->setTargetTriple(machine->getTargetTriple().str());
    ::llvm::legacy::PassManager module_pass_manager;
    module_pass_manager.add(
        ::llvm::createTargetTransformInfoWrapperPass(
            machine->getTargetIRAnalysis()));
    pass_manager_builder.populateModulePassManager(module_pass_manager);
    clk.tic();
    module_pass_manager.run(*module);
    if (::llvm::verifyModule(*module, &::llvm::errs())) {
        LUISA_ERROR_WITH_LOCATION("Failed to verify module.");
    }
    LUISA_INFO("Optimize: {} ms.", clk.toc());
    ::llvm::raw_fd_ostream file_opt{"kernel.opt.ll", ec};
    if (ec) {
        LUISA_ERROR_WITH_LOCATION(
            "Failed to create file 'kernel.opt.ll': {}.",
            ec.message());
    }
    module->print(file_opt, nullptr);

    // compile
    std::string err;
    _engine = ::llvm::EngineBuilder{std::move(module)}
                  .setErrorStr(&err)
                  .setOptLevel(::llvm::CodeGenOpt::Aggressive)
                  .setEngineKind(::llvm::EngineKind::JIT)
                  .create(machine);
    _engine->DisableLazyCompilation(true);
    _engine->DisableSymbolSearching(false);
    _engine->InstallLazyFunctionCreator([](auto name) noexcept -> void * {
        LUISA_INFO("Searching for '{}'.", name);
        return nullptr;
    });
    LUISA_ASSERT(_engine != nullptr, "Failed to create execution engine: {}.", err);
    _kernel_entry = reinterpret_cast<kernel_entry_t *>(
        _engine->getFunctionAddress("kernel_main"));
    LUISA_ASSERT(_kernel_entry != nullptr, "Failed to find kernel entry.");
}

LLVMShader::~LLVMShader() noexcept = default;

size_t LLVMShader::argument_offset(uint uid) const noexcept {
    if (auto iter = _argument_offsets.find(uid);
        iter != _argument_offsets.cend()) [[likely]] {
        return iter->second;
    }
    LUISA_ERROR_WITH_LOCATION("Invalid argument uid {}.", uid);
}

void LLVMShader::invoke(const std::byte *args, uint3 block_id) const noexcept {
    _kernel_entry(args, &block_id);
}

}// namespace luisa::compute::llvm
