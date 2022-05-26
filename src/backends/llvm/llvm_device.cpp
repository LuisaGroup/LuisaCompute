//
// Created by Mike Smith on 2022/5/23.
//

#include <backends/llvm/llvm_codegen.h>
#include <backends/llvm/llvm_device.h>

namespace luisa::compute::llvm {

LLVMDevice::LLVMDevice(const Context &ctx) noexcept : Interface{ctx} {

    static std::once_flag flag;
    std::call_once(flag, [] {
        ::llvm::InitializeNativeTarget();
        ::llvm::InitializeNativeTargetAsmPrinter();
        LLVMLinkInMCJIT();
    });
    std::string err;
    auto target_triple = ::llvm::sys::getDefaultTargetTriple();
    LUISA_INFO("Target: {}.", target_triple);
    auto target = ::llvm::TargetRegistry::lookupTarget(target_triple, err);
    if (target == nullptr) {
        LUISA_ERROR_WITH_LOCATION("Failed to get target machine: {}.", err);
    }
    ::llvm::TargetOptions options;
    options.AllowFPOpFusion = ::llvm::FPOpFusion::Fast;
    options.UnsafeFPMath = true;
    options.NoInfsFPMath = true;
    options.NoNaNsFPMath = true;
    options.NoTrappingFPMath = true;
    options.GuaranteedTailCallOpt = true;
    auto mcpu = ::llvm::sys::getHostCPUName();
    _machine = target->createTargetMachine(
        target_triple, mcpu,
#if defined(LUISA_PLATFORM_APPLE) && defined(__aarch64__)
        "+neon",
#else
        "+avx2",
#endif
        options, {}, {},
        ::llvm::CodeGenOpt::Aggressive, true);
    if (_machine == nullptr) {
        LUISA_ERROR_WITH_LOCATION("Failed to create target machine.");
    }
}

void *LLVMDevice::native_handle() const noexcept {
    return nullptr;
}

uint64_t LLVMDevice::create_buffer(size_t size_bytes) noexcept {
    return 0;
}

void LLVMDevice::destroy_buffer(uint64_t handle) noexcept {
}

void *LLVMDevice::buffer_native_handle(uint64_t handle) const noexcept {
    return nullptr;
}

uint64_t LLVMDevice::create_texture(PixelFormat format, uint dimension, uint width, uint height, uint depth, uint mipmap_levels) noexcept {
    return 0;
}

void LLVMDevice::destroy_texture(uint64_t handle) noexcept {
}

void *LLVMDevice::texture_native_handle(uint64_t handle) const noexcept {
    return nullptr;
}

uint64_t LLVMDevice::create_bindless_array(size_t size) noexcept {
    return 0;
}

void LLVMDevice::destroy_bindless_array(uint64_t handle) noexcept {
}
void LLVMDevice::emplace_buffer_in_bindless_array(uint64_t array, size_t index, uint64_t handle, size_t offset_bytes) noexcept {
}
void LLVMDevice::emplace_tex2d_in_bindless_array(uint64_t array, size_t index, uint64_t handle, Sampler sampler) noexcept {
}
void LLVMDevice::emplace_tex3d_in_bindless_array(uint64_t array, size_t index, uint64_t handle, Sampler sampler) noexcept {
}
bool LLVMDevice::is_resource_in_bindless_array(uint64_t array, uint64_t handle) const noexcept {
    return false;
}
void LLVMDevice::remove_buffer_in_bindless_array(uint64_t array, size_t index) noexcept {
}
void LLVMDevice::remove_tex2d_in_bindless_array(uint64_t array, size_t index) noexcept {
}
void LLVMDevice::remove_tex3d_in_bindless_array(uint64_t array, size_t index) noexcept {
}
uint64_t LLVMDevice::create_stream(bool for_present) noexcept {
    return 0;
}
void LLVMDevice::destroy_stream(uint64_t handle) noexcept {
}
void LLVMDevice::synchronize_stream(uint64_t stream_handle) noexcept {
}
void LLVMDevice::dispatch(uint64_t stream_handle, const CommandList &list) noexcept {
}
void LLVMDevice::dispatch(uint64_t stream_handle, luisa::span<const CommandList> lists) noexcept {
    Interface::dispatch(stream_handle, lists);
}
void LLVMDevice::dispatch(uint64_t stream_handle, move_only_function<void()> &&func) noexcept {
}
void *LLVMDevice::stream_native_handle(uint64_t handle) const noexcept {
    return nullptr;
}
uint64_t LLVMDevice::create_swap_chain(uint64_t window_handle, uint64_t stream_handle, uint width, uint height, bool allow_hdr, uint back_buffer_size) noexcept {
    return 0;
}
void LLVMDevice::destroy_swap_chain(uint64_t handle) noexcept {
}
PixelStorage LLVMDevice::swap_chain_pixel_storage(uint64_t handle) noexcept {
    return PixelStorage::BYTE2;
}

void LLVMDevice::present_display_in_stream(uint64_t stream_handle, uint64_t swapchain_handle, uint64_t image_handle) noexcept {
}

uint64_t LLVMDevice::create_shader(Function kernel, std::string_view meta_options) noexcept {

    // codegen
    auto llvm_ctx = luisa::make_unique<::llvm::LLVMContext>();
    LLVMCodegen codegen{*llvm_ctx};
    auto module = codegen.emit(kernel);
    module->print(::llvm::errs(), nullptr);

    // optimize
    ::llvm::PassManagerBuilder pass_manager_builder;
    pass_manager_builder.OptLevel = ::llvm::CodeGenOpt::Aggressive;
    pass_manager_builder.Inliner = ::llvm::createFunctionInliningPass(
        pass_manager_builder.OptLevel, 0, false);
    pass_manager_builder.LoopsInterleaved = true;
    pass_manager_builder.LoopVectorize = true;
    pass_manager_builder.SLPVectorize = true;
    pass_manager_builder.MergeFunctions = true;
    pass_manager_builder.EnablePGOCSInstrGen = false;
    pass_manager_builder.EnablePGOCSInstrUse = false;
    pass_manager_builder.EnablePGOInstrGen = false;
    pass_manager_builder.CallGraphProfile = false;
    pass_manager_builder.PerformThinLTO = true;
    _machine->adjustPassManager(pass_manager_builder);
    module->setDataLayout(_machine->createDataLayout());

    for (auto i = 0u; i < 2u; i++) {
        ::llvm::errs() << "\nOptimization Pass #" << i << "\n";
        // optimize: function passes
        ::llvm::legacy::FunctionPassManager function_pass_manager{module.get()};
        pass_manager_builder.populateFunctionPassManager(function_pass_manager);
        function_pass_manager.add(::llvm::createTargetTransformInfoWrapperPass(
            _machine->getTargetIRAnalysis()));
        pass_manager_builder.populateFunctionPassManager(function_pass_manager);
        for (auto &&f : module->functions()) {
            function_pass_manager.doInitialization();
            function_pass_manager.run(f);
            function_pass_manager.doFinalization();
        }
        ::llvm::errs() << "\nOptimization Pass #" << i << ": After Function Passes" << "\n";
        module->print(::llvm::errs(), nullptr);
        // optimize: module passes
        ::llvm::legacy::PassManager module_pass_manager;
        module_pass_manager.add(
            ::llvm::createTargetTransformInfoWrapperPass(
                _machine->getTargetIRAnalysis()));
        pass_manager_builder.populateModulePassManager(module_pass_manager);
        module_pass_manager.run(*module);
    }
    ::llvm::errs() << "\nAfter Optimization" << "\n";
    module->print(::llvm::errs(), nullptr);
    return 0;
}

void LLVMDevice::destroy_shader(uint64_t handle) noexcept {
}
uint64_t LLVMDevice::create_event() noexcept {
    return 0;
}
void LLVMDevice::destroy_event(uint64_t handle) noexcept {
}
void LLVMDevice::signal_event(uint64_t handle, uint64_t stream_handle) noexcept {
}
void LLVMDevice::wait_event(uint64_t handle, uint64_t stream_handle) noexcept {
}
void LLVMDevice::synchronize_event(uint64_t handle) noexcept {
}
uint64_t LLVMDevice::create_mesh(uint64_t v_buffer, size_t v_offset, size_t v_stride, size_t v_count, uint64_t t_buffer, size_t t_offset, size_t t_count, AccelUsageHint hint) noexcept {
    return 0;
}
void LLVMDevice::destroy_mesh(uint64_t handle) noexcept {
}
uint64_t LLVMDevice::create_accel(AccelUsageHint hint) noexcept {
    return 0;
}
void LLVMDevice::destroy_accel(uint64_t handle) noexcept {
}

}// namespace luisa::compute::llvm

LUISA_EXPORT_API luisa::compute::Device::Interface *create(const luisa::compute::Context &ctx, std::string_view) noexcept {
    return luisa::new_with_allocator<luisa::compute::llvm::LLVMDevice>(ctx);
}

LUISA_EXPORT_API void destroy(luisa::compute::Device::Interface *device) noexcept {
    luisa::delete_with_allocator(device);
}
