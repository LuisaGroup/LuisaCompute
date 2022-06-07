//
// Created by Mike Smith on 2022/5/23.
//

#include <backends/llvm/llvm_codegen.h>
#include <backends/llvm/llvm_device.h>

//
// Created by Mike Smith on 2022/2/7.
//

#include <fstream>
#include <streambuf>

#include <core/stl.h>
#include <backends/llvm/llvm_stream.h>
#include <backends/llvm/llvm_device.h>
#include <backends/llvm/llvm_event.h>
#include <backends/llvm/llvm_shader.h>

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
    LUISA_ASSERT(target != nullptr, "Failed to get target machine: {}.", err);
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
    LUISA_ASSERT(_machine != nullptr, "Failed to create target machine.");
}

void *LLVMDevice::native_handle() const noexcept {
    return reinterpret_cast<void *>(reinterpret_cast<uint64_t>(this));
}

uint64_t LLVMDevice::create_buffer(size_t size_bytes) noexcept {
    return reinterpret_cast<uint64_t>(luisa::allocate<std::byte>(size_bytes));
}

void LLVMDevice::destroy_buffer(uint64_t handle) noexcept {
    luisa::deallocate(reinterpret_cast<std::byte *>(handle));
}

void *LLVMDevice::buffer_native_handle(uint64_t handle) const noexcept {
    return reinterpret_cast<void *>(handle);
}

uint64_t LLVMDevice::create_texture(PixelFormat format, uint dimension, uint width, uint height, uint depth, uint mipmap_levels) noexcept {
    //    auto texture = luisa::new_with_allocator<LLVMTexture>(
    //        format, dimension, make_uint3(width, height, depth), mipmap_levels);
    //    return reinterpret_cast<uint64_t>(texture);
    return 0;
}

void LLVMDevice::destroy_texture(uint64_t handle) noexcept {
    //    luisa::delete_with_allocator(reinterpret_cast<LLVMTexture *>(handle));
}

void *LLVMDevice::texture_native_handle(uint64_t handle) const noexcept {
    return reinterpret_cast<void *>(handle);
}

uint64_t LLVMDevice::create_bindless_array(size_t size) noexcept {
    //    auto array = luisa::new_with_allocator<LLVMBindlessArray>(size);
    //    return reinterpret_cast<uint64_t>(array);
    return 0;
}

void LLVMDevice::destroy_bindless_array(uint64_t handle) noexcept {
    //    luisa::delete_with_allocator(reinterpret_cast<LLVMBindlessArray *>(handle));
}

void LLVMDevice::emplace_buffer_in_bindless_array(uint64_t array, size_t index, uint64_t handle, size_t offset_bytes) noexcept {
    //    reinterpret_cast<LLVMBindlessArray *>(array)->emplace_buffer(
    //        index, reinterpret_cast<const void *>(handle), offset_bytes);
}

void LLVMDevice::emplace_tex2d_in_bindless_array(uint64_t array, size_t index, uint64_t handle, Sampler sampler) noexcept {
    //    reinterpret_cast<LLVMBindlessArray *>(array)->emplace_tex2d(
    //        index, reinterpret_cast<const LLVMTexture *>(handle), sampler);
}

void LLVMDevice::emplace_tex3d_in_bindless_array(uint64_t array, size_t index, uint64_t handle, Sampler sampler) noexcept {
    //    reinterpret_cast<LLVMBindlessArray *>(array)->emplace_tex3d(
    //        index, reinterpret_cast<const LLVMTexture *>(handle), sampler);
}

bool LLVMDevice::is_resource_in_bindless_array(uint64_t array, uint64_t handle) const noexcept {
    //    return reinterpret_cast<LLVMBindlessArray *>(array)->uses_resource(handle);
    return true;
}

void LLVMDevice::remove_buffer_in_bindless_array(uint64_t array, size_t index) noexcept {
    //    reinterpret_cast<LLVMBindlessArray *>(array)->remove_buffer(index);
}

void LLVMDevice::remove_tex2d_in_bindless_array(uint64_t array, size_t index) noexcept {
    //    reinterpret_cast<LLVMBindlessArray *>(array)->remove_tex2d(index);
}

void LLVMDevice::remove_tex3d_in_bindless_array(uint64_t array, size_t index) noexcept {
    //    reinterpret_cast<LLVMBindlessArray *>(array)->remove_tex3d(index);
}

uint64_t LLVMDevice::create_stream(bool for_present) noexcept {
    return reinterpret_cast<uint64_t>(luisa::new_with_allocator<LLVMStream>());
}

void LLVMDevice::destroy_stream(uint64_t handle) noexcept {
    luisa::delete_with_allocator(reinterpret_cast<LLVMStream *>(handle));
}

void LLVMDevice::synchronize_stream(uint64_t stream_handle) noexcept {
    reinterpret_cast<LLVMStream *>(stream_handle)->synchronize();
}

void LLVMDevice::dispatch(uint64_t stream_handle, const CommandList &list) noexcept {
    auto stream = reinterpret_cast<LLVMStream *>(stream_handle);
    stream->dispatch(list);
}

void LLVMDevice::dispatch(uint64_t stream_handle, move_only_function<void()> &&func) noexcept {
    auto stream = reinterpret_cast<LLVMStream *>(stream_handle);
    stream->dispatch(std::move(func));
}

void *LLVMDevice::stream_native_handle(uint64_t handle) const noexcept {
    return reinterpret_cast<void *>(handle);
}

uint64_t LLVMDevice::create_shader(Function kernel, std::string_view meta_options) noexcept {
    auto shader = luisa::new_with_allocator<LLVMShader>(this, kernel);
    return reinterpret_cast<uint64_t>(shader);
}

void LLVMDevice::destroy_shader(uint64_t handle) noexcept {
    luisa::delete_with_allocator(reinterpret_cast<LLVMShader *>(handle));
}

uint64_t LLVMDevice::create_event() noexcept {
    return reinterpret_cast<uint64_t>(luisa::new_with_allocator<LLVMEvent>());
}

void LLVMDevice::destroy_event(uint64_t handle) noexcept {
    luisa::delete_with_allocator(reinterpret_cast<LLVMEvent *>(handle));
}

void LLVMDevice::signal_event(uint64_t handle, uint64_t stream_handle) noexcept {
    reinterpret_cast<LLVMStream *>(stream_handle)->signal(reinterpret_cast<LLVMEvent *>(handle));
}

void LLVMDevice::wait_event(uint64_t handle, uint64_t stream_handle) noexcept {
    reinterpret_cast<LLVMStream *>(stream_handle)->wait(reinterpret_cast<LLVMEvent *>(handle));
}

void LLVMDevice::synchronize_event(uint64_t handle) noexcept {
    reinterpret_cast<LLVMEvent *>(handle)->wait();
}

uint64_t LLVMDevice::create_mesh(
    uint64_t v_buffer, size_t v_offset, size_t v_stride, size_t v_count,
    uint64_t t_buffer, size_t t_offset, size_t t_count, AccelUsageHint hint) noexcept {
    //    auto mesh = luisa::new_with_allocator<LLVMMesh>(
    //        _rtc_device, hint,
    //        v_buffer, v_offset, v_stride, v_count,
    //        t_buffer, t_offset, t_count);
    //    return reinterpret_cast<uint64_t>(mesh);
    return 0;
}

void LLVMDevice::destroy_mesh(uint64_t handle) noexcept {
//    luisa::delete_with_allocator(reinterpret_cast<LLVMMesh *>(handle));
}

uint64_t LLVMDevice::create_accel(AccelUsageHint hint) noexcept {
//    auto accel = luisa::new_with_allocator<LLVMAccel>(_rtc_device, hint);
//    return reinterpret_cast<uint64_t>(accel);
return 0;
}

void LLVMDevice::destroy_accel(uint64_t handle) noexcept {
//    luisa::delete_with_allocator(reinterpret_cast<LLVMAccel *>(handle));
}

uint64_t LLVMDevice::create_swap_chain(
    uint64_t window_handle, uint64_t stream_handle, uint width, uint height,
    bool allow_hdr, uint back_buffer_count) noexcept {
    LUISA_ERROR_WITH_LOCATION("Not implemented.");
}

void LLVMDevice::destroy_swap_chain(uint64_t handle) noexcept {
    LUISA_ERROR_WITH_LOCATION("Not implemented.");
}

PixelStorage LLVMDevice::swap_chain_pixel_storage(uint64_t handle) noexcept {
    LUISA_ERROR_WITH_LOCATION("Not implemented.");
}

void LLVMDevice::present_display_in_stream(
    uint64_t stream_handle, uint64_t swap_chain_handle, uint64_t image_handle) noexcept {
    LUISA_ERROR_WITH_LOCATION("Not implemented.");
}

}// namespace luisa::compute::llvm

LUISA_EXPORT_API luisa::compute::Device::Interface *create(const luisa::compute::Context &ctx, std::string_view) noexcept {
    return luisa::new_with_allocator<luisa::compute::llvm::LLVMDevice>(ctx);
}

LUISA_EXPORT_API void destroy(luisa::compute::Device::Interface *device) noexcept {
    luisa::delete_with_allocator(device);
}
