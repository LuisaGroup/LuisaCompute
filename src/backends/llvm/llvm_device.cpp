//
// Created by Mike Smith on 2022/5/23.
//

#include <core/stl.h>
#include <backends/llvm/llvm_stream.h>
#include <backends/llvm/llvm_device.h>
#include <backends/llvm/llvm_event.h>
#include <backends/llvm/llvm_shader.h>
#include <backends/llvm/llvm_codegen.h>
#include <backends/llvm/llvm_texture.h>
#include <backends/llvm/llvm_mesh.h>
#include <backends/llvm/llvm_accel.h>
#include <backends/llvm/llvm_bindless_array.h>

namespace luisa::compute::llvm {

LLVMDevice::LLVMDevice(const Context &ctx) noexcept
    : Interface{ctx}, _rtc_device{rtcNewDevice(nullptr)} {
    static std::once_flag flag;
    std::call_once(flag, [] {
        ::llvm::InitializeNativeTarget();
        ::llvm::InitializeNativeTargetAsmPrinter();
    });
    // build JIT engine
    ::llvm::orc::LLJITBuilder jit_builder;
    if (auto host = ::llvm::orc::JITTargetMachineBuilder::detectHost()) {
        ::llvm::TargetOptions options;
        options.AllowFPOpFusion = ::llvm::FPOpFusion::Fast;
        options.UnsafeFPMath = true;
        options.NoInfsFPMath = true;
        options.NoNaNsFPMath = true;
        options.NoTrappingFPMath = true;
        options.NoSignedZerosFPMath = true;
#if LLVM_VERSION_MAJOR >= 14
        options.ApproxFuncFPMath = true;
#endif
        options.EnableIPRA = true;
        options.StackSymbolOrdering = true;
        options.EnableMachineFunctionSplitter = true;
        options.EnableMachineOutliner = true;
        options.NoTrapAfterNoreturn = true;
        host->setOptions(options);
        host->setCodeGenOptLevel(::llvm::CodeGenOpt::Aggressive);
#ifdef __aarch64__
        host->addFeatures({"+neon"});
#else
        host->addFeatures({"+avx2"});
#endif
        LUISA_INFO("LLVM JIT target: triplet = {}, features = {}.",
                   host->getTargetTriple().str(),
                   host->getFeatures().getString());
        if (auto machine = host->createTargetMachine()) {
            _target_machine = std::move(machine.get());
        } else {
            ::llvm::handleAllErrors(machine.takeError(), [&](const ::llvm::ErrorInfoBase &e) {
                LUISA_WARNING_WITH_LOCATION("JITTargetMachineBuilder::createTargetMachine(): {}.", e.message());
            });
            LUISA_ERROR_WITH_LOCATION("Failed to create target machine.");
        }
        jit_builder.setJITTargetMachineBuilder(std::move(*host));
    } else {
        ::llvm::handleAllErrors(host.takeError(), [&](const ::llvm::ErrorInfoBase &e) {
            LUISA_WARNING_WITH_LOCATION("JITTargetMachineBuilder::detectHost(): {}.", e.message());
        });
        LUISA_ERROR_WITH_LOCATION("Failed to detect host.");
    }

    if (auto expected_jit = jit_builder.create()) {
        _jit = std::move(expected_jit.get());
    } else {
        ::llvm::handleAllErrors(expected_jit.takeError(), [](const ::llvm::ErrorInfoBase &err) {
            LUISA_WARNING_WITH_LOCATION("LLJITBuilder::create(): {}", err.message());
        });
        LUISA_ERROR_WITH_LOCATION("Failed to create LLJIT.");
    }

    // map symbols
    if (auto generator = ::llvm::orc::DynamicLibrarySearchGenerator::GetForCurrentProcess(
            _jit->getDataLayout().getGlobalPrefix())) {
        _jit->getMainJITDylib().addGenerator(std::move(generator.get()));
    } else {
        ::llvm::handleAllErrors(generator.takeError(), [](const ::llvm::ErrorInfoBase &err) {
            LUISA_WARNING_WITH_LOCATION("DynamicLibrarySearchGenerator::GetForCurrentProcess(): {}", err.message());
        });
        LUISA_ERROR_WITH_LOCATION("Failed to add generator.");
    }
    ::llvm::orc::SymbolMap symbol_map{
        {_jit->mangleAndIntern("texture.read.2d.int"), ::llvm::JITEvaluatedSymbol::fromPointer(&texture_read_2d_int)},
        {_jit->mangleAndIntern("texture.read.3d.int"), ::llvm::JITEvaluatedSymbol::fromPointer(&texture_read_3d_int)},
        {_jit->mangleAndIntern("texture.read.2d.uint"), ::llvm::JITEvaluatedSymbol::fromPointer(&texture_read_2d_uint)},
        {_jit->mangleAndIntern("texture.read.3d.uint"), ::llvm::JITEvaluatedSymbol::fromPointer(&texture_read_3d_uint)},
        {_jit->mangleAndIntern("texture.read.2d.float"), ::llvm::JITEvaluatedSymbol::fromPointer(&texture_read_2d_float)},
        {_jit->mangleAndIntern("texture.read.3d.float"), ::llvm::JITEvaluatedSymbol::fromPointer(&texture_read_3d_float)},
        {_jit->mangleAndIntern("texture.write.2d.int"), ::llvm::JITEvaluatedSymbol::fromPointer(&texture_write_2d_int)},
        {_jit->mangleAndIntern("texture.write.3d.int"), ::llvm::JITEvaluatedSymbol::fromPointer(&texture_write_3d_int)},
        {_jit->mangleAndIntern("texture.write.2d.uint"), ::llvm::JITEvaluatedSymbol::fromPointer(&texture_write_2d_uint)},
        {_jit->mangleAndIntern("texture.write.3d.uint"), ::llvm::JITEvaluatedSymbol::fromPointer(&texture_write_3d_uint)},
        {_jit->mangleAndIntern("texture.write.2d.float"), ::llvm::JITEvaluatedSymbol::fromPointer(&texture_write_2d_float)},
        {_jit->mangleAndIntern("texture.write.3d.float"), ::llvm::JITEvaluatedSymbol::fromPointer(&texture_write_3d_float)},
        {_jit->mangleAndIntern("accel.trace.closest"), ::llvm::JITEvaluatedSymbol::fromPointer(&accel_trace_closest)},
        {_jit->mangleAndIntern("accel.trace.any"), ::llvm::JITEvaluatedSymbol::fromPointer(&accel_trace_any)},
        {_jit->mangleAndIntern("bindless.texture.2d.read"), ::llvm::JITEvaluatedSymbol::fromPointer(&bindless_texture_2d_read)},
        {_jit->mangleAndIntern("bindless.texture.3d.read"), ::llvm::JITEvaluatedSymbol::fromPointer(&bindless_texture_3d_read)},
        {_jit->mangleAndIntern("bindless.texture.2d.sample"), ::llvm::JITEvaluatedSymbol::fromPointer(&bindless_texture_2d_sample)},
        {_jit->mangleAndIntern("bindless.texture.3d.sample"), ::llvm::JITEvaluatedSymbol::fromPointer(&bindless_texture_3d_sample)},
        {_jit->mangleAndIntern("bindless.texture.2d.sample.level"), ::llvm::JITEvaluatedSymbol::fromPointer(&bindless_texture_2d_sample_level)},
        {_jit->mangleAndIntern("bindless.texture.3d.sample.level"), ::llvm::JITEvaluatedSymbol::fromPointer(&bindless_texture_3d_sample_level)},
        {_jit->mangleAndIntern("bindless.texture.2d.sample.grad"), ::llvm::JITEvaluatedSymbol::fromPointer(&bindless_texture_2d_sample_grad)},
        {_jit->mangleAndIntern("bindless.texture.3d.sample.grad"), ::llvm::JITEvaluatedSymbol::fromPointer(&bindless_texture_3d_sample_grad)}};
    if (auto error = _jit->getMainJITDylib().define(
            ::llvm::orc::absoluteSymbols(std::move(symbol_map)))) {
        ::llvm::handleAllErrors(std::move(error), [](const ::llvm::ErrorInfoBase &err) {
            LUISA_WARNING_WITH_LOCATION("LLJIT::define(): {}", err.message());
        });
        LUISA_ERROR_WITH_LOCATION("Failed to define symbols.");
    }
}

void *LLVMDevice::native_handle() const noexcept {
    return reinterpret_cast<void *>(reinterpret_cast<uint64_t>(this));
}

uint64_t LLVMDevice::create_buffer(size_t size_bytes) noexcept {
    return reinterpret_cast<uint64_t>(
        luisa::allocate_with_allocator<std::byte>(size_bytes));
}

void LLVMDevice::destroy_buffer(uint64_t handle) noexcept {
    luisa::deallocate_with_allocator(reinterpret_cast<std::byte *>(handle));
}

void *LLVMDevice::buffer_native_handle(uint64_t handle) const noexcept {
    return reinterpret_cast<void *>(handle);
}

uint64_t LLVMDevice::create_texture(PixelFormat format, uint dimension, uint width, uint height, uint depth, uint mipmap_levels) noexcept {
    auto texture = luisa::new_with_allocator<LLVMTexture>(
        pixel_format_to_storage(format), dimension,
        make_uint3(width, height, depth), mipmap_levels);
    return reinterpret_cast<uint64_t>(texture);
}

void LLVMDevice::destroy_texture(uint64_t handle) noexcept {
    luisa::delete_with_allocator(reinterpret_cast<LLVMTexture *>(handle));
}

void *LLVMDevice::texture_native_handle(uint64_t handle) const noexcept {
    return reinterpret_cast<void *>(handle);
}

uint64_t LLVMDevice::create_bindless_array(size_t size) noexcept {
    auto array = luisa::new_with_allocator<LLVMBindlessArray>(size);
    return reinterpret_cast<uint64_t>(array);
}

void LLVMDevice::destroy_bindless_array(uint64_t handle) noexcept {
    luisa::delete_with_allocator(reinterpret_cast<LLVMBindlessArray *>(handle));
}

void LLVMDevice::emplace_buffer_in_bindless_array(uint64_t array, size_t index, uint64_t handle, size_t offset_bytes) noexcept {
    reinterpret_cast<LLVMBindlessArray *>(array)->emplace_buffer(
        index, reinterpret_cast<const void *>(handle), offset_bytes);
}

void LLVMDevice::emplace_tex2d_in_bindless_array(uint64_t array, size_t index, uint64_t handle, Sampler sampler) noexcept {
    reinterpret_cast<LLVMBindlessArray *>(array)->emplace_tex2d(
        index, reinterpret_cast<const LLVMTexture *>(handle), sampler);
}

void LLVMDevice::emplace_tex3d_in_bindless_array(uint64_t array, size_t index, uint64_t handle, Sampler sampler) noexcept {
    reinterpret_cast<LLVMBindlessArray *>(array)->emplace_tex3d(
        index, reinterpret_cast<const LLVMTexture *>(handle), sampler);
}

bool LLVMDevice::is_resource_in_bindless_array(uint64_t array, uint64_t handle) const noexcept {
    return reinterpret_cast<LLVMBindlessArray *>(array)->uses_resource(handle);
}

void LLVMDevice::remove_buffer_in_bindless_array(uint64_t array, size_t index) noexcept {
    reinterpret_cast<LLVMBindlessArray *>(array)->remove_buffer(index);
}

void LLVMDevice::remove_tex2d_in_bindless_array(uint64_t array, size_t index) noexcept {
    reinterpret_cast<LLVMBindlessArray *>(array)->remove_tex2d(index);
}

void LLVMDevice::remove_tex3d_in_bindless_array(uint64_t array, size_t index) noexcept {
    reinterpret_cast<LLVMBindlessArray *>(array)->remove_tex3d(index);
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
    // FIXME: allow parallel compilation
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
    auto mesh = luisa::new_with_allocator<LLVMMesh>(
        _rtc_device, hint,
        v_buffer, v_offset, v_stride, v_count,
        t_buffer, t_offset, t_count);
    return reinterpret_cast<uint64_t>(mesh);
}

void LLVMDevice::destroy_mesh(uint64_t handle) noexcept {
    luisa::delete_with_allocator(reinterpret_cast<LLVMMesh *>(handle));
}

uint64_t LLVMDevice::create_accel(AccelUsageHint hint) noexcept {
    auto accel = luisa::new_with_allocator<LLVMAccel>(_rtc_device, hint);
    return reinterpret_cast<uint64_t>(accel);
}

void LLVMDevice::destroy_accel(uint64_t handle) noexcept {
    luisa::delete_with_allocator(reinterpret_cast<LLVMAccel *>(handle));
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

LLVMDevice::~LLVMDevice() noexcept {
    rtcReleaseDevice(_rtc_device);
}

}// namespace luisa::compute::llvm

LUISA_EXPORT_API luisa::compute::Device::Interface *create(const luisa::compute::Context &ctx, std::string_view) noexcept {
    return luisa::new_with_allocator<luisa::compute::llvm::LLVMDevice>(ctx);
}

LUISA_EXPORT_API void destroy(luisa::compute::Device::Interface *device) noexcept {
    luisa::delete_with_allocator(device);
}
