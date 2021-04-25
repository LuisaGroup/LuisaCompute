//
// Created by Mike Smith on 2021/3/24.
//

#import <core/clock.h>
#import <runtime/context.h>
#import <backends/metal/metal_codegen.h>
#import <backends/metal/metal_compiler.h>
#import <backends/metal/metal_device.h>

namespace luisa::compute::metal {

MetalCompiler::KernelItem MetalCompiler::_compile(uint32_t uid) noexcept {

    LUISA_INFO("Compiling kernel #{}.", uid);

    Clock clock;

    auto ast = Function::kernel(uid);
    Codegen::Scratch scratch;
    MetalCodegen codegen{scratch};
    codegen.emit(ast);

    auto s = scratch.view();
    auto hash = xxh3_hash64(s.data(), s.size());
    LUISA_VERBOSE(
        "Generated source (hash = 0x{:016x}) for kernel #{} in {} ms:\n\n{}",
        hash, uid, clock.toc(), s);

    // try memory cache
    {
        std::scoped_lock lock{_cache_mutex};
        if (auto iter = _cache.find(hash); iter != _cache.cend()) {
            LUISA_VERBOSE_WITH_LOCATION(
                "Cache hit for kernel #{}. Compilation skipped.", uid);
            return iter->second;
        }
    }

    // compile from source
    auto src = [[NSString alloc] initWithBytes:s.data()
                                        length:s.size()
                                      encoding:NSUTF8StringEncoding];

    auto options = [[MTLCompileOptions alloc] init];
    options.fastMathEnabled = true;
    options.languageVersion = MTLLanguageVersion2_3;
    options.libraryType = MTLLibraryTypeExecutable;

    __autoreleasing NSError *error = nullptr;
    auto library = [_device->handle() newLibraryWithSource:src options:options error:&error];
    if (error != nullptr) [[unlikely]] {
        auto error_msg = [error.description cStringUsingEncoding:NSUTF8StringEncoding];
        LUISA_WARNING("Output while compiling kernel #{}: {}", uid, error_msg);
        if (library == nullptr || error.code == MTLLibraryErrorCompileFailure) [[unlikely]] {
            LUISA_ERROR_WITH_LOCATION("Failed to compile kernel #{}.", uid);
        }
        error = nullptr;
    }

    auto name = fmt::format("kernel_{}", uid);
    __autoreleasing auto objc_name = @(name.c_str());
    auto func = [library newFunctionWithName:objc_name];
    if (func == nullptr) [[unlikely]] {
        LUISA_ERROR_WITH_LOCATION(
            "Failed to find function '{}' in compiled Metal library for kernel #{}.",
            name, uid);
    }

    auto block_size = ast.block_size();
    auto desc = [[MTLComputePipelineDescriptor alloc] init];
    desc.computeFunction = func;
    desc.threadGroupSizeIsMultipleOfThreadExecutionWidth = true;
    desc.maxTotalThreadsPerThreadgroup = block_size.x * block_size.y * block_size.z;
    desc.label = objc_name;
    auto pso = [_device->handle() newComputePipelineStateWithDescriptor:desc
                                                                options:MTLPipelineOptionNone
                                                             reflection:nullptr
                                                                  error:&error];
    if (error != nullptr) [[unlikely]] {
        LUISA_ERROR_WITH_LOCATION(
            "Failed to create pipeline state object for kernel #{}: {}.",
            uid, [error.description cStringUsingEncoding:NSUTF8StringEncoding]);
    }

    MTLAutoreleasedArgument reflection;
    auto encoder = [func newArgumentEncoderWithBufferIndex:0 reflection:&reflection];
    auto members = reflection.bufferStructType.members;

    std::scoped_lock lock{_cache_mutex};
    return _cache.try_emplace(hash, pso, encoder, members).first->second;
}

MetalCompiler::KernelItem MetalCompiler::kernel(uint32_t uid) noexcept {
    {
        std::scoped_lock lock{_kernel_mutex};
        if (auto iter = _kernels.find(uid); iter != _kernels.cend()) {
            return iter->second;
        }
    }

    Clock clock;
    auto item = _compile(uid);
    LUISA_VERBOSE_WITH_LOCATION(
        "Compiled source for kernel #{} in {} ms.",
        uid, clock.toc());

    std::scoped_lock lock{_kernel_mutex};
    return _kernels.try_emplace(uid, item).first->second;
}

}
