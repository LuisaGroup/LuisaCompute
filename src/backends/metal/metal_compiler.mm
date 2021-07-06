//
// Created by Mike Smith on 2021/3/24.
//

#import <core/clock.h>
#import <runtime/context.h>
#import <backends/metal/metal_codegen.h>
#import <backends/metal/metal_compiler.h>
#import <backends/metal/metal_device.h>

namespace luisa::compute::metal {

MetalShader MetalCompiler::compile(Function kernel) noexcept {

    auto hash_string = std::string{hash_to_string(kernel.hash())};
    LUISA_INFO("Compiling kernel #{}.", hash_string);

    Clock clock;

    Codegen::Scratch scratch;
    MetalCodegen codegen{scratch};
    codegen.emit(kernel);

    auto s = scratch.view();
    auto hash = xxh3_hash64(s.data(), s.size());
    LUISA_VERBOSE(
        "Generated source (hash = 0x{:016x}) for kernel #{} in {} ms:\n\n{}",
        hash, hash_string, clock.toc(), s);

    // try memory cache
    {
        std::scoped_lock lock{_cache_mutex};
        if (auto iter = _cache.find(hash); iter != _cache.cend()) {
            LUISA_VERBOSE_WITH_LOCATION(
                "Cache hit for kernel #{}. Compilation skipped.", hash_string);
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
        LUISA_WARNING("Output while compiling kernel #{}: {}", hash_string, error_msg);
        if (library == nullptr || error.code == MTLLibraryErrorCompileFailure) [[unlikely]] {
            LUISA_ERROR_WITH_LOCATION("Failed to compile kernel #{}.", hash_string);
        }
        error = nullptr;
    }

    auto name = fmt::format("kernel_{}", hash_string);
    __autoreleasing auto objc_name = @(name.c_str());
    auto func = [library newFunctionWithName:objc_name];
    if (func == nullptr) [[unlikely]] {
        LUISA_ERROR_WITH_LOCATION(
            "Failed to find function '{}' in compiled Metal library for kernel #{}.",
            name, hash_string);
    }

    auto block_size = kernel.block_size();
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
            hash_string, [error.description cStringUsingEncoding:NSUTF8StringEncoding]);
    }

    MTLAutoreleasedArgument reflection;
    auto encoder = [func newArgumentEncoderWithBufferIndex:0 reflection:&reflection];
    auto members = reflection.bufferStructType.members;

    // TODO: LRU
    std::scoped_lock lock{_cache_mutex};
    return _cache.try_emplace(hash, pso, encoder, members).first->second;
}

}
