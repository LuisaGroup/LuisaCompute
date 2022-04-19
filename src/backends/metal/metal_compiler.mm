//
// Created by Mike Smith on 2021/3/24.
//

#import <core/clock.h>
#import <runtime/context.h>
#import <backends/metal/metal_codegen.h>
#import <backends/metal/metal_compiler.h>
#import <backends/metal/metal_device.h>

namespace luisa::compute::metal {

MetalShader MetalCompiler::compile(
    Function kernel,
    std::string_view meta_options) noexcept {// TODO: meta-options

    auto hash = hash64(meta_options, kernel.hash());
    if (auto shader = _cache->fetch(hash)) { return *shader; }

    LUISA_INFO("Compiling kernel_{:016X}.", kernel.hash());
    Clock clock;

    static thread_local Codegen::Scratch scratch;
    scratch.clear();
    MetalCodegen codegen{scratch};
    codegen.emit(kernel);

    // compile from source
    auto source = scratch.view();
    auto src = [[NSString alloc] initWithBytes:source.data()
                                        length:source.size()
                                      encoding:NSUTF8StringEncoding];

    auto options = [[MTLCompileOptions alloc] init];
    options.fastMathEnabled = YES;
    options.libraryType = MTLLibraryTypeExecutable;

    __autoreleasing NSError *error = nullptr;
    auto library = [_device->handle() newLibraryWithSource:src options:options error:&error];
    if (error != nullptr) [[unlikely]] {
        auto error_msg = [error.description cStringUsingEncoding:NSUTF8StringEncoding];
        LUISA_WARNING("Output while compiling kernel_{:016X}: {}", kernel.hash(), error_msg);
        if (library == nullptr || error.code == MTLLibraryErrorCompileFailure) [[unlikely]] {
            LUISA_ERROR_WITH_LOCATION("Failed to compile kernel_{:016X}.", kernel.hash());
        }
        error = nullptr;
    }

    auto name = fmt::format("kernel_{:016X}", kernel.hash());
    __autoreleasing auto objc_name = @(name.c_str());
    auto func = [library newFunctionWithName:objc_name];
    if (func == nullptr) [[unlikely]] {
        LUISA_ERROR_WITH_LOCATION(
            "Failed to find function '{}' in "
            "compiled Metal library for kernel_{:016X}.",
            name, kernel.hash());
    }

    auto block_size = kernel.block_size();
    auto desc = [[MTLComputePipelineDescriptor alloc] init];
    desc.computeFunction = func;
    desc.threadGroupSizeIsMultipleOfThreadExecutionWidth = YES;
    desc.maxTotalThreadsPerThreadgroup = block_size.x * block_size.y * block_size.z;
    desc.label = objc_name;
    auto pso = [_device->handle() newComputePipelineStateWithDescriptor:desc
                                                                options:MTLPipelineOptionNone
                                                             reflection:nullptr
                                                                  error:&error];
    if (error != nullptr) [[unlikely]] {
        LUISA_ERROR_WITH_LOCATION(
            "Failed to create pipeline state object for kernel_{:016X}: {}.",
            kernel.hash(),
            [error.description cStringUsingEncoding:NSUTF8StringEncoding]);
    }

    // update cache
    MetalShader shader{pso};
    _cache->update(hash, shader);
    return shader;
}

}
