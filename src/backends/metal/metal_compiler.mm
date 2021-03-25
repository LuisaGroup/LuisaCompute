//
// Created by Mike Smith on 2021/3/24.
//

#include <backends/metal/metal_codegen.h>
#include <backends/metal/metal_compiler.h>

namespace luisa::compute::metal {

id<MTLComputePipelineState> MetalCompiler::_compile(uint32_t uid) noexcept {
    
    LUISA_INFO("Compiling kernel #{}.", uid);
    auto t0 = std::chrono::high_resolution_clock::now();
    compile::Codegen::Scratch scratch;
    MetalCodegen codegen{scratch};
    codegen.emit(Function::kernel(uid));
    auto t1 = std::chrono::high_resolution_clock::now();
    
    auto s = scratch.view();
    auto hash = xxh3_hash64(s.data(), s.size());
    
    using namespace std::chrono_literals;
    LUISA_VERBOSE(
        "Generated source (hash = 0x{:016x}) for kernel #{} in {} ms:\n\n{}",
        hash, uid, (t1 - t0) / 1ns * 1e-6, s);

    // try cache
    {
        std::scoped_lock lock{_cache_mutex};
        if (auto iter = std::find_if(
                _cache.cbegin(),
                _cache.cend(),
                [hash](auto &&item) noexcept { return item.hash == hash; });
            iter != _cache.cend()) { return iter->pso; }
    };

    // compile from source
    auto src = [[NSString alloc] initWithBytes:s.data()
                                        length:s.size()
                                      encoding:NSUTF8StringEncoding];

    static auto options = [] {
        auto o = [[MTLCompileOptions alloc] init];
        o.fastMathEnabled = true;
        o.languageVersion = MTLLanguageVersion2_3;
        o.libraryType = MTLLibraryTypeExecutable;
        return o;
    }();

    __autoreleasing NSError *error = nullptr;
    auto library = [_device newLibraryWithSource:src options:options error:&error];
    if (error != nullptr) {
        auto error_msg = [error.description cStringUsingEncoding:NSUTF8StringEncoding];
        LUISA_WARNING("Output while compiling kernel #{}: {}", uid, error_msg);
        if (library == nullptr || error.code == MTLLibraryErrorCompileFailure) {
            LUISA_ERROR_WITH_LOCATION("Failed to compile kernel #{}.", uid);
        }
        error = nullptr;
    }

    auto name = fmt::format("kernel_{}", uid);
    __autoreleasing auto objc_name = @(name.c_str());
    auto func = [library newFunctionWithName:objc_name];
    if (func == nullptr) {
        LUISA_ERROR_WITH_LOCATION(
            "Failed to find function '{}' in compiled Metal library for kernel #{}.",
            name, uid);
    }

    auto desc = [[MTLComputePipelineDescriptor alloc] init];
    desc.computeFunction = func;
    desc.threadGroupSizeIsMultipleOfThreadExecutionWidth = true;
    desc.label = objc_name;
    auto pso = [_device newComputePipelineStateWithDescriptor:desc
                                                      options:MTLPipelineOptionNone
                                                   reflection:nullptr
                                                        error:&error];
    if (error != nullptr) {
        LUISA_ERROR_WITH_LOCATION(
            "Failed to create pipeline state object for kernel #{}: {}.",
            uid, [error.description cStringUsingEncoding:NSUTF8StringEncoding]);
    }

    if (std::scoped_lock lock{_cache_mutex};
        std::find_if(
            _cache.cbegin(),
            _cache.cend(),
            [hash](auto &&item) noexcept { return item.hash == hash; })
        == _cache.cend()) { _cache.emplace_back(hash, pso); }

    return pso;
}

void MetalCompiler::prepare(uint32_t uid) noexcept {

    if (std::scoped_lock lock{_kernel_mutex};
        std::find_if(_kernels.cbegin(),
                     _kernels.cend(),
                     [uid](auto &&handle) noexcept { return handle.uid == uid; })
        != _kernels.cend()) { return; }

    auto kernel = std::async(std::launch::async, [uid, this] {
        auto t0 = std::chrono::high_resolution_clock::now();
        auto k = _compile(uid);
        auto t1 = std::chrono::high_resolution_clock::now();
        using namespace std::chrono_literals;
        LUISA_VERBOSE_WITH_LOCATION(
            "Compiled source for kernel #{} in {} ms.",
            uid, (t1 - t0) / 1ns * 1e-6);
        return PipelineState{k};
    });

    std::scoped_lock lock{_kernel_mutex};
    if (std::find_if(
            _kernels.cbegin(), _kernels.cend(),
            [uid](auto &&handle) noexcept { return handle.uid == uid; })
        == _kernels.cend()) {
        _kernels.emplace_back(uid, std::move(kernel));
    }
}

id<MTLComputePipelineState> MetalCompiler::kernel(uint32_t uid) noexcept {
    prepare(uid);
    std::scoped_lock lock{_kernel_mutex};
    auto iter = std::find_if(
        _kernels.begin(), _kernels.end(),
        [uid](auto &&handle) noexcept { return handle.uid == uid; });
    return iter->pso.get().handle;
}

}
