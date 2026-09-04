#include "metal_tile_codegen.h"
#include "../metal_device.h"
#include "../metal_compiler.h"
#include "../metal_shader.h"
#include <luisa/core/clock.h>
#include <algorithm>
#ifdef LUISA_METAL_TILE_TIRX
#include <luisa/tile/bridge/tirx/compiler.h>
#include <luisa/tile/bridge/tirx/lower.h>
#endif

namespace luisa::compute::metal {

ShaderCreationInfo MetalDevice::create_tile_kernel(const ShaderOption &option, const tile::Function &kernel,
                                                   const tile::CompileOptions &tile_options,
                                                   tile::KernelMetadata &metadata) noexcept {
    return with_autorelease_pool([&] {
        metadata = {};
        if (tile_options.xir != nullptr) {
            metadata.error = "Metal cannot use the CPU XIR execution planner";
            return ShaderCreationInfo::make_invalid();
        }
        if (option.compile_only) {
            metadata.error = "Tile compile-only archives are not supported yet";
            return ShaderCreationInfo::make_invalid();
        }
        if (tile_options.lowering == tile::Lowering::TIRX) {
#ifdef LUISA_METAL_TILE_TIRX
            Clock codegen_clock;
            auto fail = [&](luisa::string_view message) {
                metadata.error = message;
                return ShaderCreationInfo::make_invalid();
            };
            auto attached = false;
            if (auto parent = kernel.parent_module()) {
                for (auto function : parent->functions()) { attached |= function == &kernel; }
            }
            if (!attached) { return fail("Tile function must belong to its owning module"); }
            auto lowered = tile::bridge::tirx::lower(kernel);
            if (!lowered) { return fail(lowered.error); }
            auto root = kernel.body().block(0u);
            for (auto &arg : root->arguments()) {
                auto volume = arg->type().index_space()->static_volume();
                if (arg->type().scalar_type() != tile::ScalarType::FLOAT32 || !volume || *volume == 0u ||
                    *volume > INT32_MAX || *volume > SIZE_MAX / sizeof(float)) {
                    return fail("Metal TIRx Runtime currently requires nonempty, static, int32-addressable FP32 buffers");
                }
                auto usage = Usage::NONE;
                for (auto use : arg->use_list()) {
                    if (use->index() != 0u) { return fail("Unknown Tile view argument effect"); }
                    switch (use->user()->kind()) {
                        case tile::OperationKind::VIEW_LOAD: usage = static_cast<Usage>(to_underlying(usage) | to_underlying(Usage::READ)); break;
                        case tile::OperationKind::VIEW_STORE: usage = static_cast<Usage>(to_underlying(usage) | to_underlying(Usage::WRITE)); break;
                        default: return fail("Unknown Tile view argument effect");
                    }
                }
                metadata.arguments.emplace_back(tile::KernelArgument{tile::ScalarType::FLOAT32, *volume * sizeof(float), usage});
            }
            auto max_threads = static_cast<uint32_t>(_handle->maxThreadsPerThreadgroup().width);
            auto shared = _handle->maxThreadgroupMemoryLength();
            auto options = tile_options.tirx ? *tile_options.tirx : tile::bridge::tirx::CompileOptions{};
            auto matrix_capable = _handle->supportsFamily(MTL::GPUFamilyApple7);
            if (!tile_options.tirx) { options.cooperative_matrix = matrix_capable; }
            if (options.cooperative_matrix && !matrix_capable) { return fail("Selected Metal device lacks FP32 cooperative matrices"); }
            options.target = luisa::format(R"({{"kind":"metal","thread_warp_size":32,"max_num_threads":{},"max_shared_memory_per_block":{}}})", max_threads, shared);
            // SplitHostDevice's pointer ABI requires disjoint writable args.
            // The Runtime wrapper checks actual BufferView ranges at launch.
            options.noalias = true;
            metadata.disjoint_writes = true;
            if (tile_options.threads_per_group != 0u) {
                if (options.planner.threads_per_group != 0u && options.planner.threads_per_group != tile_options.threads_per_group) {
                    return fail("Conflicting TIRx and Runtime thread constraints");
                }
                options.planner.threads_per_group = tile_options.threads_per_group;
            }
            auto compiled = tile::bridge::tirx::compile_device(std::move(lowered.value), kernel.name(), options);
            if (!compiled) { return fail(compiled.error); }
            auto &artifact = compiled.artifact;
            auto language_version = MTL::LanguageVersion3_0;
            if (artifact.requires_metal4) {
                if (__builtin_available(macOS 26.0, iOS 26.0, *)) {
                    if (!matrix_capable) { return fail("TIRx MPP requires Apple GPU family 7 or newer"); }
                    language_version = MTL::LanguageVersion4_0;
                } else {
                    return fail("TIRx MPP requires macOS/iOS 26 or newer");
                }
            }
            if (artifact.format != tile::bridge::tirx::DeviceArtifact::Format::METAL_SOURCE || artifact.buffer_arguments.size() > 31u) {
                return fail("Unsupported Metal device artifact format or buffer binding capacity");
            }
            auto block_size = make_uint3(artifact.block[0], artifact.block[1], artifact.block[2]);
            auto threads = uint64_t{block_size.x} * block_size.y * block_size.z;
            if (threads > max_threads || (tile_options.threads_per_group && threads != tile_options.threads_per_group)) {
                return fail("TIRx device launch exceeds capacity or conflicts with the exact thread constraint");
            }
            for (auto i = 0u; i < 3u; i++) {
                if (artifact.grid[i] > UINT32_MAX / artifact.block[i]) { return fail("TIRx dispatch extent overflows Runtime uint32 ABI"); }
                metadata.dispatch_size[i] = artifact.grid[i] * artifact.block[i];
            }
            metadata.source = std::move(artifact.source);
            metadata.realization = luisa::format("TIRx -> Metal source -> Luisa Runtime; {} threads/group; {} group plans; direct-buffer ABI; fast_math={}; mpp={}",
                                                 threads, compiled.plans.size(), option.enable_fast_math, artifact.requires_metal4);
            auto codegen_ms = codegen_clock.toc();
            MetalShaderMetadata shader_metadata{};
            shader_metadata.block_size = block_size;
            for (auto &arg : metadata.arguments) {
                shader_metadata.argument_usages.emplace_back(arg.usage);
                shader_metadata.argument_sampled.emplace_back(0u);
            }
            Clock compile_clock;
            auto pipeline = _compiler->compile_buffer_kernel(metadata.source, artifact.entry, option, shader_metadata, metadata.error, language_version);
            if (!pipeline.entry) { return ShaderCreationInfo::make_invalid(); }
            if ((options.cooperative_matrix && pipeline.entry->threadExecutionWidth() != 32u) ||
                pipeline.entry->maxTotalThreadsPerThreadgroup() < threads || pipeline.entry->staticThreadgroupMemoryLength() > shared) {
                return fail("Compiled TIRx pipeline exceeds selected execution/resource limits");
            }
            auto compile_ms = compile_clock.toc();
            auto lines = static_cast<size_t>(std::count(metadata.source.begin(), metadata.source.end(), '\n'));
            auto shader = luisa::new_with_allocator<MetalShader>(
                this, std::move(pipeline), std::move(shader_metadata.argument_usages),
                std::move(shader_metadata.argument_sampled), luisa::vector<MetalShader::Argument>{},
                std::move(shader_metadata.format_types), block_size, shader_metadata.checksum,
                metadata.source.size(), lines, codegen_ms, compile_ms,
                MetalShaderBinding::DIRECT_BUFFERS, std::move(artifact.buffer_arguments));
            ShaderCreationInfo info{};
            info.handle = reinterpret_cast<uint64_t>(shader);
            info.native_handle = shader->pso();
            info.block_size = block_size;
            return info;
#else
            metadata.error = "Metal backend was built without the optional TIRx bridge";
            return ShaderCreationInfo::make_invalid();
#endif
        }
        if (tile_options.lowering != tile::Lowering::NATIVE || tile_options.tirx != nullptr) {
            metadata.error = "Invalid Tile lowering choice or TIRx options passed to native lowering";
            return ShaderCreationInfo::make_invalid();
        }
        if (__builtin_available(macOS 26.0, iOS 26.0, *)) {
            if (!_handle->supportsFamily(MTL::GPUFamilyApple7)) {
                metadata.error = "Native MPP requires Apple GPU family 7 or newer";
                return ShaderCreationInfo::make_invalid();
            }
            Clock codegen_clock;
            auto code = lower_tile_to_mpp(kernel, tile_options, static_cast<uint32_t>(_handle->maxThreadsPerThreadgroup().width));
            auto block_size = code.block_size;
            metadata = std::move(code.metadata);
            if (!metadata.error.empty()) { return ShaderCreationInfo::make_invalid(); }
            auto codegen_ms = codegen_clock.toc();
            MetalShaderMetadata shader_metadata{};
            shader_metadata.block_size = block_size;
            for (auto &arg : metadata.arguments) {
                shader_metadata.argument_types.emplace_back("buffer<float>");
                shader_metadata.argument_usages.emplace_back(arg.usage);
                shader_metadata.argument_sampled.emplace_back(0u);
            }
            Clock compile_clock;
            auto pipeline = _compiler->compile(metadata.source, option, shader_metadata, MTL::LanguageVersion4_0, &metadata.error);
            if (!pipeline.entry || !pipeline.indirect_entry) { return ShaderCreationInfo::make_invalid(); }
            if (pipeline.entry->threadExecutionWidth() != 32u || pipeline.entry->maxTotalThreadsPerThreadgroup() < block_size.x ||
                pipeline.entry->staticThreadgroupMemoryLength() > _handle->maxThreadgroupMemoryLength()) {
                metadata.error = "Compiled MPP pipeline exceeds the selected execution/resource limits";
                return ShaderCreationInfo::make_invalid();
            }
            auto compile_ms = compile_clock.toc();
            auto lines = static_cast<size_t>(std::count(metadata.source.begin(), metadata.source.end(), '\n'));
            auto shader = luisa::new_with_allocator<MetalShader>(
                this, std::move(pipeline), std::move(shader_metadata.argument_usages),
                std::move(shader_metadata.argument_sampled), luisa::vector<MetalShader::Argument>{},
                std::move(shader_metadata.format_types), block_size, shader_metadata.checksum,
                metadata.source.size(), lines, codegen_ms, compile_ms);
            ShaderCreationInfo info{};
            info.handle = reinterpret_cast<uint64_t>(shader);
            info.native_handle = shader->pso();
            info.block_size = block_size;
            return info;
        }
        metadata.error = "Native MPP requires macOS/iOS 26 or newer";
        return ShaderCreationInfo::make_invalid();
    });
}

}// namespace luisa::compute::metal
