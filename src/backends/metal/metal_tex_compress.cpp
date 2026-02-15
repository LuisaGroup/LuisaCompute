#include <luisa/core/logging.h>
#include <luisa/runtime/stream.h>
#include <luisa/runtime/buffer.h>

#include "metal_device.h"
#include "metal_stream.h"
#include "metal_buffer.h"
#include "metal_texture.h"
#include "metal_tex_compress.h"

namespace luisa::compute::metal {

namespace {

#ifdef LUISA_BIN_2_OBJ
#define LUISA_METAL_TEX_COMPRESS_DECL(name)             \
    extern "C" const uint8_t _binary_##name##_patched_metal_start[]; \
    extern "C" const uint8_t _binary_##name##_patched_metal_end[];

LUISA_METAL_TEX_COMPRESS_DECL(BC6HEncode_EncodeBlockCS)
LUISA_METAL_TEX_COMPRESS_DECL(BC6HEncode_TryModeG10CS)
LUISA_METAL_TEX_COMPRESS_DECL(BC6HEncode_TryModeLE10CS)
LUISA_METAL_TEX_COMPRESS_DECL(BC7Encode_EncodeBlockCS)
LUISA_METAL_TEX_COMPRESS_DECL(BC7Encode_TryMode02CS)
LUISA_METAL_TEX_COMPRESS_DECL(BC7Encode_TryMode137CS)
LUISA_METAL_TEX_COMPRESS_DECL(BC7Encode_TryMode456CS)
#undef LUISA_METAL_TEX_COMPRESS_DECL
#else
#include "metal_tex_compress_embedded.h"
#endif

constexpr auto metal_texture_compress_thread_group_size = 64u;
constexpr auto metal_texture_compress_format_bc6h_uf16 = 95u;
constexpr auto metal_texture_compress_format_bc7_unorm = 98u;

struct alignas(16) BCEncode_Config {
    uint g_tex_width;
    uint g_num_block_x;
    uint g_format;
    uint g_mode_id;
    uint g_start_block_id;
    uint g_num_total_blocks;
    float g_alpha_weight;
};

struct BCEncode_ArgumentBuffer {
    BCEncode_Config cbCS;
    MTL::ResourceID g_Input;
    uint64_t g_InBuff;
    uint64_t g_OutBuff;
};

void dispatch_bc_encode_shader(MTL::ComputePipelineState *shader,
                               const BCEncode_Config &config,
                               MTL::Texture *input,
                               MTL::Buffer *in_buffer, size_t in_buffer_offset,
                               MTL::Buffer *out_buffer, size_t out_buffer_offset,
                               MTL::CommandBuffer *command_buffer,
                               uint thread_group_count) noexcept {
    BCEncode_ArgumentBuffer args{.cbCS = config,
                                 .g_Input = input == nullptr ? MTL::ResourceID{} : input->gpuResourceID(),
                                 .g_InBuff = in_buffer == nullptr ? 0u : in_buffer->gpuAddress() + in_buffer_offset,
                                 .g_OutBuff = out_buffer == nullptr ? 0u : out_buffer->gpuAddress() + out_buffer_offset};
    auto command_encoder = command_buffer->computeCommandEncoder(MTL::DispatchTypeConcurrent);
    command_encoder->setComputePipelineState(shader);
    command_encoder->setBytes(&args, sizeof(BCEncode_ArgumentBuffer), 0u);
    if (input != nullptr) { command_encoder->useResource(input, MTL::ResourceUsageRead); }
    if (in_buffer != nullptr) { command_encoder->useResource(in_buffer, MTL::ResourceUsageRead); }
    if (out_buffer != nullptr) { command_encoder->useResource(out_buffer, MTL::ResourceUsageWrite); }
    command_encoder->dispatchThreadgroups(MTL::Size{thread_group_count, 1u, 1u},
                                          MTL::Size{metal_texture_compress_thread_group_size, 1u, 1u});
    command_encoder->endEncoding();
}

}// namespace

MetalTexCompressExt::MetalTexCompressExt(MetalDevice *device) noexcept : _device{device} {
    auto compile_shader = [device = device->handle()](luisa::string_view f, luisa::string_view s) noexcept {
        LUISA_VERBOSE("Compiling texture compression shader: {}.", f);
        auto source = NS::TransferPtr(NS::String::alloc()->init(
            const_cast<char *>(s.data()), s.size(), NS::UTF8StringEncoding, false));
        auto compile_options = NS::TransferPtr(MTL::CompileOptions::alloc()->init());
        compile_options->setFastMathEnabled(true);
        compile_options->setOptimizationLevel(MTL::LibraryOptimizationLevelDefault);
        compile_options->setLibraryType(MTL::LibraryTypeExecutable);
        compile_options->setMaxTotalThreadsPerThreadgroup(metal_texture_compress_thread_group_size);
        compile_options->setLanguageVersion(MTL::LanguageVersion3_0);
        NS::Error *compile_error = nullptr;
        auto library = device->newLibrary(source.get(), compile_options.get(), &compile_error);
        if (compile_error != nullptr) {
            LUISA_WARNING("Errors during texture compression shader compilation: {}.",
                          compile_error->localizedDescription()->utf8String());
        }
        LUISA_ASSERT(library, "Failed to compile texture compression shader.");
        auto func_name = NS::TransferPtr(NS::String::alloc()->init(
            const_cast<char *>(f.data()), f.size(), NS::UTF8StringEncoding, false));
        auto func = NS::TransferPtr(library->newFunction(func_name.get()));
        auto pipeline_desc = NS::TransferPtr(MTL::ComputePipelineDescriptor::alloc()->init());
        pipeline_desc->setComputeFunction(func.get());
        pipeline_desc->setMaxTotalThreadsPerThreadgroup(metal_texture_compress_thread_group_size);
        pipeline_desc->setThreadGroupSizeIsMultipleOfThreadExecutionWidth(true);
        pipeline_desc->setLabel(func_name.get());
        NS::Error *pipeline_error = nullptr;
        auto pipeline = NS::TransferPtr(device->newComputePipelineState(
            pipeline_desc.get(), MTL::PipelineOptionNone, nullptr, &pipeline_error));
        if (pipeline_error != nullptr) {
            LUISA_WARNING("Errors during texture compression pipeline creation: {}.",
                          pipeline_error->localizedDescription()->utf8String());
        }
        LUISA_ASSERT(pipeline, "Failed to create texture compression pipeline.");
        return pipeline;
    };
#ifdef LUISA_BIN_2_OBJ
#define LUISA_COMPUTE_METAL_TEX_COMPRESS_SOURCE_VIEW(name)                              \
    luisa::string_view{reinterpret_cast<const char *>(_binary_##name##_patched_metal_start), \
                       static_cast<size_t>(_binary_##name##_patched_metal_end - _binary_##name##_patched_metal_start)}
#else
#define LUISA_COMPUTE_METAL_TEX_COMPRESS_SOURCE_VIEW(name)                    \
    luisa::string_view{luisa_compute_metal_texture_compress_##name##_patched, \
                       luisa_compute_metal_texture_compress_##name##_patched_size}
#endif
    _bc7_encode_try_mode_456 = compile_shader("TryMode456CS", LUISA_COMPUTE_METAL_TEX_COMPRESS_SOURCE_VIEW(BC7Encode_TryMode456CS));
    _bc7_encode_try_mode_137 = compile_shader("TryMode137CS", LUISA_COMPUTE_METAL_TEX_COMPRESS_SOURCE_VIEW(BC7Encode_TryMode137CS));
    _bc7_encode_try_mode_02 = compile_shader("TryMode02CS", LUISA_COMPUTE_METAL_TEX_COMPRESS_SOURCE_VIEW(BC7Encode_TryMode02CS));
    _bc7_encode_encode_block = compile_shader("EncodeBlockCS", LUISA_COMPUTE_METAL_TEX_COMPRESS_SOURCE_VIEW(BC7Encode_EncodeBlockCS));
    _bc6h_encode_try_mode_g10 = compile_shader("TryModeG10CS", LUISA_COMPUTE_METAL_TEX_COMPRESS_SOURCE_VIEW(BC6HEncode_TryModeG10CS));
    _bc6h_encode_try_mode_le10 = compile_shader("TryModeLE10CS", LUISA_COMPUTE_METAL_TEX_COMPRESS_SOURCE_VIEW(BC6HEncode_TryModeLE10CS));
    _bc6h_encode_encode_block = compile_shader("EncodeBlockCS", LUISA_COMPUTE_METAL_TEX_COMPRESS_SOURCE_VIEW(BC6HEncode_EncodeBlockCS));
}

TexCompressExt::Result MetalTexCompressExt::check_builtin_shader() noexcept {
    return TexCompressExt::Result::Success;
}

TexCompressExt::Result MetalTexCompressExt::compress_bc6h(Stream &stream, const ImageView<float> &src, const BufferView<uint> &result) noexcept {
    auto blocks = luisa::max(1u, (src.size() + 3u) / 4u);
    auto total_block_count = blocks.x * blocks.y;
    auto err1_buffer_alloca = NS::TransferPtr(_device->handle()->newBuffer(total_block_count * sizeof(uint4), MTL::ResourceStorageModePrivate));
    auto err1_buffer = err1_buffer_alloca.get();
    auto err1_buffer_offset = static_cast<size_t>(0u);
    auto err2_buffer = reinterpret_cast<MetalBuffer *>(result.handle())->handle();
    auto err2_buffer_offset = result.offset_bytes();
    LUISA_DEBUG_ASSERT(result.size_bytes() >= err1_buffer->length(), "Output buffer too small for BC6H compression.");
    BCEncode_Config config{.g_tex_width = src.size().x,
                           .g_num_block_x = blocks.x,
                           .g_format = metal_texture_compress_format_bc6h_uf16,
                           .g_mode_id = 0u,
                           .g_start_block_id = 0u,
                           .g_num_total_blocks = total_block_count,
                           .g_alpha_weight = 0.f};
    auto command_buffer = reinterpret_cast<MetalStream *>(stream.handle())->queue()->commandBuffer();// will capture err1_buffer
    auto texture = reinterpret_cast<MetalTexture *>(src.handle())->handle(src.level());
    dispatch_bc_encode_shader(_bc6h_encode_try_mode_g10.get(), config, texture,
                              nullptr, 0ull, err1_buffer, err1_buffer_offset,
                              command_buffer, std::max<uint>(1u, (total_block_count + 3u) / 4u));
    for (auto i = 0u; i < 10u; i++) {
        config.g_mode_id = i;
        auto in_buffer = (i % 2u == 0u) ? err1_buffer : err2_buffer;
        auto in_buffer_offset = (i % 2u == 0u) ? err1_buffer_offset : err2_buffer_offset;
        auto out_buffer = (i % 2u == 0u) ? err2_buffer : err1_buffer;
        auto out_buffer_offset = (i % 2u == 0u) ? err2_buffer_offset : err1_buffer_offset;
        dispatch_bc_encode_shader(_bc6h_encode_try_mode_le10.get(), config, texture,
                                  in_buffer, in_buffer_offset, out_buffer, out_buffer_offset,
                                  command_buffer, std::max<uint>(1u, (total_block_count + 1u) / 2u));
    }
    dispatch_bc_encode_shader(_bc6h_encode_encode_block.get(), config, texture,
                              err1_buffer, err1_buffer_offset, err2_buffer, err2_buffer_offset,
                              command_buffer, std::max<uint>(1u, (total_block_count + 1u) / 2u));
    command_buffer->commit();
    return TexCompressExt::Result::Success;
}

TexCompressExt::Result MetalTexCompressExt::compress_bc7(Stream &stream, const ImageView<float> &src, const BufferView<uint> &result, float alpha_importance) noexcept {
    auto blocks = luisa::max(1u, (src.size() + 3u) / 4u);
    auto total_block_count = blocks.x * blocks.y;
    auto err1_buffer = reinterpret_cast<MetalBuffer *>(result.handle())->handle();
    auto err1_buffer_offset = result.offset_bytes();
    auto err2_buffer_alloc = NS::TransferPtr(_device->handle()->newBuffer(total_block_count * sizeof(uint4), MTL::ResourceStorageModePrivate));
    auto err2_buffer = err2_buffer_alloc.get();
    auto err2_buffer_offset = static_cast<size_t>(0u);
    LUISA_DEBUG_ASSERT(result.size_bytes() >= err1_buffer->length(), "Output buffer too small for BC7 compression.");
    BCEncode_Config config{.g_tex_width = src.size().x,
                           .g_num_block_x = blocks.x,
                           .g_format = metal_texture_compress_format_bc7_unorm,
                           .g_mode_id = 0u,
                           .g_start_block_id = 0u,
                           .g_num_total_blocks = total_block_count,
                           .g_alpha_weight = alpha_importance};
    auto command_buffer = reinterpret_cast<MetalStream *>(stream.handle())->queue()->commandBuffer();// will capture err2_buffer
    auto texture = reinterpret_cast<MetalTexture *>(src.handle())->handle(src.level());
    dispatch_bc_encode_shader(_bc7_encode_try_mode_456.get(), config, texture,
                              nullptr, 0ull, err1_buffer, err1_buffer_offset,
                              command_buffer, std::max<uint>(1u, (total_block_count + 3u) / 4u));
    // try mode 137
    for (auto i = 0u; i < 3u; i++) {
        constexpr uint modes[] = {1u, 3u, 7u};
        // Mode 1: err1 -> err2
        // Mode 3: err2 -> err1
        // Mode 7: err1 -> err2
        config.g_mode_id = modes[i];
        auto in_buffer = (i % 2u == 0u) ? err1_buffer : err2_buffer;
        auto in_buffer_offset = (i % 2u == 0u) ? err1_buffer_offset : err2_buffer_offset;
        auto out_buffer = (i % 2u == 0u) ? err2_buffer : err1_buffer;
        auto out_buffer_offset = (i % 2u == 0u) ? err2_buffer_offset : err1_buffer_offset;
        dispatch_bc_encode_shader(_bc7_encode_try_mode_137.get(), config, texture,
                                  in_buffer, in_buffer_offset, out_buffer, out_buffer_offset,
                                  command_buffer, total_block_count);
    }
    // try mode 02
    for (auto i = 0u; i < 2u; i++) {
        constexpr uint modes[] = {0u, 2u};
        // Mode 0: err2 -> err1
        // Mode 2: err1 -> err2
        config.g_mode_id = modes[i];
        auto in_buffer = (i % 2u == 0u) ? err2_buffer : err1_buffer;
        auto in_buffer_offset = (i % 2u == 0u) ? err2_buffer_offset : err1_buffer_offset;
        auto out_buffer = (i % 2u == 0u) ? err1_buffer : err2_buffer;
        auto out_buffer_offset = (i % 2u == 0u) ? err1_buffer_offset : err2_buffer_offset;
        dispatch_bc_encode_shader(_bc7_encode_try_mode_02.get(), config, texture,
                                  in_buffer, in_buffer_offset, out_buffer, out_buffer_offset,
                                  command_buffer, total_block_count);
    }
    dispatch_bc_encode_shader(_bc7_encode_encode_block.get(), config, texture,
                              err2_buffer, err2_buffer_offset, err1_buffer, err1_buffer_offset,
                              command_buffer, std::max<uint>(1u, (total_block_count + 3u) / 4u));
    command_buffer->commit();
    return TexCompressExt::Result::Success;
}

}// namespace luisa::compute::metal
