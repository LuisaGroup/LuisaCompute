#include <luisa/core/logging.h>
#include <luisa/runtime/stream.h>
#include <luisa/runtime/buffer.h>

#include "metal_device.h"
#include "metal_stream.h"
#include "metal_command_encoder.h"
#include "metal_buffer.h"
#include "metal_texture.h"
#include "metal_tex_compress.h"

namespace luisa::compute::metal {

namespace {

#include "metal_tex_compress_air/BC6HEncode_EncodeBlockCS.inc"
#include "metal_tex_compress_air/BC6HEncode_TryModeG10CS.inc"
#include "metal_tex_compress_air/BC6HEncode_TryModeLE10CS.inc"
#include "metal_tex_compress_air/BC7Encode_EncodeBlockCS.inc"
#include "metal_tex_compress_air/BC7Encode_TryMode02CS.inc"
#include "metal_tex_compress_air/BC7Encode_TryMode137CS.inc"
#include "metal_tex_compress_air/BC7Encode_TryMode456CS.inc"

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
                               MetalCommandEncoder &encoder,
                               uint thread_group_count) noexcept {
    BCEncode_ArgumentBuffer args{.cbCS = config,
                                 .g_Input = input == nullptr ? MTL::ResourceID{} : input->gpuResourceID(),
                                 .g_InBuff = in_buffer == nullptr ? 0u : in_buffer->gpuAddress() + in_buffer_offset,
                                 .g_OutBuff = out_buffer == nullptr ? 0u : out_buffer->gpuAddress() + out_buffer_offset};
    auto command_encoder = encoder.compute_encoder();
    command_encoder->setComputePipelineState(shader);
    auto table = encoder.argument_table(1u);
    table->setAddress(encoder.upload(&args, sizeof(args)), 0u);
    command_encoder->setArgumentTable(table);
    encoder.use_resource(shader);
    encoder.use_resource(input);
    encoder.use_resource(in_buffer);
    encoder.use_resource(out_buffer);
    command_encoder->dispatchThreadgroups(MTL::Size{thread_group_count, 1u, 1u},
                                          MTL::Size{metal_texture_compress_thread_group_size, 1u, 1u});
    command_encoder->endEncoding();
}

}// namespace

MetalTexCompressExt::MetalTexCompressExt(MetalDevice *device) noexcept : _device{device} {
    auto load_shader = [device](
                           luisa::string_view f,
                           const unsigned char *data,
                           size_t size) noexcept {
        LUISA_VERBOSE("Loading texture compression AIR shader: {}.", f);
        auto func_name = NS::TransferPtr(NS::String::alloc()->init(
            const_cast<char *>(f.data()), f.size(), NS::UTF8StringEncoding, false));
        auto library_data = dispatch_data_create(
            data, size, nullptr, DISPATCH_DATA_DESTRUCTOR_DEFAULT);
        NS::Error *library_error = nullptr;
        auto library = NS::TransferPtr(
            device->handle()->newLibrary(library_data, &library_error));
        dispatch_release(library_data);
        if (library_error != nullptr) {
            LUISA_WARNING("Errors while loading texture compression AIR: {}.",
                          library_error->localizedDescription()->utf8String());
        }
        LUISA_ASSERT(library, "Failed to load texture compression AIR shader.");
        library->setLabel(func_name.get());
        auto func_desc = NS::TransferPtr(
            MTL4::LibraryFunctionDescriptor::alloc()->init());
        func_desc->setLibrary(library.get());
        func_desc->setName(func_name.get());
        auto pipeline_desc = NS::TransferPtr(
            MTL4::ComputePipelineDescriptor::alloc()->init());
        pipeline_desc->setComputeFunctionDescriptor(func_desc.get());
        pipeline_desc->setMaxTotalThreadsPerThreadgroup(metal_texture_compress_thread_group_size);
        pipeline_desc->setThreadGroupSizeIsMultipleOfThreadExecutionWidth(true);
        pipeline_desc->setLabel(func_name.get());
        NS::Error *pipeline_error = nullptr;
        auto pipeline = NS::TransferPtr(
            device->metal4_compiler()->newComputePipelineState(
                pipeline_desc.get(), nullptr, &pipeline_error));
        if (pipeline_error != nullptr) {
            LUISA_WARNING("Errors during texture compression pipeline creation: {}.",
                          pipeline_error->localizedDescription()->utf8String());
        }
        LUISA_ASSERT(pipeline, "Failed to create texture compression pipeline.");
        return pipeline;
    };
#define LUISA_METAL4_TEX_COMPRESS_LIBRARY(name)             \
    luisa_compute_metal_texture_compress_##name##_metallib, \
        luisa_compute_metal_texture_compress_##name##_metallib_len
    _bc7_encode_try_mode_456 = load_shader(
        "TryMode456CS",
        LUISA_METAL4_TEX_COMPRESS_LIBRARY(BC7Encode_TryMode456CS));
    _bc7_encode_try_mode_137 = load_shader(
        "TryMode137CS",
        LUISA_METAL4_TEX_COMPRESS_LIBRARY(BC7Encode_TryMode137CS));
    _bc7_encode_try_mode_02 = load_shader(
        "TryMode02CS",
        LUISA_METAL4_TEX_COMPRESS_LIBRARY(BC7Encode_TryMode02CS));
    _bc7_encode_encode_block = load_shader(
        "EncodeBlockCS",
        LUISA_METAL4_TEX_COMPRESS_LIBRARY(BC7Encode_EncodeBlockCS));
    _bc6h_encode_try_mode_g10 = load_shader(
        "TryModeG10CS",
        LUISA_METAL4_TEX_COMPRESS_LIBRARY(BC6HEncode_TryModeG10CS));
    _bc6h_encode_try_mode_le10 = load_shader(
        "TryModeLE10CS",
        LUISA_METAL4_TEX_COMPRESS_LIBRARY(BC6HEncode_TryModeLE10CS));
    _bc6h_encode_encode_block = load_shader(
        "EncodeBlockCS",
        LUISA_METAL4_TEX_COMPRESS_LIBRARY(BC6HEncode_EncodeBlockCS));
#undef LUISA_METAL4_TEX_COMPRESS_LIBRARY
}

TexCompressExt::Result MetalTexCompressExt::check_builtin_shader() noexcept {
    return TexCompressExt::Result::Success;
}

TexCompressExt::Result MetalTexCompressExt::compress_bc6h(Stream &stream, const ImageView<float> &src, const BufferView<uint> &result) noexcept {
    auto blocks = luisa::max(1u, (src.size() + 3u) / 4u);
    auto total_block_count = blocks.x * blocks.y;
    auto err1_buffer = _device->handle()->newBuffer(
        total_block_count * sizeof(uint4),
        MTL::ResourceStorageModePrivate |
            MTL::ResourceHazardTrackingModeTracked);
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
    auto metal_stream = reinterpret_cast<MetalStream *>(stream.handle());
    MetalCommandEncoder encoder{metal_stream};
    encoder.add_callback(FunctionCallbackContext::create(
        [err1_buffer]() noexcept { err1_buffer->release(); }));
    auto texture = reinterpret_cast<MetalTextureBase *>(src.handle())->handle(src.level());
    dispatch_bc_encode_shader(_bc6h_encode_try_mode_g10.get(), config, texture,
                              nullptr, 0ull, err1_buffer, err1_buffer_offset,
                              encoder, std::max<uint>(1u, (total_block_count + 3u) / 4u));
    for (auto i = 0u; i < 10u; i++) {
        config.g_mode_id = i;
        auto in_buffer = (i % 2u == 0u) ? err1_buffer : err2_buffer;
        auto in_buffer_offset = (i % 2u == 0u) ? err1_buffer_offset : err2_buffer_offset;
        auto out_buffer = (i % 2u == 0u) ? err2_buffer : err1_buffer;
        auto out_buffer_offset = (i % 2u == 0u) ? err2_buffer_offset : err1_buffer_offset;
        dispatch_bc_encode_shader(_bc6h_encode_try_mode_le10.get(), config, texture,
                                  in_buffer, in_buffer_offset, out_buffer, out_buffer_offset,
                                  encoder, std::max<uint>(1u, (total_block_count + 1u) / 2u));
    }
    dispatch_bc_encode_shader(_bc6h_encode_encode_block.get(), config, texture,
                              err1_buffer, err1_buffer_offset, err2_buffer, err2_buffer_offset,
                              encoder, std::max<uint>(1u, (total_block_count + 1u) / 2u));
    static_cast<void>(encoder.submit({}));
    return TexCompressExt::Result::Success;
}

TexCompressExt::Result MetalTexCompressExt::compress_bc7(Stream &stream, const ImageView<float> &src, const BufferView<uint> &result, float alpha_importance) noexcept {
    auto blocks = luisa::max(1u, (src.size() + 3u) / 4u);
    auto total_block_count = blocks.x * blocks.y;
    auto err1_buffer = reinterpret_cast<MetalBuffer *>(result.handle())->handle();
    auto err1_buffer_offset = result.offset_bytes();
    auto err2_buffer = _device->handle()->newBuffer(
        total_block_count * sizeof(uint4),
        MTL::ResourceStorageModePrivate |
            MTL::ResourceHazardTrackingModeTracked);
    auto err2_buffer_offset = static_cast<size_t>(0u);
    LUISA_DEBUG_ASSERT(result.size_bytes() >= err1_buffer->length(), "Output buffer too small for BC7 compression.");
    BCEncode_Config config{.g_tex_width = src.size().x,
                           .g_num_block_x = blocks.x,
                           .g_format = metal_texture_compress_format_bc7_unorm,
                           .g_mode_id = 0u,
                           .g_start_block_id = 0u,
                           .g_num_total_blocks = total_block_count,
                           .g_alpha_weight = alpha_importance};
    auto metal_stream = reinterpret_cast<MetalStream *>(stream.handle());
    MetalCommandEncoder encoder{metal_stream};
    encoder.add_callback(FunctionCallbackContext::create(
        [err2_buffer]() noexcept { err2_buffer->release(); }));
    auto texture = reinterpret_cast<MetalTextureBase *>(src.handle())->handle(src.level());
    dispatch_bc_encode_shader(_bc7_encode_try_mode_456.get(), config, texture,
                              nullptr, 0ull, err1_buffer, err1_buffer_offset,
                              encoder, std::max<uint>(1u, (total_block_count + 3u) / 4u));
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
                                  encoder, total_block_count);
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
                                  encoder, total_block_count);
    }
    dispatch_bc_encode_shader(_bc7_encode_encode_block.get(), config, texture,
                              err2_buffer, err2_buffer_offset, err1_buffer, err1_buffer_offset,
                              encoder, std::max<uint>(1u, (total_block_count + 3u) / 4u));
    static_cast<void>(encoder.submit({}));
    return TexCompressExt::Result::Success;
}

}// namespace luisa::compute::metal
