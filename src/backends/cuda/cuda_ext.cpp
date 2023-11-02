//
// Created by Hercier on 2023/4/6.
//
#include "cuda_ext.h"
#include <cuda.h>
#include "cuda_device.h"
#include "cuda_buffer.h"
#include <luisa/runtime/stream.h>
namespace luisa::compute::cuda {
void CUDADenoiserExt::_init(Stream &stream, DenoiserMode mode, DenoiserInput data, uint2 resolution) noexcept {
    _mode = mode;
    auto cuda_stream = reinterpret_cast<CUDAStream *>(stream.handle())->handle();
    auto optix_ctx = _device->handle().optix_context();
    _resolution = resolution;
    _layers.clear();
    bool has_aov = data.aovs != nullptr && data.aov_size != 0;
    if (_mode.upscale) {
        _mode.kernel_pred = 1;
    }
    if (has_aov) {
        _mode.kernel_pred = 1;
    }
    optix::DenoiserOptions options = {};
    options.guideAlbedo = data.normal && bool(*data.normal);
    options.guideNormal = data.albedo && bool(*data.albedo);
    bool guideFlow = data.flow && bool(*data.flow);
    bool guideTrust = data.flowtrust && bool(*data.flowtrust);
    auto out_scale = 1u;
    if (_mode.upscale) {
        out_scale = 2u;
    }
    optix::DenoiserModelKind model_kind = optix::DENOISER_MODEL_KIND_HDR;
    if (_mode.kernel_pred) {
        model_kind = _mode.temporal ? optix::DENOISER_MODEL_KIND_TEMPORAL_AOV : optix::DENOISER_MODEL_KIND_AOV;
    } else {
        model_kind = _mode.temporal ? optix::DENOISER_MODEL_KIND_TEMPORAL : optix::DENOISER_MODEL_KIND_HDR;
    }
    LUISA_CHECK_OPTIX(optix::api().denoiserCreate(optix_ctx, model_kind, &options, &_denoiser));

    optix::DenoiserSizes denoiser_sizes;
    LUISA_CHECK_OPTIX(optix::api().denoiserComputeMemoryResources(_denoiser, resolution.x, resolution.y, &denoiser_sizes));
    _scratch_size = static_cast<uint32_t>(denoiser_sizes.withoutOverlapScratchSizeInBytes);
    _overlap = 0u;
    if (_mode.kernel_pred) {
        LUISA_CHECK_CUDA(cuMemAllocAsync(&_avg_color, 3 * sizeof(float), cuda_stream));
    } else {
        LUISA_CHECK_CUDA(cuMemAllocAsync(&_intensity, sizeof(float), cuda_stream));
    }
    LUISA_CHECK_CUDA(cuMemAllocAsync(&_scratch, _scratch_size, cuda_stream));
    LUISA_CHECK_CUDA(cuMemAllocAsync(&_state, denoiser_sizes.stateSizeInBytes, cuda_stream));
    _state_size = static_cast<uint32_t>(denoiser_sizes.stateSizeInBytes);

    auto createOptixImage2D = [&](Buffer<float> const &input) {
        optix::Image2D res;

        res.data = reinterpret_cast<CUDABuffer *>(input.handle())->handle();
        res.width = resolution.x;
        res.height = resolution.y;
        res.rowStrideInBytes = input.size_bytes() / resolution.y;
        res.pixelStrideInBytes = input.size_bytes() / resolution.y / resolution.x;
        if (res.pixelStrideInBytes == 4 * sizeof(float))
            res.format = optix::PIXEL_FORMAT_FLOAT4;
        else {
            res.format = optix::PIXEL_FORMAT_FLOAT3;
        }
        return res;
    };
    auto create_as = [&](Buffer<float> const &input, uint scale) {
        optix::Image2D res;
        LUISA_CHECK_CUDA(cuMemAllocAsync(&res.data, input.size_bytes() * scale * scale, cuda_stream));
        res.width = resolution.x * scale;
        res.height = resolution.y * scale;
        res.rowStrideInBytes = input.size_bytes() / resolution.y * scale;
        res.pixelStrideInBytes = input.size_bytes() / resolution.y / resolution.x;
        if (res.pixelStrideInBytes == 4 * sizeof(float))
            res.format = optix::PIXEL_FORMAT_FLOAT4;
        else {
            res.format = optix::PIXEL_FORMAT_FLOAT3;
        }
        return res;
    };
    optix::DenoiserLayer layer = {};
    LUISA_ASSERT(data.beauty != nullptr && *data.beauty, "input image(beauty) is invalid!");
    layer.input = createOptixImage2D(*data.beauty);
    layer.output = create_as(*data.beauty, out_scale);
    if (options.guideAlbedo)
        _guideLayer.albedo = createOptixImage2D(*data.albedo);
    if (options.guideNormal)
        _guideLayer.normal = createOptixImage2D(*data.normal);
    if (_mode.temporal) {
        layer.previousOutput = create_as(*data.beauty, out_scale);
        if (guideFlow) {
            _guideLayer.flow = createOptixImage2D(*data.flow);
        }
        if (!_mode.upscale) {
            LUISA_CHECK_CUDA(cuMemcpyAsync(layer.previousOutput.data, reinterpret_cast<CUDABuffer *>(data.beauty->handle())->handle(),
                                           data.beauty->size_bytes(), cuda_stream));
            LUISA_CHECK_CUDA(cuMemcpyAsync(layer.output.data, reinterpret_cast<CUDABuffer *>(data.beauty->handle())->handle(),
                                           data.beauty->size_bytes(), cuda_stream));
        }      
        CUdeviceptr internalMemIn = 0;
        CUdeviceptr internalMemOut = 0;
        size_t internalSize = out_scale * out_scale* resolution.x*resolution.y * denoiser_sizes.internalGuideLayerPixelSizeInBytes;
        LUISA_CHECK_CUDA(cuMemAllocAsync(&internalMemIn, internalSize,cuda_stream));
        LUISA_CHECK_CUDA(cuMemAllocAsync(&internalMemOut, internalSize,cuda_stream));
        _guideLayer.previousOutputInternalGuideLayer.data = internalMemIn;
        _guideLayer.previousOutputInternalGuideLayer.width = out_scale * resolution.x;
        _guideLayer.previousOutputInternalGuideLayer.height = out_scale * resolution.y;
        _guideLayer.previousOutputInternalGuideLayer.pixelStrideInBytes = unsigned(denoiser_sizes.internalGuideLayerPixelSizeInBytes);
        _guideLayer.previousOutputInternalGuideLayer.rowStrideInBytes = 
            _guideLayer.previousOutputInternalGuideLayer.width * _guideLayer.previousOutputInternalGuideLayer.pixelStrideInBytes;
        _guideLayer.previousOutputInternalGuideLayer.format = optix::PIXEL_FORMAT_INTERNAL_GUIDE_LAYER;
        _guideLayer.outputInternalGuideLayer = _guideLayer.previousOutputInternalGuideLayer;
        _guideLayer.outputInternalGuideLayer.data = internalMemOut;
        if( data.flowtrust )
            _guideLayer.flowTrustworthiness = createOptixImage2D(*data.flowtrust);
    }
    _layers.push_back(layer);

    for (auto i = 0u; i < data.aov_size; i++) {
        layer = {};
        layer.input = createOptixImage2D(*data.aovs[i]);
        layer.output = create_as(*data.aovs[i], out_scale);
        if (_mode.temporal) {
            // First frame initializaton.
            layer.previousOutput = create_as(*data.aovs[i], out_scale);
            if (!_mode.upscale) {
                LUISA_CHECK_CUDA(cuMemcpyAsync(layer.previousOutput.data, reinterpret_cast<CUDABuffer *>(data.aovs[i]->handle())->handle(),
                                               data.beauty->size_bytes(), cuda_stream));
                LUISA_CHECK_CUDA(cuMemcpyAsync(layer.output.data, reinterpret_cast<CUDABuffer *>(data.aovs[i]->handle())->handle(),
                                               data.beauty->size_bytes(), cuda_stream));
            }
        }
        _layers.push_back(layer);
    }
    if (_mode.aov_diffuse_id != -1) {
        _layers[_mode.aov_diffuse_id].type = optix::DenoiserAOVType::DENOISER_AOV_TYPE_DIFFUSE;
    }
    if (_mode.aov_reflection_id != -1) {
        _layers[_mode.aov_reflection_id].type = optix::DenoiserAOVType::DENOISER_AOV_TYPE_REFLECTION;
	}
    if (_mode.aov_refract_id != -1) {
        _layers[_mode.aov_refract_id].type = optix::DenoiserAOVType::DENOISER_AOV_TYPE_REFRACTION;
    }
    if (_mode.aov_specular_id != -1) {
        _layers[_mode.aov_specular_id].type = optix::DENOISER_AOV_TYPE_SPECULAR;
    }

    LUISA_CHECK_OPTIX(optix::api().denoiserSetup(_denoiser, cuda_stream, resolution.x + 2 * _overlap,
                                                 resolution.y + 2 * _overlap, _state, _state_size, _scratch, _scratch_size));
}

void CUDADenoiserExt::_process(Stream &stream, DenoiserInput data) noexcept {
    auto cuda_stream = reinterpret_cast<CUDAStream *>(stream.handle())->handle();
    // auto optix_ctx = _device->handle().optix_context();
    optix::DenoiserParams _params = {};
    //_params.denoiseAlpha = _mode.alphamode ? optix::DENOISER_ALPHA_MODE_ALPHA_AS_AOV : optix::DENOISER_ALPHA_MODE_COPY;
    _params.hdrIntensity = _intensity;
    _params.hdrAverageColor = _avg_color;
    _params.blendFactor = 0.0f;
    _params.temporalModeUsePreviousLayers = 0;
    LUISA_ASSERT(data.beauty != nullptr && *data.beauty, "input image(beauty) is invalid!");
    _layers[0].input.data = reinterpret_cast<CUDABuffer *>(data.beauty->handle())->handle();

    if (_mode.temporal)
        _guideLayer.flow.data = reinterpret_cast<CUDABuffer *>(data.flow->handle())->handle();

    if (data.albedo)
        _guideLayer.albedo.data = reinterpret_cast<CUDABuffer *>(data.albedo->handle())->handle();

    if (data.normal)
        _guideLayer.normal.data = reinterpret_cast<CUDABuffer *>(data.normal->handle())->handle();

    if (data.flowtrust)
        _guideLayer.flowTrustworthiness.data = reinterpret_cast<CUDABuffer *>(data.flowtrust->handle())->handle();

    for (size_t i = 0; i < data.aov_size; i++)
        _layers[i + 1].input.data = reinterpret_cast<CUDABuffer *>(data.aovs[i]->handle())->handle();

    if (_mode.temporal) {
        optix::Image2D temp = _guideLayer.previousOutputInternalGuideLayer;
        _guideLayer.previousOutputInternalGuideLayer = _guideLayer.outputInternalGuideLayer;
        _guideLayer.outputInternalGuideLayer = temp;

        for (size_t i = 0; i < _layers.size(); i++) {
            temp = _layers[i].previousOutput;
            _layers[i].previousOutput = _layers[i].output;
            _layers[i].output = temp;
        }
        _params.temporalModeUsePreviousLayers = 1;
    }

    if (_intensity) {
        LUISA_CHECK_OPTIX(optix::api().denoiserComputeIntensity(
            _denoiser,
            cuda_stream,
            &_layers[0].input,
            _intensity,
            _scratch,
            _scratch_size));
    }

    if (_avg_color) {
        LUISA_CHECK_OPTIX(optix::api().denoiserComputeAverageColor(
            _denoiser,
            cuda_stream,
            &_layers[0].input,
            _avg_color,
            _scratch,
            _scratch_size));
    }
    LUISA_CHECK_OPTIX(optix::api().denoiserInvoke(
        _denoiser,
        cuda_stream,
        &_params,
        _state,
        _state_size,
        &_guideLayer,
        _layers.data(),
        static_cast<unsigned int>(_layers.size()),
        0,
        0,
        _scratch,
        _scratch_size));
}

void CUDADenoiserExt::_get_result(Stream &stream, Buffer<float> &output, int index) noexcept {
    auto cuda_stream = reinterpret_cast<CUDAStream *>(stream.handle())->handle();
    LUISA_CHECK_CUDA(cuMemcpyAsync(reinterpret_cast<CUDABuffer *>(output.handle())->handle(), _layers[index + 1].output.data, output.size_bytes(), cuda_stream));
}

void CUDADenoiserExt::_destroy(Stream &stream) noexcept {
    auto cuda_stream = reinterpret_cast<CUDAStream *>(stream.handle())->handle();
    LUISA_CHECK_OPTIX(optix::api().denoiserDestroy(_denoiser));
    LUISA_CHECK_CUDA(cuMemFreeAsync(_intensity, cuda_stream));
    LUISA_CHECK_CUDA(cuMemFreeAsync(_avg_color, cuda_stream));
    LUISA_CHECK_CUDA(cuMemFreeAsync(_scratch, cuda_stream));
    LUISA_CHECK_CUDA(cuMemFreeAsync(_state, cuda_stream));
    LUISA_CHECK_CUDA(cuMemFreeAsync(_guideLayer.previousOutputInternalGuideLayer.data,cuda_stream));
    LUISA_CHECK_CUDA(cuMemFreeAsync(_guideLayer.outputInternalGuideLayer.data,cuda_stream));
    for (auto i = 0u; i < _layers.size(); i++) {
        LUISA_CHECK_CUDA(cuMemFreeAsync(_layers[i].output.data, cuda_stream));
        LUISA_CHECK_CUDA(cuMemFreeAsync(_layers[i].previousOutput.data, cuda_stream));
    }
}

void CUDADenoiserExt::denoise(Stream &stream, uint2 resolution, Buffer<float> const &image, Buffer<float> &output,
                              Buffer<float> const &normal, Buffer<float> const &albedo, Buffer<float> **aovs, uint aov_size) noexcept {
    DenoiserMode mode{};
    mode.alphamode = 0;
    mode.kernel_pred = 0;
    mode.temporal = 0;
    mode.upscale = 0;

    DenoiserInput data{};
    data.beauty = &image;
    data.normal = &normal;
    data.albedo = &albedo;
    data.flow = nullptr;
    data.flowtrust = nullptr;
    data.aovs = aovs;
    data.aov_size = aov_size;
    _device->with_handle([&] {
        _init(stream, mode, data, resolution);
        _process(stream, data);
        _get_result(stream, output, -1);
        _destroy(stream);
    });
}
//initialize the denoiser. you should give valid data for the first pass, especially when using temporal mode.
void CUDADenoiserExt::init(Stream &stream, DenoiserMode mode, DenoiserInput data, uint2 resolution) noexcept {
    _device->with_handle([&] { _init(stream, mode, data, resolution); });
}
//process the given data.
void CUDADenoiserExt::process(Stream &stream, DenoiserInput data) noexcept {
    _device->with_handle([&] { _process(stream, data); });
}
//require for result of certain aov layer. -1 for beauty pass
void CUDADenoiserExt::get_result(Stream &stream, Buffer<float> &output, int index) noexcept {
    _device->with_handle([&] { _get_result(stream, output, index); });
}
//clear all the memory usage on device
void CUDADenoiserExt::destroy(Stream &stream) noexcept {
    _device->with_handle([&] { _destroy(stream); });
}
}// namespace luisa::compute::cuda

