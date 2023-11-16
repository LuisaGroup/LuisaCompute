//
// Created by Hercier on 2023/4/6.
//
#pragma once
#include <luisa/core/logging.h>
#include <luisa/backends/ext/denoiser_ext.h>
#include <luisa/runtime/image.h>
#include <luisa/runtime/buffer.h>
#include <luisa/runtime/stream.h>

#include "../optix_api.h"
#include "../cuda_device.h"

#if LUISA_BACKEND_ENABLE_OIDN
#include "../../common/oidn_denoiser.h"
namespace luisa::compute::cuda {
class CUDADenoiserExt final : public DenoiserExt {
    CUDADevice *_device;
public:
    explicit CUDADenoiserExt(CUDADevice *device) noexcept : _device(device) {
    }
    luisa::shared_ptr<Denoiser> create(uint64_t stream) noexcept override;
    luisa::shared_ptr<Denoiser> create(Stream &stream) noexcept override {
        return create(stream.handle());
    }
};
}// namespace luisa::compute::cuda
#endif

namespace luisa::compute::cuda {

class [[deprecated]] CUDAOldDenoiserExt final : public OldDenoiserExt {
    CUDADevice *_device;
    std::vector<optix::DenoiserLayer> _layers;
    optix::Denoiser _denoiser = nullptr;
    uint32_t _scratch_size = 0;
    uint32_t _state_size = 0;
    uint32_t _overlap = 0u;
    CUdeviceptr _avg_color = 0;
    CUdeviceptr _scratch = 0;
    CUdeviceptr _intensity = 0;
    CUdeviceptr _state = 0;
    uint2 _resolution;
    optix::DenoiserGuideLayer _guideLayer = {};
    void _denoise(Stream &stream, uint2 resolution, Buffer<float> const &image, Buffer<float> &output,
                  Buffer<float> const &normal, Buffer<float> const &albedo, Buffer<float> *aovs, uint aov_size) noexcept;
    void _init(Stream &stream, DenoiserMode mode, DenoiserInput data, uint2 resolution) noexcept;
    void _process(Stream &stream, DenoiserInput input) noexcept;
    void _get_result(Stream &stream, Buffer<float> &output, int index) noexcept;
    void _destroy(Stream &stream) noexcept;

public:
    CUDAOldDenoiserExt(CUDADevice *device) noexcept : _device(device) {
    }
    ~CUDAOldDenoiserExt() noexcept {
    }
    void init(Stream &stream, DenoiserMode mode, DenoiserInput data, uint2 resolution) noexcept override;

    void process(Stream &stream, DenoiserInput input) noexcept override;

    void get_result(Stream &stream, Buffer<float> &output, int index) noexcept override;
    void destroy(Stream &stream) noexcept override;
    void denoise(Stream &stream, uint2 resolution, Buffer<float> const &image, Buffer<float> &output,
                 Buffer<float> const &normal, Buffer<float> const &albedo, Buffer<float> **aovs, uint aov_size) noexcept override;
};

}// namespace luisa::compute::cuda
