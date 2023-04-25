//
// Created by Hercier on 2023/4/6.
//
#include <backends/cuda/optix_api.h>
#include <core/logging.h>
#include <backends/cuda/cuda_device.h>
#include <backends/ext/denoiser_ext.h>
#include <runtime/image.h>
#include <runtime/buffer.h>

namespace luisa::compute::cuda {
class CUDADenoiserExt : public DenoiserExt {
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
        Buffer<float> const &normal, Buffer<float> const &albedo, Buffer<float> *aovs, uint aov_size);
    void _init(Stream &stream, DenoiserMode mode, DenoiserInput data, uint2 resolution);
    void _process(Stream &stream, DenoiserInput input);
    void _get_result(Stream &stream, Buffer<float> &output, int index);
    void _destroy(Stream &stream);

public:
    CUDADenoiserExt(CUDADevice *device) : _device(device) {
    }
    ~CUDADenoiserExt() {
    }
    void init(Stream &stream, DenoiserMode mode, DenoiserInput data, uint2 resolution) override;
    
    void process(Stream &stream, DenoiserInput input) override;
    
    void get_result(Stream &stream, Buffer<float> &output, int index) override;
    void destroy(Stream &stream) override;
    void denoise(Stream &stream, uint2 resolution, Buffer<float> const &image, Buffer<float> &output, 
        Buffer<float> const &normal, Buffer<float> const &albedo, Buffer<float> **aovs, uint aov_size) override;
};
}// namespace luisa::compute::cuda