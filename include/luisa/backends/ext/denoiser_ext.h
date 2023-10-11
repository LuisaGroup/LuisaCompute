#pragma once

#include <luisa/runtime/rhi/device_interface.h>

namespace luisa::compute {

template<typename T>
class Buffer;

class Stream;

class DenoiserExt : public DeviceExtension {

public:
    struct DenoiserMode {
        bool kernel_pred = false;//using kernel prediction model, automatically use this for aov model
        bool temporal = false;   //temporal denoise mode
        bool alphamode = false;  //alpha channel denoiser mode
        bool upscale = false;    //upscaling
        int aov_refract_id = -1;//aov channel index
        int aov_specular_id = -1;
        int aov_reflection_id = -1;
        int aov_diffuse_id = -1;
    } _mode;

    struct DenoiserInput {
        const Buffer<float> *beauty = nullptr;
        const Buffer<float> *normal = nullptr;
        const Buffer<float> *albedo = nullptr;
        const Buffer<float> *flow = nullptr;     //all 0 for the first frame (if is used)
        const Buffer<float> *flowtrust = nullptr;
        Buffer<float> **aovs = nullptr;
        uint aov_size = 0;
    };

    static constexpr luisa::string_view name = "DenoiserExt";

    //A simple integration for denoising a single image.
    virtual void denoise(Stream &stream, uint2 resolution, Buffer<float> const &image, Buffer<float> &output,
                         Buffer<float> const &normal, Buffer<float> const &albedo, Buffer<float> **aovs, uint aov_size) noexcept = 0;
    //initialize the denoiser on device. if using temporal mode, data should be valid when initializing for better performance
    virtual void init(Stream &stream, DenoiserMode mode, DenoiserInput data, uint2 resolution) noexcept = 0;
    //denoise a certain image.
    virtual void process(Stream &stream, DenoiserInput input) noexcept = 0;
    //get the result for one input aov layer, index=-1 for beauty. The output should be created before sent in.
    virtual void get_result(Stream &stream, Buffer<float> &output, int index = -1) noexcept = 0;
    //clear the memory usage on device and the internal states held. You should call this before a new initialization.
    virtual void destroy(Stream &stream) noexcept = 0;
};

}// namespace luisa::compute

