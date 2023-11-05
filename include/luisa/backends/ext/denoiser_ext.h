#pragma once

#include <luisa/runtime/rhi/device_interface.h>

namespace luisa::compute {

template<typename T>
class Buffer;
template<typename T>
class BufferView;

class Stream;

class DenoiserExt : public DeviceExtension {
public:
    static constexpr luisa::string_view name = "DenoiserExt";
    enum class PrefilterMode : uint32_t {
        NONE,
        FAST,
        ACCURATE
    };
    enum class ImageFormat : uint32_t {
        FLOAT1,
        FLOAT2,
        FLOAT3,
        FLOAT4,
        HALF1,
        HALF2,
        HALF3,
        HALF4,
    };
    enum class ImageColorSpace : uint32_t {
        HDR,
        LDR_LINEAR,
        LDR_SRGB
    };
    struct Image {
        ImageFormat format = ImageFormat::FLOAT4;
        uint64_t buffer_handle = -1;
        void *device_ptr = nullptr;
        size_t offset{};
        size_t pixel_stride{};
        size_t row_stride{};
        size_t size_bytes{};
        ImageColorSpace color_space = ImageColorSpace::HDR;
        float input_scale = 1.0f;
    };
    struct Feature {
        luisa::string_view name;
        Image image;
    };

    struct DenoiserInput {
        luisa::span<const Image> inputs;
        luisa::span<const Image> outputs;
        // if prefilter is enabled, the feature images might be prefiltered **in-place**
        luisa::span<const Feature> features;
        PrefilterMode prefilter_mode = PrefilterMode::NONE;
        bool noisy_features = false;
        uint32_t width = 0u;
        uint32_t height = 0u;
    };
    class Denoiser;
    virtual luisa::shared_ptr<Denoiser> create(uint64_t stream) noexcept = 0;
    class Denoiser : public luisa::enable_shared_from_this<Denoiser> {
    public:
        virtual void init(const DenoiserInput &input) noexcept = 0;
        virtual void execute(bool async = true) noexcept = 0;
        virtual ~Denoiser() noexcept = default;
    };
};

class [[deprecated]] OldDenoiserExt : public DeviceExtension {

public:
    struct DenoiserMode {
        bool kernel_pred = false;//using kernel prediction model, automatically use this for aov model
        bool temporal = false;   //temporal denoise mode
        bool alphamode = false;  //alpha channel denoiser mode
        bool upscale = false;    //upscaling
        int aov_refract_id = -1; //aov channel index
        int aov_specular_id = -1;
        int aov_reflection_id = -1;
        int aov_diffuse_id = -1;
    } _mode;

    struct DenoiserInput {
        const Buffer<float> *beauty = nullptr;
        const Buffer<float> *normal = nullptr;
        const Buffer<float> *albedo = nullptr;
        const Buffer<float> *flow = nullptr;//all 0 for the first frame (if is used)
        const Buffer<float> *flowtrust = nullptr;
        Buffer<float> **aovs = nullptr;
        uint aov_size = 0;
    };

    static constexpr luisa::string_view name = "OldDenoiserExt";

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
