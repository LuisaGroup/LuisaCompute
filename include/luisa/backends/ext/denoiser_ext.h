#pragma once

#include <luisa/runtime/rhi/device_interface.h>
#include <luisa/core/logging.h>

namespace luisa::compute {

template<typename T>
class Buffer;
template<typename T>
class BufferView;

class Stream;

class DenoiserExt : public DeviceExtension {

protected:
    ~DenoiserExt() noexcept = default;

public:
    static constexpr luisa::string_view name = "DenoiserExt";
    enum class PrefilterMode : uint32_t {
        NONE,
        FAST,
        ACCURATE
    };
    enum class FilterQuality : uint32_t {
        DEFAULT,
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
    static constexpr size_t size(ImageFormat fmt) {
        switch (fmt) {
            case ImageFormat::FLOAT1:
                return 4;
            case ImageFormat::FLOAT2:
                return 8;
            case ImageFormat::FLOAT3:
                return 12;
            case ImageFormat::FLOAT4:
                return 16;
            case ImageFormat::HALF1:
                return 2;
            case ImageFormat::HALF2:
                return 4;
            case ImageFormat::HALF3:
                return 6;
            case ImageFormat::HALF4:
                return 8;
            default:
                return 0;
        }
    }
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
        luisa::string name;
        Image image;
    };

    struct DenoiserInput {
        luisa::vector<Image> inputs;
        luisa::vector<Image> outputs;
        // if prefilter is enabled, the feature images might be prefiltered **in-place**
        luisa::vector<Feature> features;
        PrefilterMode prefilter_mode = PrefilterMode::NONE;
        FilterQuality filter_quality = FilterQuality::DEFAULT;
        bool noisy_features = false;
        uint32_t width = 0u;
        uint32_t height = 0u;
    private:
        template<class T>
        Image buffer_to_image(const BufferView<T> &buffer, ImageFormat format, ImageColorSpace cs, float input_scale) {
            LUISA_ASSERT(size(format) <= sizeof(T), "Invalid format");
            LUISA_ASSERT(buffer.size() == width * height, "Buffer size mismatch.");
            return Image{
                format,
                buffer.handle(),
                buffer.native_handle(),
                buffer.offset_bytes(),
                buffer.stride(),
                buffer.stride() * width,
                buffer.size_bytes(),
                cs,
                input_scale};
        }
    public:
        DenoiserInput() noexcept = delete;
        DenoiserInput(uint32_t width, uint32_t height) noexcept
            : width{width}, height{height} {}
        template<class T, class U>
        void push_noisy_image(const BufferView<T> &input,
                              const BufferView<U> &output,
                              ImageFormat format,
                              ImageColorSpace cs = ImageColorSpace::HDR,
                              float input_scale = 1.0f) noexcept {
            inputs.push_back(buffer_to_image(input, format, cs, input_scale));
            outputs.push_back(buffer_to_image(output, format, cs, input_scale));
        }
        template<class T>
        void push_feature_image(const luisa::string &feature_name,
                                const BufferView<T> &feature,
                                ImageFormat format,
                                ImageColorSpace cs = ImageColorSpace::HDR,
                                float input_scale = 1.0f) noexcept {
            features.push_back(Feature{
                feature_name,
                buffer_to_image(feature, format, cs, input_scale)});
        }
    };
    class Denoiser;
    virtual luisa::shared_ptr<Denoiser> create(uint64_t stream) noexcept = 0;

    virtual luisa::shared_ptr<Denoiser> create(Stream &stream) noexcept = 0;
    class Denoiser : public luisa::enable_shared_from_this<Denoiser> {
    public:
        virtual void init(const DenoiserInput &input) noexcept = 0;
        virtual void execute(bool async) noexcept = 0;
        void execute() noexcept { execute(true); }
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
