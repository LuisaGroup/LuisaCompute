#if LUISA_BACKEND_ENABLE_OIDN

#include <luisa/runtime/stream.h>

#include "metal_stream.h"
#include "metal_buffer.h"
#include "metal_denoiser.h"

namespace luisa::compute::metal {

class MetalOidnDenoiser final : public OidnDenoiser {
public:
    using OidnDenoiser::OidnDenoiser;
    void execute(bool async) noexcept override {
        auto lock = luisa::make_unique<std::shared_lock<std::shared_mutex>>(_mutex);
        exec_filters();
        if (!async) {
            _oidn_device.sync();
        } else {
            auto cmd_list = CommandList{};
            cmd_list.add_callback([lock_ = std::move(lock), this]() mutable {
                LUISA_ASSERT(lock_, "Callback called twice.");
                lock_.reset();
            });
            _device->dispatch(_stream, std::move(cmd_list));
        }
    }

protected:
    oidn::BufferRef get_buffer(const DenoiserExt::Image &img, bool read) noexcept override {
        auto buffer = reinterpret_cast<MetalBuffer *>(img.buffer_handle);
        return oidnNewSharedBufferFromMetal(_oidn_device.getHandle(), buffer->handle());
    }

public:
    void init(const DenoiserExt::DenoiserInput &input) noexcept override {
        std::unique_lock lock{_mutex};

        reset();
        auto get_format = [](DenoiserExt::ImageFormat fmt) noexcept {
            if (fmt == DenoiserExt::ImageFormat::FLOAT1) return oidn::Format::Float;
            if (fmt == DenoiserExt::ImageFormat::FLOAT2) return oidn::Format::Float2;
            if (fmt == DenoiserExt::ImageFormat::FLOAT3) return oidn::Format::Float3;
            if (fmt == DenoiserExt::ImageFormat::FLOAT4) return oidn::Format::Float4;
            if (fmt == DenoiserExt::ImageFormat::HALF1) return oidn::Format::Half;
            if (fmt == DenoiserExt::ImageFormat::HALF2) return oidn::Format::Half2;
            if (fmt == DenoiserExt::ImageFormat::HALF3) return oidn::Format::Half3;
            if (fmt == DenoiserExt::ImageFormat::HALF4) return oidn::Format::Half4;
            LUISA_ERROR_WITH_LOCATION("Invalid image format: {}.", (int)fmt);
        };
        auto set_filter_properties = [&](oidn::FilterRef &filter, const DenoiserExt::Image &image) noexcept {
            switch (image.color_space) {
                case DenoiserExt::ImageColorSpace::HDR:
                    filter.set("hdr", true);
                    break;
                case DenoiserExt::ImageColorSpace::LDR_LINEAR:
                    filter.set("hdr", false);
                    filter.set("srgb", false);
                    break;
                case DenoiserExt::ImageColorSpace::LDR_SRGB:
                    filter.set("hdr", false);
                    filter.set("srgb", true);
                    break;
                default:
                    LUISA_ERROR_WITH_LOCATION("Invalid image color space: {}.", (int)image.color_space);
            }
            if (image.input_scale != 1.0) {
                filter.set("inputScale", image.input_scale);
            }
            if (input.filter_quality == DenoiserExt::FilterQuality::FAST) {
                filter.set("filter", oidn::Quality::Balanced);
            } else if (input.filter_quality == DenoiserExt::FilterQuality::ACCURATE) {
                filter.set("filter", oidn::Quality::High);
            }
        };
        auto set_prefilter_properties = [&](oidn::FilterRef &filter) noexcept {
            if (input.prefilter_mode == DenoiserExt::PrefilterMode::NONE) return;
            if (input.prefilter_mode == DenoiserExt::PrefilterMode::FAST) {
                filter.set("quality", oidn::Quality::Balanced);
            } else if (input.prefilter_mode == DenoiserExt::PrefilterMode::ACCURATE) {
                filter.set("quality", oidn::Quality::High);
            }
        };
        bool has_albedo = false;
        bool has_normal = false;
        const DenoiserExt::Image *albedo_image = nullptr;
        const DenoiserExt::Image *normal_image = nullptr;
        if (input.prefilter_mode != DenoiserExt::PrefilterMode::NONE) {
            for (auto &f : input.features) {
                if (f.name == "albedo") {
                    LUISA_ASSERT(!has_albedo, "Albedo feature already set.");
                    LUISA_ASSERT(!_albedo_prefilter, "Albedo prefilter already set.");
                    _albedo_prefilter = _oidn_device.newFilter("RT");
                    _albedo_buffer = get_buffer(f.image, true);
                    _albedo_prefilter.setImage("color", _albedo_buffer, get_format(f.image.format), input.width, input.height, f.image.offset, f.image.pixel_stride, f.image.row_stride);
                    _albedo_prefilter.setImage("output", _albedo_buffer, get_format(f.image.format), input.width, input.height, f.image.offset, f.image.pixel_stride, f.image.row_stride);
                    _albedo_prefilter.commit();
                    has_albedo = true;
                    albedo_image = &f.image;
                    set_prefilter_properties(_albedo_prefilter);
                } else if (f.name == "normal") {
                    LUISA_ASSERT(!has_normal, "Normal feature already set.");
                    LUISA_ASSERT(!_normal_prefilter, "Normal prefilter already set.");
                    _normal_prefilter = _oidn_device.newFilter("RT");
                    _normal_buffer = get_buffer(f.image, true);
                    _normal_prefilter.setImage("color", _normal_buffer, get_format(f.image.format), input.width, input.height, f.image.offset, f.image.pixel_stride, f.image.row_stride);
                    _normal_prefilter.setImage("output", _normal_buffer, get_format(f.image.format), input.width, input.height, f.image.offset, f.image.pixel_stride, f.image.row_stride);
                    _normal_prefilter.commit();
                    has_normal = true;
                    normal_image = &f.image;
                    set_prefilter_properties(_normal_prefilter);
                } else {
                    LUISA_ERROR_WITH_LOCATION("Invalid feature name: {}.", f.name);
                }
            }
        }
        LUISA_ASSERT(!input.inputs.empty(), "Empty input.");
        LUISA_ASSERT(input.inputs.size() == input.outputs.size(), "Input/output count mismatch.");
        for (auto i = 0; i < input.inputs.size(); i++) {
            auto filter = _oidn_device.newFilter("RT");
            auto &in = input.inputs[i];
            auto &out = input.outputs[i];
            auto input_buffer = get_buffer(in, true);
            auto output_buffer = get_buffer(out, false);
            filter.setImage("color", input_buffer, get_format(in.format), input.width, input.height, in.offset, in.pixel_stride, in.row_stride);
            filter.setImage("output", output_buffer, get_format(out.format), input.width, input.height, out.offset, out.pixel_stride, out.row_stride);
            if (has_albedo) {
                filter.setImage("albedo", _albedo_buffer, get_format(albedo_image->format), input.width, input.height, albedo_image->offset, albedo_image->pixel_stride, albedo_image->row_stride);
            }
            if (has_normal) {
                filter.setImage("normal", _normal_buffer, get_format(normal_image->format), input.width, input.height, normal_image->offset, normal_image->pixel_stride, normal_image->row_stride);
            }
            set_filter_properties(filter, in);

            if (input.prefilter_mode != DenoiserExt::PrefilterMode::NONE || !input.noisy_features) {
                filter.set("cleanAux", true);
            }
            filter.commit();
            _filters.emplace_back(std::move(filter));
            _input_buffers.emplace_back(std::move(input_buffer));
            _output_buffers.emplace_back(std::move(output_buffer));
        }
    }
};

luisa::shared_ptr<DenoiserExt::Denoiser> MetalDenoiserExt::create(uint64_t stream) noexcept {
    auto metal_stream = reinterpret_cast<MetalStream *>(stream);
    auto oidn_device = oidn::newMetalDevice(metal_stream->queue());
    return luisa::make_shared<MetalOidnDenoiser>(_device, std::move(oidn_device), stream);
}

luisa::shared_ptr<DenoiserExt::Denoiser> MetalDenoiserExt::create(Stream &stream) noexcept {
    return this->create(stream.handle());
}

}// namespace luisa::compute::metal

#endif
