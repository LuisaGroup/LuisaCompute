#include "oidn_denoiser.h"
#include <luisa/core/logging.h>
namespace luisa::compute {
OidnDenoiser::OidnDenoiser(DeviceInterface *device, oidn::DeviceRef &&oidn_device, uint64_t stream, bool is_cpu) noexcept
    : _device(device), _oidn_device(std::move(oidn_device)), _stream(stream), _is_cpu(is_cpu) {
    _oidn_device.setErrorFunction([](void *, oidn::Error err, const char *message) noexcept {
        switch (err) {
            case oidn::Error::None:
                break;
            case oidn::Error::Cancelled:
                LUISA_WARNING_WITH_LOCATION("OIDN denoiser cancelled: {}.", message);
                break;
            default:
                LUISA_ERROR_WITH_LOCATION("OIDN denoiser error {}: `{}`.", magic_enum::enum_name(err), message);
        }
    });
}

void OidnDenoiser::reset() noexcept {
    _filters = {};
    _albedo_prefilter = {};
    _normal_prefilter = {};
}
void OidnDenoiser::init(const DenoiserExt::DenoiserInput &input) noexcept {
    std::scoped_lock lock{_mutex};
    reset();
    auto get_shared_buffer = [&](const DenoiserExt::Image &img) noexcept {
        LUISA_ASSERT(img.buffer_handle != -1u, "Invalid buffer handle.");
        LUISA_ASSERT(img.device_ptr != nullptr, "Invalid device pointer.");
        return _oidn_device.newBuffer((byte *)img.device_ptr + img.offset, img.size_bytes);
    };
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
    };
    bool has_albedo = false;
    bool has_normal = false;
    if (input.prefilter_mode != DenoiserExt::PrefilterMode::NONE) {
        for (auto &f : input.features) {
            if (f.name == "albedo") {
                LUISA_ASSERT(!has_albedo, "Albedo feature already set.");
                LUISA_ASSERT(!_albedo_prefilter, "Albedo prefilter already set.");
                _albedo_prefilter = _oidn_device.newFilter("RT");
                _albedo_prefilter.setImage("color", get_shared_buffer(f.image), get_format(f.image.format), input.width, input.height, 0, f.image.pixel_stride, f.image.row_stride);
                _albedo_prefilter.setImage("output", get_shared_buffer(f.image), get_format(f.image.format), input.width, input.height, 0, f.image.pixel_stride, f.image.row_stride);
                _albedo_prefilter.commit();
                has_albedo = true;
            } else if (f.name == "normal") {
                LUISA_ASSERT(!has_normal, "Normal feature already set.");
                LUISA_ASSERT(!_normal_prefilter, "Normal prefilter already set.");
                _normal_prefilter = _oidn_device.newFilter("RT");
                _normal_prefilter.setImage("color", get_shared_buffer(f.image), get_format(f.image.format), input.width, input.height, 0, f.image.pixel_stride, f.image.row_stride);
                _normal_prefilter.setImage("output", get_shared_buffer(f.image), get_format(f.image.format), input.width, input.height, 0, f.image.pixel_stride, f.image.row_stride);
                _normal_prefilter.commit();
                has_normal = true;
            } else {
                LUISA_ERROR_WITH_LOCATION("Invalid feature name: {}.", f.name);
            }
        }
    }
    LUISA_ASSERT(input.inputs.size() == input.outputs.size(), "Input/output count mismatch.");
    for (auto i = 0; i < input.inputs.size(); i++) {
        auto filter = _oidn_device.newFilter("RT");
        auto &in = input.inputs[i];
        auto &out = input.outputs[i];
        filter.setImage("color", get_shared_buffer(in), get_format(in.format), input.width, input.height, 0, in.pixel_stride, in.row_stride);
        filter.setImage("output", get_shared_buffer(out), get_format(out.format), input.width, input.height, 0, out.pixel_stride, out.row_stride);
        if (has_albedo) {
            filter.setImage("albedo", get_shared_buffer(in), get_format(in.format), input.width, input.height, 0, in.pixel_stride, in.row_stride);
        }
        if (has_normal) {
            filter.setImage("normal", get_shared_buffer(in), get_format(in.format), input.width, input.height, 0, in.pixel_stride, in.row_stride);
        }
        set_filter_properties(filter, in);

        if (input.prefilter_mode != DenoiserExt::PrefilterMode::NONE || input.noisy_features) {
            filter.set("cleanAux", true);
        }
        filter.commit();
        _filters.emplace_back(std::move(filter));
    }
}
void OidnDenoiser::execute(bool async) noexcept {
    auto lock = std::unique_lock{_mutex};
    auto cmd_list = CommandList{};
    if (_albedo_prefilter) _albedo_prefilter.executeAsync();
    if (_normal_prefilter) _normal_prefilter.executeAsync();
    for (auto &f : _filters) {
        f.executeAsync();
    }
    if (!async) {
        _oidn_device.sync();
    } else {
        if (!_is_cpu) {

            cmd_list.add_callback([lock = std::move(lock), this]() mutable {
                lock.release();
            });
            _device->dispatch(_stream, std::move(cmd_list));
        } else {
            cmd_list.add_callback([lock = std::move(lock), this]() mutable {
                _oidn_device.sync();
                lock.release();
            });
            _device->dispatch(_stream, std::move(cmd_list));
        }
    }
}

}// namespace luisa::compute