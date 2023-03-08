#pragma once

#include <runtime/device.h>
#include <runtime/buffer.h>
#include <runtime/rhi/command.h>
#include <runtime/buffer.h>
#include <runtime/image.h>
#include <runtime/bindless_array.h>

namespace luisa::compute {

class LC_RUNTIME_API CustomPass {

private:
    luisa::vector<CustomCommand::ResourceBinding> _bindings;
    luisa::string _name;
    StreamTag _stream_tag;

private:
    template<typename T>
    void _emplace(luisa::string name, Usage usage, BufferView<T> buffer) noexcept {
        CustomCommand::ResourceBinding bindings;
        bindings.name = std::move(name);
        bindings.usage = usage;
        bindings.resource_view = CustomCommand::BufferView{
            .handle = buffer.handle(),
            .start_byte = buffer.offset_bytes(),
            .size_byte = buffer.size_bytes()};
        _bindings.emplace_back(std::move(bindings));
    }

    template<typename T>
    void _emplace(luisa::string name, Usage usage, const Buffer<T> &buffer) noexcept {
        _emplace(std::move(name), usage, buffer.view());
    }

    template<typename T>
    void _emplace(luisa::string name, Usage usage, ImageView<T> image) noexcept {
        CustomCommand::ResourceBinding bindings;
        bindings.name = std::move(name);
        bindings.usage = usage;
        bindings.resource_view = CustomCommand::TextureView{
            .handle = image.handle(),
            .start_mip = image.level(),
            .size_mip = 1u};
        _bindings.emplace_back(std::move(bindings));
    }

    template<typename T>
    void _emplace(luisa::string name, Usage usage, const Image<T> &image) noexcept {
        CustomCommand::ResourceBinding bindings;
        bindings.name = std::move(name);
        bindings.usage = usage;
        bindings.resource_view = CustomCommand::TextureView{
            .handle = image.handle(),
            .start_mip = 0u,
            .size_mip = image.mip_levels()};
        _bindings.emplace_back(std::move(bindings));
    }

    template<typename T>
    void _emplace(luisa::string name, Usage usage, VolumeView<T> volume) noexcept {
        CustomCommand::ResourceBinding bindings;
        bindings.name = std::move(name);
        bindings.usage = usage;
        bindings.resource_view = CustomCommand::TextureView{
            .handle = volume.handle(),
            .start_mip = volume.level(),
            .size_mip = 1u};
        _bindings.emplace_back(std::move(bindings));
    }

    template<typename T>
    void _emplace(luisa::string name, Usage usage, const Volume<T> &volume) noexcept {
        CustomCommand::ResourceBinding bindings;
        bindings.name = std::move(name);
        bindings.usage = usage;
        bindings.resource_view = CustomCommand::TextureView{
            .handle = volume.handle(),
            .start_mip = 0u,
            .size_mip = volume.mip_levels()};
        _bindings.emplace_back(std::move(bindings));
    }

    void _emplace(luisa::string name, Usage usage, const BindlessArray &array) noexcept {
        CustomCommand::ResourceBinding bindings;
        bindings.name = std::move(name);
        bindings.usage = usage;
        bindings.resource_view = CustomCommand::BindlessView{.handle = array.handle()};
        _bindings.emplace_back(std::move(bindings));
    }

    // defined in rtx/mesh.cpp
    void _emplace(luisa::string &&name, Usage usage, const Mesh &v) noexcept;

    // defined in rtx/accel.cpp
    void _emplace(luisa::string name, Usage usage, const Accel &accel) noexcept;

public:
    CustomPass(luisa::string name,
               StreamTag stream_tag,
               size_t reserved_capacity = 8u) noexcept;
    ~CustomPass() noexcept;
    luisa::unique_ptr<Command> build() &noexcept;
    luisa::unique_ptr<Command> build() &&noexcept;
    template<typename T>
    void emplace(luisa::string name, Usage usage, T &&resource) noexcept {
        _emplace(std::move(name), usage, std::forward<T>(resource));
    }
};

}// namespace luisa::compute
