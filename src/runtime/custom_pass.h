#pragma once

#include <runtime/device.h>
#include <runtime/buffer.h>
#include <runtime/command.h>
#include <runtime/buffer.h>
#include <runtime/image.h>
#include <runtime/bindless_array.h>

namespace luisa::compute {

namespace custompass_detail {

template<typename T>
struct CustomResFilter {
    static constexpr bool LegalType = false;
};

}// namespace custompass_detail

class LC_RUNTIME_API CustomPass {
private:
    template<typename T>
    friend struct custompass_detail::CustomResFilter;
    luisa::vector<CustomCommand::ResourceBinding> _bindings;
    luisa::string _name;
    StreamTag _stream_tag;

public:
    CustomPass(luisa::string &&name,
               StreamTag stream_tag,
               size_t capacity = 8) noexcept;
    ~CustomPass() noexcept;
    luisa::unique_ptr<Command> build() &noexcept;
    luisa::unique_ptr<Command> build() &&noexcept;
    template<typename T>
        requires(custompass_detail::CustomResFilter<T>::LegalType)
    void emplace(luisa::string &&name, Usage usage, T const &resource);
};

namespace custompass_detail {

template<typename T>
struct CustomResFilter<Buffer<T>> {
    static constexpr bool LegalType = true;
    static void emplace(luisa::string &&name, Usage usage, CustomPass *cmd, Buffer<T> const &v) {
        CustomCommand::ResourceBinding bindings;
        bindings.name = std::move(name);
        bindings.usage = usage;
        bindings.resource_view = CustomCommand::BufferView{
            .handle = v.handle(),
            .start_byte = 0,
            .size_byte = v.size_bytes()};
        cmd->_bindings.emplace_back(std::move(bindings));
    }
};

template<typename T>
struct CustomResFilter<BufferView<T>> {
    static constexpr bool LegalType = true;
    static void emplace(luisa::string &&name, Usage usage, CustomPass *cmd, BufferView<T> const &v) {
        CustomCommand::ResourceBinding bindings;
        bindings.name = std::move(name);
        bindings.usage = usage;
        bindings.resource_view = CustomCommand::BufferView{
            .handle = v.handle(),
            .start_byte = v.offset_bytes(),
            .size_byte = v.size_bytes()};
        cmd->_bindings.emplace_back(std::move(bindings));
    }
};

template<typename T>
struct CustomResFilter<Image<T>> {
    static constexpr bool LegalType = true;
    static void emplace(luisa::string &&name, Usage usage, CustomPass *cmd, Image<T> const &v) {
        CustomCommand::ResourceBinding bindings;
        bindings.name = std::move(name);
        bindings.usage = usage;
        bindings.resource_view = CustomCommand::TextureView{
            .handle = v.handle(),
            .start_mip = 0,
            .size_mip = v.mip_levels()};
        cmd->_bindings.emplace_back(std::move(bindings));
    }
};

template<typename T>
struct CustomResFilter<ImageView<T>> {
    static constexpr bool LegalType = true;
    static void emplace(luisa::string &&name, Usage usage, CustomPass *cmd, ImageView<T> const &v) {
        CustomCommand::ResourceBinding bindings;
        bindings.name = std::move(name);
        bindings.usage = usage;
        bindings.resource_view = CustomCommand::TextureView{
            .handle = v.handle(),
            .start_mip = v.level(),
            .size_mip = 1};
        cmd->_bindings.emplace_back(std::move(bindings));
    }
};

template<>
struct CustomResFilter<BindlessArray> {
    static constexpr bool LegalType = true;
    static void emplace(luisa::string &&name, Usage usage, CustomPass *cmd, BindlessArray const &v) {
        CustomCommand::ResourceBinding bindings;
        bindings.name = std::move(name);
        bindings.usage = usage;
        bindings.resource_view = CustomCommand::BindlessView{
            .handle = v.handle()};
        cmd->_bindings.emplace_back(std::move(bindings));
    }
};

}// namespace custompass_detail

template<typename T>
    requires(custompass_detail::CustomResFilter<T>::LegalType)
void CustomPass::emplace(luisa::string &&name, Usage usage, T const &resource) {
    custompass_detail::CustomResFilter<T>::emplace(std::move(name), usage, this, resource);
}

}// namespace luisa::compute
