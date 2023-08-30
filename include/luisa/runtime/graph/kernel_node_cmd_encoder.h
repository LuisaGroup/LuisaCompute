#pragma once
#include <luisa/runtime/rhi/command_encoder.h>

namespace luisa::compute::graph {
class LC_RUNTIME_API KernelNodeCmdEncoder : public luisa::compute::ShaderDispatchCmdEncoder {
private:
    friend class GraphBuilder;
    uint3 _dispatch_size;
public:
    // should be able to update the argument buffer
    KernelNodeCmdEncoder(size_t arg_count, size_t uniform_size) noexcept;

    template<typename T>
    KernelNodeCmdEncoder &operator<<(T data) noexcept {
        _encode_uniform(&data, sizeof(T));
        return *this;
    }

    void update_uniform(size_t i, const void *data) noexcept;

    [[nodiscard]] auto uniform(const Argument::Uniform &u) const noexcept {
        return luisa::span{_argument_buffer}.subspan(u.offset, u.size);
    }

    template<typename T>
    KernelNodeCmdEncoder &operator<<(BufferView<T> buffer) noexcept {
        _encode_buffer(buffer.handle(), buffer.offset_bytes(), buffer.size_bytes());
        return *this;
    }

    template<typename T>
    KernelNodeCmdEncoder &operator<<(Buffer<T> buffer) noexcept {
        *this << buffer.view();
        return *this;
    }

    //template<typename T>
    //KernelNodeCmdEncoder &operator<<(ImageView<T> image) noexcept {
    //    _encode_texture(image.handle(), image.level());
    //    return *this;
    //}

    //template<typename T>
    //KernelNodeCmdEncoder &operator<<(VolumeView<T> volume) noexcept {
    //    _encode_texture(volume.handle(), volume.level());
    //    return *this;
    //}

    //KernelNodeCmdEncoder &operator<<(const ByteBuffer &buffer) noexcept;

    //template<typename T>
    //KernelNodeCmdEncoder &operator<<(const Image<T> &image) noexcept {
    //    return *this << image.view();
    //}

    //template<typename T>
    //KernelNodeCmdEncoder &operator<<(const Volume<T> &volume) noexcept {
    //    return *this << volume.view();
    //}

    //template<typename T>
    //KernelNodeCmdEncoder &operator<<(const SOA<T> &soa) noexcept;

    //template<typename T>
    //KernelNodeCmdEncoder &operator<<(SOAView<T> soa) noexcept;

    //// see definition in rtx/accel.cpp
    //KernelNodeCmdEncoder &operator<<(const Accel &accel) noexcept;

    //// see definition in runtime/bindless_array.cpp
    //KernelNodeCmdEncoder &operator<<(const BindlessArray &array) noexcept;

    //// see definition in runtime/dispatch_buffer.cpp
    //KernelNodeCmdEncoder &operator<<(const IndirectDispatchBuffer &array) noexcept;

    void update_buffer(size_t i, uint64_t handle, size_t offset, size_t size) noexcept;

    auto arguments() noexcept {
        return span<Argument>{reinterpret_cast<Argument *>(_argument_buffer.data()), _argument_count};
    }

    const auto arguments() const noexcept {
        return span<const Argument>{reinterpret_cast<const Argument *>(_argument_buffer.data()), _argument_count};
    }

    uint3 dispatch_size() const noexcept { return _dispatch_size; }
};
}// namespace luisa::compute::graph