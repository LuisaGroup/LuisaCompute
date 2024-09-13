#pragma once

#include <luisa/runtime/device.h>
#include <luisa/runtime/buffer.h>
#include <luisa/runtime/rtx/triangle.h>

namespace luisa::compute {
namespace detail {
LC_RUNTIME_API void check_mesh_vert_align(size_t v_stride, size_t dst);
}// namespace detail
class Accel;

// Mesh is buttom-level acceleration structure(BLAS) for ray-tracing, it present triangle-mesh only, custom intersection see ProceduralPrimitive
class LC_RUNTIME_API Mesh final : public Resource {

public:
    using BuildRequest = AccelBuildRequest;

private:
    uint _triangle_count{};
    uint64_t _v_buffer{};
    void *_v_buffer_native_handle{};
    size_t _v_buffer_offset_bytes{};
    size_t _v_buffer_size_bytes{};
    size_t _v_buffer_total_size_bytes{};
    size_t _v_stride{};

    uint64_t _t_buffer{};
    void *_t_buffer_native_handle{};
    size_t _t_buffer_offset_bytes{};
    size_t _t_buffer_size_bytes{};
    size_t _t_buffer_total_size_bytes{};

public:
    // WARNING: transform buffers' data without rebuild acceleration structure is undefined behavior
    template<typename VertexType>
    [[nodiscard]] BufferView<VertexType> vertex_buffer() const noexcept {
        detail::check_mesh_vert_align(_v_stride, sizeof(VertexType));
        return {_v_buffer_native_handle, _v_buffer, sizeof(VertexType), _v_buffer_offset_bytes, _v_buffer_size_bytes / sizeof(VertexType), _v_buffer_total_size_bytes / sizeof(VertexType)};
    }

    [[nodiscard]] BufferView<Triangle> triangle_buffer() const noexcept {
        return {_t_buffer_native_handle, _t_buffer, sizeof(Triangle), _t_buffer_offset_bytes, _t_buffer_size_bytes / sizeof(Triangle), _t_buffer_total_size_bytes / sizeof(Triangle)};
    }

private:
    friend class Device;

    template<typename VBuffer, typename TBuffer>
        requires is_buffer_or_view_v<VBuffer> &&
                 is_buffer_or_view_v<TBuffer> &&
                 std::same_as<buffer_element_t<TBuffer>, Triangle>
    [[nodiscard]] static ResourceCreationInfo _create_resource(
        DeviceInterface *device, const AccelOption &option,
        const VBuffer &vertex_buffer [[maybe_unused]],
        const TBuffer &triangle_buffer [[maybe_unused]]) noexcept {
        return device->create_mesh(option);
    }

private:
    template<typename VBuffer, typename TBuffer>
    Mesh(DeviceInterface *device,
         const VBuffer &vertex_buffer, size_t vertex_stride,
         const TBuffer &triangle_buffer,
         const AccelOption &option) noexcept
        : Resource{device, Resource::Tag::MESH,
                   _create_resource(device, option, vertex_buffer, triangle_buffer)} {

        BufferView vb_view{vertex_buffer};
        BufferView tri_view{triangle_buffer};
        _v_buffer = vb_view.handle();
        _v_buffer_native_handle = vb_view.native_handle();
        _v_buffer_offset_bytes = vb_view.offset_bytes();
        _v_buffer_size_bytes = vb_view.size_bytes();
        _v_buffer_total_size_bytes = vb_view.total_size_bytes();
        _v_stride = vertex_stride;
        _t_buffer = tri_view.handle();
        _t_buffer_native_handle = tri_view.native_handle();
        _t_buffer_offset_bytes = tri_view.offset_bytes();
        _t_buffer_size_bytes = tri_view.size_bytes();
        _t_buffer_total_size_bytes = tri_view.total_size_bytes();
    }

    template<typename VBuffer, typename TBuffer>
    Mesh(DeviceInterface *device, const VBuffer &vertex_buffer, const TBuffer &triangle_buffer,
         const AccelOption &option) noexcept
        : Mesh{device, vertex_buffer, vertex_buffer.stride(), triangle_buffer, option} {
    }

public:
    Mesh() noexcept = default;
    ~Mesh() noexcept override;
    Mesh(Mesh &&) noexcept = default;
    Mesh(Mesh const &) noexcept = delete;
    Mesh &operator=(Mesh &&rhs) noexcept {
        _move_from(std::move(rhs));
        return *this;
    }
    Mesh &operator=(Mesh const &) noexcept = delete;
    using Resource::operator bool;
    // build triangle based bottom-level acceleration structure
    [[nodiscard]] luisa::unique_ptr<Command> build(BuildRequest request = BuildRequest::PREFER_UPDATE) noexcept;
    [[nodiscard]] auto triangle_count() const noexcept {
        _check_is_valid();
        return _triangle_count;
    }
};

template<typename VBuffer, typename TBuffer>
Mesh Device::create_mesh(VBuffer &&vertices, TBuffer &&triangles, const AccelOption &option) noexcept {
    return this->_create<Mesh>(std::forward<VBuffer>(vertices), std::forward<TBuffer>(triangles), option);
}

template<typename VBuffer, typename TBuffer>
Mesh Device::create_mesh(VBuffer &&vertices, size_t vertex_stride, TBuffer &&triangles, const AccelOption &option) noexcept {
    return this->_create<Mesh>(std::forward<VBuffer>(vertices), vertex_stride, std::forward<TBuffer>(triangles), option);
}

}// namespace luisa::compute
