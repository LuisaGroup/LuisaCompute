//
// Created by Mike Smith on 2022/2/7.
//

#pragma once

#include <embree3/rtcore_device.h>
#include <runtime/device.h>

namespace luisa::compute::ispc {

/**
 * @brief The device class of ISPC
 * 
 */
class ISPCDevice final : public Device::Interface {

private:
    RTCDevice _rtc_device;

public:
    /**
     * @brief Construct a new ISPCDevice object
     * 
     * @param ctx context
     */
    explicit ISPCDevice(const Context &ctx) noexcept;
    ~ISPCDevice() noexcept override;
    /**
     * @brief Handle of this class
     * 
     * @return void* 
     */
    void *native_handle() const noexcept override;
    /**
     * @brief Create a buffer object
     * 
     * @param size_bytes size of buffer
     * @return handle of buffer
     */
    uint64_t create_buffer(size_t size_bytes) noexcept override;
    /**
     * @brief Destrory given buffer
     * 
     * @param handle handle of buffer
     */
    void destroy_buffer(uint64_t handle) noexcept override;
    /**
     * @brief Return buffer's native handle
     * 
     * @param handle handle of buffer
     * @return void* 
     */
    void *buffer_native_handle(uint64_t handle) const noexcept override;
    /**
     * @brief Create a texture object
     * 
     * @param format texture's pixel format
     * @param dimension texture's dimension
     * @param width width
     * @param height height
     * @param depth depth
     * @param mipmap_levels mipmap levels
     * @return handle of texture
     */
    uint64_t create_texture(PixelFormat format, uint dimension, uint width, uint height, uint depth, uint mipmap_levels) noexcept override;
    /**
     * @brief Destroy given texture
     * 
     * @param handle handle of texture
     */
    void destroy_texture(uint64_t handle) noexcept override;
    /**
     * @brief Return the native handle of texture
     * 
     * @param handle handle of tgexture
     * @return void* 
     */
    void *texture_native_handle(uint64_t handle) const noexcept override;
    /**
     * @brief Create a bindless array object
     * 
     * @param size size of bindless array
     * @return handle of bindless array
     */
    uint64_t create_bindless_array(size_t size) noexcept override;
    /**
     * @brief Destroy bindless array
     * 
     * @param handle handle of bindless array
     */
    void destroy_bindless_array(uint64_t handle) noexcept override;
    /**
     * @brief Emplace a buffer in bindless array
     * 
     * @param array handle of bindless array
     * @param index place to emplace
     * @param handle handle of buffer
     * @param offset_bytes offset of buffer
     */
    void emplace_buffer_in_bindless_array(uint64_t array, size_t index, uint64_t handle, size_t offset_bytes) noexcept override;
    /**
     * @brief Emplace a 2D texture in bindless array
     * 
     * @param array handle of bindless array
     * @param index place to emplace
     * @param handle handle of 2D texture
     * @param sampler sampler of 2D texture
     */
    void emplace_tex2d_in_bindless_array(uint64_t array, size_t index, uint64_t handle, Sampler sampler) noexcept override;
    /**
     * @brief Emplace a 3D texture in bindless array
     * 
     * @param array handle of bindless array
     * @param index place to emplace
     * @param handle handle of 3D texture
     * @param sampler sampler of 3D texture
     */
    void emplace_tex3d_in_bindless_array(uint64_t array, size_t index, uint64_t handle, Sampler sampler) noexcept override;
    /**
     * @brief If buffer is in bindless array
     * 
     * @param array handle of bindless array
     * @param handle handle of buffer
     * @return true 
     * @return false 
     */
    bool is_buffer_in_bindless_array(uint64_t array, uint64_t handle) const noexcept override;
    /**
     * @brief If texture is in bindless array
     * 
     * @param array handle of bindless array
     * @param handle handle of texture
     * @return true 
     * @return false 
     */
    bool is_texture_in_bindless_array(uint64_t array, uint64_t handle) const noexcept override;
    /**
     * @brief Remove buffer in bindless array
     * 
     * @param array handle of bidnless array
     * @param index place to remove
     */
    void remove_buffer_in_bindless_array(uint64_t array, size_t index) noexcept override;
    /**
     * @brief Remove 2D texture in bindless array
     * 
     * @param array handle of bidnless array
     * @param index place to remove
     */
    void remove_tex2d_in_bindless_array(uint64_t array, size_t index) noexcept override;
    /**
     * @brief Remove 3D texture in bindless array
     * 
     * @param array handle of bidnless array
     * @param index place to remove
     */
    void remove_tex3d_in_bindless_array(uint64_t array, size_t index) noexcept override;
    /**
     * @brief Create a stream object
     * 
     * @return handle of stream
     */
    uint64_t create_stream() noexcept override;
    /**
     * @brief Destrory a stream object
     * 
     * @param handle handle of stream
     */
    void destroy_stream(uint64_t handle) noexcept override;
    /**
     * @brief Synchronize a stream
     * 
     * @param stream_handle handle of stream
     */
    void synchronize_stream(uint64_t stream_handle) noexcept override;
    /**
     * @brief Dispatch commands to stream
     * 
     * @param stream_handle handle of stream
     * @param list list of commands
     */
    void dispatch(uint64_t stream_handle, const CommandList &list) noexcept override;
    /**
     * @brief Return native handle of stream
     * 
     * @param handle handle of stream
     * @return void* 
     */
    void *stream_native_handle(uint64_t handle) const noexcept override;
    /**
     * @brief Create a shader object
     * 
     * @param kernel kernel of shader
     * @param meta_options meta options
     * @return handle of shader
     */
    uint64_t create_shader(Function kernel, std::string_view meta_options) noexcept override;
    /**
     * @brief Destroy a shader object
     * 
     * @param handle handle of shader
     */
    void destroy_shader(uint64_t handle) noexcept override;
    /**
     * @brief Create a event object
     * 
     * @return handle of event
     */
    uint64_t create_event() noexcept override;
    /**
     * @brief Destroy a event object
     * 
     * @param handle handle of event
     */
    void destroy_event(uint64_t handle) noexcept override;
    /**
     * @brief Signal an event in stream
     * 
     * @param handle handle of event
     * @param stream_handle handle of stream
     */
    void signal_event(uint64_t handle, uint64_t stream_handle) noexcept override;
    /**
     * @brief Wait an event of stream
     * 
     * @param handle handle of event
     * @param stream_handle handle of stream
     */
    void wait_event(uint64_t handle, uint64_t stream_handle) noexcept override;
    /**
     * @brief Synchronize event
     * 
     * @param handle handle of event
     */
    void synchronize_event(uint64_t handle) noexcept override;
    /**
     * @brief Create a mesh object
     * 
     * @param v_buffer handle of vertice buffer
     * @param v_offset offset of vertice buffer
     * @param v_stride stride of vertice buffer
     * @param v_count count of vertices
     * @param t_buffer handle of triangle buffer
     * @param t_offset offset of triangle buffer
     * @param t_count count of triangles
     * @param hint mesh's build hint
     * @return handle of mesh
     */
    uint64_t create_mesh(uint64_t v_buffer, size_t v_offset, size_t v_stride, size_t v_count, uint64_t t_buffer, size_t t_offset, size_t t_count, AccelBuildHint hint) noexcept override;
    /**
     * @brief Destroy a mesh object
     * 
     * @param handle handle of mesh
     */
    void destroy_mesh(uint64_t handle) noexcept override;
    /**
     * @brief Create a accel object
     * 
     * @param hint build hint
     * @return handle of accel structure
     */
    uint64_t create_accel(AccelBuildHint hint) noexcept override;
    /**
     * @brief Add an instance of mesh at the back of accel
     * 
     * @param accel handle of accel
     * @param mesh handle of mesh
     * @param transform mesh's transform
     * @param visible mesh's visibility
     */
    void emplace_back_instance_in_accel(uint64_t accel, uint64_t mesh, float4x4 transform, bool visible) noexcept override;
    /**
     * @brief Pop the latest instance
     * 
     * @param accel handle of accel
     */
    void pop_back_instance_in_accel(uint64_t accel) noexcept override;
    /**
     * @brief Set the instance in accel object
     * 
     * @param accel handle of accel
     * @param index place to set
     * @param mesh new mesh
     * @param transform new transform
     * @param visible new visibility
     */
    void set_instance_in_accel(uint64_t accel, size_t index, uint64_t mesh, float4x4 transform, bool visible) noexcept override;
    /**
     * @brief Set the instance transform in accel object
     * 
     * @param accel handle of accel
     * @param index place to set
     * @param transform new transform
     */
    void set_instance_transform_in_accel(uint64_t accel, size_t index, float4x4 transform) noexcept override;
    /**
     * @brief Set the instance visibility in accel object
     * 
     * @param accel handle of accel
     * @param index place to set
     * @param visible new visibility
     */
    void set_instance_visibility_in_accel(uint64_t accel, size_t index, bool visible) noexcept override;
    /**
     * @brief Return if buffer is in accel
     * 
     * @param accel handle of accel
     * @param buffer handle of buffer
     * @return true 
     * @return false 
     */
    bool is_buffer_in_accel(uint64_t accel, uint64_t buffer) const noexcept override;
    /**
     * @brief Return if mesh is in accel
     * 
     * @param accel handle of accel
     * @param mesh handle of mesh
     * @return true 
     * @return false 
     */
    bool is_mesh_in_accel(uint64_t accel, uint64_t mesh) const noexcept override;
    /**
     * @brief Return the vertex buffer from mesh object
     * 
     * @param mesh_handle handle of mesh
     * @return handle of vertex buffer
     */
    uint64_t get_vertex_buffer_from_mesh(uint64_t mesh_handle) const noexcept override;
    /**
     * @brief Return the triangle buffer from mesh object
     * 
     * @param mesh_handle handle of mesh
     * @return handle of triangle buffer
     */
    uint64_t get_triangle_buffer_from_mesh(uint64_t mesh_handle) const noexcept override;
    /**
     * @brief Destroy a accel object 
     * 
     * @param handle handle of accel
     */
    void destroy_accel(uint64_t handle) noexcept override;
};

}