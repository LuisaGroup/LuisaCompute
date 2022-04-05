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
    /**
     * @brief Dispatch a host function in the stream
     *
     * @param stream_handle handle of the stream
     * @param func host function to dispatch
     */
    void dispatch(uint64_t stream_handle, move_only_function<void()> &&func) noexcept override;
    /**
     * @brief Create a swap-chain for the window
     *
     * @param window_handle handle of the window
     * @param stream_handle handle of the stream
     * @param width frame width
     * @param height frame height
     * @param allow_hdr should support HDR content o not
     * @param back_buffer_count number of backed buffers (for multiple buffering)
     * @return handle of the swap-chain
     */
    uint64_t create_swap_chain(uint64_t window_handle, uint64_t stream_handle, uint width, uint height, bool allow_hdr, uint back_buffer_count) noexcept override;
    /**
     * @brief Destroy the swap-chain
     *
     * @param handle handle of the swap-chain
     */
    void destroy_swap_chain(uint64_t handle) noexcept override;
    /**
     * @brief Query pixel storage of the swap-chain
     * @param handle handle of the swap-chain
     * @return pixel storage of the swap-chain
     */
    PixelStorage swap_chain_pixel_storage(uint64_t handle) noexcept override;
    /**
     * @brief Present display in the stream
     *
     * @param stream_handle handle of the stream
     * @param swap_chain_handle handle of the swap-chain
     * @param image_handle handle of the 2D texture to display
     */
    void present_display_in_stream(uint64_t stream_handle, uint64_t swap_chain_handle, uint64_t image_handle) noexcept override;
};

}