#include "device.h"
#include "event.h"
#include "stream.h"
#include "accel.h"
#include "buffer.h"
#include "texture.h"
#include "depth_buffer.h"
#include "bindless_array.h"
#include "mesh.h"
#include "procedural_primitives.h"
#include "shader.h"
#include "swap_chain.h"
#include <ast/function_builder.h>
#include "raster_ext_impl.h"
#include <core/logging.h>
namespace lc::validation {
Device::Device(Context &&ctx, luisa::shared_ptr<DeviceInterface> &&native) noexcept
    : DeviceInterface{std::move(ctx)},
      _native{std::move(native)} {
    auto raster_ext = static_cast<RasterExt *>(_native->extension(RasterExt::name));
    constexpr size_t i = sizeof(ExtPtr);
    if (raster_ext) {
        auto raster_impl = new RasterExtImpl(raster_ext);
        exts.try_emplace(
            RasterExt::name,
            ExtPtr{
                raster_impl,
                detail::ext_deleter<DeviceExtension>{[](DeviceExtension *ptr) {
                    delete static_cast<RasterExtImpl *>(ptr);
                }}});
    }
}
BufferCreationInfo Device::create_buffer(const Type *element, size_t elem_count) noexcept {
    auto buffer = _native->create_buffer(element, elem_count);
    new Buffer{buffer.handle};
    return buffer;
}
BufferCreationInfo Device::create_buffer(const ir::CArc<ir::Type> *element, size_t elem_count) noexcept {
    auto buffer = _native->create_buffer(element, elem_count);
    new Buffer{buffer.handle};
    return buffer;
}
void Device::destroy_buffer(uint64_t handle) noexcept {
    RWResource::dispose(handle);
    _native->destroy_buffer(handle);
}

// texture
ResourceCreationInfo Device::create_texture(
    PixelFormat format, uint dimension,
    uint width, uint height, uint depth,
    uint mipmap_levels) noexcept {
    auto tex = _native->create_texture(format, dimension, width, height, depth, mipmap_levels);
    new Texture{tex.handle, dimension};
    return tex;
}
void Device::destroy_texture(uint64_t handle) noexcept {
    RWResource::dispose(handle);
    _native->destroy_texture(handle);
}

// bindless array
ResourceCreationInfo Device::create_bindless_array(size_t size) noexcept {
    auto arr = _native->create_bindless_array(size);
    // TODO: bindless range check maybe?
    new BindlessArray{arr.handle};
    return arr;
}
void Device::destroy_bindless_array(uint64_t handle) noexcept {
    RWResource::dispose(handle);
    _native->destroy_bindless_array(handle);
}

// stream
ResourceCreationInfo Device::create_stream(StreamTag stream_tag) noexcept {
    auto str = _native->create_stream(stream_tag);
    new Stream(str.handle, stream_tag);
    return str;
}
void Device::destroy_stream(uint64_t handle) noexcept {
    RWResource::dispose(handle);
    _native->destroy_stream(handle);
}
void Device::synchronize_stream(uint64_t stream_handle) noexcept {
    RWResource::get<Stream>(stream_handle)->sync();
    _native->synchronize_stream(stream_handle);
}
void Device::dispatch(
    uint64_t stream_handle, CommandList &&list) noexcept {
    auto str = RWResource::get<Stream>(stream_handle);
    str->dispatch(_native.get(), list);
    str->check_compete();
    _native->dispatch(stream_handle, std::move(list));
}

// swap chain
SwapChainCreationInfo Device::create_swap_chain(
    uint64_t window_handle, uint64_t stream_handle,
    uint width, uint height, bool allow_hdr,
    bool vsync, uint back_buffer_size) noexcept {
    auto chain = _native->create_swap_chain(window_handle, stream_handle, width, height, allow_hdr, vsync, back_buffer_size);
    new SwapChain(chain.handle);
    return chain;
}
void Device::destroy_swap_chain(uint64_t handle) noexcept {
    RWResource::dispose(handle);
    _native->destroy_swap_chain(handle);
}
void Device::present_display_in_stream(uint64_t stream_handle, uint64_t swapchain_handle, uint64_t image_handle) noexcept {
    auto stream = RWResource::get<Stream>(stream_handle);
    stream->dispatch();
    RWResource::get<Texture>(image_handle)->set(stream, Usage::READ, Range{});
    RWResource::get<SwapChain>(swapchain_handle)->set(stream, Usage::WRITE, Range{});
    RWResource::get<Stream>(stream_handle)->check_compete();
    _native->present_display_in_stream(stream_handle, swapchain_handle, image_handle);
}

// kernel
ShaderCreationInfo Device::create_shader(const ShaderOption &option, Function kernel) noexcept {
    auto shader = _native->create_shader(option, kernel);
    new Shader(shader.handle, kernel.bound_arguments());
    return shader;
}
ShaderCreationInfo Device::create_shader(const ShaderOption &option, const ir::KernelModule *kernel) noexcept {
    auto shader = _native->create_shader(option, kernel);
    // TODO: IR binding test
    // 
    return shader;
}
ShaderCreationInfo Device::load_shader(luisa::string_view name, luisa::span<const Type *const> arg_types) noexcept {
    auto shader = _native->load_shader(name, arg_types);
    new Shader(shader.handle, {});
    return shader;
}
void Device::destroy_shader(uint64_t handle) noexcept {
    RWResource::dispose(handle);
    _native->destroy_shader(handle);
}
// event
ResourceCreationInfo Device::create_event() noexcept {
    auto evt = _native->create_event();
    new Event(evt.handle);
    return evt;
}
void Device::destroy_event(uint64_t handle) noexcept {
    RWResource::dispose(handle);
    _native->destroy_event(handle);
}
void Device::signal_event(uint64_t handle, uint64_t stream_handle) noexcept {
    auto evt = RWResource::get<Event>(handle);
    auto stream = RWResource::get<Stream>(stream_handle);
    stream->signal(evt);
    _native->signal_event(handle, stream_handle);
}
void Device::wait_event(uint64_t handle, uint64_t stream_handle) noexcept {
    auto evt = RWResource::get<Event>(handle);
    auto stream = RWResource::get<Stream>(stream_handle);
    stream->wait(evt);
    _native->wait_event(handle, stream_handle);
}
void Device::synchronize_event(uint64_t handle) noexcept {
    auto evt = RWResource::get<Event>(handle);
    evt->sync();
    _native->synchronize_event(handle);
}

// accel
ResourceCreationInfo Device::create_mesh(
    const AccelOption &option) noexcept {
    auto mesh = _native->create_mesh(option);
    new Mesh(mesh.handle);
    return mesh;
}
void Device::destroy_mesh(uint64_t handle) noexcept {
    RWResource::dispose(handle);
    _native->destroy_mesh(handle);
}

ResourceCreationInfo Device::create_procedural_primitive(
    const AccelOption &option) noexcept {
    auto prim = _native->create_procedural_primitive(option);
    new ProceduralPrimitives(prim.handle);
    return prim;
}
void Device::destroy_procedural_primitive(uint64_t handle) noexcept {
    RWResource::dispose(handle);
    _native->destroy_procedural_primitive(handle);
}

ResourceCreationInfo Device::create_accel(const AccelOption &option) noexcept {
    auto acc = _native->create_accel(option);
    new Accel(acc.handle);
    return acc;
}
void Device::destroy_accel(uint64_t handle) noexcept {
    RWResource::dispose(handle);
    _native->destroy_accel(handle);
}
void *Device::native_handle() const noexcept {
    return _native->native_handle();
}
Usage Device::shader_argument_usage(uint64_t handle, size_t index) noexcept {
    return _native->shader_argument_usage(handle, index);
}
// query
luisa::string Device::query(luisa::string_view property) noexcept {
    return _native->query(property);
}
DeviceExtension *Device::extension(luisa::string_view name) noexcept {
    auto iter = exts.find(name);
    if (iter != exts.end()) return iter->second.get();
    return _native->extension(name);
}
Device::~Device() {
    exts.clear();
}
void Device::set_name(luisa::compute::Resource::Tag resource_tag, uint64_t resource_handle, luisa::string_view name) noexcept {

    RWResource::get<RWResource>(resource_handle)->name = name;
    _native->set_name(resource_tag, resource_handle, name);
}
VSTL_EXPORT_C void destroy(DeviceInterface *d) {
    delete d;
}
VSTL_EXPORT_C DeviceInterface *create(Context &&ctx, luisa::shared_ptr<DeviceInterface> &&native) {
    return new Device{std::move(ctx), std::move(native)};
}
}// namespace lc::validation