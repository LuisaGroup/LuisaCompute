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
static std::mutex g_global_mtx;
std::mutex &Device::global_mtx() noexcept {
    return g_global_mtx;
}
static uint64_t origin_handle(uint64_t handle) {
    if (handle != invalid_resource_handle)
        return reinterpret_cast<Resource *>(handle)->handle();
    return handle;
}
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
    std::lock_guard lck{g_global_mtx};
    auto buffer = _native->create_buffer(element, elem_count);
    buffer.handle = reinterpret_cast<uint64_t>(new Buffer(buffer.handle));
    return buffer;
}
BufferCreationInfo Device::create_buffer(const ir::CArc<ir::Type> *element, size_t elem_count) noexcept {
    std::lock_guard lck{g_global_mtx};
    auto buffer = _native->create_buffer(element, elem_count);
    buffer.handle = reinterpret_cast<uint64_t>(new Buffer(buffer.handle));
    return buffer;
}
void Device::destroy_buffer(uint64_t handle) noexcept {
    std::lock_guard lck{g_global_mtx};
    auto buffer = reinterpret_cast<Buffer *>(handle);
    _native->destroy_buffer(buffer->handle());
    delete buffer;
}

// texture
ResourceCreationInfo Device::create_texture(
    PixelFormat format, uint dimension,
    uint width, uint height, uint depth,
    uint mipmap_levels) noexcept {
    std::lock_guard lck{g_global_mtx};
    auto tex = _native->create_texture(format, dimension, width, height, depth, mipmap_levels);
    tex.handle = reinterpret_cast<uint64_t>(new Texture(tex.handle, dimension));
    return tex;
}
void Device::destroy_texture(uint64_t handle) noexcept {
    std::lock_guard lck{g_global_mtx};
    auto tex = reinterpret_cast<Texture *>(handle);
    _native->destroy_texture(tex->handle());
    delete tex;
}

// bindless array
ResourceCreationInfo Device::create_bindless_array(size_t size) noexcept {
    std::lock_guard lck{g_global_mtx};
    auto arr = _native->create_bindless_array(size);
    arr.handle = reinterpret_cast<uint64_t>(new BindlessArray(arr.handle));
    return arr;
}
void Device::destroy_bindless_array(uint64_t handle) noexcept {
    std::lock_guard lck{g_global_mtx};
    auto arr = reinterpret_cast<BindlessArray *>(handle);
    _native->destroy_bindless_array(arr->handle());
    delete arr;
}

// stream
ResourceCreationInfo Device::create_stream(StreamTag stream_tag) noexcept {
    std::lock_guard lck{g_global_mtx};
    auto str = _native->create_stream(stream_tag);
    str.handle = reinterpret_cast<uint64_t>(new Stream(str.handle, stream_tag));
    return str;
}
void Device::destroy_stream(uint64_t handle) noexcept {
    std::lock_guard lck{g_global_mtx};
    auto str = reinterpret_cast<Stream *>(handle);
    _native->destroy_stream(str->handle());
    delete str;
}
void Device::synchronize_stream(uint64_t stream_handle) noexcept {
    std::lock_guard lck{g_global_mtx};
    reinterpret_cast<Stream *>(stream_handle)->sync();
    _native->synchronize_stream(origin_handle(stream_handle));
}
void Device::dispatch(
    uint64_t stream_handle, CommandList &&list) noexcept {
    std::lock_guard lck{g_global_mtx};
    reinterpret_cast<Stream *>(stream_handle)->dispatch(_native.get(), list);
    reinterpret_cast<Stream *>(stream_handle)->check_compete();
    _native->dispatch(origin_handle(stream_handle), std::move(list));
}

// swap chain
SwapChainCreationInfo Device::create_swap_chain(
    uint64_t window_handle, uint64_t stream_handle,
    uint width, uint height, bool allow_hdr,
    bool vsync, uint back_buffer_size) noexcept {
    std::lock_guard lck{g_global_mtx};
    auto chain = _native->create_swap_chain(window_handle, origin_handle(stream_handle), width, height, allow_hdr, vsync, back_buffer_size);
    chain.handle = reinterpret_cast<uint64_t>(new SwapChain(chain.handle));
    return chain;
}
void Device::destroy_swap_chain(uint64_t handle) noexcept {
    std::lock_guard lck{g_global_mtx};
    auto chain = reinterpret_cast<SwapChain *>(handle);
    _native->destroy_swap_chain(chain->handle());
    delete chain;
}
void Device::present_display_in_stream(uint64_t stream_handle, uint64_t swapchain_handle, uint64_t image_handle) noexcept {
    std::lock_guard lck{g_global_mtx};//TODO
    auto stream = reinterpret_cast<Stream *>(stream_handle);
    stream->dispatch();
    reinterpret_cast<Texture *>(image_handle)->set(stream, Usage::READ, Range{});
    reinterpret_cast<SwapChain *>(swapchain_handle)->set(stream, Usage::WRITE, Range{});
    reinterpret_cast<Stream *>(stream_handle)->check_compete();
    _native->present_display_in_stream(origin_handle(stream_handle), origin_handle(swapchain_handle), origin_handle(image_handle));
}

// kernel
ShaderCreationInfo Device::create_shader(const ShaderOption &option, Function kernel) noexcept {
    std::lock_guard lck{g_global_mtx};
    auto old_bound = Shader::fallback_binding(const_cast<luisa::compute::detail::FunctionBuilder *>(kernel.builder())->_bound_arguments);
    auto shader = _native->create_shader(option, kernel);
    shader.handle = reinterpret_cast<uint64_t>(new Shader(shader.handle, old_bound));
    const_cast<luisa::compute::detail::FunctionBuilder *>(kernel.builder())->_bound_arguments = std::move(old_bound);
    return shader;
}
ShaderCreationInfo Device::create_shader(const ShaderOption &option, const ir::KernelModule *kernel) noexcept {
    std::lock_guard lck{g_global_mtx};
    auto shader = _native->create_shader(option, kernel);
    // TODO: IR binding test
    shader.handle = reinterpret_cast<uint64_t>(new Shader(shader.handle, {}));
    return shader;
}
ShaderCreationInfo Device::load_shader(luisa::string_view name, luisa::span<const Type *const> arg_types) noexcept {
    std::lock_guard lck{g_global_mtx};
    auto shader = _native->load_shader(name, arg_types);
    shader.handle = reinterpret_cast<uint64_t>(new Shader(shader.handle, {}));
    return shader;
}
void Device::destroy_shader(uint64_t handle) noexcept {
    std::lock_guard lck{g_global_mtx};
    auto shader = reinterpret_cast<Shader *>(handle);
    _native->destroy_shader(shader->handle());
    delete shader;
}
// event
ResourceCreationInfo Device::create_event() noexcept {
    std::lock_guard lck{g_global_mtx};
    auto evt = _native->create_event();
    evt.handle = reinterpret_cast<uint64_t>(new Event(evt.handle));
    return evt;
}
void Device::destroy_event(uint64_t handle) noexcept {
    std::lock_guard lck{g_global_mtx};
    auto evt = reinterpret_cast<Event *>(handle);
    _native->destroy_event(evt->handle());
    delete evt;
}
void Device::signal_event(uint64_t handle, uint64_t stream_handle) noexcept {
    std::lock_guard lck{g_global_mtx};
    auto evt = reinterpret_cast<Event *>(handle);
    auto stream = reinterpret_cast<Stream *>(stream_handle);
    stream->signal(evt);
    _native->signal_event(origin_handle(handle), origin_handle(stream_handle));
}
void Device::wait_event(uint64_t handle, uint64_t stream_handle) noexcept {
    std::lock_guard lck{g_global_mtx};
    auto evt = reinterpret_cast<Event *>(handle);
    auto stream = reinterpret_cast<Stream *>(stream_handle);
    stream->wait(evt);
    _native->wait_event(origin_handle(handle), origin_handle(stream_handle));
}
void Device::synchronize_event(uint64_t handle) noexcept {
    std::lock_guard lck{g_global_mtx};
    auto evt = reinterpret_cast<Event *>(handle);
    evt->sync();
    _native->synchronize_event(origin_handle(handle));
}

// accel
ResourceCreationInfo Device::create_mesh(
    const AccelOption &option) noexcept {
    std::lock_guard lck{g_global_mtx};
    auto mesh = _native->create_mesh(option);
    mesh.handle = reinterpret_cast<uint64_t>(new Mesh(mesh.handle));
    return mesh;
}
void Device::destroy_mesh(uint64_t handle) noexcept {
    std::lock_guard lck{g_global_mtx};
    auto mesh = reinterpret_cast<Mesh *>(handle);
    _native->destroy_mesh(mesh->handle());
    delete mesh;
}

ResourceCreationInfo Device::create_procedural_primitive(
    const AccelOption &option) noexcept {
    std::lock_guard lck{g_global_mtx};
    auto prim = _native->create_procedural_primitive(option);
    prim.handle = reinterpret_cast<uint64_t>(new ProceduralPrimitives(prim.handle));
    return prim;
}
void Device::destroy_procedural_primitive(uint64_t handle) noexcept {
    std::lock_guard lck{g_global_mtx};
    auto prim = reinterpret_cast<ProceduralPrimitives *>(handle);
    _native->destroy_procedural_primitive(prim->handle());
    delete prim;
}

ResourceCreationInfo Device::create_accel(const AccelOption &option) noexcept {
    std::lock_guard lck{g_global_mtx};
    auto acc = _native->create_accel(option);
    acc.handle = reinterpret_cast<uint64_t>(new Accel(acc.handle));
    return acc;
}
void Device::destroy_accel(uint64_t handle) noexcept {
    std::lock_guard lck{g_global_mtx};
    auto acc = reinterpret_cast<Accel *>(handle);
    _native->destroy_accel(acc->handle());
    delete acc;
}
void *Device::native_handle() const noexcept {
    return _native->native_handle();
}
Usage Device::shader_arg_usage(uint64_t handle, size_t index) noexcept {
    return _native->shader_arg_usage(handle, index);
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
    std::lock_guard lck{g_global_mtx};
    reinterpret_cast<Resource *>(resource_handle)->name = name;
    _native->set_name(resource_tag, origin_handle(resource_handle), name);
}
VSTL_EXPORT_C void destroy(DeviceInterface *d) {
    delete d;
}
VSTL_EXPORT_C DeviceInterface *create(Context &&ctx, luisa::shared_ptr<DeviceInterface> &&native) {
    return new Device{std::move(ctx), std::move(native)};
}
}// namespace lc::validation