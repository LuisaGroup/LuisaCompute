#include "device.h"
#include "event.h"
#include "stream.h"
#include "accel.h"
#include "buffer.h"
#include "texture.h"
#include "bindless_array.h"
#include "mesh.h"
#include "curve.h"
#include "motion_instance.h"
#include "procedural_primitives.h"
#include "shader.h"
#include "sparse_heap.h"
#include "swap_chain.h"
#include <luisa/ast/function_builder.h>
#include "raster_ext_impl.h"
#include "dstorage_ext_impl.h"
#include "pinned_mem_impl.h"
#include <luisa/core/logging.h>
#include <luisa/runtime/rhi/command.h>
#include <luisa/backends/ext/registry.h>

namespace lc::validation {

static vstd::unordered_map<uint64_t, StreamOption> stream_options;
static std::mutex stream_mtx;

Device::Device(Context &&ctx, luisa::shared_ptr<DeviceInterface> &&native) noexcept
    : DeviceInterface{std::move(ctx)},
      _native{std::move(native)} {
    auto raster_ext = static_cast<RasterExt *>(_native->extension(RasterExt::name));
    auto dstorage_ext = static_cast<DStorageExt *>(_native->extension(DStorageExt::name));
    auto pinned_ext = static_cast<PinnedMemoryExt *>(_native->extension(PinnedMemoryExt::name));
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
    if (dstorage_ext) {
        auto dstorage_impl = new DStorageExtImpl(dstorage_ext, this);
        exts.try_emplace(
            DStorageExt::name,
            ExtPtr{
                dstorage_impl,
                detail::ext_deleter<DeviceExtension>{[](DeviceExtension *ptr) {
                    delete static_cast<DStorageExtImpl *>(ptr);
                }}});
    }
    if (pinned_ext) {
        auto pinned_ext_impl = new PinnedMemoryExtImpl(pinned_ext);
        exts.try_emplace(
            PinnedMemoryExt::name,
            ExtPtr{
                pinned_ext_impl,
                detail::ext_deleter<DeviceExtension>{[](DeviceExtension *ptr) {
                    delete static_cast<PinnedMemoryExtImpl *>(ptr);
                }}});
    }
}
BufferCreationInfo Device::create_buffer(const Type *element,
                                         size_t elem_count,
                                         void *external_memory) noexcept {
    auto buffer = _native->create_buffer(element, elem_count, external_memory);
    new Buffer{buffer.handle, 0};
    return buffer;
}
BufferCreationInfo Device::create_buffer(const ir::CArc<ir::Type> *element,
                                         size_t elem_count,
                                         void *external_memory) noexcept {
    auto buffer = _native->create_buffer(element, elem_count, external_memory);
    new Buffer{buffer.handle, 0};
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
    uint mipmap_levels, bool simultaneous_access, bool allow_raster_target) noexcept {
    auto tex = _native->create_texture(format, dimension, width, height, depth, mipmap_levels, simultaneous_access, allow_raster_target);
    new Texture{tex.handle, dimension, simultaneous_access, uint3(0, 0, 0), format};
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
void Device::add_custom_stream(uint64_t handle, StreamOption &&opt) {
    std::lock_guard lck{stream_mtx};
    stream_options.force_emplace(handle, std::move(opt));
}

// stream
ResourceCreationInfo Device::create_stream(StreamTag stream_tag) noexcept {
    auto str = _native->create_stream(stream_tag);
    new Stream(str.handle, stream_tag);
    {
        std::lock_guard lck{stream_mtx};
        auto &opt = stream_options.try_emplace(str.handle).first->second;
        switch (stream_tag) {
            case StreamTag::COMPUTE:
                opt.func = static_cast<StreamFunc>(
                    luisa::to_underlying(StreamFunc::Compute) |
                    luisa::to_underlying(StreamFunc::Copy) |
                    luisa::to_underlying(StreamFunc::Sync) |
                    luisa::to_underlying(StreamFunc::Signal) |
                    luisa::to_underlying(StreamFunc::Wait));
                break;
            case StreamTag::GRAPHICS:
                opt.func = StreamFunc::All;
                opt.supported_custom.emplace(to_underlying(CustomCommandUUID::RASTER_DRAW_SCENE));
                opt.supported_custom.emplace(to_underlying(CustomCommandUUID::RASTER_CLEAR_DEPTH));
                break;
            case StreamTag::COPY:
                opt.func = static_cast<StreamFunc>(
                    luisa::to_underlying(StreamFunc::Copy) |
                    luisa::to_underlying(StreamFunc::Sync) |
                    luisa::to_underlying(StreamFunc::Signal) |
                    luisa::to_underlying(StreamFunc::Wait));
                break;
            default:
                break;
        }
    }
    return str;
}
void Device::destroy_stream(uint64_t handle) noexcept {
    RWResource::get<Stream>(handle)->sync();
    RWResource::dispose(handle);
    {
        std::lock_guard lck{stream_mtx};
        stream_options.erase(handle);
    }
    _native->destroy_stream(handle);
}
void Device::synchronize_stream(uint64_t stream_handle) noexcept {
    check_stream(stream_handle, StreamFunc::Sync);
    RWResource::get<Stream>(stream_handle)->sync();
    _native->synchronize_stream(stream_handle);
}
void Device::dispatch(
    uint64_t stream_handle, CommandList &&list) noexcept {
    auto str = RWResource::get<Stream>(stream_handle);
    str->dispatch(_native.get(), list);
    str->check_compete();
    list._callbacks.emplace(
        list._callbacks.begin(),
        [str, executed_layer = str->executed_layer()]() {
            str->sync_layer(executed_layer);
        });
    _native->dispatch(stream_handle, std::move(list));
}

void Device::set_stream_log_callback(
    uint64_t stream_handle,
    const StreamLogCallback &callback) noexcept {
    _native->set_stream_log_callback(stream_handle, callback);
}

// swap chain
SwapchainCreationInfo Device::create_swapchain(
    const SwapchainOption &option, uint64_t stream_handle) noexcept {
    check_stream(stream_handle, StreamFunc::Swapchain);
    auto chain = _native->create_swapchain(option, stream_handle);
    new SwapChain(chain.handle);
    return chain;
}
void Device::destroy_swap_chain(uint64_t handle) noexcept {
    RWResource::dispose(handle);
    _native->destroy_swap_chain(handle);
}
void Device::present_display_in_stream(uint64_t stream_handle, uint64_t swapchain_handle, uint64_t image_handle) noexcept {
    check_stream(stream_handle, StreamFunc::Swapchain);
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
void Device::signal_event(uint64_t handle, uint64_t stream_handle, uint64_t fence) noexcept {
    check_stream(stream_handle, StreamFunc::Signal);
    auto evt = RWResource::get<Event>(handle);
    auto stream = RWResource::get<Stream>(stream_handle);
    stream->signal(evt, fence);
    _native->signal_event(handle, stream_handle, fence);
}
void Device::wait_event(uint64_t handle, uint64_t stream_handle, uint64_t fence) noexcept {
    check_stream(stream_handle, StreamFunc::Wait);
    auto evt = RWResource::get<Event>(handle);
    auto stream = RWResource::get<Stream>(stream_handle);
    stream->wait(evt, fence);
    _native->wait_event(handle, stream_handle, fence);
}
bool Device::is_event_completed(uint64_t handle, uint64_t fence) const noexcept {
    return _native->is_event_completed(handle, fence);
}
void Device::synchronize_event(uint64_t handle, uint64_t fence) noexcept {
    auto evt = RWResource::get<Event>(handle);
    evt->sync(fence);
    _native->synchronize_event(handle, fence);
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

ResourceCreationInfo Device::create_curve(const AccelOption &option) noexcept {
    auto curve = _native->create_curve(option);
    new Curve(curve.handle);
    return curve;
}

void Device::destroy_curve(uint64_t handle) noexcept {
    RWResource::dispose(handle);
    _native->destroy_curve(handle);
}

ResourceCreationInfo Device::create_motion_instance(const AccelMotionOption &option) noexcept {
    auto motion = _native->create_motion_instance(option);
    new MotionInstance(motion.handle);
    return motion;
}

void Device::destroy_motion_instance(uint64_t handle) noexcept {
    RWResource::dispose(handle);
    _native->destroy_motion_instance(handle);
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
void Device::check_stream(uint64_t stream, StreamFunc func, uint64_t custom_cmd_id) {
    auto stream_ptr = RWResource::get<RWResource>(stream);
    if (!stream_ptr) {
        LUISA_ERROR("Invalid stream.");
    }
    auto ite = stream_options.find(stream);
    if (ite == stream_options.end()) {
        LUISA_ERROR("Invalid stream.");
    }
    if (!ite->second.check_stream_func(func, custom_cmd_id)) {
        LUISA_ERROR("{} do not support function \"{}\"", stream_ptr->get_name(), luisa::to_string(func));
    }
}
VSTL_EXPORT_C void destroy(DeviceInterface *d) {
    delete d;
}
VSTL_EXPORT_C DeviceInterface *create(Context &&ctx, luisa::shared_ptr<DeviceInterface> &&native) {
    return new Device{std::move(ctx), std::move(native)};
}
SparseTextureCreationInfo Device::create_sparse_texture(
    PixelFormat format, uint dimension,
    uint width, uint height, uint depth,
    uint mipmap_levels, bool simultaneous_access) noexcept {
    auto tex = _native->create_sparse_texture(format, dimension, width, height, depth, mipmap_levels, simultaneous_access);
    new Texture{tex.handle, dimension, simultaneous_access, tex.tile_size, format};
    return tex;
}
void Device::destroy_sparse_texture(uint64_t handle) noexcept {
    RWResource::dispose(handle);
    _native->destroy_sparse_texture(handle);
}
void Device::update_sparse_resources(
    uint64_t stream_handle,
    luisa::vector<SparseUpdateTile> &&update_cmds) noexcept {
    for (auto &&i : update_cmds) {
        luisa::visit(
            [&]<typename T>(T const &t) {
                if constexpr (std::is_same_v<T, SparseTextureMapOperation>) {
                    auto tex = RWResource::get<Texture>(i.handle);
                    auto dst_byte_size = pixel_format_size(tex->format(), t.tile_count * tex->tile_size());
                    auto heap = RWResource::get<SparseHeap>(t.allocated_heap);
                    if (dst_byte_size > heap->size()) {
                        LUISA_ERROR("Map size out of range. Required size: {}, heap size: {}", dst_byte_size, heap->size());
                    }

                } else if constexpr (std::is_same_v<T, SparseBufferMapOperation>) {
                    auto heap = RWResource::get<SparseHeap>(t.allocated_heap);
                    auto buffer = RWResource::get<Buffer>(i.handle);
                    auto dst_byte_size = buffer->tile_size() * t.tile_count;
                    if (dst_byte_size > heap->size()) {
                        LUISA_ERROR("Map size out of range. Required size: {}, heap size: {}", dst_byte_size, heap->size());
                    }
                }
            },
            i.operations);
    }
    _native->update_sparse_resources(stream_handle, std::move(update_cmds));
}
SparseBufferCreationInfo Device::create_sparse_buffer(const Type *element, size_t elem_count) noexcept {
    auto buffer = _native->create_sparse_buffer(element, elem_count);
    new Buffer{buffer.handle, buffer.tile_size_bytes};
    return buffer;
}

void Device::destroy_sparse_buffer(uint64_t handle) noexcept {
    RWResource::dispose(handle);
    _native->destroy_sparse_buffer(handle);
}
ResourceCreationInfo Device::allocate_sparse_buffer_heap(size_t byte_size) noexcept {
    auto r = _native->allocate_sparse_buffer_heap(byte_size);
    new SparseHeap(r.handle, byte_size);
    return r;
}
void Device::deallocate_sparse_buffer_heap(uint64_t handle) noexcept {
    RWResource::dispose(handle);
    _native->deallocate_sparse_buffer_heap(handle);
}
ResourceCreationInfo Device::allocate_sparse_texture_heap(size_t byte_size, bool is_compressed_type) noexcept {
    auto r = _native->allocate_sparse_texture_heap(byte_size, is_compressed_type);
    new SparseHeap(r.handle, byte_size);
    return r;
}
void Device::deallocate_sparse_texture_heap(uint64_t handle) noexcept {
    RWResource::dispose(handle);
    _native->deallocate_sparse_texture_heap(handle);
}
}// namespace lc::validation
