//
// Created by Mike Smith on 2022/5/23.
//

#include <luisa/core/intrin.h>

#include <llvm/ExecutionEngine/Orc/LLJIT.h>
#include <llvm/MC/TargetRegistry.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/IR/Verifier.h>
#include <llvm/ExecutionEngine/ExecutionEngine.h>

#include <luisa/core/stl.h>
#include <luisa/core/logging.h>
#include <luisa/core/clock.h>

#include "fallback_stream.h"
#include "fallback_device.h"
#include "fallback_texture.h"
#include "fallback_mesh.h"
#include "fallback_curve.h"
#include "fallback_proc_prim.h"
#include "fallback_motion_instance.h"
#include "fallback_accel.h"
#include "fallback_bindless_array.h"
#include "fallback_shader.h"
#include "fallback_codegen.h"
#include "fallback_buffer.h"
#include "fallback_event.h"
#include "fallback_swapchain.h"

// extensions
#include "fallback_tex_compress.h"
#include "../common/embree_accel_archive.h"

namespace luisa::compute::fallback {

FallbackDevice::FallbackDevice(Context &&ctx, const DeviceConfig *config) noexcept
    : DeviceInterface{std::move(ctx)} {

    if (config == nullptr || config->binary_io == nullptr) {
        auto use_lmdb =
            config != nullptr && config->use_lmdb;
        _default_io =
            luisa::make_unique<DefaultBinaryIO>(
                context(), false, use_lmdb);
        _io = _default_io.get();
    } else {
        _io = config->binary_io;
    }

    // Shader dispatch establishes its FP policy on the executing worker.
    // Creating a device must not change the caller's floating-point state.

    // embree
    _rtc_device = rtcNewDevice(nullptr);
    rtcSetDeviceErrorFunction(
        _rtc_device,
        [](void *, RTCError code, const char *message) {
            if (code != RTC_ERROR_NONE) {
                LUISA_ERROR_WITH_LOCATION("Embree error (code = {}): {}",
                                          luisa::to_underlying(code), message);
            }
        },
        nullptr);
    if (luisa_fallback_bvh_stats_enabled()) {
        rtcSetDeviceMemoryMonitorFunction(
            _rtc_device,
            [](void *, ssize_t bytes, bool) noexcept -> bool {
                luisa_fallback_bvh_stats().record_alloc(bytes);
                return true;
            },
            nullptr);
    }

    // llvm
    static std::once_flag flag;
    std::call_once(flag, [] {
        ::llvm::InitializeNativeTarget();
        ::llvm::InitializeNativeTargetAsmPrinter();
    });
}

namespace api {
void luisa_fallback_dump_api_profile_if_enabled() noexcept;
}

FallbackDevice::~FallbackDevice() noexcept {
    api::luisa_fallback_dump_api_profile_if_enabled();
    if (luisa_fallback_bvh_stats_enabled()) {
        auto &s = luisa_fallback_bvh_stats();
        std::fprintf(
            stdout,
            "[fallback-bvh-stats] peak_bytes=%lld retained_bytes=%lld blas_commits=%llu blas_ms=%.3f tlas_commits=%llu tlas_ms=%.3f direct_meshes=%llu instanced_slots=%llu\n",
            static_cast<long long>(s.peak_bytes.load()),
            static_cast<long long>(s.retained_after_accel_bytes.load()),
            static_cast<unsigned long long>(s.blas_commits.load()),
            static_cast<double>(s.blas_commit_ns.load()) * 1e-6,
            static_cast<unsigned long long>(s.tlas_commits.load()),
            static_cast<double>(s.tlas_commit_ns.load()) * 1e-6,
            static_cast<unsigned long long>(s.last_direct_meshes.load()),
            static_cast<unsigned long long>(s.last_instanced_slots.load()));
        std::fflush(stdout);
    }
    _accel_archive_ext.reset();
    rtcReleaseDevice(_rtc_device);
}

void *FallbackDevice::native_handle() const noexcept {
    return reinterpret_cast<void *>(reinterpret_cast<uint64_t>(this));
}

void FallbackDevice::destroy_buffer(uint64_t handle) noexcept {
    luisa::delete_with_allocator(reinterpret_cast<FallbackBuffer *>(handle));
}

void FallbackDevice::destroy_texture(uint64_t handle) noexcept {
    luisa::delete_with_allocator(reinterpret_cast<FallbackTexture *>(handle));
}

void FallbackDevice::destroy_bindless_array(uint64_t handle) noexcept {
    luisa::delete_with_allocator(reinterpret_cast<FallbackBindlessArray *>(handle));
}

void FallbackDevice::destroy_stream(uint64_t handle) noexcept {
    luisa::delete_with_allocator(reinterpret_cast<FallbackStream *>(handle));
}

void FallbackDevice::synchronize_stream(uint64_t stream_handle) noexcept {
    reinterpret_cast<FallbackStream *>(stream_handle)->synchronize();
}

void FallbackDevice::destroy_shader(uint64_t handle) noexcept {
    luisa::delete_with_allocator(reinterpret_cast<FallbackShader *>(handle));
}

void FallbackDevice::destroy_event(uint64_t handle) noexcept {
    luisa::delete_with_allocator(reinterpret_cast<FallbackEvent *>(handle));
}

void FallbackDevice::destroy_mesh(uint64_t handle) noexcept {
    luisa::delete_with_allocator(reinterpret_cast<FallbackMesh *>(handle));
}

void FallbackDevice::destroy_accel(uint64_t handle) noexcept {
    luisa::delete_with_allocator(reinterpret_cast<FallbackAccel *>(handle));
}

void FallbackDevice::destroy_swapchain(uint64_t handle) noexcept {
    luisa::delete_with_allocator(reinterpret_cast<FallbackSwapchain *>(handle));
}

void FallbackDevice::present_display_in_stream(uint64_t stream_handle,
                                               uint64_t swap_chain_handle,
                                               uint64_t image_handle) noexcept {
    auto stream = reinterpret_cast<FallbackStream *>(stream_handle);
    auto chain = reinterpret_cast<FallbackSwapchain *>(swap_chain_handle);
    auto image = reinterpret_cast<FallbackTexture *>(image_handle);
    chain->present(stream, image);
}

uint FallbackDevice::compute_warp_size() const noexcept {
    return 1;
}
size_t FallbackDevice::compute_max_shared_memory_size() const noexcept {
    return max_shared_memory_size;
}
uint64_t FallbackDevice::memory_granularity() const noexcept {
    return 1;
}

BufferCreationInfo FallbackDevice::create_buffer(const Type *element, size_t elem_count, void *external_memory) noexcept {
    BufferCreationInfo info{};
    if (element == Type::of<void>()) {
        info.element_stride = 1u;
    } else {
        info.element_stride = element->size();
    }
    info.total_size_bytes = info.element_stride * elem_count;
    auto buffer = external_memory == nullptr ?
                      luisa::new_with_allocator<FallbackBuffer>(info.total_size_bytes) :
                      luisa::new_with_allocator<FallbackBuffer>(static_cast<std::byte *>(external_memory), info.total_size_bytes);
    info.handle = reinterpret_cast<uint64_t>(buffer);
    info.native_handle = reinterpret_cast<void *>(buffer->data());
    return info;
}

ResourceCreationInfo FallbackDevice::create_texture(PixelFormat format, uint dimension,
                                                    uint width, uint height, uint depth,
                                                    uint mipmap_levels, void *external_native_handle,
                                                    bool simultaneous_access, bool allow_raster_target) noexcept {
    if (external_native_handle == nullptr) {
        auto texture = luisa::new_with_allocator<FallbackTexture>(
            pixel_format_to_storage(format), dimension,
            make_uint3(width, height, depth), mipmap_levels);
        return {
            .handle = reinterpret_cast<uint64_t>(texture),
            .native_handle = texture->native_handle(),
        };
    }
    auto texture = luisa::new_with_allocator<FallbackTexture>(
        pixel_format_to_storage(format), dimension,
        make_uint3(width, height, depth), mipmap_levels,
        static_cast<std::byte *>(external_native_handle));
    return {
        .handle = reinterpret_cast<uint64_t>(texture),
        .native_handle = texture->native_handle(),
    };
}

ResourceCreationInfo FallbackDevice::create_stream(StreamTag stream_tag) noexcept {
    auto stream = luisa::new_with_allocator<FallbackStream>();
    return {
        .handle = reinterpret_cast<uint64_t>(stream),
        .native_handle = native_handle(),
    };
}

void FallbackDevice::dispatch(uint64_t stream_handle, CommandList &&list) noexcept {
    auto stream = reinterpret_cast<FallbackStream *>(stream_handle);
    stream->dispatch(std::move(list));
}

void FallbackDevice::set_stream_log_callback(uint64_t stream_handle, const DeviceInterface::StreamLogCallback &callback) noexcept {
    auto stream = reinterpret_cast<FallbackStream *>(stream_handle);
    stream->queue()->set_log_callback(callback);
}

SwapchainCreationInfo FallbackDevice::create_swapchain(const SwapchainOption &option, uint64_t stream_handle) noexcept {
    auto stream = reinterpret_cast<FallbackStream *>(stream_handle);
    auto sc = luisa::new_with_allocator<FallbackSwapchain>(stream, option);
    return {ResourceCreationInfo{.handle = reinterpret_cast<uint64_t>(sc),
                                 .native_handle = nullptr},
            (option.wants_hdr ? PixelStorage::FLOAT4 : PixelStorage::BYTE4)};
}

ShaderCreationInfo FallbackDevice::create_shader(const ShaderOption &option, Function kernel) noexcept {
    if (kernel.allowed_warp_size().value_or(1) != 1) [[unlikely]] {
        LUISA_ERROR("Fallback backend only support warp size 1.");
    }
    Clock clk;
    auto shader = luisa::new_with_allocator<FallbackShader>(this, option, kernel);
    LUISA_VERBOSE("Shader compilation took {} ms.", clk.toc());
    ShaderCreationInfo info{};
    info.handle = reinterpret_cast<uint64_t>(shader);
    info.native_handle = reinterpret_cast<void *>(shader->native_handle());
    info.block_size = kernel.block_size();
    return info;
}

ShaderCreationInfo FallbackDevice::load_shader(luisa::string_view name, luisa::span<const Type *const> arg_types) noexcept {
    return ShaderCreationInfo();
}

Usage FallbackDevice::shader_argument_usage(uint64_t handle, size_t index) noexcept {
    return reinterpret_cast<FallbackShader *>(handle)->argument_usage(index);
}

void FallbackDevice::signal_event(uint64_t handle, uint64_t stream_handle, uint64_t fence_value) noexcept {
    auto event = reinterpret_cast<FallbackEvent *>(handle);
    auto stream = reinterpret_cast<FallbackStream *>(stream_handle);
    stream->queue()->enqueue([event, fence_value] {
        event->signal(fence_value);
    });
}

void FallbackDevice::wait_event(uint64_t handle, uint64_t stream_handle, uint64_t fence_value) noexcept {
    auto event = reinterpret_cast<FallbackEvent *>(handle);
    auto stream = reinterpret_cast<FallbackStream *>(stream_handle);
    stream->queue()->enqueue([event, fence_value] {
        event->wait(fence_value);
    });
}

bool FallbackDevice::is_event_completed(uint64_t handle, uint64_t fence_value) const noexcept {
    auto event = reinterpret_cast<FallbackEvent *>(handle);
    return event->is_completed(fence_value);
}

void FallbackDevice::synchronize_event(uint64_t handle, uint64_t fence_value) noexcept {
    auto event = reinterpret_cast<FallbackEvent *>(handle);
    event->wait(fence_value);
}

ResourceCreationInfo FallbackDevice::create_mesh(const AccelOption &option) noexcept {
    auto mesh = luisa::new_with_allocator<FallbackMesh>(_rtc_device, option);
    return {.handle = reinterpret_cast<uint64_t>(mesh),
            .native_handle = mesh->handle()};
}

ResourceCreationInfo FallbackDevice::create_procedural_primitive(const AccelOption &option) noexcept {
    auto prim = luisa::new_with_allocator<FallbackProceduralPrim>(_rtc_device, option);
    return {.handle = reinterpret_cast<uint64_t>(prim),
            .native_handle = prim->handle()};
}

void FallbackDevice::destroy_procedural_primitive(uint64_t handle) noexcept {
    auto prim = reinterpret_cast<FallbackProceduralPrim *>(handle);
    luisa::delete_with_allocator(prim);
}

ResourceCreationInfo FallbackDevice::create_curve(const AccelOption &option) noexcept {
    auto curve = luisa::new_with_allocator<FallbackCurve>(_rtc_device, option);
    return {.handle = reinterpret_cast<uint64_t>(curve),
            .native_handle = curve->handle()};
}

void FallbackDevice::destroy_curve(uint64_t handle) noexcept {
    auto curve = reinterpret_cast<FallbackCurve *>(handle);
    luisa::delete_with_allocator(curve);
}

ResourceCreationInfo FallbackDevice::create_motion_instance(const AccelMotionOption &option) noexcept {
    auto instance = luisa::new_with_allocator<FallbackMotionInstance>(option);
    return {.handle = reinterpret_cast<uint64_t>(instance), .native_handle = instance};
}

void FallbackDevice::destroy_motion_instance(uint64_t handle) noexcept {
    auto instance = reinterpret_cast<FallbackMotionInstance *>(handle);
    luisa::delete_with_allocator(instance);
}

ResourceCreationInfo FallbackDevice::create_accel(const AccelOption &option) noexcept {
    auto accel = luisa::new_with_allocator<FallbackAccel>(_rtc_device, option);
    return ResourceCreationInfo{.handle = reinterpret_cast<uint64_t>(accel),
                                .native_handle = accel->handle()};
}

string FallbackDevice::query(luisa::string_view property) noexcept {
    return DeviceInterface::query(property);
}

DeviceExtension *FallbackDevice::extension(luisa::string_view name) noexcept {
    if (name == TexCompressExt::name) {
        std::scoped_lock lock{_ext_mutex};
        if (_tex_compress_ext == nullptr) { _tex_compress_ext = create_fallback_tex_compress_ext(); }
        return _tex_compress_ext.get();
    }
    if (name == AccelArchiveExt::name) {
        std::scoped_lock lock{_ext_mutex};
        if (_accel_archive_ext == nullptr) {
            EmbreeAccelArchiveOps ops{};
            ops.backend_kind = AccelArchiveBackendKind::FALLBACK;
            ops.rtc_device = _rtc_device;
            ops.device_fingerprint = "embree4_fallback_cpu";
            ops.user_ctx = this;
            ops.validate_stream = [](void *, uint64_t stream_handle) noexcept -> bool {
                return stream_handle != invalid_resource_handle && stream_handle != 0u;
            };
            ops.synchronize_stream = [](void *, uint64_t stream_handle) noexcept {
                reinterpret_cast<FallbackStream *>(stream_handle)->synchronize();
            };
            ops.get_mesh_export_state = [](void *, uint64_t mesh_handle, EmbreeMeshExportState &out_state) noexcept -> AccelArchiveResult {
                const auto *base = reinterpret_cast<const FallbackPrimBase *>(mesh_handle);
                if (base->is_motion_instance() || !static_cast<const FallbackPrim *>(base)->is_mesh()) {
                    return accel_archive::make_archive_error(
                        AccelArchiveStatus::INVALID_ARGUMENT,
                        "Resource handle does not denote a FallbackMesh.");
                }
                const auto *mesh = static_cast<const FallbackMesh *>(base);
                out_state.option = mesh->option();
                out_state.is_built = mesh->is_built();
                out_state.triangle_count = mesh->last_triangle_count();
                out_state.committed_rtc_scene = out_state.is_built ?
                                                    static_cast<void *>(mesh->committed_handle()) :
                                                    nullptr;
                return {};
            };
            ops.resolve_geometry_binding = [](void *, uint64_t mesh_handle, const AccelArchiveMeshDesc &expected_desc,
                                              const AccelArchiveGeometryBinding &binding,
                                              EmbreeResolvedTriangleGeometry &out_geom) noexcept -> AccelArchiveResult {
                const auto *base = reinterpret_cast<const FallbackPrimBase *>(mesh_handle);
                if (base->is_motion_instance() || !static_cast<const FallbackPrim *>(base)->is_mesh()) {
                    return accel_archive::make_archive_error(
                        AccelArchiveStatus::INVALID_ARGUMENT,
                        "Resource handle does not denote a FallbackMesh.");
                }
                const auto *mesh = static_cast<const FallbackMesh *>(base);
                if (binding.vertex_buffer == invalid_resource_handle || binding.vertex_buffer == 0u ||
                    binding.triangle_buffer == invalid_resource_handle || binding.triangle_buffer == 0u) {
                    return accel_archive::make_archive_error(
                        AccelArchiveStatus::INVALID_ARGUMENT,
                        "Embree BLAS import requires valid rebound vertex and triangle buffer handles.");
                }
                auto *vbuf = reinterpret_cast<FallbackBuffer *>(binding.vertex_buffer);
                auto *tbuf = reinterpret_cast<FallbackBuffer *>(binding.triangle_buffer);
                return validate_embree_triangle_binding(
                    mesh->option(),
                    expected_desc,
                    binding,
                    static_cast<const std::byte *>(vbuf->data()),
                    vbuf->size_bytes(),
                    static_cast<const std::byte *>(tbuf->data()),
                    tbuf->size_bytes(),
                    out_geom);
            };
            ops.import_mesh_bvh = [](void *, uint64_t mesh_handle, const EmbreeResolvedTriangleGeometry &geom,
                                     luisa::span<const std::byte> bvh_payload, EmbreeArchiveImportFn import_fn,
                                     size_t &out_unpacked_bytes, EmbreeArchiveStatus &out_status) noexcept -> AccelArchiveResult {
                auto *mesh = static_cast<FallbackMesh *>(reinterpret_cast<FallbackPrimBase *>(mesh_handle));
                static_cast<void>(mesh->import_from_archive(
                    geom, bvh_payload, import_fn, out_unpacked_bytes, out_status));
                return {};
            };
            _accel_archive_ext = luisa::make_unique<EmbreeAccelArchiveExt>(std::move(ops));
        }
        return _accel_archive_ext.get();
    }
    return DeviceInterface::extension(name);
}

void FallbackDevice::set_name(luisa::compute::Resource::Tag resource_tag, uint64_t resource_handle, luisa::string_view name) noexcept {
}

SparseBufferCreationInfo FallbackDevice::create_sparse_buffer(const Type *element, size_t elem_count) noexcept {
    return DeviceInterface::create_sparse_buffer(element, elem_count);
}

ResourceCreationInfo FallbackDevice::allocate_sparse_buffer_heap(size_t byte_size) noexcept {
    return DeviceInterface::allocate_sparse_buffer_heap(byte_size);
}

void FallbackDevice::deallocate_sparse_buffer_heap(uint64_t handle) noexcept {
    DeviceInterface::deallocate_sparse_buffer_heap(handle);
}

void FallbackDevice::update_sparse_resources(uint64_t stream_handle, vector<SparseUpdateTile> &&textures_update) noexcept {
    DeviceInterface::update_sparse_resources(stream_handle, std::move(textures_update));
}

void FallbackDevice::destroy_sparse_buffer(uint64_t handle) noexcept {
    DeviceInterface::destroy_sparse_buffer(handle);
}

ResourceCreationInfo FallbackDevice::allocate_sparse_texture_heap(size_t byte_size) noexcept {
    return DeviceInterface::allocate_sparse_texture_heap(byte_size);
}

void FallbackDevice::deallocate_sparse_texture_heap(uint64_t handle) noexcept {
    DeviceInterface::deallocate_sparse_texture_heap(handle);
}

SparseTextureCreationInfo FallbackDevice::create_sparse_texture(PixelFormat format, uint dimension, uint width, uint height, uint depth, uint mipmap_levels, bool simultaneous_access) noexcept {
    return DeviceInterface::create_sparse_texture(format, dimension, width, height, depth, mipmap_levels, simultaneous_access);
}

void FallbackDevice::destroy_sparse_texture(uint64_t handle) noexcept {
    DeviceInterface::destroy_sparse_texture(handle);
}

ResourceCreationInfo FallbackDevice::create_bindless_array(size_t size, BindlessSlotType type) noexcept {
    LUISA_ASSERT(type == BindlessSlotType::MULTIPLE);
    auto array = luisa::new_with_allocator<FallbackBindlessArray>(size);
    return ResourceCreationInfo{
        .handle = reinterpret_cast<uint64_t>(array),
        .native_handle = array->native_handle(),
    };
}

ResourceCreationInfo FallbackDevice::create_event() noexcept {
    auto event = luisa::new_with_allocator<FallbackEvent>();
    return ResourceCreationInfo{
        .handle = reinterpret_cast<uint64_t>(event),
        .native_handle = event->native_handle(),
    };
}

}// namespace luisa::compute::fallback

LUISA_EXPORT_API luisa::compute::DeviceInterface *create(luisa::compute::Context &&ctx,
                                                         const luisa::compute::DeviceConfig *config) noexcept {
    return luisa::new_with_allocator<
        luisa::compute::fallback::FallbackDevice>(
        std::move(ctx), config);
}

LUISA_EXPORT_API void destroy(luisa::compute::DeviceInterface *device) noexcept {
    luisa::delete_with_allocator(device);
}

LUISA_EXPORT_API void backend_device_names(luisa::vector<luisa::string> &names) noexcept {
    names.clear();
    names.emplace_back(luisa::cpu_name());
}

#include "../common/export_version.inl.h"
