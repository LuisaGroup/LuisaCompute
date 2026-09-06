#pragma once
#include <luisa/runtime/rhi/device_interface.h>
#include <luisa/runtime/rhi/command.h>
#include <luisa/runtime/byte_buffer.h>
#include <luisa/runtime/buffer.h>
#include <luisa/runtime/image.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>
#include <luisa/runtime/volume.h>
#include <luisa/vstl/meta_lib.h>
#include <luisa/backends/ext/native_resource_ext.hpp>
#include <luisa/backends/ext/registry.h>
#include <luisa/backends/ext/cuda/cuda_config_ext.h>
namespace luisa::compute {
class VkCudaInterop;
namespace vk_cuda_interop {
struct Signal {
    VkCudaInterop *ext;
    uint64_t handle;
    uint64_t fence;
    void operator()(DeviceInterface *device, uint64_t stream_handle) const && noexcept;
};
struct Wait {
    VkCudaInterop *ext;
    uint64_t handle;
    uint64_t fence;
    void operator()(DeviceInterface *device, uint64_t stream_handle) const && noexcept;
};

// Options for compiling (or ingesting) a CUDA kernel "shader" for in-command-buffer
// dispatch through VK_NV_cuda_kernel_launch (VkCudaModuleNV + VkCudaFunctionNV pair).
struct CudaKernelShaderOption {
    luisa::string_view source;                  // CUDA C++ source, or PTX text when source_is_ptx
    luisa::string_view kernel_name;             // __global__ entry point name; NVRTC mangles C++
                                                // symbols, so the kernel must be declared
                                                // extern "C" (or the mangled name passed here)
    bool source_is_ptx{false};                  // true when `source` holds precompiled PTX
    luisa::vector<luisa::string> compile_options{};// extra NVRTC compile options (source compiles only)
    luisa::vector<char> *ptx_output{nullptr};   // optional sink receiving the compiled PTX text
};

// Custom dispatch command launching a CUDA kernel (created through
// VkCudaInterop::create_cuda_kernel_shader) inside a Vulkan command buffer.
//
// Buffer argument packing convention: every Argument::Buffer is passed to the
// CUDA kernel as a raw 64-bit device address (VkDeviceAddress + argument
// offset); declare the corresponding kernel parameter as a plain pointer
// (e.g. `float *data`). Argument::Uniform values are passed by value from the
// embedded uniform blob, in argument order.
class CudaKernelLaunchCommand final : public CustomDispatchCommand, public ShaderDispatchCommandBase {

private:
    uint3 _grid_dim;
    uint3 _block_dim;
    uint32_t _shared_mem_bytes;
    // One entry per non-uniform argument, in argument order.
    luisa::vector<Usage> _argument_usages;

    template<typename Self, typename Visitor>
    static void traverse(Self &&self, Visitor &visitor) noexcept {
        auto usage_index = 0u;
        auto next_usage = [&self, &usage_index]() noexcept {
            LUISA_ASSERT(usage_index < self._argument_usages.size(),
                         "CUDA kernel launch command has fewer argument usages than resource arguments.");
            return self._argument_usages[usage_index++];
        };
        for (auto &&arg : self.arguments()) {
            switch (arg.tag) {
                case Argument::Tag::BUFFER:
                    visitor.visit(arg.buffer, next_usage());
                    break;
                case Argument::Tag::TEXTURE:
                    visitor.visit(arg.texture, next_usage());
                    break;
                case Argument::Tag::BINDLESS_ARRAY:
                    visitor.visit(arg.bindless_array, next_usage());
                    break;
                case Argument::Tag::ACCEL:
                    visitor.visit(arg.accel, next_usage());
                    break;
                case Argument::Tag::UNIFORM: break;
                default: break;
            }
        }
        LUISA_ASSERT(usage_index == self._argument_usages.size(),
                     "CUDA kernel launch command has {} argument usages for the resource arguments.",
                     self._argument_usages.size());
    }

public:
    CudaKernelLaunchCommand(uint64_t cuda_function_handle,
                            uint3 grid_dim, uint3 block_dim,
                            uint32_t shared_mem_bytes,
                            luisa::vector<std::byte> &&argument_buffer,
                            size_t argument_count,
                            luisa::vector<Usage> &&argument_usages) noexcept
        : ShaderDispatchCommandBase{cuda_function_handle,
                                    std::move(argument_buffer),
                                    argument_count},
          _grid_dim{grid_dim},
          _block_dim{block_dim},
          _shared_mem_bytes{shared_mem_bytes},
          _argument_usages{std::move(argument_usages)} {
        LUISA_ASSERT(grid_dim.x > 0u && grid_dim.y > 0u && grid_dim.z > 0u,
                     "CUDA kernel launch grid dimension must be nonzero.");
        LUISA_ASSERT(block_dim.x > 0u && block_dim.y > 0u && block_dim.z > 0u,
                     "CUDA kernel launch block dimension must be nonzero.");
    }
    CudaKernelLaunchCommand(CudaKernelLaunchCommand const &) = delete;
    CudaKernelLaunchCommand(CudaKernelLaunchCommand &&) noexcept = default;

public:
    [[nodiscard]] uint64_t custom_cmd_uuid() const noexcept override {
        return luisa::to_underlying(CustomCommandUUID::VK_CUDA_LAUNCH_KERNEL);
    }
    [[nodiscard]] StreamTag stream_tag() const noexcept override {
        return StreamTag::COMPUTE;
    }
    [[nodiscard]] uint64_t cuda_function() const noexcept { return handle(); }
    [[nodiscard]] uint3 grid_dim() const noexcept { return _grid_dim; }
    [[nodiscard]] uint3 block_dim() const noexcept { return _block_dim; }
    [[nodiscard]] uint32_t shared_mem_bytes() const noexcept { return _shared_mem_bytes; }
    [[nodiscard]] luisa::span<const Usage> argument_usages() const noexcept { return _argument_usages; }
    // The reorder budget counts threads, so report grid * block (saturating).
    [[nodiscard]] uint3 max_dispatch_size() const noexcept override {
        auto saturate_mul = [](uint32_t a, uint32_t b) noexcept {
            auto v = static_cast<uint64_t>(a) * static_cast<uint64_t>(b);
            constexpr auto m = std::numeric_limits<uint32_t>::max();
            return static_cast<uint32_t>(v > m ? m : v);
        };
        return uint3{saturate_mul(_grid_dim.x, _block_dim.x),
                     saturate_mul(_grid_dim.y, _block_dim.y),
                     saturate_mul(_grid_dim.z, _block_dim.z)};
    }
    [[nodiscard]] bool requires_resource_state_isolation() const noexcept override { return true; }
    void traverse_arguments(ArgumentVisitor &visitor) const noexcept override {
        traverse(*this, visitor);
    }
    void traverse_arguments(MutableArgumentVisitor &visitor) noexcept override {
        traverse(*this, visitor);
    }
    // Un-hide the generic-lambda traverse_arguments adapters.
    using CustomDispatchCommand::traverse_arguments;
};

// Header-only builder collecting buffer/texture/uniform arguments and packing
// them into a CudaKernelLaunchCommand. Uniform values are copied into the
// command; resource arguments are referenced by handle and must outlive the
// dispatch.
class KernelLauncher {

private:
    luisa::vector<Argument> _arguments;
    luisa::vector<Usage> _argument_usages;
    luisa::vector<std::byte> _uniform_blob;

    [[nodiscard]] Argument &_create_argument() noexcept {
        return _arguments.emplace_back();
    }

public:
    KernelLauncher() noexcept = default;
    // Buffers are passed to the CUDA kernel as raw 64-bit device addresses
    // (VkDeviceAddress + offset); declare the kernel parameter as a pointer.
    KernelLauncher &add_buffer(uint64_t handle, size_t offset, size_t size, Usage usage) noexcept {
        auto &&arg = _create_argument();
        arg.tag = Argument::Tag::BUFFER;
        arg.buffer = Argument::Buffer{handle, offset, size};
        _argument_usages.emplace_back(usage);
        return *this;
    }
    template<typename T>
    KernelLauncher &add_buffer(const BufferView<T> &view, Usage usage) noexcept {
        return add_buffer(view.handle(), view.offset_bytes(), view.size_bytes(), usage);
    }
    KernelLauncher &add_texture(uint64_t handle, uint32_t level, Usage usage) noexcept {
        auto &&arg = _create_argument();
        arg.tag = Argument::Tag::TEXTURE;
        arg.texture = Argument::Texture{handle, level};
        _argument_usages.emplace_back(usage);
        return *this;
    }
    // Scalar/aggregate kernel parameters passed by value.
    KernelLauncher &add_uniform(const void *data, size_t size, size_t alignment) noexcept {
        LUISA_DEBUG_ASSERT(alignment > 0u && alignment <= 16u,
                           "Invalid uniform alignment {}.", alignment);
        auto offset = luisa::align(_uniform_blob.size(), alignment);
        luisa::vector_resize(_uniform_blob, offset + size);
        std::memcpy(_uniform_blob.data() + offset, data, size);
        auto &&arg = _create_argument();
        arg.tag = Argument::Tag::UNIFORM;
        arg.uniform = Argument::Uniform{offset, size, alignment};
        return *this;
    }
    template<typename T>
    KernelLauncher &add_uniform(const T &value) noexcept {
        return add_uniform(&value, sizeof(T), alignof(T));
    }
    [[nodiscard]] luisa::unique_ptr<CudaKernelLaunchCommand> build(
        uint64_t cuda_function_handle,
        uint3 grid_dim, uint3 block_dim,
        uint32_t shared_mem_bytes = 0u) && noexcept {
        auto argument_header_size = _arguments.size() * sizeof(Argument);
        luisa::vector<std::byte> argument_buffer;
        luisa::vector_resize(argument_buffer, argument_header_size + _uniform_blob.size());
        if (argument_header_size > 0u) {
            std::memcpy(argument_buffer.data(), _arguments.data(), argument_header_size);
        }
        if (!_uniform_blob.empty()) {
            std::memcpy(argument_buffer.data() + argument_header_size,
                        _uniform_blob.data(), _uniform_blob.size());
        }
          // Shift uniform offsets past the argument header.
          if (argument_header_size > 0u) {
              auto *args = std::launder(reinterpret_cast<Argument *>(argument_buffer.data()));
              for (auto i = 0u; i < _arguments.size(); ++i) {
                  if (args[i].tag == Argument::Tag::UNIFORM) {
                      args[i].uniform.offset += argument_header_size;
                  }
              }
          }
        return luisa::make_unique<CudaKernelLaunchCommand>(
            cuda_function_handle, grid_dim, block_dim, shared_mem_bytes,
            std::move(argument_buffer), _arguments.size(),
            std::move(_argument_usages));
    }
};
}// namespace vk_cuda_interop

class VkCudaInterop : public DeviceExtension {
public:
    static constexpr luisa::string_view name = "VkCudaInterop";

public:
    [[nodiscard]] virtual BufferCreationInfo create_interop_buffer(const Type *element, size_t elem_count) noexcept = 0;
    [[nodiscard]] virtual ResourceCreationInfo create_interop_texture(
        PixelFormat format, uint dimension,
        uint width, uint height, uint depth,
        uint mipmap_levels, bool simultaneous_access, bool allow_raster_target) noexcept = 0;
    virtual void vk_signal(uint64_t cuda_event_handle, uint64_t vk_stream, uint64_t fence_index) noexcept = 0;
    virtual void vk_wait(uint64_t cuda_event_handle, uint64_t vk_stream, uint64_t fence_index) noexcept = 0;

public:
    [[nodiscard]] virtual CUDADeviceConfigExt::ExternalVkDevice get_external_vk_device() const noexcept = 0;
    virtual void cuda_buffer(uint64_t vk_buffer_handle, uint64_t *cuda_ptr, uint64_t *cuda_handle /*CUexternalMemory* */) noexcept = 0;
    [[nodiscard]] virtual /*CUexternalMemory* */ uint64_t cuda_texture(uint64_t vk_texture_handle) noexcept = 0;
    virtual void unmap(void *cuda_ptr, void *cuda_handle) noexcept = 0;
    [[nodiscard]] virtual int cuda_device_index() const noexcept = 0;
    [[nodiscard]] virtual DeviceInterface *device() noexcept = 0;

public:
    // VK_NV_cuda_kernel_launch support: true only when the backend was built
    // with CUDA interop and the device extension/feature could be enabled
    // (owned logical devices only).
    [[nodiscard]] virtual bool cuda_kernel_launch_supported() const noexcept = 0;
    // Compiles CUDA C++ source with NVRTC (or ingests precompiled PTX when
    // option.source_is_ptx) into a CUDA kernel shader usable with
    // vk_cuda_interop::KernelLauncher/CudaKernelLaunchCommand.
    // Returns 0 when CUDA kernel launch is unsupported.
    [[nodiscard]] virtual uint64_t create_cuda_kernel_shader(
        const vk_cuda_interop::CudaKernelShaderOption &option) noexcept = 0;
    virtual void destroy_cuda_kernel_shader(uint64_t handle) noexcept = 0;

    vk_cuda_interop::Signal vk_signal(TimelineEvent const &cuda_event, uint64_t fence_index) noexcept {
        return vk_cuda_interop::Signal{
            this,
            cuda_event.handle(),
            fence_index};
    }
    vk_cuda_interop::Wait vk_wait(TimelineEvent const &cuda_event, uint64_t fence_index) noexcept {
        return vk_cuda_interop::Wait{
            this,
            cuda_event.handle(),
            fence_index};
    }
    vk_cuda_interop::Signal vk_signal(Event const &cuda_event) noexcept {
        auto signal = cuda_event.signal();
        return vk_cuda_interop::Signal{
            this,
            signal.handle,
            signal.fence};
    }
    vk_cuda_interop::Wait vk_wait(Event const &cuda_event, uint64_t fence = std::numeric_limits<uint64_t>::max()) noexcept {
        auto wait = cuda_event.wait(fence);
        return vk_cuda_interop::Wait{
            this,
            wait.handle,
            wait.fence};
    }
    template<typename T>
    Buffer<T> create_buffer(size_t elem_count) noexcept {
        return Buffer<T>{device(), create_interop_buffer(Type::of<T>(), elem_count)};
    }
    ByteBuffer create_byte_buffer(size_t size_bytes) noexcept {
        return ByteBuffer{device(), create_interop_buffer(Type::of<void>(), size_bytes)};
    }
    template<typename T>
    Image<T> create_image(PixelStorage pixel, uint width, uint height, uint mip_levels = 1u, bool simultaneous_access = false, bool allow_raster_target = false) noexcept {
        return Image<T>{
            device(),
            create_interop_texture(pixel_storage_to_format<T>(pixel), 2, width, height, 1, mip_levels, simultaneous_access, allow_raster_target),
            pixel,
            uint2(width, height),
            mip_levels};
    }
    template<typename T>
    Image<T> create_image(PixelStorage pixel, uint2 size, uint mip_levels = 1u, bool simultaneous_access = false, bool allow_raster_target = false) noexcept {
        return Image<T>{
            device(),
            create_interop_texture(pixel_storage_to_format<T>(pixel), 2, size.x, size.y, 1, mip_levels, simultaneous_access, allow_raster_target),
            pixel,
            size,
            mip_levels};
    }
    template<typename T>
    Volume<T> create_volume(PixelStorage pixel, uint width, uint height, uint volume, uint mip_levels = 1u, bool simultaneous_access = false, bool allow_raster_target = false) noexcept {
        return Volume<T>{
            device(),
            create_interop_texture(pixel_storage_to_format<T>(pixel), 3, width, height, volume, mip_levels, simultaneous_access, allow_raster_target),
            pixel,
            uint3(width, height, volume),
            mip_levels};
    }
    template<typename T>
    Volume<T> create_volume(PixelStorage pixel, uint3 size, uint mip_levels = 1u, bool simultaneous_access = false, bool allow_raster_target = false) noexcept {
        return Volume<T>{
            device(),
            create_interop_texture(pixel_storage_to_format<T>(pixel), 3, size.x, size.y, size.z, mip_levels, simultaneous_access, allow_raster_target),
            pixel,
            size,
            mip_levels};
    }

    virtual ~VkCudaInterop() noexcept = default;
};
LUISA_MARK_STREAM_EVENT_TYPE(vk_cuda_interop::Signal)
LUISA_MARK_STREAM_EVENT_TYPE(vk_cuda_interop::Wait)
namespace vk_cuda_interop {
inline void Signal::operator()(DeviceInterface *device, uint64_t stream_handle) const && noexcept {
    ext->vk_signal(handle, stream_handle, fence);
}
inline void Wait::operator()(DeviceInterface *device, uint64_t stream_handle) const && noexcept {
    ext->vk_wait(handle, stream_handle, fence);
}
}// namespace vk_cuda_interop
}// namespace luisa::compute