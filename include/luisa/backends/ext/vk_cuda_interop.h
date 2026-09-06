#pragma once
#include <array>
#include <utility>
#include <luisa/core/concepts.h>
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

// Attaches an explicit Usage to a resource argument. NVRTC-compiled CUDA
// kernels have no AST, so read/write intent cannot be inferred like in the
// DSL; use the read()/write()/read_write() helpers at the call site, or pass
// a bare view (which defaults to READ_WRITE).
template<typename T>
struct UsageArg {
    T resource;
    Usage usage;
};

// Per-argument adapter carrying a view and its usage through the
// CudaKernelInvoke ratchet. Typed kernels (CudaKernelT) construct these
// internally from the usage baked into their signature; the untyped path
// accepts bare views (READ_WRITE) or usage-tagged views (read()/write()/
// read_write()).
template<typename View>
struct ResourceArg {
    View view;
    Usage usage;
    ResourceArg(View v) noexcept
        : view{v}, usage{Usage::READ_WRITE} {}
    ResourceArg(UsageArg<View> a) noexcept
        : view{a.resource}, usage{a.usage} {}
};

namespace detail {

// Resource types accepted by the typed CUDA kernel launch API. Everything
// else that is trivially copyable is packed as a by-value uniform.
template<typename T>
struct is_cuda_launch_resource : std::false_type {};
template<typename T>
struct is_cuda_launch_resource<BufferView<T>> : std::true_type {};
template<typename T>
struct is_cuda_launch_resource<Buffer<T>> : std::true_type {};
template<typename T>
struct is_cuda_launch_resource<ImageView<T>> : std::true_type {};
template<typename T>
struct is_cuda_launch_resource<Image<T>> : std::true_type {};
template<typename T>
struct is_cuda_launch_resource<VolumeView<T>> : std::true_type {};
template<typename T>
struct is_cuda_launch_resource<Volume<T>> : std::true_type {};
template<>
struct is_cuda_launch_resource<ByteBuffer> : std::true_type {};
template<>
struct is_cuda_launch_resource<ByteBufferView> : std::true_type {};

template<typename T>
inline constexpr auto is_cuda_launch_resource_v =
    is_cuda_launch_resource<std::remove_cvref_t<T>>::value;

// Maps owning resources to their view types; views map to themselves.
template<typename T>
struct cuda_resource_view {
    using type = T;
};
template<typename T>
struct cuda_resource_view<Buffer<T>> {
    using type = BufferView<T>;
};
template<>
struct cuda_resource_view<ByteBuffer> {
    using type = ByteBufferView;
};
template<typename T>
struct cuda_resource_view<Image<T>> {
    using type = ImageView<T>;
};
template<typename T>
struct cuda_resource_view<Volume<T>> {
    using type = VolumeView<T>;
};
template<typename T>
using cuda_resource_view_t = typename cuda_resource_view<std::remove_cvref_t<T>>::type;

template<typename T>
struct is_cuda_launch_texture : std::false_type {};
template<typename T>
struct is_cuda_launch_texture<ImageView<T>> : std::true_type {};
template<typename T>
struct is_cuda_launch_texture<VolumeView<T>> : std::true_type {};
template<typename T>
inline constexpr auto is_cuda_launch_texture_v =
    is_cuda_launch_texture<std::remove_cvref_t<T>>::value;

template<typename T>
struct is_usage_arg : std::false_type {};
template<typename T>
struct is_usage_arg<UsageArg<T>> : std::true_type {};
template<typename T>
struct is_resource_arg : std::false_type {};
template<typename T>
struct is_resource_arg<ResourceArg<T>> : std::true_type {};

// True when the signature parameter is a resource type (owning or view).
template<typename T>
inline constexpr auto is_cuda_arg_resource_v =
    is_cuda_launch_resource_v<T> ||
    is_cuda_launch_resource_v<cuda_resource_view_t<T>>;

}// namespace detail

///@name Usage tagging helpers for typed CUDA kernel launches
///@{
template<typename T>
[[nodiscard]] UsageArg<BufferView<T>> read(BufferView<T> view) noexcept { return {view, Usage::READ}; }
template<typename T>
[[nodiscard]] UsageArg<BufferView<T>> write(BufferView<T> view) noexcept { return {view, Usage::WRITE}; }
template<typename T>
[[nodiscard]] UsageArg<BufferView<T>> read_write(BufferView<T> view) noexcept { return {view, Usage::READ_WRITE}; }
template<typename T>
[[nodiscard]] UsageArg<ImageView<T>> read(ImageView<T> view) noexcept { return {view, Usage::READ}; }
template<typename T>
[[nodiscard]] UsageArg<ImageView<T>> write(ImageView<T> view) noexcept { return {view, Usage::WRITE}; }
template<typename T>
[[nodiscard]] UsageArg<ImageView<T>> read_write(ImageView<T> view) noexcept { return {view, Usage::READ_WRITE}; }
template<typename T>
[[nodiscard]] UsageArg<VolumeView<T>> read(VolumeView<T> view) noexcept { return {view, Usage::READ}; }
template<typename T>
[[nodiscard]] UsageArg<VolumeView<T>> write(VolumeView<T> view) noexcept { return {view, Usage::WRITE}; }
template<typename T>
[[nodiscard]] UsageArg<VolumeView<T>> read_write(VolumeView<T> view) noexcept { return {view, Usage::READ_WRITE}; }
[[nodiscard]] inline UsageArg<ByteBufferView> read(ByteBufferView view) noexcept { return {view, Usage::READ}; }
[[nodiscard]] inline UsageArg<ByteBufferView> write(ByteBufferView view) noexcept { return {view, Usage::WRITE}; }
[[nodiscard]] inline UsageArg<ByteBufferView> read_write(ByteBufferView view) noexcept { return {view, Usage::READ_WRITE}; }
///@}

// DSL-style ratchet over KernelLauncher: arguments are fold-encoded through
// operator<< (mirroring detail::ShaderInvokeBase), and the rvalue-qualified
// dispatch(...) builds the CudaKernelLaunchCommand.
class CudaKernelInvoke {

private:
    KernelLauncher _launcher;
    uint64_t _function;

    template<typename R>
    static void _encode_resource(KernelLauncher &launcher, R &&resource, Usage usage) noexcept {
        using Res = std::remove_cvref_t<R>;
        using View = detail::cuda_resource_view_t<Res>;
        static_assert(detail::is_cuda_launch_resource_v<Res>,
                      "CUDA kernel launch arguments must be buffers, images, "
                      "volumes, or trivially-copyable uniforms.");
        // Owning resources are converted to views; views pass through.
        auto view = [&]() noexcept -> View {
            if constexpr (std::is_same_v<Res, View>) {
                return resource;
            } else {
                return resource.view();
            }
        }();
        if constexpr (detail::is_cuda_launch_texture_v<View>) {
            launcher.add_texture(view.handle(), view.level(), usage);
        } else {
            launcher.add_buffer(view.handle(), view.offset_bytes(), view.size_bytes(), usage);
        }
    }

public:
    explicit CudaKernelInvoke(uint64_t cuda_function_handle) noexcept
        : _function{cuda_function_handle} {}
    CudaKernelInvoke(CudaKernelInvoke const &) = delete;
    CudaKernelInvoke(CudaKernelInvoke &&) noexcept = default;

    // Usage-tagged resource arguments.
    template<typename R>
    CudaKernelInvoke &operator<<(UsageArg<R> arg) noexcept {
        _encode_resource(_launcher, arg.resource, arg.usage);
        return *this;
    }
    // Typed-kernel resource arguments (bare or usage-tagged).
    template<typename R>
    CudaKernelInvoke &operator<<(ResourceArg<R> arg) noexcept {
        _encode_resource(_launcher, arg.view, arg.usage);
        return *this;
    }
    // Bare resource arguments default to READ_WRITE (barrier-safe).
    template<typename R>
        requires detail::is_cuda_launch_resource_v<R>
    CudaKernelInvoke &operator<<(R &&resource) noexcept {
        _encode_resource(_launcher, std::forward<R>(resource), Usage::READ_WRITE);
        return *this;
    }
    // Trivially-copyable by-value uniforms.
    template<typename T>
        requires(!detail::is_cuda_launch_resource_v<T> &&
                 !detail::is_usage_arg<std::remove_cvref_t<T>>::value &&
                 !detail::is_resource_arg<std::remove_cvref_t<T>>::value &&
                 std::is_trivially_copyable_v<std::remove_cvref_t<T>>)
    CudaKernelInvoke &operator<<(T &&value) noexcept {
        _launcher.add_uniform(std::forward<T>(value));
        return *this;
    }

    // CUDA launches need both grid and block dimensions (unlike the DSL's
    // thread-count dispatch). Returns the same command type as
    // KernelLauncher::build, so `stream << ...` works unchanged.
    [[nodiscard]] luisa::unique_ptr<CudaKernelLaunchCommand>
    dispatch(uint3 grid_dim, uint3 block_dim, uint32_t shared_mem_bytes = 0u) && noexcept {
        return std::move(_launcher).build(_function, grid_dim, block_dim, shared_mem_bytes);
    }
    // Ergonomic 1D overload: grid = ceil(thread_count_x / block_dim.x).
    [[nodiscard]] luisa::unique_ptr<CudaKernelLaunchCommand>
    dispatch(uint32_t thread_count_x, uint3 block_dim, uint32_t shared_mem_bytes = 0u) && noexcept {
        LUISA_ASSERT(block_dim.x > 0u, "CUDA kernel launch block dimension must be nonzero.");
        auto grid = uint3{(thread_count_x + block_dim.x - 1u) / block_dim.x, 1u, 1u};
        return std::move(*this).dispatch(grid, block_dim, shared_mem_bytes);
    }
};

// Untyped DSL-style CUDA kernel: `cuda_kernel(args...).dispatch(grid, block)`.
// Usage must be supplied via read()/write()/read_write() wrappers, or bare
// views default to READ_WRITE.
class CudaKernel {

protected:
    uint64_t _handle;

public:
    explicit CudaKernel(uint64_t cuda_function_handle) noexcept
        : _handle{cuda_function_handle} {}

    template<typename... Args>
    [[nodiscard]] CudaKernelInvoke operator()(Args &&...args) const noexcept {
        CudaKernelInvoke invoke{_handle};
        static_cast<void>((invoke << ... << std::forward<Args>(args)));
        return invoke;
    }
};

// Usage-carrying signature parameter for typed CUDA kernels. A kernel's
// per-argument read/write intent is a property of the kernel signature, not
// of an individual launch, so it is declared once when creating the typed
// kernel (create_cuda_kernel(...).kernel<CudaArg<Buffer<float>, Usage::READ>,
// ...>()) and baked into the CudaKernelT instance; dispatch sites then pass
// bare views only, mirroring the DSL where usage is inferred from the
// compiled kernel and never appears at the call site.
template<typename T, Usage U = Usage::READ_WRITE>
struct CudaArg {
    using resource_type = std::remove_cvref_t<T>;
    static constexpr auto usage = U;
};

// Traits mapping a declared typed-kernel signature parameter to its call-site
// argument type and baked usage:
// - CudaArg<T, U> (T an owning resource or view): call site passes the bare
//   view; usage is U.
// - bare resource type (Buffer<float>, ByteBuffer, Image<float>,
//   Volume<float>, BufferView<float>, ...): same, defaulting to READ_WRITE
//   (barrier-safe, matching the untyped path's bare-view behavior).
// - anything else (trivially-copyable scalars/aggregates): passed by const
//   reference as a by-value uniform; carries no usage.
template<typename T>
struct cuda_arg_traits {
    using type = std::remove_cvref_t<T>;
    static constexpr bool is_resource = detail::is_cuda_arg_resource_v<type>;
    static constexpr Usage usage = Usage::READ_WRITE;
    using view_type = std::conditional_t<is_resource,
                                         detail::cuda_resource_view_t<type>,
                                         void>;
    using arg_type = std::conditional_t<is_resource, view_type, const type &>;
};
template<typename T, Usage U>
struct cuda_arg_traits<CudaArg<T, U>> {
    using type = std::remove_cvref_t<T>;
    static_assert(detail::is_cuda_arg_resource_v<type>,
                  "CudaArg<T, U> requires a CUDA launch resource type "
                  "(buffer/image/volume/byte-buffer, owning or view).");
    static constexpr bool is_resource = true;
    static constexpr Usage usage = U;
    using view_type = detail::cuda_resource_view_t<type>;
    using arg_type = view_type;
};
template<typename T>
using cuda_arg_traits_t = cuda_arg_traits<std::remove_cvref_t<T>>;

// Typed DSL-style CUDA kernel mirroring Shader<dim, Args...>: the host-side
// argument list is checked against the declared signature, and each resource
// parameter's Usage is declared in the signature (via CudaArg<T, U>, or
// READ_WRITE for a bare resource type) and baked into the instance — dispatch
// sites pass bare views only. Declare by-value parameters as their scalar
// types. The CUDA source itself is opaque, so this checks shape, not the
// device signature (same trust level as AOT Shader<dim, Args...> loaded from
// file); the declared usages are likewise trusted, not verified against the
// NVRTC source. Per-launch usage overrides are only available through the
// untyped CudaKernel / KernelLauncher path.
template<concepts::non_cvref... Args>
class CudaKernelT final : public CudaKernel {

private:
    template<typename Arg, typename A>
    static void _encode_arg(CudaKernelInvoke &invoke, A &&arg) noexcept {
        using traits = cuda_arg_traits_t<Arg>;
        if constexpr (traits::is_resource) {
            // Construct ResourceArg explicitly so overload resolution against
            // CudaKernelInvoke::operator<< is deterministic.
            invoke << ResourceArg<typename traits::view_type>{
                UsageArg<typename traits::view_type>{arg, traits::usage}};
        } else {
            invoke << arg;
        }
    }

    // Positional pairing of Args... with the call-site arguments; a plain
    // comma fold would lose the per-position Arg association.
    template<size_t... I>
    [[nodiscard]] CudaKernelInvoke _invoke(std::index_sequence<I...>,
                                           typename cuda_arg_traits_t<Args>::arg_type... args) const noexcept {
        CudaKernelInvoke invoke{this->_handle};
        (static_cast<void>(I, _encode_arg<Args>(invoke, args)), ...);
        return invoke;
    }

public:
    using CudaKernel::CudaKernel;

    [[nodiscard]] static constexpr size_t arg_count() noexcept {
        return sizeof...(Args);
    }

    // The per-resource-argument usages baked into this instance, in argument
    // order (uniform parameters are skipped), matching the layout of
    // CudaKernelLaunchCommand::argument_usages().
    [[nodiscard]] static constexpr auto argument_usages() noexcept {
        std::array<Usage, (0u + ... + (cuda_arg_traits_t<Args>::is_resource ? 1u : 0u))> usages{};
        auto index = 0u;
        auto append = [&index, &usages]<typename Arg>() noexcept {
            if constexpr (cuda_arg_traits_t<Arg>::is_resource) {
                usages[index++] = cuda_arg_traits_t<Arg>::usage;
            }
        };
        (append.template operator()<Args>(), ...);
        return usages;
    }

    [[nodiscard]] CudaKernelInvoke operator()(typename cuda_arg_traits_t<Args>::arg_type... args) const noexcept {
        return _invoke(std::index_sequence_for<Args...>{}, args...);
    }
};

// RAII owner of a CUDA kernel shader handle created through
// VkCudaInterop::create_cuda_kernel_shader; destroys it on destruction.
// Defined at the end of this header (needs the complete VkCudaInterop type).
class CudaShader;

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

    // RAII wrapper around create_cuda_kernel_shader; the returned object
    // destroys the shader handle on destruction. Evaluates to false when
    // CUDA kernel launch is unsupported or compilation failed.
    [[nodiscard]] vk_cuda_interop::CudaShader create_cuda_kernel(
        const vk_cuda_interop::CudaKernelShaderOption &option) noexcept;

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

// RAII owner of a CUDA kernel shader handle created through
// VkCudaInterop::create_cuda_kernel_shader; destroys it on destruction.
class CudaShader {

private:
    VkCudaInterop *_ext{nullptr};
    uint64_t _handle{0u};

    void _destroy() noexcept {
        if (_handle != 0u) {
            _ext->destroy_cuda_kernel_shader(_handle);
            _handle = 0u;
        }
    }

public:
    CudaShader() noexcept = default;
    CudaShader(VkCudaInterop &ext, uint64_t handle) noexcept
        : _ext{&ext}, _handle{handle} {}
    ~CudaShader() noexcept { _destroy(); }
    CudaShader(CudaShader const &) = delete;
    CudaShader &operator=(CudaShader const &) = delete;
    CudaShader(CudaShader &&rhs) noexcept
        : _ext{rhs._ext}, _handle{rhs._handle} {
        rhs._handle = 0u;
    }
    CudaShader &operator=(CudaShader &&rhs) noexcept {
        if (this != &rhs) {
            _destroy();
            _ext = rhs._ext;
            _handle = rhs._handle;
            rhs._handle = 0u;
        }
        return *this;
    }
    explicit operator bool() const noexcept { return _handle != 0u; }
    [[nodiscard]] uint64_t handle() const noexcept { return _handle; }
    uint64_t release() noexcept {
        auto handle = _handle;
        _handle = 0u;
        return handle;
    }
    // Typed DSL-style kernel bound to this shader's handle.
    template<concepts::non_cvref... KArgs>
    [[nodiscard]] CudaKernelT<KArgs...> kernel() const noexcept {
        return CudaKernelT<KArgs...>{_handle};
    }
    // Untyped DSL-style kernel bound to this shader's handle.
    [[nodiscard]] CudaKernel kernel() const noexcept {
        return CudaKernel{_handle};
    }
};

}// namespace vk_cuda_interop

inline vk_cuda_interop::CudaShader VkCudaInterop::create_cuda_kernel(
    const vk_cuda_interop::CudaKernelShaderOption &option) noexcept {
    return vk_cuda_interop::CudaShader{*this, create_cuda_kernel_shader(option)};
}

}// namespace luisa::compute