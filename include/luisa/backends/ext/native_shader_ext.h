#pragma once

#include <algorithm>
#include <cstring>
#include <new>

#include <luisa/core/basic_traits.h>
#include <luisa/core/basic_types.h>
#include <luisa/core/logging.h>
#include <luisa/core/stl/format.h>
#include <luisa/core/stl/memory.h>
#include <luisa/core/stl/string.h>
#include <luisa/core/stl/vector.h>
#include <luisa/ast/usage.h>
#include <luisa/runtime/rhi/command.h>
#include <luisa/runtime/rhi/device_interface.h>
#include <luisa/runtime/rhi/resource.h>
#include <luisa/runtime/buffer.h>
#include <luisa/runtime/byte_buffer.h>
#include <luisa/backends/ext/registry.h>

namespace luisa::compute {

// Native shader injection (bypassing the DSL/AST codegen path).
//
// `NativeShaderExt` lets an application hand the runtime native shader source
// (HLSL, GLSL, or - on the CUDA backend - CUDA C++ for NVRTC), have it compiled
// to the backend's bytecode format through the backend's own compiler, reflect
// its resource bindings, create a backend compute-shader instance from the
// compiled bytecode, and dispatch it from a `Stream` with
// `NativeShaderDispatchCommand` (built by `NativeShaderLauncher`).
//
// Contract notes (read before use):
//  * Compute shaders only. Ray tracing and rasterization native injection are
//    out of scope for this API.
//  * Native shaders are JIT-only: they do not participate in `ShaderSerializer`
//    / AOT caching, and `LUISA_DUMP_SOURCE` does not dump them. An application
//    that wants to persist them should serialize `NativeShaderCompileResult`
//    (binary + reflection) in its own pipeline.
//  * The per-argument `Usage` supplied to the launcher is the contract the
//    command-reorder pass relies on (see `CustomDispatchCommand`): a declared
//    READ is a read-only contract whose ranges may merge with other readers,
//    a declared WRITE is exclusive over its range. Declaring READ for a
//    resource the shader writes silently breaks synchronization. `build()`
//    cross-checks the declared usages against the reflected binding kinds
//    (an SRV bound as WRITE is always rejected; a UAV bound as READ is
//    rejected unless `allow_usage_override` is requested explicitly).
//  * The launcher's `block_size` must equal the shader's `[numthreads]`
//    (HLSL) / `local_size_*` (GLSL) workgroup size; `load` reports the
//    reflected size and `build()` asserts they match.
//  * Uniform values passed through `add_uniform` are bound as the backend's
// "push constant" currency: a Vulkan push-constant range, or DirectX root
// 32-bit constants. The shader must therefore declare a matching block
// (`layout(push_constant) uniform ...` in GLSL, `[[vk::push_constant]]
// ConstantBuffer<...>` in HLSL for Vulkan, and a `cbuffer ... : register(b0)`
// block for DirectX), and the application must declare the block's size
// through `NativeShaderCompileInfo::push_constant_size` (on the Vulkan side
// the size is also read back from the module when it is left 0). On DirectX
// the reflected `cbuffer` at `register(b0)` *is* that block: it is fed by the
// launcher's uniform values and is therefore not reported as a resource to
// bind (declare the uniform block at another register if you want to bind a
// buffer to it instead, and pass `push_constant_size = 0`).
//  * Textures, samplers and acceleration structures are reflected and reported,
// but `load()` rejects them for now on both backends: buffers and uniform
// blocks are the supported resource classes of this iteration.
//
//  * Lifetime: a `NativeShader` instance (and any shader handle returned by
// `load`) must be destroyed - `NativeShader::reset()`, the RAII destructor or
// `destroy_shader()` - before the `Device` that created it, and its
// dispatches must have been synchronized before destruction, exactly like any
// other GPU resource. Instances that are still alive when the device's
// extension is torn down are released by the backend with a warning; they are
// never silently dropped.
//
// Backend support matrix (this iteration):
// | HLSL | GLSL | CUDA C++ (NVRTC)
//  DirectX (dx) | DXIL + DXC reflection | rejected (fail-closed) | rejected (fail-closed)
//  Vulkan (vk) | SPIR-V via DXC | SPIR-V via glslang | rejected (fail-closed)
//  CUDA (cuda) | rejected (fail-closed) | rejected (fail-closed) | PTX via NVRTC
//
// Resource kinds supported per backend (buffers and uniform blocks only):
//  * dx: constant buffers (CBV), structured buffers (SRV/UAV) and byte-address
// buffers (SRV/UAV). Typed buffers, textures, samplers and acceleration
// structures are reflected but rejected at `load()` with an explicit error.
//  * vk: constant buffers (uniform buffer descriptors), structured/byte-address
// /typed buffers (storage buffer descriptors). Samplers, sampled images,
// storage images and acceleration structures are reflected but rejected at
// `load()` with an explicit error.
//  * cuda: buffers, and buffers only - every pointer-typed parameter of the
// `__global__` entry is one binding. There is no resource-class metadata in
// PTX, so the reflection is read from the kernel signature in the source and
// cross-checked against the compiled parameter list: `const T*` becomes a
// read-only binding (`StructuredBuffer`, `Usage::READ`) and `T*` a writable one
// (`RWStructuredBuffer`, `Usage::READ_WRITE`). CUDA has no `ConstantBuffer`,
// `Sampler`, `Texture2D` or acceleration-structure classes in this API: a
// non-pointer parameter is not a binding at all, it is a plain scalar kernel
// parameter (see the CUDA route notes below).
//
// CUDA route in detail (`NativeShaderLanguage::CUDA_NVRTC`):
//  * The source is CUDA C++ compiled by NVRTC (the backend's `luisa_nvrtc`
// helper), and the resulting PTX is loaded with the CUDA driver API. Helper
// `__device__` functions, headers included by the runtime, and multiple
// `__global__` functions per translation unit are all fine; `entry_point`
// selects the kernel (it may be left at the default `main` when the source
// declares exactly one `__global__` function).
//  * Kernel parameter layout: the pointer parameters of the entry function are
// the launcher's resource arguments, in declaration order, each supplied as
// the buffer's 64-bit device address (buffer base + the `BufferView`'s byte
// offset). Every non-pointer parameter is a scalar kernel parameter whose
// value comes from `add_uniform`, in declaration order, and whose size must
// equal the parameter's compiled size (a mismatch - or a dispatch that supplies
// the wrong number of buffers/scalars - is a contract violation and is reported
// as an error). (So the kernel signature
// `void k(const float *src, float k, float *dst)` takes `src` and `dst` as
// `add_buffer*()` arguments and `k` as an `add_uniform()` argument, and the
// reflection table is `[src(READ), dst(READ_WRITE)]`.)
//  * `NativeShaderCompileInfo::push_constant_size` is the *total* size of those
// scalar parameters in bytes: it may be left 0 (the value is reflected from the
// compiled parameter list) but must match when it is nonzero. There is no push-
// constant *block* on the CUDA route - the values travel as ordinary kernel
// parameters, so a scalar parameter may be a small struct passed by value
// (`add_uniform(my_struct)` uses `sizeof(my_struct)` bytes).
//  * `block_size` comes from `NativeShaderCompileInfo::block_size`; when it is
// left 0 the kernel's declared maximum thread count (`__launch_bounds__(N)`,
// i.e. PTX `.maxntid`) is used, and the compile fails closed when the kernel
// declares neither. `shader_model` is ignored by the CUDA route.
//  * `load()` therefore rejects nothing that the CUDA route reflects: a native
// CUDA shader is buffers + scalars by construction. Non-pointer parameters are
// not reported as bindings (they are the uniform values), and buffer bindings
// carry `register_index` = the parameter index in the kernel signature (space
// 0), which is exactly the order `add_buffer*()` binds them by default.
//
// Binding a resource: `NativeShaderLauncher` accepts a reflection index
// (`add_buffer_by_index`, unambiguous), a `(register, space)` pair (HLSL
// register namespaces can collide, e.g. `register(t0)` and `register(b0)`), or
// positional arguments in the canonical `(space, register)` order that
// `NativeShaderMetadata::bindings` reports.

enum class NativeShaderLanguage : uint8_t {
    HLSL, // HLSL source, DirectX (DXIL) or Vulkan (SPIR-V via DXC)
    GLSL, // GLSL source, Vulkan only (SPIR-V via glslang)
    // CUDA C++ source compiled by NVRTC (CUDA backend only). Buffers are the
    // pointer parameters of the `__global__` entry, and non-pointer parameters
    // are scalar kernel parameters fed by `add_uniform`; see the CUDA route
    // notes above.
    CUDA_NVRTC
};

// Reflection kind of one shader resource.
enum class NativeShaderResourceKind : uint8_t {
    ConstantBuffer,       // cbuffer / std140 UBO
    Sampler,              // SamplerState / sampler
    StructuredBuffer,     // SRV StructuredBuffer
    RWStructuredBuffer,   // UAV RWStructuredBuffer
    ByteAddressBuffer,    // SRV ByteAddressBuffer
    RWByteAddressBuffer,  // UAV RWByteAddressBuffer
    TypedBuffer,          // SRV Buffer<T>
    RWTypedBuffer,        // UAV RWBuffer<T>
    Texture2D,            // SRV Texture2D
    RWTexture2D,          // UAV RWTexture2D
    Texture3D,            // SRV Texture3D
    RWTexture3D,          // UAV RWTexture3D
    AccelerationStructure // reserved (not supported yet)
};

// Binding-class -> default usage. SRV/CBV/sampler classes are reads, UAV
// classes are read-write. The default can be tightened (UAV -> READ under
// `allow_usage_override`) or upgraded for SRVs at `load()`.
[[nodiscard]] constexpr Usage native_shader_default_usage(NativeShaderResourceKind kind) noexcept {
    switch (kind) {
        case NativeShaderResourceKind::RWStructuredBuffer:
        case NativeShaderResourceKind::RWByteAddressBuffer:
        case NativeShaderResourceKind::RWTypedBuffer:
        case NativeShaderResourceKind::RWTexture2D:
        case NativeShaderResourceKind::RWTexture3D:
            return Usage::READ_WRITE;
        default: return Usage::READ;
    }
}

struct NativeShaderResourceBinding {
    NativeShaderResourceKind kind{NativeShaderResourceKind::StructuredBuffer};
    uint register_index{0u}; // HLSL register / GLSL binding
    uint space_index{0u};    // HLSL space / GLSL set
    uint array_size{1u};     // 1 == non-array
    Usage usage{Usage::READ};// effective usage (reflection default or override)
    uint32_t stride{0u};     // buffer element stride in bytes (0 when unknown)
    uint32_t size_bytes{0u}; // constant-buffer size in bytes (0 when unknown)
};

struct NativeShaderCompileInfo {
    NativeShaderLanguage language{NativeShaderLanguage::HLSL};
    luisa::string_view source;
    // Entry point of the compiled module: the HLSL function name, `main` for
    // GLSL, or the `__global__` function name for the CUDA route (where the
    // default `main` is understood as "the only kernel in the source", and is
    // rejected when the source declares several of them).
    luisa::string_view entry_point{"main"};
    luisa::string_view file_name; // diagnostics only
    uint shader_model{65u}; // DX SM for HLSL (ignored by GLSL and CUDA)
    // Optional workgroup size: 0 => take `[numthreads]` / `local_size_*`
    // (HLSL/GLSL) or the kernel's `__launch_bounds__` (CUDA).
    uint3 block_size{0u, 0u, 0u};
    // Uniform bytes: bytes; 0 => no push/root constant block. On the CUDA route
    // this is the total size of the scalar kernel parameters (0 => reflect it).
    uint32_t push_constant_size{0u};
    bool optimize{true};
    bool enable_fast_math{false};
    bool enable_debug_info{false};
};

struct NativeShaderCompileResult {
    luisa::vector<std::byte> binary; // DXIL, SPIR-V words, or NUL-terminated PTX
    luisa::vector<NativeShaderResourceBinding> bindings;  // reflection
    luisa::string error; // empty == success
    NativeShaderLanguage language{NativeShaderLanguage::HLSL};
    uint3 block_size{0u, 0u, 0u};
    uint32_t shader_model{0u};
    // Push-constant / root-constant block size carried over from the compile
    // info, so that `load(result)` is self-contained.
    uint32_t push_constant_size{0u};
    // The entry point `load(result)` must use. Empty means "derive it from the
    // binary" (the Vulkan and DirectX routes re-reflect their module, which
    // carries a single entry point); the CUDA route fills it in, because a PTX
    // module can hold every `__global__` function of the translation unit.
    luisa::string entry_point;
    [[nodiscard]] bool ok() const noexcept { return error.empty() && !binary.empty(); }
};

struct NativeShaderMetadata {
    uint64_t handle{invalid_resource_handle}; // backend shader instance pointer
    uint3 block_size{0u, 0u, 0u};
    luisa::vector<NativeShaderResourceBinding> bindings; // canonical order
    NativeShaderLanguage language{NativeShaderLanguage::HLSL};
    uint32_t push_constant_size{0u};
    [[nodiscard]] bool valid() const noexcept { return handle != invalid_resource_handle; }
};

class NativeShaderExt;  // fwd (create_shader returns a NativeShader)
class NativeShader;

class NativeShaderExt : public DeviceExtension {
protected:
    DeviceInterface *_device{nullptr};
    ~NativeShaderExt() noexcept = default;

public:
    static constexpr luisa::string_view name = "NativeShaderExt";
    explicit NativeShaderExt(DeviceInterface *device) noexcept : _device{device} {}

    // Compile native source to bytecode + reflection. Never throws: a failure
    // is reported through `NativeShaderCompileResult::error` with an empty
    // binary (fail-closed).
    [[nodiscard]] virtual NativeShaderCompileResult compile(
        const NativeShaderCompileInfo &info) noexcept = 0;

    // Create a backend compute-shader instance from a previous compile result.
    // `usage_override`, when non-empty, must have exactly one entry per
    // reflected binding (in `NativeShaderCompileResult::bindings` order) and
    // replaces the reflected usage per binding (subject to the SRV/UAV
    // consistency checks). Returns a metadata block whose `handle` is
    // `invalid_resource_handle` on failure.
    [[nodiscard]] virtual NativeShaderMetadata load(
        const NativeShaderCompileResult &result,
        luisa::span<const Usage> usage_override = {}) noexcept = 0;

    virtual void destroy_shader(uint64_t handle) noexcept = 0;

    // compile + load convenience; the returned shader is invalid when either
    // step fails (use `compile()` directly when the error message is needed).
    [[nodiscard]] NativeShader create_shader(const NativeShaderCompileInfo &info) noexcept;
};

// ---------------------------------------------------------------------------
// dispatch
// ---------------------------------------------------------------------------

// A compute dispatch of a native shader. Being a `CustomDispatchCommand`, the
// command-reorder pass tracks every resource argument with its declared usage
// and honours `max_dispatch_size()` in the per-layer thread budget.
class NativeShaderDispatchCommand final : public CustomDispatchCommand,
                                          public ShaderDispatchCommandBase {

private:
    // Exact number of threads the dispatch launches (grid * block); the
    // reorder budget counts threads.
    uint3 _dispatch_size;
    uint3 _block_size;
    // One entry per non-uniform argument, in the argument order.
    luisa::vector<Usage> _argument_usages;

    template<typename Self, typename Visitor>
    static void traverse(Self &&self, Visitor &visitor) noexcept {
        auto usage_index = 0u;
        auto next_usage = [&self, &usage_index]() noexcept {
            LUISA_ASSERT(usage_index < self._argument_usages.size(),
                         "Native shader dispatch has fewer argument usages "
                         "({}) than resource arguments.",
                         self._argument_usages.size());
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
            }
        }
        LUISA_ASSERT(usage_index == self._argument_usages.size(),
                     "Native shader dispatch consumed {} argument usages for "
                     "{} resource arguments.",
                     self._argument_usages.size(), usage_index);
    }

public:
    NativeShaderDispatchCommand(
        uint64_t shader_handle,
        uint3 dispatch_size,
        uint3 block_size,
        luisa::vector<std::byte> &&argument_buffer,
        size_t argument_count,
        luisa::vector<Usage> &&argument_usages) noexcept
        : ShaderDispatchCommandBase{shader_handle,
                                    std::move(argument_buffer),
                                    argument_count},
          _dispatch_size{dispatch_size},
          _block_size{block_size},
          _argument_usages{std::move(argument_usages)} {
        LUISA_ASSERT(shader_handle != invalid_resource_handle,
                     "Native shader dispatch requires a valid shader handle.");
        LUISA_ASSERT(dispatch_size.x > 0u && dispatch_size.y > 0u &&
                         dispatch_size.z > 0u,
                     "Native shader dispatch size must be nonzero.");
        LUISA_ASSERT(block_size.x > 0u && block_size.y > 0u &&
                         block_size.z > 0u,
                     "Native shader block size must be nonzero.");
    }
    NativeShaderDispatchCommand(NativeShaderDispatchCommand const &) = delete;
    NativeShaderDispatchCommand(NativeShaderDispatchCommand &&) noexcept = default;

public:
    [[nodiscard]] uint64_t custom_cmd_uuid() const noexcept override {
        return luisa::to_underlying(CustomCommandUUID::NATIVE_SHADER_DISPATCH);
    }
    [[nodiscard]] StreamTag stream_tag() const noexcept override {
        return StreamTag::COMPUTE;
    }
    [[nodiscard]] uint64_t shader_handle() const noexcept { return handle(); }
    // Exact thread count (the reorder budget's unit).
    [[nodiscard]] uint3 dispatch_size() const noexcept { return _dispatch_size; }
    [[nodiscard]] uint3 block_size() const noexcept { return _block_size; }
    [[nodiscard]] luisa::span<const Usage> argument_usages() const noexcept {
        return _argument_usages;
    }
    [[nodiscard]] uint3 max_dispatch_size() const noexcept override {
        return _dispatch_size;
    }
    // The reorder pass tracks the shader's accesses through its declared
    // argument usages; see the contract note at the top of this header.
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
// them into a `NativeShaderDispatchCommand`.
//
// Resource arguments are resolved against the shader's reflected binding table
// (canonical order: space, then register). Prefer the binding-aware overloads
// (an explicit `register`/`space` pair, or a resource name for DirectX): the
// positional overloads fill the *canonical* order, which is what `load()`
// reports in `NativeShaderMetadata::bindings`.
//
// Use `NativeShader::launcher()` to obtain a pre-parameterised launcher.
class NativeShaderLauncher {

public:
    struct Entry {
        Argument argument{};
        Usage usage{Usage::NONE};
        uint register_index{0u};
        uint space_index{0u};
        uint32_t binding_index{0u};
        bool binding_explicit{false};
        bool index_explicit{false};
    };

    // The resolved dispatch plan: resource arguments in canonical binding order
    // followed by the uniform arguments, the usage of every resource argument,
    // and the packed argument/uniform buffer.
    struct ResolvedPlan {
        luisa::vector<Argument> arguments;
        luisa::vector<Usage> usages;
        luisa::vector<std::byte> argument_buffer;
        luisa::string error;
        [[nodiscard]] bool ok() const noexcept { return error.empty(); }
    };

private:
    uint64_t _shader_handle{invalid_resource_handle};
    uint3 _block_size{0u, 0u, 0u};
    luisa::vector<NativeShaderResourceBinding> _bindings;
    luisa::vector<Entry> _resources;
    luisa::vector<std::byte> _uniform_blob;
    bool _allow_usage_override{false};

    [[nodiscard]] Entry &_create_entry(Usage usage) noexcept {
        auto &entry = _resources.emplace_back();
        entry.usage = usage;
        return entry;
    }
    [[nodiscard]] ResolvedPlan _plan() const noexcept;
    [[nodiscard]] luisa::unique_ptr<NativeShaderDispatchCommand>
    _build(uint64_t shader_handle, uint3 thread_count, uint3 block_size) && noexcept;

public:
    NativeShaderLauncher() noexcept = default;
    NativeShaderLauncher(uint64_t shader_handle,
                         uint3 block_size,
                         luisa::span<const NativeShaderResourceBinding> bindings) noexcept;
    NativeShaderLauncher(NativeShaderLauncher const &) = delete;
    NativeShaderLauncher(NativeShaderLauncher &&) noexcept = default;

    // Permits binding a UAV-class resource with a READ-only declaration (the
    // shader promises not to write through it). Off by default: an SRV bound
    // as WRITE is always rejected.
    NativeShaderLauncher &set_allow_usage_override(bool allow) noexcept {
        _allow_usage_override = allow;
        return *this;
    }
    [[nodiscard]] uint64_t shader_handle() const noexcept { return _shader_handle; }
    [[nodiscard]] uint3 block_size() const noexcept { return _block_size; }
    [[nodiscard]] luisa::span<const NativeShaderResourceBinding> bindings() const noexcept {
        return _bindings;
    }
    // Validates the collected arguments against the reflected binding table and
    // returns an empty string when the launcher is dispatchable. `build()`
    // asserts exactly this; applications that want a soft failure (or their own
    // diagnostics) can call it first.
    [[nodiscard]] luisa::string validate() const noexcept { return _plan().error; }
    // The resolved plan (canonical argument order, per-argument usages).
    [[nodiscard]] ResolvedPlan plan() const noexcept { return _plan(); }

    // ---- resources (positional: canonical reflected order) ---------------
    NativeShaderLauncher &add_buffer(uint64_t handle, size_t offset,
                                     size_t size, Usage usage) noexcept;
    template<typename T>
    NativeShaderLauncher &add_buffer(const BufferView<T> &view, Usage usage) noexcept {
        return add_buffer(view.handle(), view.offset_bytes(), view.size_bytes(), usage);
    }
    NativeShaderLauncher &add_texture(uint64_t handle, uint32_t level,
                                      Usage usage) noexcept;

    // ---- resources (binding-aware: recommended) --------------------------
    NativeShaderLauncher &add_buffer(uint32_t register_index, uint32_t space_index,
                                     uint64_t handle, size_t offset, size_t size,
                                     Usage usage) noexcept;
    template<typename T>
    NativeShaderLauncher &add_buffer(uint32_t register_index, uint32_t space_index,
                                     const BufferView<T> &view, Usage usage) noexcept {
        return add_buffer(register_index, space_index, view.handle(),
                          view.offset_bytes(), view.size_bytes(), usage);
    }
    NativeShaderLauncher &add_texture(uint32_t register_index, uint32_t space_index,
                                      uint64_t handle, uint32_t level,
                                      Usage usage) noexcept;

    // ---- resources (explicit binding index) ------------------------------
    // `binding_index` indexes the shader's reflection table, i.e. the order of
    // `NativeShaderMetadata::bindings` / `NativeShader::bindings()`. This is
    // the only unambiguous selector for a DirectX shader whose HLSL register
    // namespaces collide (`register(t0)` and `cbuffer ... : register(b0)` are
    // both bind point 0); prefer it over the (register, space) form whenever
    // the shader mixes namespaces.
    NativeShaderLauncher &add_buffer_by_index(uint32_t binding_index,
                                              uint64_t handle, size_t offset,
                                              size_t size, Usage usage) noexcept;
    template<typename T>
    NativeShaderLauncher &add_buffer_by_index(uint32_t binding_index,
                                              const BufferView<T> &view,
                                              Usage usage) noexcept {
        return add_buffer_by_index(binding_index, view.handle(),
                                   view.offset_bytes(), view.size_bytes(), usage);
    }
    NativeShaderLauncher &add_texture_by_index(uint32_t binding_index,
                                               uint64_t handle, uint32_t level,
                                               Usage usage) noexcept;

    // ---- uniforms (push constants / root 32-bit constants) ---------------
    NativeShaderLauncher &add_uniform(const void *data, size_t size,
                                      size_t alignment) noexcept;
    template<typename T>
    NativeShaderLauncher &add_uniform(const T &value) noexcept {
        return add_uniform(&value, sizeof(T), alignof(T));
    }

    // Exact-thread-count launch using this launcher's shader and block size.
    [[nodiscard]] luisa::unique_ptr<NativeShaderDispatchCommand>
    build(uint3 thread_count) && noexcept;
    // Standalone form: parameters the launcher was not built with.
    [[nodiscard]] luisa::unique_ptr<NativeShaderDispatchCommand>
    build(uint64_t shader_handle, uint3 thread_count, uint3 block_size) && noexcept;
};

// RAII owner of a shader instance created through `NativeShaderExt::load`;
// destroys it on destruction (or on `reset()`).
class NativeShader {

private:
    NativeShaderExt *_ext{nullptr};
    uint64_t _handle{invalid_resource_handle};
    NativeShaderMetadata _meta;

public:
    NativeShader() noexcept = default;
    NativeShader(NativeShaderExt &ext, NativeShaderMetadata meta) noexcept
        : _ext{&ext}, _handle{meta.handle}, _meta{std::move(meta)} {
        LUISA_ASSERT(_handle != invalid_resource_handle,
                     "NativeShader requires a valid shader handle.");
    }
    ~NativeShader() noexcept { reset(); }
    NativeShader(NativeShader const &) = delete;
    NativeShader &operator=(NativeShader const &) = delete;
    NativeShader(NativeShader &&other) noexcept
        : _ext{other._ext}, _handle{other._handle}, _meta{std::move(other._meta)} {
        other._ext = nullptr;
        other._handle = invalid_resource_handle;
    }
    NativeShader &operator=(NativeShader &&other) noexcept {
        if (this != &other) {
            reset();
            _ext = other._ext;
            _handle = other._handle;
            _meta = std::move(other._meta);
            other._ext = nullptr;
            other._handle = invalid_resource_handle;
        }
        return *this;
    }
    [[nodiscard]] explicit operator bool() const noexcept {
        return _handle != invalid_resource_handle;
    }
    [[nodiscard]] uint64_t handle() const noexcept { return _handle; }
    [[nodiscard]] uint3 block_size() const noexcept { return _meta.block_size; }
    [[nodiscard]] NativeShaderLanguage language() const noexcept { return _meta.language; }
    [[nodiscard]] luisa::span<const NativeShaderResourceBinding> bindings() const noexcept {
        return _meta.bindings;
    }
    [[nodiscard]] NativeShaderMetadata const &metadata() const noexcept { return _meta; }
    void reset() noexcept {
        if (_ext != nullptr && _handle != invalid_resource_handle) {
            _ext->destroy_shader(_handle);
        }
        _ext = nullptr;
        _handle = invalid_resource_handle;
        _meta = NativeShaderMetadata{};
    }
    // A launcher pre-parameterised with this shader's handle, block size and
    // reflected binding table.
    [[nodiscard]] NativeShaderLauncher launcher() const noexcept {
        return NativeShaderLauncher{_handle, _meta.block_size, _meta.bindings};
    }
};

inline NativeShader NativeShaderExt::create_shader(
    const NativeShaderCompileInfo &info) noexcept {
    auto result = compile(info);
    if (!result.ok()) { return NativeShader{}; }
    auto metadata = load(result);
    if (!metadata.valid()) { return NativeShader{}; }
    return NativeShader{*this, std::move(metadata)};
}

// ---------------------------------------------------------------------------
// launcher implementation
// ---------------------------------------------------------------------------

namespace detail {

// Canonical reflection order: (space, register, kind). Positional launcher
// arguments follow this order, exactly like `NativeShaderMetadata::bindings`.
[[nodiscard]] inline bool native_shader_binding_less(
    const NativeShaderResourceBinding &a,
    const NativeShaderResourceBinding &b) noexcept {
    if (a.space_index != b.space_index) { return a.space_index < b.space_index; }
    if (a.register_index != b.register_index) { return a.register_index < b.register_index; }
    return luisa::to_underlying(a.kind) < luisa::to_underlying(b.kind);
}

// Sorts a reflection table into the canonical order in place.
inline void native_shader_canonicalize_bindings(
    luisa::vector<NativeShaderResourceBinding> &bindings) noexcept {
    std::stable_sort(bindings.begin(), bindings.end(),
                     native_shader_binding_less);
}

}// namespace detail

inline NativeShaderLauncher::NativeShaderLauncher(
    uint64_t shader_handle, uint3 block_size,
    luisa::span<const NativeShaderResourceBinding> bindings) noexcept
    : _shader_handle{shader_handle},
      _block_size{block_size},
      _bindings{bindings.begin(), bindings.end()} {
    // Binding order is the launcher's positional order, so normalise the table
    // into the canonical (space, register) order that `load()` reports.
    detail::native_shader_canonicalize_bindings(_bindings);
}

inline NativeShaderLauncher &NativeShaderLauncher::add_buffer(
    uint64_t handle, size_t offset, size_t size, Usage usage) noexcept {
    auto &entry = _create_entry(usage);
    entry.argument.tag = Argument::Tag::BUFFER;
    entry.argument.buffer = Argument::Buffer{handle, offset, size};
    return *this;
}

inline NativeShaderLauncher &NativeShaderLauncher::add_buffer(
    uint32_t register_index, uint32_t space_index, uint64_t handle,
    size_t offset, size_t size, Usage usage) noexcept {
    auto &entry = _create_entry(usage);
    entry.argument.tag = Argument::Tag::BUFFER;
    entry.argument.buffer = Argument::Buffer{handle, offset, size};
    entry.register_index = register_index;
    entry.space_index = space_index;
    entry.binding_explicit = true;
    return *this;
}

inline NativeShaderLauncher &NativeShaderLauncher::add_texture(
    uint64_t handle, uint32_t level, Usage usage) noexcept {
    auto &entry = _create_entry(usage);
    entry.argument.tag = Argument::Tag::TEXTURE;
    entry.argument.texture = Argument::Texture{handle, level};
    return *this;
}

inline NativeShaderLauncher &NativeShaderLauncher::add_texture(
    uint32_t register_index, uint32_t space_index, uint64_t handle,
    uint32_t level, Usage usage) noexcept {
    auto &entry = _create_entry(usage);
    entry.argument.tag = Argument::Tag::TEXTURE;
    entry.argument.texture = Argument::Texture{handle, level};
    entry.register_index = register_index;
    entry.space_index = space_index;
    entry.binding_explicit = true;
    return *this;
}

inline NativeShaderLauncher &NativeShaderLauncher::add_buffer_by_index(
    uint32_t binding_index, uint64_t handle, size_t offset, size_t size,
    Usage usage) noexcept {
    auto &entry = _create_entry(usage);
    entry.argument.tag = Argument::Tag::BUFFER;
    entry.argument.buffer = Argument::Buffer{handle, offset, size};
    entry.binding_index = binding_index;
    entry.index_explicit = true;
    return *this;
}

inline NativeShaderLauncher &NativeShaderLauncher::add_texture_by_index(
    uint32_t binding_index, uint64_t handle, uint32_t level,
    Usage usage) noexcept {
    auto &entry = _create_entry(usage);
    entry.argument.tag = Argument::Tag::TEXTURE;
    entry.argument.texture = Argument::Texture{handle, level};
    entry.binding_index = binding_index;
    entry.index_explicit = true;
    return *this;
}

inline NativeShaderLauncher &NativeShaderLauncher::add_uniform(
    const void *data, size_t size, size_t alignment) noexcept {
    LUISA_ASSERT(data != nullptr && size > 0u,
                 "Native shader uniform requires data and a nonzero size.");
    LUISA_ASSERT(alignment > 0u && alignment <= 16u,
                 "Invalid native shader uniform alignment {}.", alignment);
    auto offset = luisa::align(_uniform_blob.size(), alignment);
    luisa::vector_resize(_uniform_blob, offset + size);
    std::memcpy(_uniform_blob.data() + offset, data, size);
    auto &entry = _create_entry(Usage::NONE);
    entry.argument.tag = Argument::Tag::UNIFORM;
    entry.argument.uniform = Argument::Uniform{offset, size, alignment};
    return *this;
}

inline luisa::unique_ptr<NativeShaderDispatchCommand>
NativeShaderLauncher::build(uint3 thread_count) && noexcept {
    LUISA_ASSERT(_shader_handle != invalid_resource_handle,
                 "Native shader launcher build() without a shader handle; use "
                 "build(handle, thread_count, block_size) or "
                 "NativeShader::launcher().");
    LUISA_ASSERT(_block_size.x > 0u && _block_size.y > 0u && _block_size.z > 0u,
                 "Native shader launcher build() without a block size.");
    auto handle = _shader_handle;
    auto block_size = _block_size;
    return std::move(*this)._build(handle, thread_count, block_size);
}

inline luisa::unique_ptr<NativeShaderDispatchCommand>
NativeShaderLauncher::build(uint64_t shader_handle, uint3 thread_count,
                            uint3 block_size) && noexcept {
    return std::move(*this)._build(shader_handle, thread_count, block_size);
}

inline NativeShaderLauncher::ResolvedPlan
NativeShaderLauncher::_plan() const noexcept {
    ResolvedPlan plan;
    // Resolve every resource argument against the reflected binding table.
    auto binding_count = _bindings.size();
    luisa::vector<bool> binding_used(binding_count, false);
    luisa::vector<std::pair<size_t, size_t>> resolved;// (binding index, entry index)
    resolved.reserve(_resources.size());
    auto next_unused_binding = [&]() noexcept -> size_t {
        for (auto i = 0u; i < binding_count; i++) {
            if (!binding_used[i]) { return i; }
        }
        return binding_count;
    };
    for (auto i = 0u; i < _resources.size(); i++) {
        auto &&entry = _resources[i];
        if (entry.argument.tag == Argument::Tag::UNIFORM) { continue; }
        auto argument_handle = [&entry]() noexcept {
            switch (entry.argument.tag) {
                case Argument::Tag::BUFFER: return entry.argument.buffer.handle;
                case Argument::Tag::TEXTURE: return entry.argument.texture.handle;
                case Argument::Tag::BINDLESS_ARRAY: return entry.argument.bindless_array.handle;
                case Argument::Tag::ACCEL: return entry.argument.accel.handle;
                case Argument::Tag::UNIFORM: break;
            }
            return invalid_resource_handle;
        }();
        if (argument_handle == invalid_resource_handle) {
            plan.error = luisa::format(
                "Native shader launcher argument {} has a null handle.", i);
            return plan;
        }
        size_t binding_index = binding_count;
        if (entry.index_explicit) {
            if (entry.binding_index >= binding_count) {
                plan.error = luisa::format(
                    "Native shader launcher argument {} selects binding index "
                    "{}, but the shader reflects only {} bindings.",
                    i, entry.binding_index, binding_count);
                return plan;
            }
            if (binding_used[entry.binding_index]) {
                plan.error = luisa::format(
                    "Native shader binding {} (register {}, space {}) was "
                    "supplied more than once.",
                    entry.binding_index,
                    _bindings[entry.binding_index].register_index,
                    _bindings[entry.binding_index].space_index);
                return plan;
            }
            binding_index = entry.binding_index;
        } else if (entry.binding_explicit) {
            // HLSL has separate register namespaces (t/u/b/s), so a shader may
            // declare `register(t0)` and `register(u0)` at the same
            // (space, register) pair; the declared usage disambiguates them
            // (a write-capable declaration targets the UAV-class binding).
            auto wants_uav = (luisa::to_underlying(entry.usage) &
                              luisa::to_underlying(Usage::WRITE)) != 0u;
            auto fallback = binding_count;
            auto pair_exists = false;
            for (auto b = 0u; b < binding_count; b++) {
                if (_bindings[b].register_index != entry.register_index ||
                    _bindings[b].space_index != entry.space_index) {
                    continue;
                }
                pair_exists = true;
                if (binding_used[b]) { continue; }
                auto is_uav = native_shader_default_usage(_bindings[b].kind) ==
                              Usage::READ_WRITE;
                if (is_uav == wants_uav) {
                    binding_index = b;
                    break;
                }
                if (fallback >= binding_count) { fallback = b; }
            }
            if (binding_index >= binding_count) { binding_index = fallback; }
            if (binding_index >= binding_count) {
                plan.error = pair_exists ?
                                 luisa::format(
                                     "Native shader binding (register {}, space {}) "
                                     "was supplied more than once.",
                                     entry.register_index, entry.space_index) :
                                 luisa::format(
                                     "Native shader has no resource at register {} "
                                     "space {} (declared explicitly by launcher "
                                     "argument {}).",
                                     entry.register_index, entry.space_index, i);
                return plan;
            }
        } else {
            binding_index = next_unused_binding();
            if (binding_index >= binding_count) {
                plan.error = luisa::format(
                    "Native shader launcher supplied more resource arguments "
                    "than the shader declares ({} bindings).",
                    binding_count);
                return plan;
            }
        }
        if (binding_used[binding_index]) {
            plan.error = luisa::format(
                "Native shader binding {} (register {}, space {}) was supplied "
                "more than once.",
                binding_index, _bindings[binding_index].register_index,
                _bindings[binding_index].space_index);
            return plan;
        }
        binding_used[binding_index] = true;
        resolved.emplace_back(binding_index, i);
    }
    for (auto b = 0u; b < binding_count; b++) {
        if (!binding_used[b]) {
            plan.error = luisa::format(
                "Native shader binding {} (register {}, space {}) was not "
                "supplied to the launcher.",
                b, _bindings[b].register_index, _bindings[b].space_index);
            return plan;
        }
    }

    // Usage cross-check against the reflected binding class (the reorder pass
    // trusts these declarations).
    auto usage_error = [this](NativeShaderResourceBinding const &binding,
                              Usage usage) noexcept -> luisa::string {
        auto writes = (luisa::to_underlying(usage) &
                       luisa::to_underlying(Usage::WRITE)) != 0u;
        auto is_uav = native_shader_default_usage(binding.kind) == Usage::READ_WRITE;
        if (usage == Usage::NONE) {
            return luisa::format(
                "Native shader binding (register {}, space {}) was declared "
                "with Usage::NONE.",
                binding.register_index, binding.space_index);
        }
        if (is_uav) {
            if (!writes && !_allow_usage_override) {
                return luisa::format(
                    "Native shader UAV binding (register {}, space {}) is bound "
                    "with a read-only Usage::READ declaration; a UAV-class "
                    "resource must be declared WRITE or READ_WRITE unless the "
                    "launcher explicitly allows the read-only override.",
                    binding.register_index, binding.space_index);
            }
        } else if (writes) {
            return luisa::format(
                "Native shader SRV/CBV binding (register {}, space {}) is bound "
                "with a write-capable usage declaration; read-only resources "
                "must be declared Usage::READ.",
                binding.register_index, binding.space_index);
        }
        return {};
    };

    // Canonical argument order: resources sorted by binding index, then the
    // uniform arguments.
    std::sort(resolved.begin(), resolved.end(),
              [](auto const &a, auto const &b) noexcept { return a.first < b.first; });
    plan.arguments.reserve(_resources.size());
    plan.usages.reserve(resolved.size());
    for (auto &&[binding_index, entry_index] : resolved) {
        auto &&entry = _resources[entry_index];
        if (auto error = usage_error(_bindings[binding_index], entry.usage);
            !error.empty()) {
            plan.error = std::move(error);
            return plan;
        }
        plan.arguments.emplace_back(entry.argument);
        plan.usages.emplace_back(entry.usage);
    }
    for (auto &&entry : _resources) {
        if (entry.argument.tag == Argument::Tag::UNIFORM) {
            plan.arguments.emplace_back(entry.argument);
        }
    }

    // Pack: [Argument array][uniform blob], with uniform offsets shifted past
    // the header (mirrors the vk/cuda interop launcher).
    auto argument_header_size = plan.arguments.size() * sizeof(Argument);
    luisa::vector_resize(plan.argument_buffer,
                         argument_header_size + _uniform_blob.size());
    if (argument_header_size > 0u) {
        std::memcpy(plan.argument_buffer.data(), plan.arguments.data(),
                    argument_header_size);
    }
    if (!_uniform_blob.empty()) {
        std::memcpy(plan.argument_buffer.data() + argument_header_size,
                    _uniform_blob.data(), _uniform_blob.size());
    }
    if (argument_header_size > 0u) {
        auto *args = std::launder(
            reinterpret_cast<Argument *>(plan.argument_buffer.data()));
        for (auto i = 0u; i < plan.arguments.size(); i++) {
            if (args[i].tag == Argument::Tag::UNIFORM) {
                args[i].uniform.offset += argument_header_size;
            }
        }
    }
    return plan;
}

inline luisa::unique_ptr<NativeShaderDispatchCommand>
NativeShaderLauncher::_build(uint64_t shader_handle, uint3 thread_count,
                             uint3 block_size) && noexcept {
    LUISA_ASSERT(shader_handle != invalid_resource_handle,
                 "Native shader dispatch requires a valid shader handle.");
    LUISA_ASSERT(thread_count.x > 0u && thread_count.y > 0u && thread_count.z > 0u,
                 "Native shader dispatch thread count must be nonzero.");
    LUISA_ASSERT(block_size.x > 0u && block_size.y > 0u && block_size.z > 0u,
                 "Native shader dispatch block size must be nonzero.");
    if (_block_size.x > 0u && _block_size.y > 0u && _block_size.z > 0u) {
        LUISA_ASSERT(_block_size.x == block_size.x &&
                         _block_size.y == block_size.y &&
                         _block_size.z == block_size.z,
                     "Native shader launcher block size ({}, {}, {}) does not "
                     "match the shader's compiled workgroup size ({}, {}, {}).",
                     _block_size.x, _block_size.y, _block_size.z,
                     block_size.x, block_size.y, block_size.z);
    }
    auto plan = _plan();
    LUISA_ASSERT(plan.error.empty(),
                 "Native shader dispatch is not dispatchable: {}",
                 plan.error);
    return luisa::make_unique<NativeShaderDispatchCommand>(
        shader_handle, thread_count, block_size,
        std::move(plan.argument_buffer), plan.arguments.size(),
        std::move(plan.usages));
}

}// namespace luisa::compute
