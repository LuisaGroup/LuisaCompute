#pragma once

#include <array>
#include <luisa/tile/dsl.h>
#include <luisa/runtime/buffer.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/shader.h>

namespace luisa::compute::tile {

namespace bridge::tirx {
struct CompileOptions;
}// namespace bridge::tirx
namespace bridge::xir {
struct PlannerOptions;
}// namespace bridge::xir
enum class Lowering : uint8_t { NATIVE,
                                TIRX };

struct CompileOptions {
    // Zero selects a legal default. Nonzero is an exact constraint, not a hint.
    uint32_t threads_per_group{0u};
    Lowering lowering{Lowering::NATIVE};
    // Optional borrowed configuration for this synchronous compilation only.
    // No TVM dependency is introduced into the public Runtime/TileIR layers.
    // The backend supplies the physical target and enforces its capabilities.
    const bridge::tirx::CompileOptions *tirx{nullptr};
    // Optional exact constraints/cost prior for the CPU XIR execution planner.
    // Unsupported backends reject this configuration instead of ignoring it.
    const bridge::xir::PlannerOptions *xir{nullptr};
};

struct KernelArgument {
    ScalarType element{ScalarType::INVALID};
    size_t minimum_size_bytes{0u};
    Usage usage{Usage::NONE};
};

struct KernelMetadata {
    uint3 dispatch_size{0u};
    luisa::vector<KernelArgument> arguments;
    // The bootstrap view-forwarding realization requires disjoint writable
    // arguments. This is checked at invocation, not asserted as buffer noalias.
    bool disjoint_writes{false};
    luisa::string error;
    luisa::string source;
    luisa::string realization;
};

class Shader final : public ShaderBase {
private:
    KernelMetadata _metadata;

public:
    Shader(DeviceInterface *device, const ShaderCreationInfo &info,
           KernelMetadata metadata) noexcept
        : ShaderBase{device, info, 0u}, _metadata{std::move(metadata)} {}
    Shader(Shader &&) noexcept = default;
    Shader &operator=(Shader &&rhs) noexcept {
        if (this != &rhs) {
            ShaderBase::operator=(std::move(rhs));
            _block_size = rhs._block_size;
            _metadata = std::move(rhs._metadata);
        }
        return *this;
    }
    [[nodiscard]] const KernelMetadata &metadata() const noexcept { return _metadata; }

    class Invocation final {
    private:
        ComputeDispatchCmdEncoder _encoder;
    public:
        explicit Invocation(ComputeDispatchCmdEncoder encoder) noexcept : _encoder{std::move(encoder)} {}
        [[nodiscard]] auto dispatch() && noexcept { return std::move(_encoder).build(); }
    };

    template<typename... Args>
        requires((is_buffer_or_view_v<Args> && scalar_cpp_type<buffer_element_t<Args>>) && ...)
    [[nodiscard]] Invocation operator()(const Args &...args) const noexcept {
        LUISA_ASSERT(static_cast<bool>(*this), "Cannot invoke native Tile shader: {}", _metadata.error);
        LUISA_ASSERT(sizeof...(Args) == _metadata.arguments.size(), "Native Tile argument count mismatch.");
        struct Binding {
            void *native;
            uint64_t handle;
            size_t offset;
            size_t size;
            ScalarType element;
        };
        auto binding = [&]<typename T>(const T &arg) noexcept {
            if constexpr (is_buffer_v<T>) {
                LUISA_ASSERT(arg.device() == device(), "Native Tile buffer belongs to another Device.");
            }
            auto view = [&] {
                if constexpr (is_buffer_v<T>) {
                    return arg.view();
                } else {
                    return arg;
                }
            }();
            LUISA_ASSERT(view && view.stride() == sizeof(buffer_element_t<T>), "Invalid native Tile buffer view.");
            return Binding{view.native_handle(), view.handle(), view.offset_bytes(), view.size_bytes(), scalar_type_v<buffer_element_t<T>>};
        };
        std::array<Binding, sizeof...(Args)> bindings{binding(args)...};
        ComputeDispatchCmdEncoder encoder{handle(), bindings.size(), 0u};
        for (auto i = size_t{0}; i < bindings.size(); i++) {
            auto &b = bindings[i];
            auto &a = _metadata.arguments[i];
            LUISA_ASSERT(a.element == b.element && b.size >= a.minimum_size_bytes, "Native Tile argument {} type/size mismatch.", i);
            if (_metadata.disjoint_writes) {
                for (auto j = size_t{0}; j < i; j++) {
                    auto &other = bindings[j];
                    auto writes = (to_underlying(a.usage) | to_underlying(_metadata.arguments[j].usage)) & to_underlying(Usage::WRITE);
                    auto same = b.handle == other.handle || (b.native != nullptr && b.native == other.native);
                    auto overlap = b.offset <= other.offset ? other.offset - b.offset < b.size : b.offset - other.offset < other.size;
                    LUISA_ASSERT(!(writes && same && overlap), "Native Tile writable arguments {} and {} overlap.", i, j);
                }
            }
            encoder.encode_buffer(b.handle, b.offset, b.size);
        }
        encoder.set_dispatch_size(_metadata.dispatch_size);
        return Invocation{std::move(encoder)};
    }
};

// Native entry deliberately lives in the opt-in Runtime adapter. TileIR and
// the ordinary Tile DSL still link only the core library, with no TVM or RHI.
[[nodiscard]] inline Shader compile(Device &device, const Kernel &kernel,
                                    const CompileOptions &tile_options = {},
                                    const ShaderOption &shader_options = {.enable_fast_math = false}) noexcept {
    KernelMetadata metadata;
    auto info = ShaderCreationInfo::make_invalid();
    if (!kernel.valid()) {
        metadata.error = "Cannot compile invalid TileIR";
    } else if (shader_options.compile_only) {
        metadata.error = "Native Tile compile-only archives are not supported yet";
    } else {
        info = device.impl()->create_tile_kernel(shader_options, kernel.function(), tile_options, metadata);
        if (!info.valid() && metadata.error.empty()) { metadata.error = "Native Tile compilation is unavailable on this device"; }
    }
    return Shader{device.impl(), info, std::move(metadata)};
}

}// namespace luisa::compute::tile
