// Host-only product of one compute-shader compilation (plan P5.1).
//
// `Device::create_shader` is split into two phases so that a batch of shaders can
// be prepared on a worker pool and published (constructed) in request order on one
// thread:
//
//  * **prepare** (`Device::_prepare_compute_shader`) does everything that only
//    produces host data: the codegen-route decision, the shader-cache lookup, XIR
//    -> SPIR-V code generation, the SPIR-V optimizer/validator, the bytecode
//    artifact encoding and the legacy PSO seed lookup. It creates **no** Vulkan
//    object, so it may run on any thread.
//  * **publish** (`Device::_publish_compute_shader`) turns one prepared record
//    into a `ComputeShader`: `vkCreateShaderModule`, `vkCreateComputePipelines`
//    (under the device pipeline cache's lease), the interned-layout acquisition of
//    the new shader, the artifact write into the BinaryIO, and the legacy PSO seed
//    merge. It runs on one thread at a time, in request order.
//
// `PreparedComputeShader` therefore holds no handle of any kind, and a serial
// `create_shader` is exactly `publish(prepare(...))`.
//
// `WorkerContext` is the per-worker compiler state. A worker owns one
// compatibility DXC compiler (`hlsl::ShaderCompiler`) when the build has the HLSL
// route and the shader needs it; a worker never shares that state with another
// thread or with `Device::compiler()` (the process-global instance the serial path
// uses), because nothing in this tree establishes that a DXC compiler object may
// be used from two threads at once.
#pragma once
#include <cstdint>
#include <vector>
#include <luisa/core/basic_types.h>
#include <luisa/core/stl/optional.h>
#include <luisa/core/stl/string.h>
#include <luisa/core/stl/vector.h>
#include <luisa/core/stl/memory.h>
#include <luisa/vstl/common.h>
#include <luisa/vstl/md5.h>
#include <luisa/runtime/rhi/argument.h>
#include "saved_argument_contract.h"
#include "serde_type.h"
#include "shader_interface_plan.h"
#include "../common/hlsl/shader_property.h"
namespace lc::hlsl {
class ShaderCompiler;
}// namespace lc::hlsl
namespace lc::vk {
namespace detail {
struct PreparedComputeShader {
    enum class Kind : uint8_t {
        // Nothing to publish: the compilation already reported its failure (or
        // there is simply no shader object to build, e.g. `compile_only`). The
        // caller answers with `ShaderCreationInfo::make_invalid()` semantics.
        NOTHING,
        // The request must be compiled on the publishing thread, through the
        // unchanged serial path (`_create_shader_hlsl`). `option`/`kernel` are the
        // caller's own, so nothing else has to be carried here.
        SERIAL_FALLBACK,
        // A `ComputeShader` can be constructed from the fields below.
        READY,
    };
    Kind kind{Kind::NOTHING};
    // True when the artifact came from the BinaryIO instead of code generation
    // (diagnostics only: it is what `cache_hits` counts).
    bool from_cache{false};

    // ---- construction inputs (Kind::READY) ------------------------------------
    luisa::uint3 block_size{};
    vstd::vector<hlsl::Property> properties;
    vstd::vector<SavedArgument> saved_arguments;
    vstd::vector<luisa::compute::Argument> captured;
    std::vector<uint32_t> spirv;
    bool use_tex2d_bindless{false};
    bool use_tex3d_bindless{false};
    bool use_buffer_bindless{false};
    vstd::vector<std::pair<luisa::string, luisa::compute::Type const *>> printers;
    luisa::vector<std::byte> constant_ubo_data;
    uint validation_count{0u};
    luisa::optional<uint8_t> required_subgroup_size;
    uint32_t push_constant_size{32u};
    ShaderCodegenDialect codegen_dialect{ShaderCodegenDialect::HLSL_SPIRV};
    bool enable_driver_optimization{true};
    // False for a `compile_only` request: the artifact (if any) is still written,
    // but no `Shader` object is created. Always false for `Kind::SERIAL_FALLBACK`
    // and `Kind::NOTHING`.
    bool construct_shader{true};
    // The answer `ShaderCreationInfo::block_size` carries; equals `block_size` for
    // a constructed shader and the kernel's block size for a compile-only request.
    luisa::uint3 reported_block_size{};

    // ---- publication side effects (Kind::READY, serial phase) -----------------
    // Bytecode artifact to persist, produced by the prepare phase. The bytes are
    // the encoder's output for exactly the arguments a serial compile would pass,
    // so the artifact is identical to the serial one.
    bool write_artifact{false};
    luisa::string artifact_name;
    SerdeType artifact_type{SerdeType::kCache};
    luisa::vector<std::byte> artifact_bytes;
    // Pre-P1.3 per-shader PSO blob name to fold into the device cache (cache-hit
    // path, plan P1.3). Empty means "no legacy seed".
    luisa::string pso_seed_name;

    // ---- serial fallback payload ---------------------------------------------
    bool fallback_requires_sampler_anisotropy{false};
    bool fallback_rtx{false};

    [[nodiscard]] bool ready() const noexcept { return kind == Kind::READY; }
};

// Per-worker compiler state.
//
// It is empty, and that is a *finding* of the P5.1 audit rather than an omission:
//  * XIR -> SPIR-V (`SpirvCodegenEntry::compile_spirv`) builds a fresh
//    `glslang::spv::Builder` plus a `StringScratch` per call, and the SPIR-V
//    optimizer/validator helpers (`optimize_spirv`, `validate_spirv`) construct a
//    `spvtools::Optimizer` / `spvtools::SpirvTools` per call. Nothing in that path
//    is process-global, so two workers cannot share or corrupt it.
//  * The glslang *front end* (which does keep process-global tables and is
//    serialized by `glslang_compiler.cpp`) is only used by the native GLSL shader
//    route, never by `create_shader`.
//  * The one compiler object with a documented process-global instance is the
//    compatibility DXC module (`Device::compiler()`), and the HLSL-to-SPIR-V route
//    is not prepared on a worker at all: it fails closed to the serial path, where
//    that instance is used exactly as before.
// A future change that prepares the HLSL route in parallel must give each worker
// its own `hlsl::ShaderCompiler` here (and owns it on this thread) instead of
// reaching for `Device::compiler()`.
struct WorkerContext {
    // Index of the owning worker inside the pool (diagnostics only).
    uint32_t index{0u};
    // Diagnostics: how many requests this worker prepared.
    uint64_t prepared{0u};
};
}
}// namespace lc::vk::detail