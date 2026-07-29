//
// Created by mike on 12/25/25.
//

#include <luisa/core/dll_export.h>
#include <luisa/core/clock.h>
#include <luisa/core/stl/hash.h>
#include <luisa/runtime/dispatch_buffer.h>

#include "hip_check.h"
#include "hip_buffer.h"
#include "hip_texture.h"
#include "hip_bindless_array.h"
#include "hip_stream.h"
#include "hip_event.h"
#include "hip_swapchain.h"
#include "hip_shader.h"
#include "hip_shader_native.h"
#include "hip_mesh.h"
#include "hip_curve.h"
#include "hip_motion_instance.h"
#include "hip_motion_mesh_builtin.h"
#include "hip_procedural_primitive.h"
#include "hip_accel.h"
#include "hip_pinned_memory.h"
#include "hip_device.h"

#ifdef LUISA_ENABLE_IR
#include <luisa/ir/ir2ast.h>
#endif

#ifdef LUISA_COMPUTE_ENABLE_LLVM
#include <luisa/xir/translators/ast2xir.h>
#include <luisa/xir/passes/destructure_cfg.h>
#include <luisa/xir/passes/simplify_cfg.h>
#include <luisa/xir/passes/restructure_cfg.h>
#include <luisa/xir/passes/early_return_elimination.h>
#include <luisa/xir/passes/lower_ray_query_loop.h>
#include <luisa/xir/passes/autodiff.h>
#include <luisa/xir/passes/inline.h>
#include <luisa/xir/passes/pass_pipeline.h>
#include <luisa/xir/verifier.h>
#include <llvm/Config/llvm-config.h>
#include <llvm/TargetParser/TargetParser.h>
#include "llvm_codegen/hip_codegen_llvm.h"
#endif

namespace luisa::compute::hip {

#ifdef LUISA_COMPUTE_ENABLE_LLVM
namespace {

static constexpr char hip_shader_package_magic[] = "LCHIPAOT";
static constexpr auto hip_shader_package_version = 2u;
static constexpr char hip_shader_cache_magic[] = "LCHIPCCH";
static constexpr auto hip_shader_cache_artifact_version = 1u;
// Increment whenever the HIP AST/XIR/LLVM lowering contract changes in a way
// that can alter generated code without changing the kernel AST hash.
static constexpr auto hip_shader_cache_codegen_revision = 1u;
static constexpr auto hip_shader_cache_max_artifact_size = 1ull << 30u;
static constexpr auto hip_shader_cache_payload_hash_seed =
    0x4849504341434845ull;

class HIPShaderPackageWriter {

private:
    luisa::vector<std::byte> _data;

public:
    void write_bytes(const void *data, size_t size) noexcept {
        auto bytes = static_cast<const std::byte *>(data);
        _data.insert(_data.end(), bytes, bytes + size);
    }

    void write_u8(uint8_t value) noexcept {
        _data.emplace_back(static_cast<std::byte>(value));
    }

    void write_u32(uint32_t value) noexcept {
        for (auto i = 0u; i < 4u; i++) {
            write_u8(static_cast<uint8_t>(value >> (i * 8u)));
        }
    }

    void write_u64(uint64_t value) noexcept {
        for (auto i = 0u; i < 8u; i++) {
            write_u8(static_cast<uint8_t>(value >> (i * 8u)));
        }
    }

    void write_string(luisa::string_view value) noexcept {
        write_u64(value.size());
        write_bytes(value.data(), value.size());
    }

    [[nodiscard]] luisa::vector<std::byte> finish() && noexcept {
        return std::move(_data);
    }
};

class HIPShaderPackageReader {

private:
    luisa::span<const std::byte> _data;
    size_t _offset{};

public:
    explicit HIPShaderPackageReader(luisa::span<const std::byte> data) noexcept
        : _data{data} {}

    [[nodiscard]] bool read_bytes(void *dst, size_t size) noexcept {
        if (size > _data.size() - std::min(_offset, _data.size())) { return false; }
        std::memcpy(dst, _data.data() + _offset, size);
        _offset += size;
        return true;
    }

    [[nodiscard]] bool read_u8(uint8_t &value) noexcept {
        std::byte byte{};
        if (!read_bytes(&byte, sizeof(byte))) { return false; }
        value = std::to_integer<uint8_t>(byte);
        return true;
    }

    [[nodiscard]] bool read_u32(uint32_t &value) noexcept {
        value = 0u;
        for (auto i = 0u; i < 4u; i++) {
            uint8_t byte{};
            if (!read_u8(byte)) { return false; }
            value |= static_cast<uint32_t>(byte) << (i * 8u);
        }
        return true;
    }

    [[nodiscard]] bool read_u64(uint64_t &value) noexcept {
        value = 0u;
        for (auto i = 0u; i < 8u; i++) {
            uint8_t byte{};
            if (!read_u8(byte)) { return false; }
            value |= static_cast<uint64_t>(byte) << (i * 8u);
        }
        return true;
    }

    [[nodiscard]] bool read_string(luisa::string &value) noexcept {
        uint64_t size{};
        if (!read_u64(size) || size > remaining()) { return false; }
        value.resize(static_cast<size_t>(size));
        return read_bytes(value.data(), value.size());
    }

    [[nodiscard]] size_t remaining() const noexcept {
        return _offset <= _data.size() ? _data.size() - _offset : 0u;
    }
};

struct HIPShaderPackage {
    HIPShaderMetadata metadata;
    luisa::string amdgpu_arch;
    uint32_t wave_size{};
    luisa::string code;
};

struct HIPShaderCacheArtifact {
    luisa::vector<std::byte> identity;
    HIPShaderPackage package;
};

[[nodiscard]] luisa::vector<std::byte> serialize_hip_shader_package(
    luisa::string_view code, const HIPShaderMetadata &metadata,
    luisa::string_view amdgpu_arch, uint32_t wave_size) noexcept {
    HIPShaderPackageWriter writer;
    writer.write_bytes(hip_shader_package_magic, sizeof(hip_shader_package_magic) - 1u);
    writer.write_u32(hip_shader_package_version);
    writer.write_u64(metadata.checksum);
    writer.write_u64(metadata.curve_bases.to_u64());
    writer.write_u8(static_cast<uint8_t>(metadata.kind));
    auto flags = static_cast<uint32_t>(metadata.enable_debug) |
                 static_cast<uint32_t>(metadata.requires_trace_closest) << 1u |
                 static_cast<uint32_t>(metadata.requires_trace_any) << 2u |
                 static_cast<uint32_t>(metadata.requires_ray_query) << 3u |
                 static_cast<uint32_t>(metadata.requires_printing) << 4u |
                 static_cast<uint32_t>(metadata.requires_motion_blur) << 5u |
                 static_cast<uint32_t>(metadata.requires_global_rt_stack) << 6u;
    writer.write_u32(flags);
    writer.write_u32(metadata.max_register_count);
    writer.write_u32(metadata.block_size.x);
    writer.write_u32(metadata.block_size.y);
    writer.write_u32(metadata.block_size.z);
    writer.write_u32(static_cast<uint32_t>(metadata.argument_types.size()));
    for (auto &&type : metadata.argument_types) { writer.write_string(type); }
    writer.write_u32(static_cast<uint32_t>(metadata.argument_usages.size()));
    for (auto usage : metadata.argument_usages) {
        writer.write_u32(static_cast<uint32_t>(usage));
    }
    writer.write_u32(static_cast<uint32_t>(metadata.format_types.size()));
    for (auto &&[format, type] : metadata.format_types) {
        writer.write_string(format);
        writer.write_string(type);
    }
    writer.write_string(amdgpu_arch);
    writer.write_u32(wave_size);
    writer.write_string(code);
    return std::move(writer).finish();
}

[[nodiscard]] luisa::optional<HIPShaderPackage> deserialize_hip_shader_package(
    luisa::span<const std::byte> data) noexcept {
    HIPShaderPackageReader reader{data};
    char magic[sizeof(hip_shader_package_magic) - 1u]{};
    uint32_t version{};
    if (!reader.read_bytes(magic, sizeof(magic)) ||
        std::memcmp(magic, hip_shader_package_magic, sizeof(magic)) != 0 ||
        !reader.read_u32(version) ||
        (version != 1u && version != hip_shader_package_version)) {
        return luisa::nullopt;
    }
    HIPShaderPackage package{};
    uint64_t curve_bases{};
    uint8_t kind{};
    uint32_t flags{};
    if (!reader.read_u64(package.metadata.checksum) ||
        !reader.read_u64(curve_bases) ||
        !reader.read_u8(kind) ||
        !reader.read_u32(flags) ||
        !reader.read_u32(package.metadata.max_register_count) ||
        !reader.read_u32(package.metadata.block_size.x) ||
        !reader.read_u32(package.metadata.block_size.y) ||
        !reader.read_u32(package.metadata.block_size.z)) {
        return luisa::nullopt;
    }
    if (kind != static_cast<uint8_t>(HIPShaderMetadata::Kind::COMPUTE) &&
        kind != static_cast<uint8_t>(HIPShaderMetadata::Kind::RAY_TRACING)) {
        return luisa::nullopt;
    }
    package.metadata.curve_bases = CurveBasisSet::from_u64(curve_bases);
    package.metadata.kind = static_cast<HIPShaderMetadata::Kind>(kind);
    package.metadata.enable_debug = (flags & (1u << 0u)) != 0u;
    package.metadata.requires_trace_closest = (flags & (1u << 1u)) != 0u;
    package.metadata.requires_trace_any = (flags & (1u << 2u)) != 0u;
    package.metadata.requires_ray_query = (flags & (1u << 3u)) != 0u;
    package.metadata.requires_printing = (flags & (1u << 4u)) != 0u;
    package.metadata.requires_motion_blur = (flags & (1u << 5u)) != 0u;
    package.metadata.requires_global_rt_stack =
        version >= 2u && (flags & (1u << 6u)) != 0u;
    if ((flags & ~(version >= 2u ? 0x7fu : 0x3fu)) != 0u) {
        return luisa::nullopt;
    }
    auto read_count = [&reader](uint32_t &count) noexcept {
        return reader.read_u32(count) && count <= (1u << 20u);
    };
    uint32_t count{};
    if (!read_count(count)) { return luisa::nullopt; }
    package.metadata.argument_types.resize(count);
    for (auto &type : package.metadata.argument_types) {
        if (!reader.read_string(type)) { return luisa::nullopt; }
    }
    if (!read_count(count)) { return luisa::nullopt; }
    package.metadata.argument_usages.reserve(count);
    for (auto i = 0u; i < count; i++) {
        uint32_t usage{};
        if (!reader.read_u32(usage) || usage > static_cast<uint32_t>(Usage::READ_WRITE)) {
            return luisa::nullopt;
        }
        package.metadata.argument_usages.emplace_back(static_cast<Usage>(usage));
    }
    if (!read_count(count)) { return luisa::nullopt; }
    package.metadata.format_types.resize(count);
    for (auto &[format, type] : package.metadata.format_types) {
        if (!reader.read_string(format) || !reader.read_string(type)) {
            return luisa::nullopt;
        }
    }
    if (!reader.read_string(package.amdgpu_arch) ||
        !reader.read_u32(package.wave_size) ||
        !reader.read_string(package.code) || reader.remaining() != 0u) {
        return luisa::nullopt;
    }
    if (version == 1u) {
        // Version 1 kernels used a 16-byte global-stack kernarg tail for every
        // software-stack RT shader and for every motion shader, including
        // direct motion traces. Preserve that legacy ABI when loading an old
        // package instead of launching it with a shortened argument buffer.
        auto legacy_hardware_stack =
            package.amdgpu_arch == "gfx1200" ||
            package.amdgpu_arch == "gfx1201";
        auto package_requires_hiprt =
            package.metadata.kind == HIPShaderMetadata::Kind::RAY_TRACING;
        package.metadata.requires_global_rt_stack =
            package_requires_hiprt &&
            (!legacy_hardware_stack || package.metadata.requires_motion_blur);
    }
    return package;
}

[[nodiscard]] luisa::vector<std::byte> serialize_hip_shader_cache_artifact(
    luisa::span<const std::byte> identity,
    luisa::span<const std::byte> package) noexcept {
    HIPShaderPackageWriter writer;
    writer.write_bytes(
        hip_shader_cache_magic,
        sizeof(hip_shader_cache_magic) - 1u);
    writer.write_u32(hip_shader_cache_artifact_version);
    writer.write_u64(identity.size_bytes());
    writer.write_bytes(identity.data(), identity.size_bytes());
    writer.write_u64(package.size_bytes());
    writer.write_u64(luisa::hash64(
        package.data(), package.size_bytes(),
        hip_shader_cache_payload_hash_seed));
    writer.write_bytes(package.data(), package.size_bytes());
    return std::move(writer).finish();
}

[[nodiscard]] luisa::optional<HIPShaderCacheArtifact>
deserialize_hip_shader_cache_artifact(
    luisa::span<const std::byte> data) noexcept {
    if (data.size_bytes() > hip_shader_cache_max_artifact_size) {
        return luisa::nullopt;
    }
    HIPShaderPackageReader reader{data};
    char magic[sizeof(hip_shader_cache_magic) - 1u]{};
    uint32_t version{};
    uint64_t identity_size{};
    if (!reader.read_bytes(magic, sizeof(magic)) ||
        std::memcmp(magic, hip_shader_cache_magic, sizeof(magic)) != 0 ||
        !reader.read_u32(version) ||
        version != hip_shader_cache_artifact_version ||
        !reader.read_u64(identity_size) ||
        identity_size > reader.remaining()) {
        return luisa::nullopt;
    }
    HIPShaderCacheArtifact artifact{};
    artifact.identity.resize(static_cast<size_t>(identity_size));
    if (!reader.read_bytes(
            artifact.identity.data(),
            artifact.identity.size())) {
        return luisa::nullopt;
    }
    uint64_t package_size{};
    uint64_t package_hash{};
    if (!reader.read_u64(package_size) ||
        !reader.read_u64(package_hash) ||
        package_size != reader.remaining()) {
        return luisa::nullopt;
    }
    luisa::vector<std::byte> package_bytes(
        static_cast<size_t>(package_size));
    if (!reader.read_bytes(
            package_bytes.data(), package_bytes.size()) ||
        reader.remaining() != 0u ||
        luisa::hash64(
            package_bytes.data(), package_bytes.size(),
            hip_shader_cache_payload_hash_seed) != package_hash) {
        return luisa::nullopt;
    }
    auto package = deserialize_hip_shader_package(package_bytes);
    if (!package) { return luisa::nullopt; }
    artifact.package = std::move(*package);
    return artifact;
}

}// namespace

static const bool LUISA_XIR_NORMALIZE_CFG = [] {
    if (auto env = std::getenv("LUISA_XIR_NORMALIZE_CFG")) {
        return std::string_view{env} == "1";
    }
    return false;
}();

static const bool LUISA_XIR_RESTRUCTURE_CFG = [] {
    if (auto env = std::getenv("LUISA_XIR_RESTRUCTURE_CFG")) {
        return std::string_view{env} == "1";
    }
    return false;
}();

static const bool LUISA_XIR_ELIMINATE_EARLY_RETURN = [] {
    if (auto env = std::getenv("LUISA_XIR_ELIMINATE_EARLY_RETURN")) {
        return std::string_view{env} == "1";
    }
    return false;
}();

namespace {

[[nodiscard]] luisa::vector<std::byte> make_hip_shader_cache_identity(
    Function kernel, const ShaderOption &option,
    luisa::string_view amdgpu_arch, uint32_t wave_size) noexcept {
    HIPShaderPackageWriter writer;
    // This is a canonical, fixed-width encoding. The cache filename is only an
    // index derived from these bytes; a hit is accepted only after the complete
    // identity stored in the artifact is compared byte-for-byte.
    writer.write_u32(hip_shader_cache_codegen_revision);
    writer.write_u32(hip_shader_package_version);
    writer.write_u32(LLVM_VERSION_MAJOR);
    writer.write_u32(LLVM_VERSION_MINOR);
    writer.write_u32(LLVM_VERSION_PATCH);
    writer.write_u32(HIP_VERSION_MAJOR);
    writer.write_u32(HIP_VERSION_MINOR);
    writer.write_u32(HIP_VERSION_PATCH);
    writer.write_string(HIP_VERSION_GITHASH);
    writer.write_u32(HIPRT_MAJOR_VERSION);
    writer.write_u32(HIPRT_MINOR_VERSION);
    writer.write_u32(HIPRT_PATCH_VERSION);
    writer.write_u32(HIPRT_HASH_VERSION);
    writer.write_u64(kernel.hash());
    writer.write_string(amdgpu_arch);
    writer.write_u32(wave_size);
    writer.write_u32(option.max_registers);
    writer.write_u32(static_cast<uint32_t>(
        HIPCodegenLLVMConfig::OptLevel::LEVEL_AGGRESSIVE));
    auto flags =
        static_cast<uint32_t>(option.enable_fast_math) |
        static_cast<uint32_t>(option.enable_debug_info) << 1u |
        static_cast<uint32_t>(LUISA_XIR_ELIMINATE_EARLY_RETURN) << 2u |
        static_cast<uint32_t>(LUISA_XIR_NORMALIZE_CFG) << 3u |
        static_cast<uint32_t>(LUISA_XIR_RESTRUCTURE_CFG) << 4u |
        static_cast<uint32_t>(kernel.requires_autodiff()) << 5u;
    writer.write_u32(flags);
    writer.write_string(option.native_include);
    return std::move(writer).finish();
}

[[nodiscard]] luisa::string make_hip_shader_cache_name(
    luisa::span<const std::byte> identity) noexcept {
    auto digest = luisa::hash64(
        identity.data(), identity.size_bytes(),
        hip_shader_cache_payload_hash_seed);
    return luisa::format("hip_kernel_{:016x}.cache", digest);
}

[[nodiscard]] HIPShaderMetadata make_hip_shader_metadata(
    Function kernel, const ShaderOption &option,
    bool uses_hardware_rt_stack,
    luisa::vector<std::pair<luisa::string, luisa::string>>
        format_types = {}) noexcept {
    auto builtin_callables = kernel.propagated_builtin_callables();
    auto requires_static_trace =
        builtin_callables.test(CallOp::RAY_TRACING_TRACE_CLOSEST) ||
        builtin_callables.test(CallOp::RAY_TRACING_TRACE_ANY);
    auto uses_codegen_hardware_rt_stack =
        uses_hardware_rt_stack &&
        !builtin_callables.uses_ray_query_motion_blur();
    auto requires_global_rt_stack =
        !uses_codegen_hardware_rt_stack &&
        (requires_static_trace || builtin_callables.uses_ray_query());

    luisa::vector<Usage> argument_usages;
    argument_usages.reserve(kernel.arguments().size());
    luisa::vector<luisa::string> argument_types;
    argument_types.reserve(kernel.arguments().size());
    for (auto &&arg : kernel.arguments()) {
        argument_usages.emplace_back(
            kernel.variable_usage(arg.uid()));
        argument_types.emplace_back(
            arg.type()->description());
    }
    return HIPShaderMetadata{
        .checksum = kernel.hash(),
        .curve_bases = kernel.required_curve_bases(),
        .kind = kernel.requires_raytracing() ?
                    HIPShaderMetadata::Kind::RAY_TRACING :
                    HIPShaderMetadata::Kind::COMPUTE,
        .enable_debug = option.enable_debug_info,
        .requires_trace_closest =
            builtin_callables.test(
                CallOp::RAY_TRACING_TRACE_CLOSEST) ||
            builtin_callables.test(
                CallOp::RAY_TRACING_TRACE_CLOSEST_MOTION_BLUR),
        .requires_trace_any =
            builtin_callables.test(
                CallOp::RAY_TRACING_TRACE_ANY) ||
            builtin_callables.test(
                CallOp::RAY_TRACING_TRACE_ANY_MOTION_BLUR),
        .requires_ray_query =
            builtin_callables.uses_ray_query(),
        .requires_printing = kernel.requires_printing(),
        .requires_motion_blur =
            kernel.requires_motion_blur(),
        .requires_global_rt_stack =
            requires_global_rt_stack,
        .max_register_count = option.max_registers,
        .block_size = kernel.block_size(),
        .argument_types = std::move(argument_types),
        .argument_usages = std::move(argument_usages),
        .format_types = std::move(format_types),
    };
}

[[nodiscard]] bool hip_shader_cache_package_matches(
    const HIPShaderPackage &package,
    const HIPShaderMetadata &expected_metadata,
    luisa::string_view amdgpu_arch,
    uint32_t wave_size) noexcept {
    if (package.amdgpu_arch != amdgpu_arch ||
        package.wave_size != wave_size ||
        package.code.empty()) {
        return false;
    }
    // Format descriptors are generated payload, not an AST input. All other
    // metadata is independently derived from the current kernel and options.
    auto expected = expected_metadata;
    expected.format_types = package.metadata.format_types;
    return package.metadata == expected;
}

[[nodiscard]] luisa::vector<ShaderDispatchCommand::Argument>
make_hip_bound_arguments(Function kernel) noexcept {
    luisa::vector<ShaderDispatchCommand::Argument> bound_arguments;
    bound_arguments.reserve(kernel.bound_arguments().size());
    for (auto &&arg : kernel.bound_arguments()) {
        luisa::visit(
            [&bound_arguments]<typename T>(T binding) noexcept {
                ShaderDispatchCommand::Argument argument{};
                if constexpr (std::is_same_v<T, Function::BufferBinding>) {
                    argument.tag =
                        ShaderDispatchCommand::Argument::Tag::BUFFER;
                    argument.buffer.handle = binding.handle;
                    argument.buffer.offset = binding.offset;
                    argument.buffer.size = binding.size;
                } else if constexpr (
                    std::is_same_v<T, Function::TextureBinding>) {
                    argument.tag =
                        ShaderDispatchCommand::Argument::Tag::TEXTURE;
                    argument.texture.handle = binding.handle;
                    argument.texture.level = binding.level;
                } else if constexpr (
                    std::is_same_v<T, Function::BindlessArrayBinding>) {
                    argument.tag =
                        ShaderDispatchCommand::Argument::Tag::BINDLESS_ARRAY;
                    argument.bindless_array.handle = binding.handle;
                } else if constexpr (
                    std::is_same_v<T, Function::AccelBinding>) {
                    argument.tag =
                        ShaderDispatchCommand::Argument::Tag::ACCEL;
                    argument.accel.handle = binding.handle;
                } else {
                    LUISA_ERROR_WITH_LOCATION(
                        "Unsupported binding type.");
                }
                bound_arguments.emplace_back(argument);
            },
            arg);
    }
    return bound_arguments;
}

}// namespace

[[nodiscard]] static llvm::AMDGPU::GPUKind hip_amdgpu_kind(
    luisa::string_view amdgpu_arch) noexcept {
    auto arch = llvm::StringRef{amdgpu_arch.data(), amdgpu_arch.size()};
    auto kind = llvm::AMDGPU::parseArchAMDGCN(arch);
    LUISA_ASSERT(kind != llvm::AMDGPU::GK_NONE,
                 "Unsupported AMDGPU architecture '{}'.", amdgpu_arch);
    return kind;
}

[[nodiscard]] static bool hip_wave_size_supported(
    luisa::string_view amdgpu_arch, uint32_t wave_size) noexcept {
    if (wave_size == 64u) { return true; }
    if (wave_size != 32u) { return false; }
    auto attributes = llvm::AMDGPU::getArchAttrAMDGCN(
        hip_amdgpu_kind(amdgpu_arch));
    return (attributes & llvm::AMDGPU::FEATURE_WAVE32) != 0u;
}

[[nodiscard]] static uint32_t select_hip_wave_size(
    luisa::string_view amdgpu_arch, uint32_t native_wave_size,
    luisa::optional<uint8_t> requested_wave_size,
    bool requires_hiprt, bool uses_hardware_rt_stack) noexcept {
    auto wave_size = static_cast<uint32_t>(requested_wave_size.value_or(
        static_cast<uint8_t>(native_wave_size)));
    if (!requested_wave_size) {
        if (auto env = std::getenv("LUISA_HIP_WAVE64");
            env && std::string_view{env} == "1") {
            wave_size = 64u;
        }
    }
    if (!hip_wave_size_supported(amdgpu_arch, wave_size)) {
        LUISA_ERROR_WITH_LOCATION(
            "HIP wave{} code generation is not supported on AMDGPU architecture '{}'. "
            "HIP kernels may request only native wave32 or wave64 modes supported by the target.",
            wave_size, amdgpu_arch);
    }
    if (requires_hiprt && uses_hardware_rt_stack && wave_size != 32u) {
        LUISA_ERROR_WITH_LOCATION(
            "HIPRT ray tracing on AMDGPU architecture '{}' requires wave32, "
            "but the kernel requests wave{}.",
            amdgpu_arch, wave_size);
    }
    return wave_size;
}

static void verify_xir_or_error(const xir::Module *module,
                                luisa::string_view stage) noexcept {
    auto verification = xir::xir_verify_module(module);
    if (!verification.succeeded()) {
        LUISA_ERROR_WITH_LOCATION(
            "Invalid XIR at HIP {}: {} ({} error(s) total).",
            stage, verification.errors.front().message, verification.errors.size());
    }
}
#endif

void luisa_initialize_hip() noexcept {
    static std::once_flag flag;
    std::call_once(flag, [] {
        LUISA_CHECK_HIP(hipInit(0));
    });
}

struct HIPDeviceGuard {
    hipCtx_t context{nullptr};
    explicit HIPDeviceGuard(hipCtx_t ctx) noexcept : context{ctx} {
        LUISA_ASSERT(context != nullptr, "HIP device context is null.");
        LUISA_CHECK_HIP(hipCtxPushCurrent(context));
    }
    ~HIPDeviceGuard() noexcept {
        hipCtx_t popped_context{nullptr};
        LUISA_CHECK_HIP(hipCtxPopCurrent(&popped_context));
        LUISA_ASSERT(popped_context == context,
                     "Unexpected HIP context popped from the current thread.");
    }
    HIPDeviceGuard(HIPDeviceGuard &&) noexcept = delete;
    HIPDeviceGuard(const HIPDeviceGuard &) noexcept = delete;
    HIPDeviceGuard &operator=(HIPDeviceGuard &&) noexcept = delete;
    HIPDeviceGuard &operator=(const HIPDeviceGuard &) noexcept = delete;
};

template<typename F>
decltype(auto) HIPDevice::with_device(F &&f) const noexcept {
    HIPDeviceGuard guard{_hip_context};
    return std::invoke(std::forward<F>(f));
}

HIPDevice::HIPDevice(Context &&ctx, const DeviceConfig *config) noexcept
    : DeviceInterface{std::move(ctx)},
      _default_io{config == nullptr || config->binary_io == nullptr ?
                      luisa::make_unique<DefaultBinaryIO>(context()) :
                      nullptr},
      _io{_default_io == nullptr ? config->binary_io : _default_io.get()},
      _device_id{config == nullptr || config->device_index == ~0ull ? 0 : static_cast<int>(config->device_index)} {
    auto device_count = 0;
    LUISA_CHECK_HIP(hipGetDeviceCount(&device_count));
    LUISA_ASSERT(_device_id >= 0 && _device_id < device_count,
                 "HIP device index out of range (required = {}, count = {}).",
                 _device_id, device_count);
    LUISA_CHECK_HIP(hipDeviceGet(&_hip_device, _device_id));
    LUISA_CHECK_HIP(hipDevicePrimaryCtxRetain(&_hip_context, _hip_device));

    // log device name and version
    hipDeviceProp_t prop;
    LUISA_CHECK_HIP(hipGetDeviceProperties(&prop, _device_id));
    auto arch_name = std::string_view{prop.gcnArchName};
    if (auto feature_suffix = arch_name.find(':'); feature_suffix != std::string_view::npos) {
        arch_name = arch_name.substr(0u, feature_suffix);
    }
    LUISA_ASSERT(arch_name.starts_with("gfx") && arch_name.size() > 3u,
                 "Invalid AMDGPU architecture name '{}' reported by HIP.", prop.gcnArchName);
    _amdgpu_arch.assign(arch_name.data(), arch_name.size());
    auto driver_version = 0;
    auto runtime_version = 0;
    LUISA_CHECK_HIP(hipDriverGetVersion(&driver_version));
    LUISA_CHECK_HIP(hipRuntimeGetVersion(&runtime_version));
    auto version_major = [](auto x) noexcept { return x / 10000000; };
    auto version_minor = [](auto x) noexcept { return x % 10000000 / 100000; };
    auto version_patch = [](auto x) noexcept { return x % 100000; };
    LUISA_INFO("Created HIP device {}: {} (arch = {}, cc = {}.{}, driver = {}.{}.{}, runtime = {}.{}.{}, build = {}.{}.{}).",
               _device_id, prop.name, _amdgpu_arch, prop.major, prop.minor,
               version_major(driver_version), version_minor(driver_version), version_patch(driver_version),
               version_major(runtime_version), version_minor(runtime_version), version_patch(runtime_version),
               HIP_VERSION_MAJOR, HIP_VERSION_MINOR, HIP_VERSION_PATCH);

    // hipLimitStackSize controls per-thread scratch allocation for uses_dynamic_stack=true kernels.
    // HIPRT traversal uses its hardware stack or a dedicated dynamic stack buffer, not this.
    // With full LTO inlining of HIPRT functions, no dynamic call stack is needed.
    with_device([&] {
        auto ret = hipDeviceSetLimit(hipLimitStackSize, 0u);
        if (ret != hipSuccess) {
            LUISA_WARNING("hipDeviceSetLimit(hipLimitStackSize) failed: {}",
                          hipGetErrorString(ret));
        }
    });
}

HIPDevice::~HIPDevice() noexcept {
    {
        std::scoped_lock lock{_motion_mesh_builtin_mutex};
        if (_motion_mesh_builtin != nullptr) {
            with_device([&] {
                // Module unloading is not stream-ordered. All streams should
                // already have been destroyed by the runtime, but drain the
                // context defensively before releasing the embedded module.
                LUISA_CHECK_HIP(hipDeviceSynchronize());
                _motion_mesh_builtin.reset();
            });
        }
    }
    {
        std::scoped_lock lock{_hiprt_mutex};
        if (_hiprt_context != nullptr) {
            with_device([&] {
                if (_hiprt_global_stack_buffer_initialized) {
                    LUISA_CHECK_HIPRT(hiprtDestroyGlobalStackBuffer(
                        _hiprt_context, _hiprt_global_stack_buffer));
                    _hiprt_global_stack_buffer = {};
                    _hiprt_global_stack_buffer_initialized = false;
                }
                LUISA_CHECK_HIPRT(hiprtDestroyContext(_hiprt_context));
                _hiprt_context = nullptr;
            });
        }
    }
    LUISA_CHECK_HIP(hipDevicePrimaryCtxRelease(_hip_device));
    _hip_context = nullptr;
}

HIPMotionMeshBuiltin &HIPDevice::motion_mesh_builtin() const noexcept {
    std::scoped_lock lock{_motion_mesh_builtin_mutex};
    if (_motion_mesh_builtin == nullptr) {
        // The module belongs to this device's primary context. Push it even
        // when the caller already runs under with_device(), so this accessor
        // remains safe if it is reused from another host path later.
        with_device([&] {
            _motion_mesh_builtin =
                luisa::make_unique<HIPMotionMeshBuiltin>();
        });
    }
    return *_motion_mesh_builtin;
}

hiprtContext HIPDevice::_ensure_hiprt_context_locked() const noexcept {
    if (_hiprt_context == nullptr) {
        with_device([&] {
            hipDeviceProp_t prop{};
            LUISA_CHECK_HIP(hipGetDeviceProperties(&prop, _device_id));
            hiprtContextCreationInput input{};
            input.deviceType = std::string_view{prop.name}.find("NVIDIA") != std::string_view::npos ?
                                   hiprtDeviceNVIDIA :
                                   hiprtDeviceAMD;
            input.device = static_cast<hiprtApiDevice>(_hip_device);
            input.ctxt = _hip_context;
            LUISA_CHECK_HIPRT(hiprtCreateContext(HIPRT_API_VERSION, input, _hiprt_context));
            LUISA_CHECK_HIPRT(hiprtSetLogLevel(
                _hiprt_context,
                static_cast<hiprtLogLevel>(hiprtLogLevelInfo | hiprtLogLevelWarn | hiprtLogLevelError)));
            LUISA_INFO("Created HIPRT context lazily for HIP device {}.", _device_id);
        });
    }
    return _hiprt_context;
}

hiprtContext HIPDevice::hiprt_context() const noexcept {
    std::scoped_lock lock{_hiprt_mutex};
    return _ensure_hiprt_context_locked();
}

hiprtGlobalStackBuffer HIPDevice::hiprt_global_stack_buffer() const noexcept {
    std::scoped_lock lock{_hiprt_mutex};
    auto context = _ensure_hiprt_context_locked();
    if (!_hiprt_global_stack_buffer_initialized) {
        with_device([&] {
            hiprtGlobalStackBufferInput input{};
            input.type = hiprtStackTypeDynamic;
            input.entryType = hiprtStackEntryTypeInteger;
            input.stackSize = 64u;
            // Dynamic stacks allocate for the maximum concurrently resident
            // threads and use per-wave locks. Unlike a fixed global stack, the
            // allocation is bounded by device occupancy rather than dispatch size.
            input.threadCount = 0u;
            LUISA_CHECK_HIPRT(hiprtCreateGlobalStackBuffer(
                context, input, _hiprt_global_stack_buffer));
            _hiprt_global_stack_buffer_initialized = true;
            LUISA_INFO("Created bounded HIPRT dynamic stack buffer "
                       "(stackSize={}, stackCount={}).",
                       _hiprt_global_stack_buffer.stackSize,
                       _hiprt_global_stack_buffer.stackCount);
        });
    }
    return _hiprt_global_stack_buffer;
}

void HIPDevice::set_stream_log_callback(uint64_t stream_handle, const StreamLogCallback &callback) noexcept {
    reinterpret_cast<HIPStream *>(stream_handle)->set_log_callback(callback);
}

ShaderCreationInfo HIPDevice::create_shader(const ShaderOption &option, const ir_v2::KernelModule &kernel) noexcept {
    LUISA_NOT_IMPLEMENTED();
}

ResourceCreationInfo HIPDevice::create_curve(const AccelOption &option) noexcept {
    auto context = hiprt_context();
    auto curve = with_device([&] {
        return luisa::new_with_allocator<HIPCurve>(context, option);
    });
    return {.handle = reinterpret_cast<uint64_t>(curve),
            .native_handle = curve};
}

void HIPDevice::destroy_curve(uint64_t handle) noexcept {
    with_device([&] {
        auto curve = reinterpret_cast<HIPCurve *>(handle);
        luisa::delete_with_allocator(curve);
    });
}

ResourceCreationInfo HIPDevice::create_motion_instance(const AccelMotionOption &option) noexcept {
    auto context = hiprt_context();
    auto instance = with_device([&] {
        return luisa::new_with_allocator<HIPMotionInstance>(context, option);
    });
    return {.handle = reinterpret_cast<uint64_t>(instance),
            .native_handle = instance};
}

void HIPDevice::destroy_motion_instance(uint64_t handle) noexcept {
    with_device([&] {
        auto instance = reinterpret_cast<HIPMotionInstance *>(handle);
        luisa::delete_with_allocator(instance);
    });
}

luisa::string HIPDevice::query(luisa::string_view property) noexcept {
    if (property == "amdgpu_arch") {
        return _amdgpu_arch;
    }
    return DeviceInterface::query(property);
}

DeviceExtension *HIPDevice::extension(luisa::string_view name) noexcept {
    if (name == PinnedMemoryExt::name) {
        std::scoped_lock lock{_extension_mutex};
        if (_pinned_memory_ext == nullptr) {
            _pinned_memory_ext = luisa::make_unique<HIPPinnedMemoryExt>(this);
        }
        return _pinned_memory_ext.get();
    }
    return DeviceInterface::extension(name);
}

luisa::string_view HIPDevice::get_name(uint64_t resource_handle) const noexcept {
    return DeviceInterface::get_name(resource_handle);
}

SparseBufferCreationInfo HIPDevice::create_sparse_buffer(const Type *element, size_t elem_count) noexcept {
    LUISA_NOT_IMPLEMENTED();
}

ResourceCreationInfo HIPDevice::allocate_sparse_buffer_heap(size_t byte_size) noexcept {
    LUISA_NOT_IMPLEMENTED();
}

void HIPDevice::deallocate_sparse_buffer_heap(uint64_t handle) noexcept {
    LUISA_NOT_IMPLEMENTED();
}

void HIPDevice::update_sparse_resources(uint64_t stream_handle, luisa::vector<SparseUpdateTile> &&textures_update) noexcept {
    LUISA_NOT_IMPLEMENTED();
}

void HIPDevice::destroy_sparse_buffer(uint64_t handle) noexcept {
    LUISA_NOT_IMPLEMENTED();
}

ResourceCreationInfo HIPDevice::allocate_sparse_texture_heap(size_t byte_size) noexcept {
    LUISA_NOT_IMPLEMENTED();
}

void HIPDevice::deallocate_sparse_texture_heap(uint64_t handle) noexcept {
    LUISA_NOT_IMPLEMENTED();
}

SparseTextureCreationInfo HIPDevice::create_sparse_texture(PixelFormat format, uint dimension,
                                                           uint width, uint height, uint depth,
                                                           uint mipmap_levels, bool simultaneous_access) noexcept {
    LUISA_NOT_IMPLEMENTED();
}

void HIPDevice::destroy_sparse_texture(uint64_t handle) noexcept {
    LUISA_NOT_IMPLEMENTED();
}

hipUUID_t HIPDevice::device_uuid() const noexcept {
    hipUUID_t uuid{};
    LUISA_CHECK_HIP(hipDeviceGetUuid(&uuid, _device_id));
    return uuid;
}

hipUUID_t HIPDevice::device_uuid_for_vulkan() const noexcept {
    // The value that hipDeviceGetUuid returns does not correspond with those returned
    // by mesa (see https://gitlab.freedesktop.org/mesa/mesa/-/blob/5cd3e395037250946ba2519600836341df02c8ca/src/amd/common/ac_gpu_info.c#L1366-1382)
    // and by xgl (see https://github.com/GPUOpen-Drivers/xgl/blob/4118707939c2f4783d28ce2a383184a3794ca477/icd/api/vk_physical_device.cpp#L4363-L4421)
    // Those drivers _do_ align with each other, so we can create our own UUID here.
    // \see https://github.com/ROCm-Developer-Tools/hipamd/issues/50.
    hipDeviceProp_t props;
    LUISA_CHECK_HIP(hipGetDeviceProperties(&props, device_id()));
    hipUUID_t result = {};
    auto uuid_ints = reinterpret_cast<uint32_t *>(result.bytes);
    uuid_ints[0] = props.pciDomainID;
    uuid_ints[1] = props.pciBusID;
    uuid_ints[2] = props.pciDeviceID;
    return result;
}

void *HIPDevice::native_handle() const noexcept {
    return _hip_context;
}

uint HIPDevice::compute_warp_size() const noexcept {
    int warp_size = 0;
    LUISA_CHECK_HIP(hipDeviceGetAttribute(
        &warp_size,
        hipDeviceAttributeWarpSize,
        _device_id));
    return static_cast<uint>(warp_size);
}

uint64_t HIPDevice::memory_granularity() const noexcept {
    LUISA_NOT_IMPLEMENTED();
}

BufferCreationInfo HIPDevice::create_buffer(const Type *element, size_t elem_count, void *external_memory) noexcept {
    elem_count = std::max<size_t>(elem_count, 1u);
    if (element == Type::of<IndirectKernelDispatch>()) {
        LUISA_ASSERT(external_memory == nullptr,
                     "Indirect dispatch buffers cannot import external memory.");
        auto buffer = with_device([&] {
            return HIPBuffer::create_indirect_buffer(elem_count);
        });
        BufferCreationInfo info{};
        info.handle = reinterpret_cast<uint64_t>(buffer);
        info.native_handle = buffer->handle();
        info.element_stride = sizeof(HIPBuffer::IndirectDispatch);
        info.total_size_bytes = buffer->size_bytes();
        return info;
    }
    LUISA_ASSERT(element == nullptr || element->is_basic() || element->is_structure() || element->is_array(),
                 "Invalid buffer element type {}.", element->description());
    auto elem_stride = element == nullptr ? 1u : element->size();
    auto size_bytes = elem_stride * elem_count;
    auto buffer = with_device([&] {
        return external_memory == nullptr ?
                   HIPBuffer::create_device_buffer(size_bytes) :
                   HIPBuffer::import_external_device_buffer(external_memory, size_bytes);
    });
    BufferCreationInfo info{};
    info.handle = reinterpret_cast<uint64_t>(buffer);
    info.native_handle = buffer->handle();
    info.element_stride = elem_stride;
    info.total_size_bytes = size_bytes;
    return info;
}

BufferCreationInfo HIPDevice::create_buffer(const ir::CArc<ir::Type> *element, size_t elem_count, void *external_memory) noexcept {
#ifdef LUISA_ENABLE_IR
    auto type = IR2AST::get_type(element->get());
    return create_buffer(type, elem_count, external_memory);
#else
    LUISA_ERROR_WITH_LOCATION("HIP backend was compiled without legacy IR support.");
    return BufferCreationInfo::make_invalid();
#endif
}

void HIPDevice::destroy_buffer(uint64_t handle) noexcept {
    auto buffer = reinterpret_cast<HIPBuffer *>(handle);
    with_device([&] { HIPBuffer::destroy(buffer); });
}

ResourceCreationInfo HIPDevice::create_texture(PixelFormat format, uint dimension,
                                               uint width, uint height, uint depth,
                                               uint mipmap_levels, void *external_native_handle,
                                               bool simultaneous_access, bool allow_raster_target) noexcept {
    auto p = with_device([&] {
        return external_native_handle == nullptr ?
                   HIPTexture::create_device_texture(format, dimension,
                                                     make_uint3(width, height, depth),
                                                     mipmap_levels) :
                   HIPTexture::import_external_texture(
                       reinterpret_cast<uint64_t>(external_native_handle),
                       format, dimension,
                       make_uint3(width, height, depth),
                       mipmap_levels);
    });
    return {.handle = reinterpret_cast<uint64_t>(p),
            .native_handle = p->handle()};
}

void HIPDevice::destroy_texture(uint64_t handle) noexcept {
    auto texture = reinterpret_cast<HIPTexture *>(handle);
    with_device([&] {
        luisa::delete_with_allocator(texture);
    });
}

ResourceCreationInfo HIPDevice::create_bindless_array(size_t size, BindlessSlotType type) noexcept {
    auto p = with_device([&] {
        return luisa::new_with_allocator<HIPBindlessArray>(size);
    });
    return {.handle = reinterpret_cast<uint64_t>(p),
            .native_handle = reinterpret_cast<void *>(p->handle())};
}

void HIPDevice::destroy_bindless_array(uint64_t handle) noexcept {
    with_device([&] {
        auto array = reinterpret_cast<HIPBindlessArray *>(handle);
        luisa::delete_with_allocator(array);
    });
}

ResourceCreationInfo HIPDevice::create_stream(StreamTag stream_tag) noexcept {
    auto p = with_device([&] {
        return luisa::new_with_allocator<HIPStream>(this);
    });
    return {.handle = reinterpret_cast<uint64_t>(p),
            .native_handle = p->handle()};
}

void HIPDevice::destroy_stream(uint64_t handle) noexcept {
    with_device([&] {
        auto stream = reinterpret_cast<HIPStream *>(handle);
        delete_with_allocator(stream);
    });
}

void HIPDevice::synchronize_stream(uint64_t stream_handle) noexcept {
    with_device([&] {
        auto stream = reinterpret_cast<HIPStream *>(stream_handle);
        stream->synchronize();
    });
}

void HIPDevice::dispatch(uint64_t stream_handle, CommandList &&list) noexcept {
    if (!list.empty()) {
        with_device([&] {
            auto stream = reinterpret_cast<HIPStream *>(stream_handle);
            stream->dispatch(std::move(list));
        });
    }
}

namespace {

#ifndef LUISA_BACKEND_ENABLE_VULKAN_SWAPCHAIN
void report_swapchain_not_enabled() noexcept {
    LUISA_ERROR_WITH_LOCATION("Swapchains are not enabled on the HIP backend. "
                              "You need to enable the GUI module and install "
                              "the Vulkan SDK (>= 1.1) to enable it.");
}
#endif

}// namespace

SwapchainCreationInfo HIPDevice::create_swapchain(const SwapchainOption &option, uint64_t stream_handle) noexcept {
#ifdef LUISA_BACKEND_ENABLE_VULKAN_SWAPCHAIN
    auto chain = with_device([&] {
        return new_with_allocator<HIPSwapchain>(this, option);
    });
    SwapchainCreationInfo info{};
    info.handle = reinterpret_cast<uint64_t>(chain);
    info.native_handle = chain->native_handle();
    info.storage = chain->pixel_storage();
    return info;
#else
    report_swapchain_not_enabled();
#endif
}

void HIPDevice::destroy_swapchain(uint64_t handle) noexcept {
#ifdef LUISA_BACKEND_ENABLE_VULKAN_SWAPCHAIN
    with_device([chain = reinterpret_cast<HIPSwapchain *>(handle)] {
        delete_with_allocator(chain);
    });
#else
    report_swapchain_not_enabled();
#endif
}

void HIPDevice::present_display_in_stream(uint64_t stream_handle, uint64_t swapchain_handle, uint64_t image_handle) noexcept {
#ifdef LUISA_BACKEND_ENABLE_VULKAN_SWAPCHAIN
    with_device([stream = reinterpret_cast<HIPStream *>(stream_handle),
                 chain = reinterpret_cast<HIPSwapchain *>(swapchain_handle),
                 image = reinterpret_cast<HIPTexture *>(image_handle)] {
        chain->present(stream, image);
    });
#else
    report_swapchain_not_enabled();
#endif
}

ShaderCreationInfo HIPDevice::create_shader(const ShaderOption &option, Function kernel) noexcept {
#ifdef LUISA_COMPUTE_ENABLE_LLVM
    auto builtin_callables = kernel.propagated_builtin_callables();
    auto requires_hiprt = kernel.requires_raytracing() ||
                          builtin_callables.uses_ray_query();
    auto wave_size = select_hip_wave_size(
        _amdgpu_arch, compute_warp_size(), kernel.allowed_warp_size(),
        requires_hiprt, uses_hardware_rt_stack());
    auto expected_metadata = make_hip_shader_metadata(
        kernel, option, uses_hardware_rt_stack());

    auto uses_shader_cache =
        option.name.empty() && option.enable_cache;
    luisa::vector<std::byte> cache_identity;
    luisa::string cache_name;
    if (uses_shader_cache) {
        cache_identity = make_hip_shader_cache_identity(
            kernel, option, _amdgpu_arch, wave_size);
        cache_name = make_hip_shader_cache_name(cache_identity);
    }

    luisa::optional<HIPShaderPackage> shader_package;
    if (uses_shader_cache) {
        auto stream = _io->read_shader_cache(cache_name);
        if (stream != nullptr && stream->length() != 0u) {
            if (stream->length() <= hip_shader_cache_max_artifact_size) {
                luisa::vector<std::byte> bytes(stream->length());
                stream->read(
                    luisa::span{bytes.data(), bytes.size()});
                if (auto artifact =
                        deserialize_hip_shader_cache_artifact(bytes);
                    artifact &&
                    artifact->identity == cache_identity &&
                    hip_shader_cache_package_matches(
                        artifact->package, expected_metadata,
                        _amdgpu_arch, wave_size)) {
                    shader_package =
                        std::move(artifact->package);
                    LUISA_INFO(
                        "Loaded HIP shader '{}' from cache ({} bytes).",
                        cache_name, bytes.size());
                } else {
                    LUISA_WARNING_WITH_LOCATION(
                        "HIP shader cache entry '{}' is invalid or "
                        "does not match the requested shader; recompiling.",
                        cache_name);
                }
            } else {
                LUISA_WARNING_WITH_LOCATION(
                    "HIP shader cache entry '{}' is too large ({} bytes); "
                    "recompiling.",
                    cache_name, stream->length());
            }
        }
    }

    if (!shader_package) {
        Clock translate_clk;
        auto xir_module = xir::ast_to_xir_translate(kernel, {});
        xir_module->set_name(
            luisa::format("kernel_{:016x}", kernel.hash()));
        if (!option.name.empty()) {
            xir_module->set_location(option.name);
        }
        verify_xir_or_error(
            xir_module.get(), "AST translation");
        LUISA_VERBOSE(
            "AST to XIR translation done in {} ms.",
            translate_clk.toc());

        if (kernel.requires_autodiff()) {
            auto inline_info =
                xir::inline_all_pass_run_on_module(
                    xir_module.get());
            auto autodiff_info =
                xir::autodiff_pass_run_on_module(
                    xir_module.get());
            LUISA_VERBOSE(
                "HIP XIR autodiff lowering: inlined {} call(s), "
                "transformed {} scope(s), removed {} instruction(s).",
                inline_info.inlined_call_count,
                autodiff_info.transformed_scope_count,
                autodiff_info.removed_instruction_count);
            verify_xir_or_error(
                xir_module.get(), "autodiff lowering");
        }

        if (LUISA_XIR_ELIMINATE_EARLY_RETURN) {
            auto early_return_info =
                xir::early_return_elimination_pass_run_on_module(
                    xir_module.get());
            LUISA_VERBOSE(
                "XIR early-return elimination: removed {} "
                "early return(s).",
                early_return_info.removed_return_count);
        }
        {
            xir::PassReport report;
            auto ray_query_info =
                xir::lower_ray_query_loop_pass_run_on_module(
                    xir_module.get(), &report);
            if (!ray_query_info.succeeded()) {
                LUISA_ERROR_WITH_LOCATION(
                    "HIP XIR ray-query lowering rejected {} loop(s).",
                    ray_query_info.error_count);
            }
            LUISA_VERBOSE(
                "HIP XIR ray-query lowering: outlined {} loop(s).",
                ray_query_info.lowered_loop_count);
            verify_xir_or_error(
                xir_module.get(), "ray-query lowering");
        }
        if (LUISA_XIR_NORMALIZE_CFG) {
            xir::PassPipeline cfg_pipeline;
            cfg_pipeline.add(
                "destructure-cfg",
                [](xir::Module *m, xir::PassReport &r) {
                    auto i =
                        xir::destructure_cfg_pass_run_on_module(
                            m, &r);
                    if (!i.succeeded()) {
                        LUISA_ERROR_WITH_LOCATION(
                            "HIP XIR destructuring failed "
                            "(errors={}, leaked_blocks={}).",
                            i.error_count,
                            i.leaked_block_count);
                    }
                    return i.changed();
                });
            cfg_pipeline.add(
                "simplify-cfg",
                [](xir::Module *m, xir::PassReport &r) {
                    auto i =
                        xir::simplify_cfg_pass_run_on_module(
                            m, &r);
                    return i.changed();
                });
            if (LUISA_XIR_RESTRUCTURE_CFG) {
                cfg_pipeline.add(
                    "restructure-cfg",
                    [](xir::Module *m,
                       xir::PassReport &r) {
                        auto i =
                            xir::restructure_cfg_pass_run_on_module(
                                m, &r);
                        if (!i.succeeded()) {
                            LUISA_ERROR_WITH_LOCATION(
                                "HIP XIR restructuring failed "
                                "(irreducible={}, unstructured={}, "
                                "invalid={}, iteration_limit={}).",
                                i.irreducible_region_count,
                                i.unstructured_branch_count,
                                i.invalid_construct_count,
                                i.iteration_limit_count);
                        }
                        return i.changed();
                    });
            }
            auto stats =
                cfg_pipeline.run(xir_module.get());
            stats.log("HIP backend CFG normalization");
        }
        verify_xir_or_error(
            xir_module.get(), "codegen handoff");

        HIPCodegenLLVMConfig config{
            .source_file = option.name,
            .native_include = option.native_include,
            .bindings = kernel.bound_arguments(),
            .block_size = {
                kernel.block_size().x,
                kernel.block_size().y,
                kernel.block_size().z},
            .amdgpu_arch = _amdgpu_arch,
            .wave_size = wave_size,
            .max_register_count = option.max_registers,
            .opt_level = HIPCodegenLLVMConfig::OptLevel::LEVEL_AGGRESSIVE,
            .enable_fast_math = option.enable_fast_math,
            .enable_debug_info = option.enable_debug_info,
            .requires_ray_tracing = kernel.requires_raytracing(),
            .requires_ray_query = builtin_callables.uses_ray_query(),
            .requires_motion_blur = kernel.requires_motion_blur(),
            .requires_static_trace = builtin_callables.test(CallOp::RAY_TRACING_TRACE_CLOSEST) || builtin_callables.test(CallOp::RAY_TRACING_TRACE_ANY),
            .requires_motion_ray_query = builtin_callables.uses_ray_query_motion_blur(),
            .requires_printing = kernel.requires_printing(),
            .curve_bases = kernel.required_curve_bases(),
        };

        auto codegen_result =
            hip_codegen_llvm(*xir_module, config);
        LUISA_INFO(
            "Generated AMDGPU code ({} bytes)",
            codegen_result.code.size());

        auto metadata = make_hip_shader_metadata(
            kernel, option, uses_hardware_rt_stack(),
            std::move(codegen_result.format_types));
        LUISA_ASSERT(
            metadata.requires_global_rt_stack ==
                codegen_result.requires_global_rt_stack,
            "HIP RT-stack metadata analysis disagrees with LLVM "
            "codegen (expected={}, generated={}).",
            metadata.requires_global_rt_stack,
            codegen_result.requires_global_rt_stack);
        shader_package.emplace(HIPShaderPackage{
            .metadata = std::move(metadata),
            .amdgpu_arch = _amdgpu_arch,
            .wave_size = wave_size,
            .code = std::move(codegen_result.code)});

        auto package = serialize_hip_shader_package(
            shader_package->code,
            shader_package->metadata,
            shader_package->amdgpu_arch,
            shader_package->wave_size);
        if (!option.name.empty()) {
            auto package_data =
                luisa::span{package.data(), package.size()};
            auto path = _io->write_shader_bytecode(
                option.name, package_data);
            auto saved_path = path.empty() ?
                                  option.name :
                                  luisa::string{path.string()};
            LUISA_INFO(
                "Saved HIP AOT shader package ({} bytes) to '{}'.",
                package.size(), saved_path);
        } else if (uses_shader_cache) {
            auto cache_artifact =
                serialize_hip_shader_cache_artifact(
                    cache_identity, package);
            auto path = _io->write_shader_cache(
                cache_name,
                luisa::span{
                    cache_artifact.data(),
                    cache_artifact.size()});
            auto saved_path = path.empty() ?
                                  cache_name :
                                  luisa::string{path.string()};
            LUISA_INFO(
                "Saved HIP shader cache entry ({} bytes) to '{}'.",
                cache_artifact.size(), saved_path);
        }
    }

    if (option.compile_only) {
        return ShaderCreationInfo::make_invalid();
    }

    auto bound_arguments =
        make_hip_bound_arguments(kernel);

    HIPShaderNative *shader = nullptr;
    if (shader_package->metadata.kind ==
        HIPShaderMetadata::Kind::RAY_TRACING) {
        auto rt_context = hiprt_context();
        shader = with_device([&] {
            return luisa::new_with_allocator<HIPShaderNative>(
                this, std::move(shader_package->code),
                "kernel_main", shader_package->metadata,
                rt_context,
                std::move(bound_arguments));
        });
    } else {
        shader = with_device([&] {
            return luisa::new_with_allocator<HIPShaderNative>(
                this, std::move(shader_package->code),
                "kernel_main", shader_package->metadata,
                std::move(bound_arguments));
        });
    }

    ShaderCreationInfo info{};
    info.handle = reinterpret_cast<uint64_t>(shader);
    info.native_handle = shader->handle();
    info.block_size = shader_package->metadata.block_size;
    return info;
#else
    LUISA_ERROR_WITH_LOCATION("HIP backend requires LLVM to be enabled.");
    return ShaderCreationInfo::make_invalid();
#endif
}

ShaderCreationInfo HIPDevice::create_shader(const ShaderOption &option, const ir::KernelModule *kernel) noexcept {
#ifdef LUISA_ENABLE_IR
    Clock clk;
    auto function = IR2AST::build(kernel);
    LUISA_VERBOSE("IR2AST done in {} ms.", clk.toc());
    return create_shader(option, function->function());
#else
    LUISA_ERROR_WITH_LOCATION("HIP backend was compiled without legacy IR support.");
    return ShaderCreationInfo::make_invalid();
#endif
}

ShaderCreationInfo HIPDevice::load_shader(luisa::string_view name, luisa::span<Type const *const> arg_types) noexcept {
#ifdef LUISA_COMPUTE_ENABLE_LLVM
    auto stream = _io->read_shader_bytecode(name);
    if (stream == nullptr || stream->length() == 0u) {
        LUISA_WARNING_WITH_LOCATION("HIP AOT shader package '{}' was not found.", name);
        return ShaderCreationInfo::make_invalid();
    }
    luisa::vector<std::byte> bytes(stream->length());
    stream->read(luisa::span{bytes.data(), bytes.size()});
    auto package = deserialize_hip_shader_package(bytes);
    if (!package) {
        LUISA_WARNING_WITH_LOCATION("HIP AOT shader package '{}' is invalid or unsupported.", name);
        return ShaderCreationInfo::make_invalid();
    }
    if (package->amdgpu_arch != _amdgpu_arch) {
        LUISA_WARNING_WITH_LOCATION(
            "HIP AOT shader package '{}' targets {}, but this device is {}.",
            name, package->amdgpu_arch, _amdgpu_arch);
        return ShaderCreationInfo::make_invalid();
    }
    if (!hip_wave_size_supported(_amdgpu_arch, package->wave_size)) {
        LUISA_WARNING_WITH_LOCATION(
            "HIP AOT shader package '{}' targets wave{}, which is unsupported on {}.",
            name, package->wave_size, _amdgpu_arch);
        return ShaderCreationInfo::make_invalid();
    }
    auto package_requires_hiprt =
        package->metadata.kind == HIPShaderMetadata::Kind::RAY_TRACING ||
        package->metadata.requires_ray_query;
    if (package_requires_hiprt && uses_hardware_rt_stack() &&
        package->wave_size != 32u) {
        LUISA_WARNING_WITH_LOCATION(
            "HIP AOT shader package '{}' targets wave{} ray tracing, "
            "but HIPRT on {} requires wave32.",
            name, package->wave_size, _amdgpu_arch);
        return ShaderCreationInfo::make_invalid();
    }
    if (package->metadata.argument_types.size() != arg_types.size()) {
        LUISA_WARNING_WITH_LOCATION(
            "HIP AOT shader package '{}' expects {} argument(s), but {} were requested.",
            name, package->metadata.argument_types.size(), arg_types.size());
        return ShaderCreationInfo::make_invalid();
    }
    for (auto i = 0u; i < arg_types.size(); i++) {
        if (package->metadata.argument_types[i] != arg_types[i]->description()) {
            LUISA_WARNING_WITH_LOCATION(
                "HIP AOT shader package '{}' argument {} has type '{}', not '{}'.",
                name, i, package->metadata.argument_types[i], arg_types[i]->description());
            return ShaderCreationInfo::make_invalid();
        }
    }
    auto shader = with_device([&]() noexcept -> HIPShaderNative * {
        if (package->metadata.kind == HIPShaderMetadata::Kind::RAY_TRACING) {
            return luisa::new_with_allocator<HIPShaderNative>(
                this, std::move(package->code), "kernel_main",
                package->metadata, hiprt_context());
        }
        return luisa::new_with_allocator<HIPShaderNative>(
            this, std::move(package->code), "kernel_main", package->metadata);
    });
    shader->set_name(luisa::string{name});
    ShaderCreationInfo info{};
    info.handle = reinterpret_cast<uint64_t>(shader);
    info.native_handle = shader->handle();
    info.block_size = package->metadata.block_size;
    return info;
#else
    LUISA_WARNING_WITH_LOCATION("HIP AOT loading requires LLVM support.");
    return ShaderCreationInfo::make_invalid();
#endif
}

Usage HIPDevice::shader_argument_usage(uint64_t handle, size_t index) noexcept {
    return reinterpret_cast<const HIPShader *>(handle)->argument_usage(index);
}

void HIPDevice::destroy_shader(uint64_t handle) noexcept {
    auto shader = reinterpret_cast<HIPShader *>(handle);
    with_device([&] { luisa::delete_with_allocator(shader); });
}

ResourceCreationInfo HIPDevice::create_event() noexcept {
    auto event = with_device([&] {
        return luisa::new_with_allocator<HIPEvent>(this);
    });
    return {.handle = reinterpret_cast<uint64_t>(event),
            .native_handle = event->handle()};
}

void HIPDevice::destroy_event(uint64_t handle) noexcept {
    auto event = reinterpret_cast<HIPEvent *>(handle);
    with_device([&] {
        luisa::delete_with_allocator(event);
    });
}

void HIPDevice::signal_event(uint64_t handle, uint64_t stream_handle, uint64_t fence_value) noexcept {
    auto event = reinterpret_cast<HIPEvent *>(handle);
    auto stream = reinterpret_cast<HIPStream *>(stream_handle);
    with_device([&] {
        event->signal(stream->handle(), fence_value);
    });
}

void HIPDevice::wait_event(uint64_t handle, uint64_t stream_handle, uint64_t fence_value) noexcept {
    auto event = reinterpret_cast<HIPEvent *>(handle);
    auto stream = reinterpret_cast<HIPStream *>(stream_handle);
    with_device([&] {
        event->wait(stream->handle(), fence_value);
    });
}

bool HIPDevice::is_event_completed(uint64_t handle, uint64_t fence_value) const noexcept {
    auto event = reinterpret_cast<HIPEvent *>(handle);
    return event->has_signaled(fence_value);
}

void HIPDevice::synchronize_event(uint64_t handle, uint64_t fence_value) noexcept {
    auto event = reinterpret_cast<HIPEvent *>(handle);
    event->synchronize(fence_value);
}

ResourceCreationInfo HIPDevice::create_mesh(const AccelOption &option) noexcept {
    auto context = hiprt_context();
    auto mesh = with_device([&] {
        return luisa::new_with_allocator<HIPMesh>(context, option);
    });
    return {.handle = reinterpret_cast<uint64_t>(mesh),
            .native_handle = mesh};
}

void HIPDevice::destroy_mesh(uint64_t handle) noexcept {
    with_device([=] {
        auto mesh = reinterpret_cast<HIPMesh *>(handle);
        luisa::delete_with_allocator(mesh);
    });
}

ResourceCreationInfo HIPDevice::create_procedural_primitive(const AccelOption &option) noexcept {
    auto context = hiprt_context();
    auto prim = with_device([&] {
        return luisa::new_with_allocator<HIPProceduralPrimitive>(context, option);
    });
    return {.handle = reinterpret_cast<uint64_t>(prim),
            .native_handle = prim};
}

void HIPDevice::destroy_procedural_primitive(uint64_t handle) noexcept {
    with_device([=] {
        auto prim = reinterpret_cast<HIPProceduralPrimitive *>(handle);
        luisa::delete_with_allocator(prim);
    });
}

ResourceCreationInfo HIPDevice::create_accel(const AccelOption &option) noexcept {
    auto context = hiprt_context();
    auto accel = with_device([&] {
        return luisa::new_with_allocator<HIPAccel>(context, option);
    });
    return {.handle = reinterpret_cast<uint64_t>(accel),
            .native_handle = accel};
}

void HIPDevice::destroy_accel(uint64_t handle) noexcept {
    with_device([=] {
        auto accel = reinterpret_cast<HIPAccel *>(handle);
        luisa::delete_with_allocator(accel);
    });
}

void HIPDevice::set_name(Resource::Tag resource_tag,
                         uint64_t resource_handle,
                         luisa::string_view name) noexcept {
    // ignored
}

}// namespace luisa::compute::hip

LUISA_EXPORT_API luisa::compute::DeviceInterface *create(luisa::compute::Context &&ctx,
                                                         const luisa::compute::DeviceConfig *config) noexcept {
    luisa::compute::hip::luisa_initialize_hip();
    return luisa::new_with_allocator<luisa::compute::hip::HIPDevice>(std::move(ctx), config);
}

LUISA_EXPORT_API void destroy(luisa::compute::DeviceInterface *device) noexcept {
    luisa::delete_with_allocator(device);
}

LUISA_EXPORT_API void backend_device_names(luisa::vector<luisa::string> &names) noexcept {
    names.clear();
    auto count = 0;
    luisa::compute::hip::luisa_initialize_hip();
    LUISA_CHECK_HIP(hipGetDeviceCount(&count));
    names.reserve(count);
    for (int i = 0; i < count; i++) {
        hipDeviceProp_t prop;
        LUISA_CHECK_HIP(hipGetDeviceProperties(&prop, i));
        names.emplace_back(prop.name);
    }
}

#include "../common/export_version.inl.h"
