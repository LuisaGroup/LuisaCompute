// DirectX cluster-record conversion test.
// This test covers:
// - Vulkan ClusterBuildInfo to DXR CLAS-argument ABI conversion.
// - Mixed UInt8/UInt16/UInt32 partitioning and exact operation counts.
// - Explicit destination addresses and original cluster ordering.
// - Geometry flag, stride, pointer, and truncation-bit mapping.
// - Fail-closed handling of malformed and unsupported records.

#include "ut/ut.hpp"
#include "test_device.h"

#include <array>
#include <cstdint>
#include <cstring>

#include <luisa/luisa-compute.h>
#include <luisa/backends/ext/native_shader_ext.h>
#include "hlsl_codegen.h"

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace test_dx_cluster_convert_detail {

constexpr int skip_exit_code = 77;
constexpr size_t record_count = 6u;
constexpr uint64_t pool_base = 0x100000080ull;
constexpr uint64_t pool_stride = 384u;

struct alignas(8) DxArgs {
    uint32_t cluster_id;
    uint32_t cluster_flags;
    uint16_t triangle_count;
    uint16_t vertex_count;
    uint32_t base_geometry_index_and_flags;
    uint32_t opacity_micromap_base_location;
    uint16_t vertex_buffer_stride;
    uint16_t index_buffer_stride;
    uint16_t opacity_micromap_index_buffer_stride;
    uint16_t geometry_index_and_flags_array_stride;
    uint16_t position_truncate_bit_count;
    uint16_t reserved_padding;
    uint64_t vertex_buffer;
    uint64_t index_buffer;
    uint64_t geometry_index_and_flags_array;
    uint64_t geometry_index_and_flags_index_buffer;
    uint64_t opacity_micromap_array;
    uint64_t opacity_micromap_index_buffer;
};
static_assert(sizeof(DxArgs) == 80u);
static_assert(offsetof(DxArgs, vertex_buffer) == 32u);
static_assert(offsetof(DxArgs, opacity_micromap_index_buffer) == 72u);

struct RawBuffers {
    std::array<uint32_t, 3u> counts;
    std::array<std::array<uint32_t, record_count * 20u>, 3u> args;
    std::array<std::array<uint32_t, record_count * 2u>, 3u> destinations;
    std::array<uint32_t, record_count * 2u> clas_addresses;
};

[[nodiscard]] uint32_t binding_index(
    NativeShader const &shader,
    NativeShaderResourceKind kind,
    uint32_t register_index) noexcept {
    auto bindings = shader.bindings();
    for (auto i = 0u; i < bindings.size(); ++i) {
        if (bindings[i].kind == kind &&
            bindings[i].register_index == register_index &&
            bindings[i].space_index == 0u) {
            return i;
        }
    }
    return ~0u;
}

[[nodiscard]] int run(Device &device) noexcept {
    if (device.backend_name() != "dx") {
        LUISA_WARNING("SKIPPED: [test_dx_cluster_convert] requires dx, got '{}'.",
                      device.backend_name());
        return skip_exit_code;
    }
    auto ext = device.extension<NativeShaderExt>();
    if (ext == nullptr) {
        LUISA_WARNING("SKIPPED: [test_dx_cluster_convert] NativeShaderExt is unavailable.");
        return skip_exit_code;
    }

    NativeShaderCompileInfo info;
    info.language = NativeShaderLanguage::HLSL;
    info.source = lc::hlsl::CodegenUtility::ReadInternalHLSLFile(
        "cluster_build_info.bytes");
    info.source_type = NativeShaderSourceType::SourceCode;
    info.entry_point = "main";
    info.shader_model = 62u;
    info.push_constant_size = 4u * sizeof(uint32_t);
    auto compiled = ext->compile(info);
    if (!compiled.ok()) {
        expect(false) << compiled.error.c_str();
        return 1;
    }
    auto metadata = ext->load(compiled);
    if (!metadata.valid()) {
        expect(false) << "failed to load the cluster conversion shader";
        return 1;
    }
    NativeShader shader{*ext, std::move(metadata)};

    std::array<ClusterBuildInfo, record_count> records{};
    std::array<ClusterIndexFormat, record_count> formats{
        ClusterIndexFormat::UInt16,
        ClusterIndexFormat::UInt8,
        ClusterIndexFormat::UInt32,
        ClusterIndexFormat::UInt16,
        ClusterIndexFormat::UInt32,// overwritten below to make a malformed record
        ClusterIndexFormat::UInt8,
    };
    std::array<ClusterGeometryFlags, record_count> geometry_flags{
        static_cast<ClusterGeometryFlags>(4u | 2u),
        ClusterGeometryFlags::Opaque,
        ClusterGeometryFlags::NoDuplicateAnyHitInvocation,
        static_cast<ClusterGeometryFlags>(4u | 1u),
        ClusterGeometryFlags::Opaque,
        ClusterGeometryFlags::Opaque,
    };
    for (auto i = 0u; i < record_count; ++i) {
        records[i] = make_cluster_build_info(
            100u + i, (i % 2u) + 1u, (i % 2u) + 3u,
            0x200000ull + i * 0x1000ull,
            0x100000ull + i * 0x2000ull,
            formats[i], i + 1u, geometry_flags[i]);
        records[i].vertex_buffer_stride = 12u;
        records[i].geometry_index_and_flags_buffer_stride = 4u;
        records[i].geometry_index_and_flags_buffer = 0x300000ull + i * 0x100ull;
        records[i].packed0 |= (i + 1u) << 18u;
    }
    // The malformed record must be skipped rather than reaching DXR.
    records[4].packed0 = (4u << 18u) | (3u << 9u) | 1u;
    records[4].geometry_index_and_flags_buffer = 0u;

    Stream stream = device.create_stream(StreamTag::COMPUTE);
    Buffer<uint32_t> source = device.create_buffer<uint32_t>(record_count * 16u);
    std::array<Buffer<uint32_t>, 3u> args{
        device.create_buffer<uint32_t>(record_count * 20u),
        device.create_buffer<uint32_t>(record_count * 20u),
        device.create_buffer<uint32_t>(record_count * 20u)};
    std::array<Buffer<uint32_t>, 3u> destinations{
        device.create_buffer<uint32_t>(record_count * 2u),
        device.create_buffer<uint32_t>(record_count * 2u),
        device.create_buffer<uint32_t>(record_count * 2u)};
    Buffer<uint32_t> counts = device.create_buffer<uint32_t>(3u);
    Buffer<uint32_t> clas_addresses = device.create_buffer<uint32_t>(record_count * 2u);

    std::array<uint32_t, record_count * 16u> source_words{};
    std::memcpy(source_words.data(), records.data(), sizeof(records));
    std::array<uint32_t, 3u> zero_counts{};
    stream << source.copy_from(luisa::span{source_words})
           << counts.copy_from(luisa::span{zero_counts});

    auto launcher = shader.launcher();
    auto bind = [&](NativeShaderResourceKind kind, uint32_t reg,
                    Buffer<uint32_t> const &buffer, Usage usage) {
        auto index = binding_index(shader, kind, reg);
        expect(index != ~0u) << "missing reflected cluster conversion binding";
        if (index != ~0u) {
            launcher.add_buffer_by_index(index, buffer.view(), usage);
        }
    };
    bind(NativeShaderResourceKind::StructuredBuffer, 0u, source, Usage::READ);
    for (auto format = 0u; format < 3u; ++format) {
        bind(NativeShaderResourceKind::RWStructuredBuffer,
             format, args[format], Usage::WRITE);
    }
    bind(NativeShaderResourceKind::RWStructuredBuffer, 3u, counts, Usage::WRITE);
    for (auto format = 0u; format < 3u; ++format) {
        bind(NativeShaderResourceKind::RWStructuredBuffer,
             4u + format, destinations[format], Usage::WRITE);
    }
    bind(NativeShaderResourceKind::RWStructuredBuffer, 7u,
         clas_addresses, Usage::WRITE);
    std::array<uint32_t, 4u> constants{
        static_cast<uint32_t>(record_count),
        static_cast<uint32_t>(pool_stride),
        static_cast<uint32_t>(pool_base),
        static_cast<uint32_t>(pool_base >> 32u)};
    launcher.add_uniform(constants.data(), sizeof(constants), alignof(uint32_t));
    auto validation = launcher.validate();
    expect(validation.empty()) << validation.c_str();
    if (!validation.empty()) return 1;
    stream << std::move(launcher).build(uint3(record_count, 1u, 1u));

    RawBuffers raw{};
    stream << counts.copy_to(luisa::span{raw.counts});
    for (auto format = 0u; format < 3u; ++format) {
        stream << args[format].copy_to(luisa::span{raw.args[format]})
               << destinations[format].copy_to(luisa::span{raw.destinations[format]});
    }
    stream << clas_addresses.copy_to(luisa::span{raw.clas_addresses})
           << synchronize();

    uint failures = 0u;
    auto check = [&](bool condition, luisa::string_view message) {
        if (!condition) {
            ++failures;
            LUISA_WARNING("[test_dx_cluster_convert] {}", message);
        }
    };
    check(raw.counts[0] == 2u && raw.counts[1] == 2u && raw.counts[2] == 1u,
          "mixed-index operation counts must be UInt8=2, UInt16=2, UInt32=1");
    check(failures == 0u, "reflected bindings and dispatch must succeed");

    std::array<uint64_t, record_count> addresses{};
    std::memcpy(addresses.data(), raw.clas_addresses.data(), sizeof(addresses));
    for (auto i = 0u; i < record_count; ++i) {
        auto expected = i == 4u ? 0u : pool_base + i * pool_stride;
        check(addresses[i] == expected,
              "CLAS address must preserve the original record index");
    }

    auto record_with_id = [&](uint32_t id) -> ClusterBuildInfo const * {
        for (auto const &record : records) {
            if (record.cluster_id == id) return &record;
        }
        return nullptr;
    };
    for (auto format = 0u; format < 3u; ++format) {
        for (auto slot = 0u; slot < raw.counts[format]; ++slot) {
            DxArgs converted{};
            std::memcpy(&converted, raw.args[format].data() + slot * 20u,
                        sizeof(converted));
            auto source_record = record_with_id(converted.cluster_id);
            check(source_record != nullptr, "compacted argument has an unknown cluster id");
            if (source_record == nullptr) continue;
            auto const &src = *source_record;
            auto original_index = static_cast<size_t>(converted.cluster_id - 100u);
            auto expected_format = src.index_type() == ClusterIndexFormat::UInt8 ? 0u :
                                   src.index_type() == ClusterIndexFormat::UInt16 ? 1u : 2u;
            check(format == expected_format, "record was compacted into the wrong index batch");
            check(converted.cluster_flags == (src.cluster_flags & 1u),
                  "cluster flags were not translated");
            check(converted.triangle_count == src.triangle_count() &&
                      converted.vertex_count == src.vertex_count(),
                  "triangle/vertex counts were not preserved");
            check(converted.base_geometry_index_and_flags ==
                      (src.packed1 & 0xc0ffffffu),
                  "geometry flags must preserve DXR bits 31/30 and clear reserved bits");
            check(converted.opacity_micromap_base_location == 0u,
                  "OMM base location must be zero");
            check(converted.vertex_buffer_stride == src.vertex_buffer_stride &&
                      converted.index_buffer_stride == src.index_buffer_stride &&
                      converted.opacity_micromap_index_buffer_stride ==
                          src.opacity_micromap_index_buffer_stride &&
                      converted.geometry_index_and_flags_array_stride ==
                          src.geometry_index_and_flags_buffer_stride,
                  "buffer strides were not mapped to their DXR fields");
            check(converted.position_truncate_bit_count ==
                      src.position_truncate_bit_count() &&
                      converted.reserved_padding == 0u,
                  "position truncation or reserved padding is incorrect");
            check(converted.vertex_buffer == src.vertex_buffer &&
                      converted.index_buffer == src.index_buffer &&
                      converted.geometry_index_and_flags_array ==
                          src.geometry_index_and_flags_buffer &&
                      converted.geometry_index_and_flags_index_buffer == 0u &&
                      converted.opacity_micromap_array == 0u &&
                      converted.opacity_micromap_index_buffer == 0u,
                  "device pointers were not mapped to their DXR fields");

            uint64_t destination = 0u;
            std::memcpy(&destination,
                        raw.destinations[format].data() + slot * 2u,
                        sizeof(destination));
            check(destination == pool_base + original_index * pool_stride,
                  "compacted destination does not match its original CLAS slot");
        }
    }

    expect(failures == 0u) << "cluster record conversion checks failed: " << failures;
    LUISA_INFO("[test_dx_cluster_convert] {} records, counts ({}, {}, {}), {} failure(s).",
               record_count, raw.counts[0], raw.counts[1], raw.counts[2], failures);
    return failures == 0u ? 0 : 1;
}

}// namespace test_dx_cluster_convert_detail

int main(int argc, char *argv[]) {
    auto dc = luisa::test::create_device_from_ut(argc, argv);
    if (!dc) return 0;
    boost::ut::detail::cfg::parse_arg_with_fallback(
        argc, const_cast<const char **>(argv));
    return test_dx_cluster_convert_detail::run(dc->device);
}
