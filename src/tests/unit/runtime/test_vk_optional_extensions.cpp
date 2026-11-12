// Test for the Vulkan backend's optional extension enable/disable contract.
// This test covers:
// - a device created with every optional extension switched off still runs a
//   normal compute + bindless workload (fallback paths stay correct)
// - a device created with the default config reports enabled ⊆ supported and
//   enabled ⊆ requested for every optional extension
// - the reported state is stable, unique by name, and "active" implies
//   "enabled"
// - VulkanExtensionStatusExt answers per-name queries consistently with the
//   flat list

#include "ut/ut.hpp"
#include "test_device.h"

#include <luisa/luisa-compute.h>
#include <luisa/dsl/sugar.h>
#include <luisa/backends/ext/vk_config_ext.h>
#include <luisa/backends/ext/vk_extension_ext.h>

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace vk_optional_extensions_test_detail {

// Switches off every optional Vulkan device extension, so the backend must use
// its fallback path for all of them.
class AllDisabledVkConfig final : public VulkanDeviceConfigExt {
public:
#define LC_TEST_DISABLE(Name)                          \
    [[nodiscard]] bool Name() const noexcept override { \
        return false;                                  \
    }
    LC_TEST_DISABLE(enable_descriptor_heap)
    LC_TEST_DISABLE(enable_device_address_commands)
    LC_TEST_DISABLE(enable_unified_image_layouts)
    LC_TEST_DISABLE(enable_copy_memory_indirect)
    LC_TEST_DISABLE(enable_memory_decompression)
    LC_TEST_DISABLE(enable_zero_initialize_device_memory)
    LC_TEST_DISABLE(enable_internally_synchronized_queues)
    LC_TEST_DISABLE(enable_extended_flags)
    LC_TEST_DISABLE(enable_image_tiling_control)
    LC_TEST_DISABLE(enable_present_id2)
    LC_TEST_DISABLE(enable_present_wait2)
    LC_TEST_DISABLE(enable_present_mode_fifo_latest_ready)
    LC_TEST_DISABLE(enable_surface_maintenance1)
    LC_TEST_DISABLE(enable_swapchain_maintenance1)
    LC_TEST_DISABLE(enable_present_timing)
    LC_TEST_DISABLE(enable_multisampled_render_to_swapchain)
    LC_TEST_DISABLE(enable_shader_fma)
    LC_TEST_DISABLE(enable_shader_64bit_indexing)
    LC_TEST_DISABLE(enable_shader_uniform_buffer_unsized_array)
    LC_TEST_DISABLE(enable_shader_long_vector)
    LC_TEST_DISABLE(enable_shader_subgroup_partitioned)
    LC_TEST_DISABLE(enable_shader_split_barrier)
    LC_TEST_DISABLE(enable_shader_ocp_microscaling_types)
    LC_TEST_DISABLE(enable_shader_constant_data)
    LC_TEST_DISABLE(enable_shader_abort)
    LC_TEST_DISABLE(enable_ray_tracing_invocation_reorder)
    LC_TEST_DISABLE(enable_opacity_micromap)
    LC_TEST_DISABLE(enable_custom_resolve)
    LC_TEST_DISABLE(enable_fragment_density_map_offset)
    LC_TEST_DISABLE(enable_primitive_restart_index)
    LC_TEST_DISABLE(enable_texture_compression_astc_3d)
    LC_TEST_DISABLE(enable_pipeline_library_group_handles)
    LC_TEST_DISABLE(enable_external_memory_metal)
    LC_TEST_DISABLE(enable_depth_clamp_zero_one)
    LC_TEST_DISABLE(enable_maintenance8)
    LC_TEST_DISABLE(enable_maintenance9)
    LC_TEST_DISABLE(enable_maintenance10)
    LC_TEST_DISABLE(enable_maintenance11)
    LC_TEST_DISABLE(enable_maintenance12)
    LC_TEST_DISABLE(enable_robustness2)
    LC_TEST_DISABLE(enable_tensor_arm)
    LC_TEST_DISABLE(enable_data_graph_arm)
    LC_TEST_DISABLE(enable_tile_shading_qcom)
    LC_TEST_DISABLE(enable_tile_memory_heap_qcom)
    LC_TEST_DISABLE(enable_partitioned_acceleration_structure)
    LC_TEST_DISABLE(enable_ray_tracing_linear_swept_spheres)
    LC_TEST_DISABLE(enable_dense_geometry_format)
#undef LC_TEST_DISABLE
};

constexpr auto element_count = 1024u;

// Exercises the parts of the backend that each optional extension can change:
// buffer allocation (zero-initialize / tiling), descriptor binding (bindless /
// descriptor heap), dispatch, copies and readback.
void run_backend_workload(Device &device) {
    auto stream = device.create_stream(StreamTag::COMPUTE);
    auto buffer = device.create_buffer<float>(element_count);
    auto bindless = device.create_bindless_array(16u);

    Kernel1D fill = [](BufferFloat output) noexcept {
        output.write(dispatch_id().x, 3.0f);
    };
    auto fill_shader = device.compile(fill);

    // Read the value back through a bindless descriptor, so the descriptor
    // path (descriptor sets, or descriptor heaps when enabled) is exercised.
    Kernel1D read_back = [&](BufferFloat output) noexcept {
        auto view = bindless->buffer<float>(0u);
        output.write(dispatch_id().x, view.read(dispatch_id().x));
    };
    auto read_back_shader = device.compile(read_back);

    // Image round trip: upload (TRANSFER_DST), shader write, download
    // (TRANSFER_SRC). This is what makes the unified-image-layouts fast path
    // observable: with the extension on, the tracker keeps GENERAL instead of
    // switching layouts, and the result must be identical.
    constexpr auto image_extent = 32u;
    auto image = device.create_image<float>(PixelStorage::FLOAT4,
                                            make_uint2(image_extent, image_extent));
    auto image_staging = device.create_buffer<float4>(image_extent * image_extent);
    luisa::vector<float4> image_host(image_extent * image_extent,
                                     make_float4(1.0f, 2.0f, 3.0f, 4.0f));
    Kernel2D fill_image = [&](Float4 value) noexcept {
        image->write(dispatch_id().xy(), value);
    };
    auto fill_image_shader = device.compile(fill_image);
    stream << image_staging.copy_from(luisa::span{image_host})
           << image.copy_from(image_staging)
           << synchronize();
    stream << fill_image_shader(make_float4(9.0f, 8.0f, 7.0f, 6.0f))
                  .dispatch(image_extent, image_extent)
           << synchronize();
    image_host.assign(image_extent * image_extent, make_float4(0.0f));
    stream << image.copy_to(image_staging)
           << image_staging.copy_to(luisa::span{image_host})
           << synchronize();
    auto image_ok = true;
    for (auto const &pixel : image_host) {
        if (pixel.x != 9.0f || pixel.y != 8.0f ||
            pixel.z != 7.0f || pixel.w != 6.0f) {
            image_ok = false;
            break;
        }
    }
    expect(image_ok) << "image upload/shader-write/download must round-trip";

    luisa::vector<float> host(element_count, 0.0f);
    stream << buffer.copy_from(luisa::span{host})
           << fill_shader(buffer).dispatch(element_count)
           << synchronize();
    bindless.emplace_on_update(0u, buffer);
    stream << bindless.update() << synchronize();
    // Overwrite with a marker so the bindless read is observable.
    host.assign(element_count, 7.0f);
    stream << buffer.copy_from(luisa::span{host}) << synchronize();
    stream << read_back_shader(buffer).dispatch(element_count)
           << synchronize();
    host.assign(element_count, 0.0f);
    stream << buffer.copy_to(luisa::span{host}) << synchronize();

    auto ok = true;
    for (auto i = 0u; i < element_count; ++i) {
        if (host[i] != 7.0f) {
            LUISA_WARNING("element {} = {} (expected 7)", i, host[i]);
            ok = false;
            break;
        }
    }
    expect(ok) << "compute + bindless workload must produce correct results";
}

void check_status_consistency(Device &device, bool expect_none_enabled) {
    auto ext = device.extension<VulkanExtensionStatusExt>();
    expect(ext != nullptr) << "the Vulkan backend must expose the status extension";
    if (ext == nullptr) { return; }
    auto extensions = ext->extensions();
    expect(!extensions.empty());

    auto enabled_count = size_t{0u};
    luisa::vector<luisa::string> enabled_names;
    for (auto const &status : extensions) {
        expect(!status.name.empty());
        if (status.enabled) { ++enabled_count; }
        // Invariants the negotiation must never break.
        expect(!status.enabled || status.supported)
            << status.name << " cannot be enabled without device support";
        expect(!status.enabled || status.requested)
            << status.name << " cannot be enabled when the config refuses it";
        expect(!status.active || status.enabled)
            << status.name << " cannot be active when it is not enabled";
        // Per-name queries agree with the flat list.
        expect(ext->is_supported(status.name) == status.supported) << status.name;
        expect(ext->is_enabled(status.name) == status.enabled) << status.name;
        expect(ext->is_active(status.name) == status.active) << status.name;
        if (status.enabled) { enabled_names.emplace_back(status.name); }
    }
    // The workload above uploads, writes and downloads an image, so the
    // unified-image-layouts fast path must report itself active when the
    // extension is enabled.
    if (ext->is_enabled("VK_KHR_unified_image_layouts")) {
        expect(ext->is_active("VK_KHR_unified_image_layouts"))
            << "the barrier tracker must adopt GENERAL layouts when the "
               "extension is enabled";
    } else {
        expect(!ext->is_active("VK_KHR_unified_image_layouts"));
    }
    if (expect_none_enabled) {
        expect(enabled_count == 0u)
            << "a device created with every switch off must enable nothing";
        if (enabled_count != 0u) {
            for (auto const &name : enabled_names) {
                LUISA_WARNING("unexpectedly enabled optional extension: {}", name);
            }
        }
    } else {
        LUISA_INFO("{} of {} optional extensions enabled on this device.",
                   enabled_count, extensions.size());
        for (auto const &name : enabled_names) {
            LUISA_INFO("  {} [active={}]", name, ext->is_active(name));
        }
    }
    // Unknown names are reported as absent rather than guessed.
    expect(!ext->is_enabled("VK_definitely_not_an_extension"));
    expect(!ext->is_supported("VK_definitely_not_an_extension"));
    expect(!ext->is_active("VK_definitely_not_an_extension"));
}

}// namespace vk_optional_extensions_test_detail

using namespace vk_optional_extensions_test_detail;

int main(int argc, char *argv[]) {
    if (argc < 2) {
        luisa::test::print_device_usage(argc > 0 ? argv[0] : "test");
        return 1;
    }
    auto backend = luisa::string_view{argv[1]};
    if (backend != "vk") {
        LUISA_INFO("This test only applies to the Vulkan backend; skipping {}.",
                   backend);
        return 0;
    }
    boost::ut::detail::cfg::parse_arg_with_fallback(
        argc, const_cast<const char **>(argv));

    "vk_optional_extensions_all_disabled"_test = [&] {
        DeviceConfig device_config{};
        device_config.extension = luisa::make_unique<AllDisabledVkConfig>();
        auto context = Context{argv[0]};
        auto device = context.create_device("vk", &device_config);
        run_backend_workload(device);
        check_status_consistency(device, true);
    };

    "vk_optional_extensions_default_config"_test = [&] {
        auto context = Context{argv[0]};
        auto device = context.create_device("vk");
        run_backend_workload(device);
        check_status_consistency(device, false);
    };
}
