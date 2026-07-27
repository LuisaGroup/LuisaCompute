// Capability-gated Vulkan sparse-buffer integration test.
//
// This exercises the native queue/memory path that pure planner tests cannot:
// ordinary-queue event wait -> timeline bridge -> vkQueueBindSparse -> copy /
// readback -> explicit unmap. Vulkan validation is requested before the
// process-owned instance is created.

#include "ut/ut.hpp"

#include <volk.h>

#include <luisa/backends/ext/vk_config_ext.h>
#include <luisa/luisa-compute.h>
#include <luisa/runtime/sparse_command_list.h>

#include <cstdint>
#include <cstdlib>

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

void request_vulkan_validation() noexcept {
#ifdef _WIN32
    _putenv_s("LUISA_VULKAN_VALIDATION", "1");
#else
    setenv("LUISA_VULKAN_VALIDATION", "1", 1);
#endif
}

class SparseCapabilityProbe final : public VulkanDeviceConfigExt {
private:
    bool _sparse_buffer_supported{};

public:
    void init_volk(PFN_vkGetInstanceProcAddr handler) noexcept override {
        volkInitializeCustom(handler);
    }

    void readback_vulkan_device(
        VkInstance instance,
        VkPhysicalDevice physical_device,
        VkDevice device,
        VkAllocationCallbacks *,
        VkPipelineCacheHeaderVersionOne const &,
        VkQueue,
        VkQueue,
        VkQueue,
        uint32_t,
        uint32_t,
        uint32_t,
        IDxcCompiler3 *,
        IDxcLibrary *,
        IDxcUtils *) noexcept override {
        volkLoadInstance(instance);
        volkLoadDevice(device);
        VkPhysicalDeviceFeatures features{};
        vkGetPhysicalDeviceFeatures(physical_device, &features);
        _sparse_buffer_supported =
            features.sparseBinding == VK_TRUE &&
            features.sparseResidencyBuffer == VK_TRUE;
    }

    [[nodiscard]] bool sparse_buffer_supported() const noexcept {
        return _sparse_buffer_supported;
    }
};

}// namespace

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(
        argc, const_cast<const char **>(argv));
    request_vulkan_validation();

    Context context{argc > 0 && argv != nullptr ? argv[0] : ""};
    DeviceConfig config{.headless = true};
    auto capability_probe = luisa::make_unique<SparseCapabilityProbe>();
    auto capability_probe_ptr = capability_probe.get();
    config.extension = std::move(capability_probe);
    auto device = context.create_device("vk", &config);

    if (!capability_probe_ptr->sparse_buffer_supported()) {
        LUISA_INFO(
            "Skipping Vulkan sparse-buffer integration test: physical "
            "device lacks sparseBinding or sparseResidencyBuffer.");
        return 0;
    }

    "vk_sparse_buffer_maps_copies_and_unmaps_with_queue_bridge"_test = [&] {
        auto sparse_buffer = device.create_sparse_buffer<uint32_t>(1u);
        expect(sparse_buffer.tile_size_bytes() != 0u);
        auto heap = device.allocate_sparse_buffer_heap(
            sparse_buffer.tile_size_bytes());
        auto producer = device.create_stream(StreamTag::COMPUTE);
        auto consumer = device.create_stream(StreamTag::COMPUTE);
        auto handoff = device.create_timeline_event();

        producer << handoff.signal(1u);
        SparseCommandList map;
        map << sparse_buffer.map_tile(0u, 1u, heap);
        consumer << handoff.wait(1u)
                 << map.commit();

        constexpr uint32_t expected = 0x5a17c3e9u;
        uint32_t actual{};
        consumer << sparse_buffer.view().copy_from(luisa::span{&expected, 1u})
                 << sparse_buffer.view().copy_to(luisa::span{&actual, 1u})
                 << synchronize();
        expect(actual == expected)
            << "mapped sparse-buffer data must survive a transfer round trip";

        SparseCommandList unmap;
        unmap << sparse_buffer.unmap_tile(0u, 1u);
        consumer << unmap.commit()
                 << synchronize();
    };
}
