// Test for the Vulkan backend's numeric capability record
// (plan P6.1 / P6.2 / P6.3).
//
// This test covers:
// - `VulkanExtensionStatusExt::capability_status()` agrees, field for field, with
//   the matching `Device::query` key: a non-zero field always has a non-empty
//   answer and an unknown (empty) answer always means 0 in the record. Because
//   both are read from the one record the device captured at creation, a drift
//   between the query path and the report shows up here
// - every field of the struct is exercised: the 11 unconditional VkPhysicalDeviceLimits
//   members answer non-zero and plausible values, the three conditional ones
//   (min/max subgroup size, which need Vulkan subgroup-size-control properties)
//   are either real or explicitly unknown - never derived from subgroupSize
// - the descriptor-heap state comes from the optional-extension registry: it
//   agrees with the flat `extensions()` entry and the per-name queries, and the
//   `--disable-descriptor-heap` run proves the requested/enabled/active bits
//   follow the configuration instead of being hard-coded
// - the SM-count contract (P6.2): unknown by default (`sm_count` empty,
//   `sm_count_source = unknown`), reported as `calibration_estimate` only when the
//   caller supplies a calibrated value through
//   `VulkanDeviceConfigExt::calibrated_sm_count_estimate()`, and never published
//   for an implausible input (`--sm-count` above the bound)
// - a key the backend does not know stays unanswered
//
// One process holds one Vulkan backend Device, so the three configurations are
// separate runs of this binary:
//   test_vk_capability_status vk                             (default: no calibration,
//                                                             descriptor heap on)
//   test_vk_capability_status vk --sm-count 84               (calibrated estimate)
//   test_vk_capability_status vk --sm-count 100000           (rejected -> unknown)
//   test_vk_capability_status vk --disable-descriptor-heap   (heap switched off)
#include "ut/ut.hpp"
#include "test_device.h"

#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string_view>

#include <luisa/luisa-compute.h>
#include <luisa/backends/ext/vk_config_ext.h>
#include <luisa/backends/ext/vk_extension_ext.h>
#include <luisa/runtime/device.h>

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

// The configuration switches this test needs. Everything else keeps the backend
// default, so the device is created exactly as a default one except for what the
// command line asks for.
struct CapabilityOptions {
    uint64_t sm_count{0u};
    bool disable_descriptor_heap{false};
};

CapabilityOptions g_options{};

class CapabilityStatusConfig final : public VulkanDeviceConfigExt {
public:
    [[nodiscard]] uint64_t calibrated_sm_count_estimate() const noexcept override {
        return g_options.sm_count;
    }
    [[nodiscard]] bool enable_descriptor_heap() const noexcept override {
        return !g_options.disable_descriptor_heap;
    }
};

// Removes this test's options from argv in place, so the device helper and the
// Boost.UT parser only see what they understand.
void take_options(int &argc, char **argv) noexcept {
    auto kept = 0;
    for (auto i = 0; i < argc; ++i) {
        if (std::strcmp(argv[i], "--sm-count") == 0 && i + 1 < argc) {
            g_options.sm_count = std::strtoull(argv[i + 1], nullptr, 10);
            ++i;// skip the value as well
            continue;
        }
        if (std::strcmp(argv[i], "--disable-descriptor-heap") == 0) {
            g_options.disable_descriptor_heap = true;
            continue;
        }
        argv[kept++] = argv[i];
    }
    argv[kept] = nullptr;
    argc = kept;
}

[[nodiscard]] uint64_t parse_answer(luisa::string_view answer) noexcept {
    return std::strtoull(luisa::string{answer}.c_str(), nullptr, 10);
}

// The 14 public keys of plan P6.1 with the value the P6.3 record carries for each.
struct FieldProbe {
    const char *key;
    uint64_t recorded;
    bool available;
};

[[nodiscard]] std::array<FieldProbe, 14> probes(const VulkanCapabilityStatus &s) noexcept {
    return std::array<FieldProbe, 14>{{
        {"max_memory_allocation_count", s.max_memory_allocation_count, true},
        {"min_subgroup_size", s.min_subgroup_size, s.subgroup_size_control_available},
        {"max_subgroup_size", s.max_subgroup_size, s.subgroup_size_control_available},
        {"subgroup_ops", s.subgroup_ops, s.subgroup_ops_available},
        {"max_compute_workgroup_count_x", s.max_compute_workgroup_count_x, true},
        {"max_compute_workgroup_count_y", s.max_compute_workgroup_count_y, true},
        {"max_compute_workgroup_count_z", s.max_compute_workgroup_count_z, true},
        {"max_compute_workgroup_size_x", s.max_compute_workgroup_size_x, true},
        {"max_compute_workgroup_size_y", s.max_compute_workgroup_size_y, true},
        {"max_compute_workgroup_size_z", s.max_compute_workgroup_size_z, true},
        {"max_compute_shared_memory_bytes", s.max_compute_shared_memory_bytes, true},
        {"non_coherent_atom_size", s.non_coherent_atom_size, true},
        {"min_storage_buffer_offset_alignment", s.min_storage_buffer_offset_alignment, true},
        {"max_storage_buffer_range", s.max_storage_buffer_range, true},
    }};
}

[[nodiscard]] const VulkanExtensionStatus *find_extension(
    luisa::span<const VulkanExtensionStatus> extensions,
    luisa::string_view name) noexcept {
    for (auto const &status : extensions) {
        if (status.name == name) { return &status; }
    }
    return nullptr;
}

void reg_query_status_consistency(Device &device) {
    "vk_capability_status_matches_every_query"_test = [&device] {
        auto *ext = device.extension<VulkanExtensionStatusExt>();
        expect(ext != nullptr) << "the Vulkan backend must expose the status extension";
        if (ext == nullptr) { return; }
        auto status = ext->capability_status();
        // The struct is returned by value and owns no pointer into the device.
        auto copy = status;
        expect(copy.max_compute_shared_memory_bytes == status.max_compute_shared_memory_bytes);
        for (auto const &probe : probes(status)) {
            auto answer = device.query(probe.key);
            if (!probe.available) {
                expect(answer.empty())
                    << probe.key << ": the record says this device cannot answer it, so "
                                    "the query must be empty rather than a number";
                expect(probe.recorded == 0u)
                    << probe.key << ": an unavailable limit must be 0 in the record";
                LUISA_INFO("P6.x {}: unknown on this device", probe.key);
                continue;
            }
            expect(!answer.empty())
                << probe.key << ": a field the record fills must also answer as a "
                                "query (the two read the same device properties)";
            if (answer.empty()) { continue; }
            expect(parse_answer(answer) == probe.recorded)
                << probe.key << ": query answer " << answer << " must equal the "
                << "capability_status() field " << probe.recorded;
            LUISA_INFO("P6.x {}: {}", probe.key, answer);
        }
        // The unconditional limits are real values, never 0: Vulkan has no
        // "unknown" for them.
        for (auto const &probe : probes(status)) {
            if (!probe.available || probe.recorded == 0u) {
                expect(!probe.available)
                    << probe.key << ": an available limit must carry a non-zero value";
            }
        }
        // The three VkDeviceSize / storage-range fields are reported through
        // 64-bit holds; a value that does not round-trip through the query is a
        // truncation bug, not a formatting preference.
        expect(parse_answer(device.query("non_coherent_atom_size")) ==
               status.non_coherent_atom_size);
        expect(parse_answer(device.query("max_storage_buffer_range")) ==
               status.max_storage_buffer_range);
        // A name that is not one of the 14 or the SM-count pair stays unanswered.
        expect(device.query("max_compute_workgroup_size").empty())
            << "a near-miss key must not be answered";
        expect(device.query("subgroup_size").empty())
            << "the plain subgroup size is not one of the 14 keys";
    };
}

void reg_descriptor_heap_state(Device &device) {
    "vk_capability_descriptor_heap_state"_test = [&device] {
        auto *ext = device.extension<VulkanExtensionStatusExt>();
        expect(ext != nullptr);
        if (ext == nullptr) { return; }
        auto status = ext->capability_status();
        auto heap = status.descriptor_heap;
        LUISA_INFO("VK_EXT_descriptor_heap: supported={} requested={} enabled={} "
                   "active={} (switched off by config: {})",
                   heap.supported, heap.requested, heap.enabled, heap.active,
                   g_options.disable_descriptor_heap);
        // The entry in the flat list is the same negotiation: the struct must not
        // be a second, independently maintained answer.
        auto extensions = ext->extensions();
        auto *entry = find_extension(extensions, "VK_EXT_descriptor_heap");
        expect(entry != nullptr)
            << "the registry must carry VK_EXT_descriptor_heap";
        if (entry != nullptr) {
            expect(heap.supported == entry->supported);
            expect(heap.requested == entry->requested);
            expect(heap.enabled == entry->enabled);
            expect(heap.active == entry->active);
        }
        // ... and the per-name queries agree as well.
        expect(heap.supported == ext->is_supported("VK_EXT_descriptor_heap"));
        expect(heap.enabled == ext->is_enabled("VK_EXT_descriptor_heap"));
        expect(heap.active == ext->is_active("VK_EXT_descriptor_heap"));
        // Invariants the negotiation can never break, whatever the configuration.
        expect(!heap.requested || heap.supported)
            << "the config can request an extension, not support it";
        expect(!heap.enabled || heap.requested)
            << "an extension cannot be enabled when the configuration refuses it";
        expect(!heap.active || heap.enabled)
            << "the backend cannot route work through an extension it did not enable";
        if (g_options.disable_descriptor_heap) {
            // The decisive check that the state is not hard-coded: this run switched
            // the extension off, so requested/enabled/active must all be false while
            // "supported" still reports the physical device's own answer.
            expect(!heap.requested)
                << "enable_descriptor_heap() returned false, so it must not be "
                   "reported as requested";
            expect(!heap.enabled) << "a refused extension must not be enabled";
            expect(!heap.active) << "a disabled extension cannot be active";
            expect(!ext->is_enabled("VK_EXT_descriptor_heap"));
        } else {
            // Default configuration: the switch is on, so on hardware that exposes
            // the extension it must be requested and enabled (and only hardware
            // without it may report otherwise).
            if (heap.supported) {
                expect(heap.requested)
                    << "the default configuration requests VK_EXT_descriptor_heap";
            }
        }
    };
}

void reg_sm_count(Device &device) {
    "vk_sm_count_source_is_honest"_test = [&device] {
        auto *ext = device.extension<VulkanExtensionStatusExt>();
        expect(ext != nullptr);
        if (ext == nullptr) { return; }
        auto status = ext->capability_status();
        auto answer = device.query("sm_count");
        auto source = device.query("sm_count_source");
        LUISA_INFO("P6.2 sm_count='{}' sm_count_source='{}' (calibrated input: {})",
                   answer, source, g_options.sm_count);
        expect(!source.empty())
            << "sm_count_source must always answer: unknown is an answer";
        expect(source == "unknown" || source == "calibration_estimate")
            << "sm_count_source has exactly two values (got '" << source << "')";
        // Vulkan has no physical SM-count query, so the backend has exactly one
        // legitimate way to report a number: a caller-supplied calibration.
        auto calibration_is_plausible =
            g_options.sm_count > 0u && g_options.sm_count <= 4096u;
        if (!calibration_is_plausible) {
            // No input, or an input the backend must reject: unknown, and the
            // number must not be fabricated.
            expect(source == "unknown")
                << "without a usable calibrated input the source must be unknown, "
                   "never a device measurement";
            expect(answer.empty())
                << "an unknown SM count must be empty, not 0 and not a guess";
            expect(status.sm_count == 0u);
            expect(status.sm_count_source == VulkanSmCountSource::UNKNOWN);
            if (g_options.sm_count > 0u) {
                expect(g_options.sm_count > 4096u);
                LUISA_INFO("P6.2: the implausible calibrated input {} was rejected, "
                           "so no device attribute was published",
                           g_options.sm_count);
            }
        } else {
            expect(source == "calibration_estimate")
                << "a caller-supplied estimate must be labelled as such, so no "
                   "consumer mistakes it for a device attribute";
            expect(!answer.empty());
            if (!answer.empty()) {
                expect(parse_answer(answer) == g_options.sm_count)
                    << "the reported estimate must be the caller's value, verbatim";
            }
            expect(status.sm_count == static_cast<uint32_t>(g_options.sm_count));
            expect(status.sm_count_source == VulkanSmCountSource::CALIBRATION_ESTIMATE);
            expect(std::string_view{vulkan_sm_count_source_name(status.sm_count_source)} ==
                   std::string_view{"calibration_estimate"});
        }
        // The two keys are a pair: a number without its source, or a source without
        // its number, is exactly the dishonesty this contract removes.
        expect(answer.empty() == (source == "unknown"))
            << "sm_count and sm_count_source must agree with each other";
    };
}

}// namespace

int main(int argc, char *argv[]) {
    take_options(argc, argv);
    DeviceConfig device_config;
    device_config.extension = luisa::make_unique<CapabilityStatusConfig>();
    auto dc = luisa::test::create_device_from_ut(argc, argv, &device_config);
    if (!dc) { return 0; }
    boost::ut::detail::cfg::parse_arg_with_fallback(
        argc, const_cast<const char **>(argv));
    auto &device = dc->device;
    if (device.backend_name() != "vk") {
        LUISA_INFO("Skipping: the capability record is a Vulkan-only surface.");
        return 0;
    }
    reg_query_status_consistency(device);
    reg_descriptor_heap_state(device);
    reg_sm_count(device);
    return 0;
}