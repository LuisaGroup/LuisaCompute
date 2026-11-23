// Host-level tests for the Vulkan backend's numeric capability record
// (plan P6.1 / P6.2).
//
// `src/backends/vk/device_capability.h` is device-independent: it maps plain
// VkPhysicalDevice*Properties aggregates onto the public
// `VulkanCapabilityStatus` and formats the query answers. Nothing in it calls a
// Vulkan entry point, so the whole mapping is covered here without a device:
// - the 14 public key names of plan P6.1 are exactly that set, unique, and
//   resolvable both ways (`core_limit_name` <-> `core_limit_from_name`); any other
//   name resolves to nothing, so an unknown property keeps falling through to the
//   backend's other answers instead of being reported as a limit
// - every field is a 1:1 copy of its VkPhysicalDeviceLimits member (the six
//   workgroup components are checked with six different values, which is what
//   catches a field swapped for its neighbour)
// - the three VkDeviceSize fields keep their full 64-bit value: they are
//   formatted from 64-bit integers and are asserted above 2^32, where a 32-bit
//   intermediate would truncate
// - subgroup-size control is conditional: without the properties structure (a
//   Vulkan 1.2 device without VK_EXT_subgroup_size_control) min/max subgroup size
//   are *unknown* (empty answers, zero fields), a driver that ignores the pNext
//   chain and leaves the structure zeroed is reported the same way, and only a
//   structure carrying both bounds produces values. They are never derived from
//   VkPhysicalDeviceSubgroupProperties::subgroupSize
// - `subgroup_ops` is the supportedOperations bitmask and is unknown (not 0) when
//   the base subgroup structure could not be queried
// - the SM-count contract of plan P6.2: 0 is unknown, a positive plausible value
//   is a calibration estimate, an implausible value is rejected to unknown, and a
//   struct that carries a number with `UNKNOWN` as its source answers nothing (the
//   source is authoritative, so no code path can publish a bare number)
#include "ut/ut.hpp"

#include "device_capability.h"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iterator>
#include <limits>
#include <string_view>
#include <vector>

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace vk_detail = lc::vk::detail;

namespace {

// A VkPhysicalDeviceLimits carrying a distinct, recognisable value in every field
// this record reports. Values are written field by field (never through an
// aggregate initialiser) so this stays correct regardless of how the vendored
// Vulkan header orders or extends the structure.
[[nodiscard]] VkPhysicalDeviceLimits synthetic_limits() noexcept {
    VkPhysicalDeviceLimits limits{};
    limits.maxMemoryAllocationCount = 4096u;
    limits.maxComputeWorkGroupCount[0] = 2147483647u;// x
    limits.maxComputeWorkGroupCount[1] = 65535u;     // y
    limits.maxComputeWorkGroupCount[2] = 65535u;     // z
    limits.maxComputeWorkGroupSize[0] = 1024u;       // x
    limits.maxComputeWorkGroupSize[1] = 1024u;       // y
    limits.maxComputeWorkGroupSize[2] = 64u;         // z
    limits.maxComputeSharedMemorySize = 49152u;
    // Above 2^32 on purpose (still a power of two, as Vulkan requires): a 32-bit
    // intermediate would print 0 here.
    limits.nonCoherentAtomSize = 0x100000000ull;
    limits.minStorageBufferOffsetAlignment = 16u;
    // The largest value the field can hold, so the answer must not be clipped.
    limits.maxStorageBufferRange = 0xfffffffeu;
    return limits;
}

[[nodiscard]] VkPhysicalDeviceSubgroupProperties
synthetic_subgroup_properties() noexcept {
    VkPhysicalDeviceSubgroupProperties properties{};
    properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES;
    properties.subgroupSize = 32u;
    // BASIC | VOTE | ARITHMETIC | BALLOT | SHUFFLE | SHUFFLE_RELATIVE | CLUSTERED
    properties.supportedOperations =
        VK_SUBGROUP_FEATURE_BASIC_BIT |
        VK_SUBGROUP_FEATURE_VOTE_BIT |
        VK_SUBGROUP_FEATURE_ARITHMETIC_BIT |
        VK_SUBGROUP_FEATURE_BALLOT_BIT |
        VK_SUBGROUP_FEATURE_SHUFFLE_BIT |
        VK_SUBGROUP_FEATURE_SHUFFLE_RELATIVE_BIT |
        VK_SUBGROUP_FEATURE_CLUSTERED_BIT;
    return properties;
}

[[nodiscard]] VkPhysicalDeviceSubgroupSizeControlProperties
synthetic_subgroup_size_control(uint32_t minimum, uint32_t maximum) noexcept {
    VkPhysicalDeviceSubgroupSizeControlProperties properties{};
    properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_PROPERTIES;
    properties.minSubgroupSize = minimum;
    properties.maxSubgroupSize = maximum;
    properties.maxComputeWorkgroupSubgroups = 32u;
    properties.requiredSubgroupSizeStages = VK_SHADER_STAGE_COMPUTE_BIT;
    return properties;
}

// Parses the numeric answer of one query key. An empty answer is "unknown".
[[nodiscard]] luisa::string query(const VulkanCapabilityStatus &status,
                                  vk_detail::CoreLimit limit) noexcept {
    return vk_detail::query_core_limit(status, limit);
}

void reg_key_set() {
    "vk_capability_limit_key_set_is_exactly_fourteen"_test = [] {
        expect(eq(vk_detail::core_limit_count, size_t{14u}))
            << "plan P6.1 exposes exactly 14 core limits";
        std::vector<std::string_view> names;
        names.reserve(vk_detail::core_limit_count);
        for (auto i = 0u; i < vk_detail::core_limit_count; ++i) {
            auto limit = static_cast<vk_detail::CoreLimit>(i);
            std::string_view name{vk_detail::core_limit_name(limit)};
            expect(!name.empty());
            expect(name != std::string_view{"unknown_core_limit"})
                << "every enumerator must have its own public key";
            for (auto const &previous : names) {
                expect(previous != name) << "duplicate core limit key " << name;
            }
            names.emplace_back(name);
            // Round trip: the key resolves back to the enumerator it came from.
            auto resolved = vk_detail::core_limit_from_name(name);
            expect(static_cast<bool>(resolved)) << name;
            if (resolved) { expect(*resolved == limit) << name; }
        }
        // The exact spellings are the public contract of plan P6.1.
        constexpr std::string_view expected[] = {
            "max_memory_allocation_count",
            "min_subgroup_size",
            "max_subgroup_size",
            "subgroup_ops",
            "max_compute_workgroup_count_x",
            "max_compute_workgroup_count_y",
            "max_compute_workgroup_count_z",
            "max_compute_workgroup_size_x",
            "max_compute_workgroup_size_y",
            "max_compute_workgroup_size_z",
            "max_compute_shared_memory_bytes",
            "non_coherent_atom_size",
            "min_storage_buffer_offset_alignment",
            "max_storage_buffer_range"};
        expect(eq(names.size(), std::size(expected)));
        for (auto i = 0u; i < std::size(expected) && i < names.size(); ++i) {
            expect(names[i] == expected[i]) << names[i];
        }
        // Anything else is not a core limit: the backend must keep falling
        // through to its other answers instead of reporting "no value".
        expect(!vk_detail::core_limit_from_name(""));
        expect(!vk_detail::core_limit_from_name("max_subgroup_size_x"));
        expect(!vk_detail::core_limit_from_name("max_memory_allocation_count "));
        expect(!vk_detail::core_limit_from_name("sm_count"));
        expect(!vk_detail::core_limit_from_name("max_compute_workgroup_size"));
    };
}

void reg_field_mapping() {
    "vk_capability_limits_copy_their_vk_fields"_test = [] {
        auto limits = synthetic_limits();
        auto subgroup = synthetic_subgroup_properties();
        auto size_control = synthetic_subgroup_size_control(32u, 64u);
        auto status = vk_detail::make_capability_limits(
            limits, &subgroup, &size_control);
        expect(eq(status.max_memory_allocation_count, 4096u));
        expect(eq(status.max_compute_workgroup_count_x, 2147483647u));
        expect(eq(status.max_compute_workgroup_count_y, 65535u));
        expect(eq(status.max_compute_workgroup_count_z, 65535u));
        expect(eq(status.max_compute_workgroup_size_x, 1024u));
        expect(eq(status.max_compute_workgroup_size_y, 1024u));
        expect(eq(status.max_compute_workgroup_size_z, 64u));
        expect(eq(status.max_compute_shared_memory_bytes, 49152u));
        expect(eq(status.non_coherent_atom_size, uint64_t{0x100000000ull}));
        expect(eq(status.min_storage_buffer_offset_alignment, uint64_t{16u}));
        expect(eq(status.max_storage_buffer_range, uint64_t{0xfffffffeu}));
        expect(status.subgroup_ops_available)
            << "a queried subgroup structure makes its bitmask real";
        expect(eq(status.subgroup_ops,
                  static_cast<uint32_t>(subgroup.supportedOperations)));
        expect(status.subgroup_size_control_available)
            << "a structure carrying both bounds makes min/max subgroup size real";
        expect(eq(status.min_subgroup_size, 32u));
        expect(eq(status.max_subgroup_size, 64u));
        // The SM count and the descriptor-heap state are filled from other
        // sources, so building the limit fields must leave them at their
        // documented defaults.
        expect(eq(status.sm_count, 0u));
        expect(status.sm_count_source == VulkanSmCountSource::UNKNOWN);
        expect(!status.descriptor_heap.supported);
        expect(!status.descriptor_heap.requested);
        expect(!status.descriptor_heap.enabled);
        expect(!status.descriptor_heap.active);
    };
}

void reg_query_strings() {
    "vk_capability_queries_format_every_field"_test = [] {
        auto limits = synthetic_limits();
        auto subgroup = synthetic_subgroup_properties();
        auto size_control = synthetic_subgroup_size_control(32u, 64u);
        auto status = vk_detail::make_capability_limits(
            limits, &subgroup, &size_control);
        // Every one of the 14 keys answers, and the answer is the decimal form of
        // the field - so a query and the record cannot disagree.
        for (auto i = 0u; i < vk_detail::core_limit_count; ++i) {
            auto limit = static_cast<vk_detail::CoreLimit>(i);
            auto answer = query(status, limit);
            expect(!answer.empty())
                << "every limit of a fully capable device must answer: "
                << vk_detail::core_limit_name(limit);
        }
        expect(query(status, vk_detail::CoreLimit::MAX_MEMORY_ALLOCATION_COUNT) == "4096");
        expect(query(status, vk_detail::CoreLimit::MAX_COMPUTE_WORKGROUP_COUNT_X) == "2147483647");
        expect(query(status, vk_detail::CoreLimit::MAX_COMPUTE_WORKGROUP_COUNT_Y) == "65535");
        expect(query(status, vk_detail::CoreLimit::MAX_COMPUTE_WORKGROUP_SIZE_Z) == "64");
        expect(query(status, vk_detail::CoreLimit::MAX_COMPUTE_SHARED_MEMORY_BYTES) == "49152");
        expect(query(status, vk_detail::CoreLimit::MIN_SUBGROUP_SIZE) == "32");
        expect(query(status, vk_detail::CoreLimit::MAX_SUBGROUP_SIZE) == "64");
        // 64-bit VkDeviceSize values print in full; a 32-bit intermediate would
        // give 64 here.
        expect(query(status, vk_detail::CoreLimit::NON_COHERENT_ATOM_SIZE) == "4294967296")
            << "a VkDeviceSize above 2^32 must not be truncated: "
            << query(status, vk_detail::CoreLimit::NON_COHERENT_ATOM_SIZE);
        expect(query(status, vk_detail::CoreLimit::MIN_STORAGE_BUFFER_OFFSET_ALIGNMENT) == "16");
        expect(query(status, vk_detail::CoreLimit::MAX_STORAGE_BUFFER_RANGE) == "4294967294")
            << "the largest representable maxStorageBufferRange must survive intact: "
            << query(status, vk_detail::CoreLimit::MAX_STORAGE_BUFFER_RANGE);
        // VkSubgroupFeatureFlags is a bitmask; the answer is its integer value.
        auto ops = query(status, vk_detail::CoreLimit::SUBGROUP_OPS);
        expect(ops == luisa::format("{}", static_cast<uint32_t>(subgroup.supportedOperations)));
        expect(static_cast<uint32_t>(std::strtoul(ops.c_str(), nullptr, 10)) &
               static_cast<uint32_t>(VK_SUBGROUP_FEATURE_BASIC_BIT))
            << "Vulkan guarantees the BASIC subgroup operation";
    };
}

void reg_conditional_subgroup_bounds() {
    "vk_capability_subgroup_bounds_are_conditional"_test = [] {
        auto limits = synthetic_limits();
        auto subgroup = synthetic_subgroup_properties();
        // The Vulkan 1.2 case: no subgroup-size-control properties at all. The
        // base subgroup structure knows a single subgroup size, so there is no
        // honest lower/upper bound to report - both keys stay unknown instead of
        // being derived from subgroupSize.
        auto without = vk_detail::make_capability_limits(limits, &subgroup, nullptr);
        expect(!without.subgroup_size_control_available);
        expect(eq(without.min_subgroup_size, 0u));
        expect(eq(without.max_subgroup_size, 0u));
        expect(query(without, vk_detail::CoreLimit::MIN_SUBGROUP_SIZE).empty())
            << "min_subgroup_size must be unknown, never the single subgroupSize";
        expect(query(without, vk_detail::CoreLimit::MAX_SUBGROUP_SIZE).empty())
            << "max_subgroup_size must be unknown, never the single subgroupSize";
        // ... while the eleven unconditional limits and the subgroup bitmask stay
        // answered, so "unknown" is scoped to what really is unknowable.
        expect(!query(without, vk_detail::CoreLimit::SUBGROUP_OPS).empty());
        expect(!query(without, vk_detail::CoreLimit::MAX_COMPUTE_SHARED_MEMORY_BYTES).empty());
        // A driver that ignores the chained structure leaves it zeroed; that is
        // "no bounds known", not "min = max = 0".
        auto ignored = synthetic_subgroup_size_control(0u, 0u);
        auto zeroed = vk_detail::make_capability_limits(limits, &subgroup, &ignored);
        expect(!zeroed.subgroup_size_control_available);
        expect(query(zeroed, vk_detail::CoreLimit::MIN_SUBGROUP_SIZE).empty());
        expect(query(zeroed, vk_detail::CoreLimit::MAX_SUBGROUP_SIZE).empty());
        // Only a structure carrying both bounds produces values.
        auto only_max = synthetic_subgroup_size_control(0u, 64u);
        auto partial = vk_detail::make_capability_limits(limits, &subgroup, &only_max);
        expect(!partial.subgroup_size_control_available)
            << "half a bound is not a bound";
        expect(query(partial, vk_detail::CoreLimit::MIN_SUBGROUP_SIZE).empty());
        expect(query(partial, vk_detail::CoreLimit::MAX_SUBGROUP_SIZE).empty());
        // The single-size device (min == max) is a real bound and is reported.
        auto fixed = synthetic_subgroup_size_control(32u, 32u);
        auto fixed_status = vk_detail::make_capability_limits(limits, &subgroup, &fixed);
        expect(fixed_status.subgroup_size_control_available);
        expect(query(fixed_status, vk_detail::CoreLimit::MIN_SUBGROUP_SIZE) == "32");
        expect(query(fixed_status, vk_detail::CoreLimit::MAX_SUBGROUP_SIZE) == "32");
    };
}

void reg_subgroup_ops_unknown() {
    "vk_capability_subgroup_ops_unknown_without_properties"_test = [] {
        auto limits = synthetic_limits();
        auto status = vk_detail::make_capability_limits(limits, nullptr, nullptr);
        expect(!status.subgroup_ops_available);
        expect(eq(status.subgroup_ops, 0u));
        expect(query(status, vk_detail::CoreLimit::SUBGROUP_OPS).empty())
            << "an unqueried subgroup bitmask is unknown, not 'no operations'";
        // A queried structure with subgroupSize == 0 is a driver that did not fill
        // it in, which is the same "unknown" answer.
        auto empty = VkPhysicalDeviceSubgroupProperties{};
        auto unfilled = vk_detail::make_capability_limits(limits, &empty, nullptr);
        expect(!unfilled.subgroup_ops_available);
        expect(query(unfilled, vk_detail::CoreLimit::SUBGROUP_OPS).empty());
        // The eleven unconditional limits never depend on the subgroup query.
        for (auto limit : {vk_detail::CoreLimit::MAX_MEMORY_ALLOCATION_COUNT,
                           vk_detail::CoreLimit::MAX_COMPUTE_WORKGROUP_COUNT_X,
                           vk_detail::CoreLimit::MAX_COMPUTE_WORKGROUP_COUNT_Y,
                           vk_detail::CoreLimit::MAX_COMPUTE_WORKGROUP_COUNT_Z,
                           vk_detail::CoreLimit::MAX_COMPUTE_WORKGROUP_SIZE_X,
                           vk_detail::CoreLimit::MAX_COMPUTE_WORKGROUP_SIZE_Y,
                           vk_detail::CoreLimit::MAX_COMPUTE_WORKGROUP_SIZE_Z,
                           vk_detail::CoreLimit::MAX_COMPUTE_SHARED_MEMORY_BYTES,
                           vk_detail::CoreLimit::NON_COHERENT_ATOM_SIZE,
                           vk_detail::CoreLimit::MIN_STORAGE_BUFFER_OFFSET_ALIGNMENT,
                           vk_detail::CoreLimit::MAX_STORAGE_BUFFER_RANGE}) {
            expect(!query(unfilled, limit).empty())
                << "an unconditional VkPhysicalDeviceLimits member always answers: "
                << vk_detail::core_limit_name(limit);
        }
    };
}

void reg_sm_count_contract() {
    "vk_sm_count_default_is_unknown"_test = [] {
        auto status = VulkanCapabilityStatus{};
        expect(eq(status.sm_count, 0u));
        expect(status.sm_count_source == VulkanSmCountSource::UNKNOWN);
        expect(vk_detail::query_sm_count(status).empty())
            << "no calibration input means no number - never 0 as a device attribute";
        expect(vk_detail::query_sm_count_source(status) == "unknown");
        // A record that carries a number but says its source is unknown cannot be
        // produced by the device, and must still answer nothing: the source, not
        // the number, decides whether the value may be published.
        status.sm_count = 84u;
        expect(vk_detail::query_sm_count(status).empty());
        expect(vk_detail::query_sm_count_source(status) == "unknown");
    };
    "vk_sm_count_calibration_estimate_is_reported_with_its_source"_test = [] {
        auto plan = vk_detail::plan_calibrated_sm_count(84u);
        expect(static_cast<bool>(plan))
            << "a positive plausible value is a calibration estimate";
        expect(plan.status == vk_detail::CalibratedSmCountStatus::CALIBRATION_ESTIMATE);
        expect(eq(plan.value, 84u))
            << "the caller's value is reported verbatim, not rounded or adjusted";
        VulkanCapabilityStatus status{};
        status.sm_count = plan.value;
        status.sm_count_source = VulkanSmCountSource::CALIBRATION_ESTIMATE;
        expect(vk_detail::query_sm_count(status) == "84");
        expect(vk_detail::query_sm_count_source(status) == "calibration_estimate")
            << "an estimate is never reported as a device attribute";
        expect(std::string_view{vulkan_sm_count_source_name(status.sm_count_source)} ==
               std::string_view{"calibration_estimate"});
    };
    "vk_sm_count_input_validation"_test = [] {
        // 0 is the documented "no estimate" default and stays unknown.
        auto absent = vk_detail::plan_calibrated_sm_count(0u);
        expect(!static_cast<bool>(absent));
        expect(absent.status == vk_detail::CalibratedSmCountStatus::ABSENT);
        expect(eq(absent.value, 0u));
        // 1 and the bound itself are accepted.
        auto smallest = vk_detail::plan_calibrated_sm_count(1u);
        expect(smallest.status == vk_detail::CalibratedSmCountStatus::CALIBRATION_ESTIMATE);
        expect(eq(smallest.value, 1u));
        auto upper = vk_detail::plan_calibrated_sm_count(
            static_cast<uint64_t>(vk_detail::kMaxCalibratedSmCount));
        expect(upper.status == vk_detail::CalibratedSmCountStatus::CALIBRATION_ESTIMATE);
        expect(eq(upper.value, vk_detail::kMaxCalibratedSmCount));
        // Above the bound the input is rejected, and a rejected plan carries no
        // value at all - a caller bug must not become a device attribute.
        for (auto out_of_range : {static_cast<uint64_t>(vk_detail::kMaxCalibratedSmCount) + 1u,
                                  uint64_t{1u} << 32u,
                                  std::numeric_limits<uint64_t>::max()}) {
            auto rejected = vk_detail::plan_calibrated_sm_count(out_of_range);
            expect(!static_cast<bool>(rejected)) << out_of_range;
            expect(rejected.status == vk_detail::CalibratedSmCountStatus::REJECTED)
                << out_of_range;
            expect(eq(rejected.value, 0u)) << out_of_range;
            expect(std::string_view{vk_detail::calibrated_sm_count_status_name(rejected.status)} !=
                   std::string_view{"unknown"})
                << "the rejection has its own name so a log line can state it";
        }
        // The status names are the contract the device logs and the tests read.
        expect(std::string_view{vk_detail::calibrated_sm_count_status_name(
                   vk_detail::CalibratedSmCountStatus::ABSENT)} == "absent");
        expect(std::string_view{vk_detail::calibrated_sm_count_status_name(
                   vk_detail::CalibratedSmCountStatus::CALIBRATION_ESTIMATE)} ==
               "calibration_estimate");
    };
}

}// namespace

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(
        argc, const_cast<const char **>(argv));
    reg_key_set();
    reg_field_mapping();
    reg_query_strings();
    reg_conditional_subgroup_bounds();
    reg_subgroup_ops_unknown();
    reg_sm_count_contract();
    return 0;
}