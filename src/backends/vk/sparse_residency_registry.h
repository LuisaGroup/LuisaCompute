#pragma once

#include <array>
#include <cstdint>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace lc::vk::detail {

enum class SparseResidencyResourceKind : uint8_t {
    BUFFER,
    IMAGE
};

enum class SparseResidencyRegistryStatus : uint8_t {
    SUCCESS,
    INVALID_HANDLE,
    INVALID_REGION,
    HEAP_ALREADY_REGISTERED,
    HEAP_NOT_REGISTERED,
    RESOURCE_ALREADY_REGISTERED,
    RESOURCE_NOT_REGISTERED,
    RESOURCE_KIND_MISMATCH,
    HEAP_REUSED_IN_BATCH,
    HEAP_ALREADY_ACTIVE,
    RESOURCE_RANGE_ALREADY_MAPPED,
    UNMAP_RANGE_NOT_FULLY_MAPPED,
    HEAP_HAS_ACTIVE_MAPPINGS,
    RESOURCE_HAS_ACTIVE_MAPPINGS,
    TRANSACTION_INACTIVE
};

struct SparseResidencyRegistryResult {
    SparseResidencyRegistryStatus status{
        SparseResidencyRegistryStatus::SUCCESS};
    uint64_t resource{};
    uint64_t heap{};

    [[nodiscard]] constexpr explicit operator bool() const noexcept {
        return status == SparseResidencyRegistryStatus::SUCCESS;
    }
};

struct SparseBufferResidencyRange {
    uint64_t offset{};
    uint64_t size{};
};

// Half-open, axis-aligned texel box within one non-tail mip level.
struct SparseImageResidencyBox {
    uint32_t mip_level{};
    std::array<uint64_t, 3u> offset{};
    std::array<uint64_t, 3u> extent{};
};

struct SparseResidencyMapping {
    uint64_t heap{};
    uint64_t resource{};
    SparseResidencyResourceKind resource_kind{};
    SparseBufferResidencyRange buffer{};
    SparseImageResidencyBox image{};
};

[[nodiscard]] bool sparse_buffer_residency_ranges_overlap(
    SparseBufferResidencyRange lhs,
    SparseBufferResidencyRange rhs) noexcept;
[[nodiscard]] bool sparse_image_residency_boxes_overlap(
    SparseImageResidencyBox const &lhs,
    SparseImageResidencyBox const &rhs) noexcept;
[[nodiscard]] const char *sparse_residency_registry_status_name(
    SparseResidencyRegistryStatus status) noexcept;

// Device-wide ownership registry for sparse bindings. Transactions retain the
// registry lock through native queue submission, so validation and commit form
// one serializable operation across every stream on the Device.
class SparseResidencyRegistry {
private:
    struct State {
        std::unordered_set<uint64_t> heaps;
        std::unordered_map<uint64_t, SparseResidencyResourceKind> resources;
        std::vector<SparseResidencyMapping> mappings;
    };

    mutable std::mutex _mutex;
    State _state;

public:
    class Transaction {
        friend class SparseResidencyRegistry;

    private:
        SparseResidencyRegistry *_registry{};
        std::unique_lock<std::mutex> _lock;
        State _candidate;
        std::unordered_set<uint64_t> _mapped_heaps;
        SparseResidencyRegistryResult _result{};

        explicit Transaction(SparseResidencyRegistry &registry) noexcept;
        [[nodiscard]] SparseResidencyRegistryResult _fail(
            SparseResidencyRegistryStatus status,
            uint64_t resource,
            uint64_t heap) noexcept;
        [[nodiscard]] SparseResidencyRegistryResult _validate(
            uint64_t resource,
            SparseResidencyResourceKind resource_kind,
            uint64_t heap,
            bool is_map) noexcept;

    public:
        Transaction(Transaction const &) = delete;
        Transaction(Transaction &&) noexcept = default;
        Transaction &operator=(Transaction const &) = delete;
        Transaction &operator=(Transaction &&) noexcept = default;
        ~Transaction() noexcept = default;

        [[nodiscard]] SparseResidencyRegistryResult map_buffer(
            uint64_t resource,
            uint64_t heap,
            SparseBufferResidencyRange range) noexcept;
        [[nodiscard]] SparseResidencyRegistryResult validate_resource(
            uint64_t resource,
            SparseResidencyResourceKind resource_kind) noexcept;
        [[nodiscard]] SparseResidencyRegistryResult unmap_buffer(
            uint64_t resource,
            SparseBufferResidencyRange range) noexcept;
        [[nodiscard]] SparseResidencyRegistryResult map_image(
            uint64_t resource,
            uint64_t heap,
            SparseImageResidencyBox const &box) noexcept;
        [[nodiscard]] SparseResidencyRegistryResult unmap_image(
            uint64_t resource,
            SparseImageResidencyBox const &box) noexcept;
        [[nodiscard]] SparseResidencyRegistryResult commit() noexcept;
        [[nodiscard]] auto result() const noexcept { return _result; }
    };

    SparseResidencyRegistry() noexcept = default;
    SparseResidencyRegistry(SparseResidencyRegistry const &) = delete;
    SparseResidencyRegistry(SparseResidencyRegistry &&) = delete;
    SparseResidencyRegistry &operator=(SparseResidencyRegistry const &) = delete;
    SparseResidencyRegistry &operator=(SparseResidencyRegistry &&) = delete;
    ~SparseResidencyRegistry() noexcept = default;

    [[nodiscard]] SparseResidencyRegistryResult register_heap(
        uint64_t heap) noexcept;
    [[nodiscard]] SparseResidencyRegistryResult unregister_heap(
        uint64_t heap) noexcept;
    [[nodiscard]] SparseResidencyRegistryResult register_resource(
        uint64_t resource,
        SparseResidencyResourceKind resource_kind) noexcept;
    [[nodiscard]] SparseResidencyRegistryResult unregister_resource(
        uint64_t resource) noexcept;

    [[nodiscard]] Transaction begin_transaction() noexcept;
    [[nodiscard]] std::vector<SparseResidencyMapping>
    mapping_snapshot() const noexcept;
    [[nodiscard]] bool contains_heap(uint64_t heap) const noexcept;
    [[nodiscard]] bool contains_resource(uint64_t resource) const noexcept;
    [[nodiscard]] bool heap_is_active(uint64_t heap) const noexcept;
    [[nodiscard]] bool resource_is_active(uint64_t resource) const noexcept;
};

}// namespace lc::vk::detail
