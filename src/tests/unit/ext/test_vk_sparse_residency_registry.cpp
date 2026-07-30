// Pure tests for Device-wide Vulkan sparse residency ownership planning.

#include "ut/ut.hpp"

#include "sparse_residency_registry.h"

#include <algorithm>
#include <cstdint>

using namespace boost::ut;
using namespace boost::ut::literals;
using namespace lc::vk::detail;

namespace {

constexpr uint64_t heap_a = 0x1000u;
constexpr uint64_t heap_b = 0x2000u;
constexpr uint64_t buffer_a = 0x3000u;
constexpr uint64_t buffer_b = 0x4000u;
constexpr uint64_t image_a = 0x5000u;

void register_heap(
    SparseResidencyRegistry &registry, uint64_t heap) {
    expect(static_cast<bool>(registry.register_heap(heap)));
}

void register_resource(
    SparseResidencyRegistry &registry,
    uint64_t resource,
    SparseResidencyResourceKind kind) {
    expect(static_cast<bool>(
        registry.register_resource(resource, kind)));
}

[[nodiscard]] uint64_t volume(
    SparseImageResidencyBox const &box) noexcept {
    return box.extent[0u] * box.extent[1u] * box.extent[2u];
}

}// namespace

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(
        argc, const_cast<const char **>(argv));

    "vk_sparse_registry_preserves_buffer_fragments_after_partial_unmap"_test = [] {
        SparseResidencyRegistry registry;
        register_heap(registry, heap_a);
        register_resource(
            registry, buffer_a,
            SparseResidencyResourceKind::BUFFER);
        {
            auto transaction = registry.begin_transaction();
            expect(static_cast<bool>(transaction.map_buffer(
                buffer_a, heap_a, {.offset = 0u, .size = 100u})));
            expect(static_cast<bool>(transaction.commit()));
        }
        {
            auto transaction = registry.begin_transaction();
            expect(static_cast<bool>(transaction.unmap_buffer(
                buffer_a, {.offset = 30u, .size = 40u})));
            expect(static_cast<bool>(transaction.commit()));
        }
        auto mappings = registry.mapping_snapshot();
        std::sort(
            mappings.begin(), mappings.end(),
            [](auto const &lhs, auto const &rhs) noexcept {
                return lhs.buffer.offset < rhs.buffer.offset;
            });
        expect(eq(mappings.size(), size_t{2u}));
        expect(mappings[0u].buffer.offset == 0u &&
               mappings[0u].buffer.size == 30u);
        expect(mappings[1u].buffer.offset == 70u &&
               mappings[1u].buffer.size == 30u);
        expect(registry.heap_is_active(heap_a));
        expect(registry.unregister_heap(heap_a).status ==
               SparseResidencyRegistryStatus::HEAP_HAS_ACTIVE_MAPPINGS);

        {
            auto transaction = registry.begin_transaction();
            expect(static_cast<bool>(transaction.unmap_buffer(
                buffer_a, {.offset = 0u, .size = 30u})));
            expect(static_cast<bool>(transaction.unmap_buffer(
                buffer_a, {.offset = 70u, .size = 30u})));
            expect(static_cast<bool>(transaction.commit()));
        }
        expect(!registry.heap_is_active(heap_a));
        expect(static_cast<bool>(registry.unregister_heap(heap_a)));
        expect(static_cast<bool>(
            registry.unregister_resource(buffer_a)));
    };

    "vk_sparse_registry_requires_explicit_unmap_before_rebind"_test = [] {
        SparseResidencyRegistry registry;
        register_heap(registry, heap_a);
        register_heap(registry, heap_b);
        register_resource(
            registry, buffer_a,
            SparseResidencyResourceKind::BUFFER);
        {
            auto transaction = registry.begin_transaction();
            expect(static_cast<bool>(transaction.map_buffer(
                buffer_a, heap_a, {.offset = 0u, .size = 64u})));
            expect(static_cast<bool>(transaction.commit()));
        }
        {
            auto transaction = registry.begin_transaction();
            auto overlap = transaction.map_buffer(
                buffer_a, heap_b, {.offset = 32u, .size = 32u});
            expect(overlap.status ==
                   SparseResidencyRegistryStatus::RESOURCE_RANGE_ALREADY_MAPPED);
        }
        auto unchanged = registry.mapping_snapshot();
        expect(eq(unchanged.size(), size_t{1u}));
        expect(unchanged.front().heap == heap_a &&
               unchanged.front().buffer.offset == 0u &&
               unchanged.front().buffer.size == 64u);

        {
            auto transaction = registry.begin_transaction();
            expect(static_cast<bool>(transaction.unmap_buffer(
                buffer_a, {.offset = 0u, .size = 64u})));
            expect(static_cast<bool>(transaction.commit()));
        }
        {
            auto transaction = registry.begin_transaction();
            expect(static_cast<bool>(transaction.map_buffer(
                buffer_a, heap_b, {.offset = 0u, .size = 64u})));
            expect(static_cast<bool>(transaction.map_buffer(
                buffer_a, heap_a, {.offset = 64u, .size = 64u})));
            expect(static_cast<bool>(transaction.commit()));
        }
        {
            auto transaction = registry.begin_transaction();
            expect(static_cast<bool>(transaction.unmap_buffer(
                buffer_a, {.offset = 0u, .size = 128u})))
                << "adjacent mappings from distinct heaps jointly cover the unmap";
            expect(static_cast<bool>(transaction.commit()));
        }
        expect(!registry.heap_is_active(heap_a));
        expect(!registry.heap_is_active(heap_b));
    };

    "vk_sparse_registry_rejects_live_and_same_batch_heap_reuse"_test = [] {
        SparseResidencyRegistry registry;
        register_heap(registry, heap_a);
        register_heap(registry, heap_b);
        register_resource(
            registry, buffer_a,
            SparseResidencyResourceKind::BUFFER);
        register_resource(
            registry, buffer_b,
            SparseResidencyResourceKind::BUFFER);
        {
            auto transaction = registry.begin_transaction();
            expect(static_cast<bool>(transaction.map_buffer(
                buffer_a, heap_a, {.offset = 0u, .size = 64u})));
            expect(static_cast<bool>(transaction.commit()));
        }
        {
            auto transaction = registry.begin_transaction();
            auto conflict = transaction.map_buffer(
                buffer_b, heap_a, {.offset = 0u, .size = 64u});
            expect(conflict.status ==
                   SparseResidencyRegistryStatus::HEAP_ALREADY_ACTIVE);
        }
        {
            auto transaction = registry.begin_transaction();
            expect(static_cast<bool>(transaction.unmap_buffer(
                buffer_a, {.offset = 0u, .size = 64u})));
            auto same_batch_recycle = transaction.map_buffer(
                buffer_b, heap_a, {.offset = 0u, .size = 64u});
            expect(same_batch_recycle.status ==
                   SparseResidencyRegistryStatus::HEAP_ALREADY_ACTIVE)
                << "a VkBindSparseInfo cannot recycle a heap released by "
                   "another bind in the same unordered batch";
        }
        {
            auto transaction = registry.begin_transaction();
            expect(static_cast<bool>(transaction.map_buffer(
                buffer_b, heap_b, {.offset = 0u, .size = 64u})));
            auto repeated = transaction.map_buffer(
                buffer_a, heap_b, {.offset = 64u, .size = 64u});
            expect(repeated.status ==
                   SparseResidencyRegistryStatus::HEAP_REUSED_IN_BATCH);
        }
        expect(!registry.resource_is_active(buffer_b))
            << "a failed transaction must not publish earlier operations";
    };

    "vk_sparse_registry_subtracts_orthogonal_image_boxes"_test = [] {
        SparseResidencyRegistry registry;
        register_heap(registry, heap_a);
        register_resource(
            registry, image_a,
            SparseResidencyResourceKind::IMAGE);
        auto whole = SparseImageResidencyBox{
            .mip_level = 2u,
            .offset = {0u, 0u, 0u},
            .extent = {4u, 4u, 4u}};
        auto center = SparseImageResidencyBox{
            .mip_level = 2u,
            .offset = {1u, 1u, 1u},
            .extent = {2u, 2u, 2u}};
        {
            auto transaction = registry.begin_transaction();
            expect(static_cast<bool>(transaction.map_image(
                image_a, heap_a, whole)));
            expect(static_cast<bool>(transaction.commit()));
        }
        {
            auto transaction = registry.begin_transaction();
            expect(static_cast<bool>(
                transaction.unmap_image(image_a, center)));
            expect(static_cast<bool>(transaction.commit()));
        }
        auto mappings = registry.mapping_snapshot();
        expect(eq(mappings.size(), size_t{6u}));
        auto remaining_volume = uint64_t{0u};
        for (auto const &mapping : mappings) {
            remaining_volume += volume(mapping.image);
            expect(!sparse_image_residency_boxes_overlap(
                mapping.image, center));
        }
        expect(remaining_volume == 56u)
            << "a 2x2x2 hole removes eight texels from a 4x4x4 box";
        for (auto i = 0u; i < mappings.size(); ++i) {
            for (auto j = 0u; j < i; ++j) {
                expect(!sparse_image_residency_boxes_overlap(
                    mappings[i].image, mappings[j].image));
            }
        }
    };

    "vk_sparse_registry_rejects_unmapped_or_holed_unmaps"_test = [] {
        SparseResidencyRegistry registry;
        register_heap(registry, heap_a);
        register_resource(
            registry, image_a,
            SparseResidencyResourceKind::IMAGE);
        auto whole = SparseImageResidencyBox{
            .mip_level = 0u,
            .offset = {0u, 0u, 0u},
            .extent = {4u, 4u, 1u}};
        auto hole = SparseImageResidencyBox{
            .mip_level = 0u,
            .offset = {1u, 1u, 0u},
            .extent = {2u, 2u, 1u}};
        {
            auto transaction = registry.begin_transaction();
            expect(transaction.unmap_image(image_a, whole).status ==
                   SparseResidencyRegistryStatus::UNMAP_RANGE_NOT_FULLY_MAPPED);
        }
        {
            auto transaction = registry.begin_transaction();
            expect(static_cast<bool>(transaction.map_image(
                image_a, heap_a, whole)));
            expect(static_cast<bool>(transaction.commit()));
        }
        {
            auto transaction = registry.begin_transaction();
            expect(static_cast<bool>(
                transaction.unmap_image(image_a, hole)));
            expect(static_cast<bool>(transaction.commit()));
        }
        {
            auto transaction = registry.begin_transaction();
            auto crosses_hole =
                transaction.unmap_image(image_a, whole);
            expect(crosses_hole.status ==
                   SparseResidencyRegistryStatus::UNMAP_RANGE_NOT_FULLY_MAPPED);
        }
        expect(eq(registry.mapping_snapshot().size(), size_t{4u}));
    };

    "vk_sparse_registry_guards_heap_and_resource_destruction"_test = [] {
        SparseResidencyRegistry registry;
        register_heap(registry, heap_a);
        register_resource(
            registry, buffer_a,
            SparseResidencyResourceKind::BUFFER);
        {
            auto transaction = registry.begin_transaction();
            expect(static_cast<bool>(transaction.map_buffer(
                buffer_a, heap_a, {.offset = 0u, .size = 64u})));
            expect(static_cast<bool>(transaction.commit()));
        }
        expect(registry.unregister_heap(heap_a).status ==
               SparseResidencyRegistryStatus::HEAP_HAS_ACTIVE_MAPPINGS);
        expect(registry.unregister_resource(buffer_a).status ==
               SparseResidencyRegistryStatus::RESOURCE_HAS_ACTIVE_MAPPINGS);
        expect(registry.contains_heap(heap_a));
        expect(registry.contains_resource(buffer_a));
        {
            auto transaction = registry.begin_transaction();
            expect(static_cast<bool>(transaction.unmap_buffer(
                buffer_a, {.offset = 0u, .size = 64u})));
            expect(static_cast<bool>(transaction.commit()));
        }
        expect(static_cast<bool>(registry.unregister_heap(heap_a)));
        expect(static_cast<bool>(
            registry.unregister_resource(buffer_a)));
        expect(!registry.contains_heap(heap_a));
        expect(!registry.contains_resource(buffer_a));
        {
            auto transaction = registry.begin_transaction();
            expect(transaction.map_buffer(
                                  buffer_a, heap_a,
                                  {.offset = 0u, .size = 64u})
                       .status ==
                   SparseResidencyRegistryStatus::RESOURCE_NOT_REGISTERED);
        }
    };

    "vk_sparse_registry_validates_opaque_handles_before_dereference"_test = [] {
        SparseResidencyRegistry registry;
        register_resource(
            registry, buffer_a,
            SparseResidencyResourceKind::BUFFER);
        auto transaction = registry.begin_transaction();
        expect(static_cast<bool>(transaction.validate_resource(
            buffer_a, SparseResidencyResourceKind::BUFFER)));
        expect(transaction.validate_resource(
                              buffer_a,
                              SparseResidencyResourceKind::IMAGE)
                   .status ==
               SparseResidencyRegistryStatus::RESOURCE_KIND_MISMATCH);

        SparseResidencyRegistry missing_registry;
        auto missing_transaction =
            missing_registry.begin_transaction();
        expect(missing_transaction.validate_resource(
                                      buffer_a,
                                      SparseResidencyResourceKind::BUFFER)
                   .status ==
               SparseResidencyRegistryStatus::RESOURCE_NOT_REGISTERED);
    };

    return 0;
}
