// Test for FirstFit memory allocator
// This test covers:
// - Basic construction with size and alignment
// - allocate() - First-fit allocation strategy
// - allocate_best_fit() - Best-fit allocation strategy
// - free() - Deallocation and memory reuse
// - Move semantics (move constructor and move assignment)
// - dump_free_list() - Debugging free list state
// - Fragmentation scenarios
// - Edge cases (exact fit, alignment constraints)

#include <vector>
#include <algorithm>

#include <luisa/core/first_fit.h>
#include <luisa/core/logging.h>
#include "ut/ut.hpp"

using namespace luisa;
using namespace boost::ut;
using namespace boost::ut::literals;

// Test basic construction and simple allocation
void test_basic_construction() {
    LUISA_INFO("Testing basic construction...");

    // Create a FirstFit allocator with 1024 bytes and 8-byte alignment
    FirstFit allocator(1024, 8);

    expect(static_cast<bool>(allocator.size() == 1024));
    expect(static_cast<bool>(allocator.alignment() == 8));

    LUISA_INFO("Basic construction test passed.");
}

// Test basic allocate and free
void test_basic_allocate_free() {
    LUISA_INFO("Testing basic allocate/free...");

    FirstFit allocator(1024, 8);

    // Allocate some blocks
    auto *node1 = allocator.allocate(100);
    expect(static_cast<bool>(node1 != nullptr));
    expect(static_cast<bool>(node1->offset() == 0));
    expect(static_cast<bool>(node1->size() >= 100));

    auto *node2 = allocator.allocate(200);
    expect(static_cast<bool>(node2 != nullptr));
    expect(static_cast<bool>(node2->offset() >= node1->offset() + node1->size()));

    auto *node3 = allocator.allocate(300);
    expect(static_cast<bool>(node3 != nullptr));

    LUISA_INFO("Allocated blocks at offsets: {}, {}, {}", node1->offset(), node2->offset(), node3->offset());

    // Free middle block and reallocate
    size_t old_offset = node2->offset();
    allocator.free(node2);

    // Allocate smaller block - should fit in freed space (first-fit)
    auto *node4 = allocator.allocate(150);
    expect(static_cast<bool>(node4 != nullptr));
    // Should reuse the freed space (first-fit)
    expect(static_cast<bool>(node4->offset() == old_offset));

    // Cleanup
    allocator.free(node1);
    allocator.free(node3);
    allocator.free(node4);

    LUISA_INFO("Basic allocate/free test passed.");
}

// Test allocation with different alignments
void test_alignment() {
    LUISA_INFO("Testing alignment constraints...");

    // Test with 16-byte alignment
    FirstFit allocator(1024, 16);
    expect(static_cast<bool>(allocator.alignment() == 16));

    auto *node1 = allocator.allocate(50);
    expect(static_cast<bool>(node1 != nullptr));
    expect(static_cast<bool>(node1->offset() % 16 == 0));

    auto *node2 = allocator.allocate(60);
    expect(static_cast<bool>(node2 != nullptr));
    expect(static_cast<bool>(node2->offset() % 16 == 0));

    allocator.free(node1);
    allocator.free(node2);

    // Test with 64-byte alignment (cache line)
    FirstFit allocator64(4096, 64);
    auto *node3 = allocator64.allocate(100);
    expect(static_cast<bool>(node3 != nullptr));
    expect(static_cast<bool>(node3->offset() % 64 == 0));

    allocator64.free(node3);

    LUISA_INFO("Alignment test passed.");
}

// Test first-fit allocation strategy
void test_first_fit_strategy() {
    LUISA_INFO("Testing first-fit allocation strategy...");

    FirstFit allocator(1024, 8);

    // Create fragmentation pattern: alloc A, B, C, then free A and B
    auto *nodeA = allocator.allocate(100);
    auto *nodeB = allocator.allocate(100);
    auto *nodeC = allocator.allocate(100);

    size_t offsetA = nodeA->offset();
    allocator.free(nodeA);
    allocator.free(nodeB);

    // Now we have a hole at offset 0 of size ~200
    // Allocate a block of 50 - should use first-fit (offset 0)
    auto *nodeD = allocator.allocate(50);
    expect(static_cast<bool>(nodeD != nullptr));
    expect(static_cast<bool>(nodeD->offset() == offsetA));

    allocator.free(nodeC);
    allocator.free(nodeD);

    LUISA_INFO("First-fit strategy test passed.");
}

// Test best-fit allocation strategy
void test_best_fit_strategy() {
    LUISA_INFO("Testing best-fit allocation strategy...");

    FirstFit allocator(1024, 8);

    // Create specific fragmentation: allocate and free to create holes
    auto *node1 = allocator.allocate(200);
    auto *node2 = allocator.allocate(300);
    auto *node3 = allocator.allocate(200);

    allocator.free(node1);
    allocator.free(node3);

    // Both holes are size 200. Allocate 150 using best-fit
    auto *node4 = allocator.allocate_best_fit(150);
    expect(static_cast<bool>(node4 != nullptr));
    expect(static_cast<bool>(node4->size() >= 150));

    allocator.free(node2);
    allocator.free(node4);

    LUISA_INFO("Best-fit strategy test passed.");
}

// Test memory exhaustion
void test_memory_exhaustion() {
    LUISA_INFO("Testing memory exhaustion...");

    FirstFit allocator(256, 8);

    // Allocate most of the memory
    auto *node1 = allocator.allocate(200);
    expect(static_cast<bool>(node1 != nullptr));

    // Try to allocate more than available
    auto *node2 = allocator.allocate(100);
    // Note: May or may not succeed depending on internal fragmentation
    // Just verify we don't crash

    if (node2) {
        allocator.free(node2);
    }
    allocator.free(node1);

    LUISA_INFO("Memory exhaustion test passed.");
}

// Test move semantics
void test_move_semantics() {
    LUISA_INFO("Testing move semantics...");

    // Move constructor
    {
        FirstFit allocator1(1024, 8);
        auto *node = allocator1.allocate(100);
        expect(static_cast<bool>(node != nullptr));
        size_t offset = node->offset();

        // Move construct
        FirstFit allocator2(std::move(allocator1));

        // Verify allocator2 has the allocation
        expect(static_cast<bool>(allocator2.size() == 1024));

        // Free from moved allocator
        allocator2.free(node);

        // Verify we can allocate from moved allocator
        auto *node2 = allocator2.allocate(200);
        expect(static_cast<bool>(node2 != nullptr));
        allocator2.free(node2);
    }

    // Move assignment
    {
        FirstFit allocator1(1024, 8);
        auto *node = allocator1.allocate(100);
        expect(static_cast<bool>(node != nullptr));

        FirstFit allocator2(512, 16);
        allocator2 = std::move(allocator1);

        expect(static_cast<bool>(allocator2.size() == 1024));
        expect(static_cast<bool>(allocator2.alignment() == 8));

        allocator2.free(node);
    }

    LUISA_INFO("Move semantics test passed.");
}

// Test dump_free_list for debugging
void test_dump_free_list() {
    LUISA_INFO("Testing dump_free_list...");

    FirstFit allocator(1024, 8);

    // Get initial free list dump
    auto free_list_str = allocator.dump_free_list();
    expect(static_cast<bool>(!free_list_str.empty()));
    LUISA_INFO("Initial free list: {}", free_list_str);

    // Allocate and free to create fragmentation
    auto *node1 = allocator.allocate(100);
    auto *node2 = allocator.allocate(100);
    auto *node3 = allocator.allocate(100);

    allocator.free(node2);// Create a hole

    free_list_str = allocator.dump_free_list();
    LUISA_INFO("Free list after fragmentation: {}", free_list_str);

    allocator.free(node1);
    allocator.free(node3);

    free_list_str = allocator.dump_free_list();
    LUISA_INFO("Free list after all freed: {}", free_list_str);

    LUISA_INFO("Dump free list test passed.");
}

// Test fragmentation and coalescing
void test_fragmentation_and_coalescing() {
    LUISA_INFO("Testing fragmentation and coalescing...");

    FirstFit allocator(1024, 8);

    // Create multiple allocations
    auto *node1 = allocator.allocate(100);
    auto *node2 = allocator.allocate(100);
    auto *node3 = allocator.allocate(100);
    auto *node4 = allocator.allocate(100);

    size_t offset1 = node1->offset();
    size_t offset2 = node2->offset();
    size_t offset3 = node3->offset();
    size_t offset4 = node4->offset();

    // Free in reverse order to test coalescing
    allocator.free(node4);
    allocator.free(node3);
    allocator.free(node2);
    allocator.free(node1);

    // After freeing all, we should be able to allocate the full size again
    auto *big_node = allocator.allocate(400);
    expect(static_cast<bool>(big_node != nullptr));
    expect(static_cast<bool>(big_node->offset() == offset1));

    allocator.free(big_node);

    LUISA_INFO("Fragmentation and coalescing test passed.");
}

// Test exact fit allocation
void test_exact_fit() {
    LUISA_INFO("Testing exact fit allocation...");

    FirstFit allocator(1024, 8);

    // Allocate a specific size
    auto *node1 = allocator.allocate(256);
    expect(static_cast<bool>(node1 != nullptr));
    size_t allocated_size = node1->size();

    allocator.free(node1);

    // Allocate the exact same size again
    auto *node2 = allocator.allocate(allocated_size);
    expect(static_cast<bool>(node2 != nullptr));

    allocator.free(node2);

    LUISA_INFO("Exact fit test passed.");
}

// Test multiple allocations/deallocations pattern
void test_multiple_allocs_pattern() {
    LUISA_INFO("Testing multiple allocations pattern...");

    FirstFit allocator(4096, 16);
    std::vector<FirstFit::Node *> nodes;

    // Allocate many small blocks
    for (int i = 0; i < 20; ++i) {
        auto *node = allocator.allocate(64);
        expect(static_cast<bool>(node != nullptr));
        nodes.push_back(node);
    }

    // Free every other block (create fragmentation)
    for (size_t i = 0; i < nodes.size(); i += 2) {
        allocator.free(nodes[i]);
    }

    // Allocate blocks that should fit in freed spaces
    std::vector<FirstFit::Node *> new_nodes;
    for (int i = 0; i < 10; ++i) {
        auto *node = allocator.allocate(32);
        expect(static_cast<bool>(node != nullptr));
        new_nodes.push_back(node);
    }

    // Cleanup remaining nodes
    for (size_t i = 1; i < nodes.size(); i += 2) {
        allocator.free(nodes[i]);
    }
    for (auto *node : new_nodes) {
        allocator.free(node);
    }

    LUISA_INFO("Multiple allocations pattern test passed.");
}

// Test edge case: allocate 0 bytes
void test_zero_allocation() {
    LUISA_INFO("Testing zero allocation edge case...");

    FirstFit allocator(1024, 8);

    // Try to allocate 0 bytes
    auto *node = allocator.allocate(0);
    // Implementation may return nullptr or a valid node
    if (node) {
        allocator.free(node);
    }

    LUISA_INFO("Zero allocation test passed.");
}

static auto test_first_fit_registration = [] {
    "test_basic_construction"_test = [] {
        log_level_verbose();
        test_basic_construction();
    };
    "test_basic_allocate_free"_test = [] { test_basic_allocate_free(); };
    "test_alignment"_test = [] { test_alignment(); };
    "test_first_fit_strategy"_test = [] { test_first_fit_strategy(); };
    "test_best_fit_strategy"_test = [] { test_best_fit_strategy(); };
    "test_memory_exhaustion"_test = [] { test_memory_exhaustion(); };
    "test_move_semantics"_test = [] { test_move_semantics(); };
    "test_dump_free_list"_test = [] { test_dump_free_list(); };
    "test_fragmentation_and_coalescing"_test = [] { test_fragmentation_and_coalescing(); };
    "test_exact_fit"_test = [] { test_exact_fit(); };
    "test_multiple_allocs_pattern"_test = [] { test_multiple_allocs_pattern(); };
    "test_zero_allocation"_test = [] {
        test_zero_allocation();
        LUISA_INFO("All FirstFit tests passed!");
    };
    return 0;
}();
int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));

}
