// Test for Pool class
// This test covers:
// - Basic allocate/deallocate operations
// - create/destroy for object construction/destruction
// - Thread safety with concurrent allocations
// - Memory leak detection for non-trivially destructible types
// - Move semantics
// - Block allocation behavior (block_size = 64)

#include <atomic>
#include <thread>
#include <vector>

#include <luisa/core/pool.h>
#include <luisa/core/logging.h>
#include "ut/ut.hpp"

using namespace luisa;
using namespace boost::ut;
using namespace boost::ut::literals;

// Simple struct for basic tests
struct SimpleStruct {
    int x;
    float y;

    SimpleStruct() noexcept : x(0), y(0.0f) {}
    SimpleStruct(int x_, float y_) noexcept : x(x_), y(y_) {}
};

// Non-trivially destructible struct for leak detection tests
struct ComplexStruct {
    static std::atomic<int> constructor_count;
    static std::atomic<int> destructor_count;

    int value;

    ComplexStruct() noexcept : value(0) {
        constructor_count.fetch_add(1, std::memory_order_relaxed);
    }

    explicit ComplexStruct(int v) noexcept : value(v) {
        constructor_count.fetch_add(1, std::memory_order_relaxed);
    }

    ~ComplexStruct() noexcept {
        destructor_count.fetch_add(1, std::memory_order_relaxed);
    }

    static void reset_counts() noexcept {
        constructor_count.store(0, std::memory_order_relaxed);
        destructor_count.store(0, std::memory_order_relaxed);
    }
};

std::atomic<int> ComplexStruct::constructor_count{0};
std::atomic<int> ComplexStruct::destructor_count{0};

// Test basic allocate and deallocate
void test_basic_allocate_deallocate() {
    LUISA_INFO("Testing basic allocate/deallocate...");

    Pool<SimpleStruct> pool;

    // Allocate some objects
    std::vector<SimpleStruct *> objects;
    for (int i = 0; i < 100; ++i) {
        auto *obj = pool.allocate();
        expect(static_cast<bool>(obj != nullptr));
        obj->x = i;
        obj->y = static_cast<float>(i) * 0.5f;
        objects.push_back(obj);
    }

    // Deallocate all objects
    for (auto *obj : objects) {
        pool.deallocate(obj);
    }

    // Allocate again - should reuse memory
    auto *obj1 = pool.allocate();
    auto *obj2 = pool.allocate();

    // These should be from the available pool (LIFO order)
    expect(static_cast<bool>(obj1 != nullptr && obj2 != nullptr));

    pool.deallocate(obj1);
    pool.deallocate(obj2);

    LUISA_INFO("Basic allocate/deallocate test passed.");
}

// Test create and destroy with constructors/destructors
void test_create_destroy() {
    LUISA_INFO("Testing create/destroy...");

    ComplexStruct::reset_counts();

    {
        Pool<ComplexStruct> pool;

        // Create objects
        std::vector<ComplexStruct *> objects;
        for (int i = 0; i < 50; ++i) {
            auto *obj = pool.create(i);
            expect(static_cast<bool>(obj != nullptr));
            expect(static_cast<bool>(obj->value == i));
            objects.push_back(obj);
        }

        int constructor_count = ComplexStruct::constructor_count.load(std::memory_order_relaxed);
        expect(static_cast<bool>(constructor_count == 50));

        // Destroy objects
        for (auto *obj : objects) {
            pool.destroy(obj);
        }

        int destructor_count = ComplexStruct::destructor_count.load(std::memory_order_relaxed);
        expect(static_cast<bool>(destructor_count == 50));
    }

    LUISA_INFO("Create/destroy test passed.");
}

// Test thread safety with concurrent allocations
void test_thread_safety() {
    LUISA_INFO("Testing thread safety...");

    Pool<SimpleStruct> pool;
    constexpr int num_threads = 8;
    constexpr int allocations_per_thread = 1000;

    std::vector<std::thread> threads;
    std::vector<std::vector<SimpleStruct *>> thread_objects(num_threads);

    // Concurrent allocations
    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&, t]() {
            for (int i = 0; i < allocations_per_thread; ++i) {
                auto *obj = pool.allocate();
                expect(static_cast<bool>(obj != nullptr));
                obj->x = t;
                obj->y = static_cast<float>(i);
                thread_objects[t].push_back(obj);
            }
        });
    }

    for (auto &t : threads) {
        t.join();
    }
    threads.clear();

    // Verify all objects have correct values
    for (int t = 0; t < num_threads; ++t) {
        for (auto *obj : thread_objects[t]) {
            expect(static_cast<bool>(obj->x == t));
        }
    }

    // Concurrent deallocations
    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&, t]() {
            for (auto *obj : thread_objects[t]) {
                pool.deallocate(obj);
            }
        });
    }

    for (auto &t : threads) {
        t.join();
    }

    LUISA_INFO("Thread safety test passed.");
}

// Test move semantics
void test_move_semantics() {
    LUISA_INFO("Testing move semantics...");

    Pool<SimpleStruct> pool1;

    // Allocate some objects
    std::vector<SimpleStruct *> objects;
    for (int i = 0; i < 10; ++i) {
        objects.push_back(pool1.allocate());
    }

    // Deallocate half, keep half
    for (size_t i = 0; i < objects.size() / 2; ++i) {
        pool1.deallocate(objects[i]);
    }

    // Move construct
    Pool<SimpleStruct> pool2{std::move(pool1)};

    // Move assign
    Pool<SimpleStruct> pool3;
    pool3 = std::move(pool2);

    // Should still be able to allocate from moved pool
    auto *obj = pool3.allocate();
    expect(static_cast<bool>(obj != nullptr));
    pool3.deallocate(obj);

    LUISA_INFO("Move semantics test passed.");
}

// Test block allocation behavior (block_size = 64)
void test_block_allocation() {
    LUISA_INFO("Testing block allocation (block_size = 64)...");

    Pool<SimpleStruct> pool;

    // Allocate exactly 64 objects (one block)
    std::vector<SimpleStruct *> block1;
    for (int i = 0; i < 64; ++i) {
        block1.push_back(pool.allocate());
    }

    // Allocate one more - should trigger second block
    auto *extra = pool.allocate();

    // Deallocate all
    for (auto *obj : block1) {
        pool.deallocate(obj);
    }
    pool.deallocate(extra);

    // Now allocate again - should reuse from available pool
    std::vector<SimpleStruct *> block2;
    for (int i = 0; i < 65; ++i) {
        block2.push_back(pool.allocate());
    }

    LUISA_INFO("Block allocation test passed.");
}

// Test with trivially destructible type (check_recycle = false)
void test_trivially_destructible() {
    LUISA_INFO("Testing trivially destructible type...");

    Pool<int> pool;

    std::vector<int *> ints;
    for (int i = 0; i < 100; ++i) {
        auto *p = pool.create(i);
        expect(static_cast<bool>(*p == i));
        ints.push_back(p);
    }

    for (auto *p : ints) {
        pool.destroy(p);
    }

    LUISA_INFO("Trivially destructible type test passed.");
}

// Test non-thread-safe pool
void test_non_thread_safe() {
    LUISA_INFO("Testing non-thread-safe pool...");

    Pool<SimpleStruct, false> pool;

    // Single thread operations
    std::vector<SimpleStruct *> objects;
    for (int i = 0; i < 100; ++i) {
        objects.push_back(pool.allocate());
    }

    for (auto *obj : objects) {
        pool.deallocate(obj);
    }

    LUISA_INFO("Non-thread-safe pool test passed.");
}

static auto test_pool_registration = [] {
    "test_basic_allocate_deallocate"_test = [] {
        log_level_verbose();
        LUISA_INFO("Starting Pool tests...");
        test_basic_allocate_deallocate();
    };
    "test_create_destroy"_test = [] { test_create_destroy(); };
    "test_thread_safety"_test = [] { test_thread_safety(); };
    "test_move_semantics"_test = [] { test_move_semantics(); };
    "test_block_allocation"_test = [] { test_block_allocation(); };
    "test_trivially_destructible"_test = [] { test_trivially_destructible(); };
    "test_non_thread_safe"_test = [] {
        test_non_thread_safe();
        LUISA_INFO("All Pool tests passed!");
    };
    return 0;
}();
int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));

}
