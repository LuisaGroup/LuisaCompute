// Test for Pool class
// This test covers:
// - Basic allocate/deallocate operations
// - create/destroy for object construction/destruction
// - Thread safety with concurrent allocations
// - Memory leak detection for non-trivially destructible types
// - Move semantics
// - Block allocation behavior (block_size = 64)

#include <algorithm>
#include <array>
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


// ---------------------------------------------------------------------------
// ConcurrentPool: the lock-free variant used for resource reference counts.
// ---------------------------------------------------------------------------

// Test basic allocate and deallocate on ConcurrentPool
void test_concurrent_pool_basic() {
    LUISA_INFO("Testing ConcurrentPool basic allocate/deallocate...");

    ConcurrentPool<SimpleStruct> pool;

    std::vector<SimpleStruct *> objects;
    for (int i = 0; i < 100; ++i) {
        auto *obj = pool.allocate();
        expect(static_cast<bool>(obj != nullptr));
        obj->x = i;
        obj->y = static_cast<float>(i) * 0.5f;
        objects.push_back(obj);
    }

    // All live objects must sit at distinct addresses.
    std::vector<SimpleStruct *> sorted{objects};
    std::sort(sorted.begin(), sorted.end());
    expect(static_cast<bool>(std::adjacent_find(sorted.begin(), sorted.end()) == sorted.end()))
        << "ConcurrentPool handed out the same address twice";

    for (auto *obj : objects) { pool.deallocate(obj); }

    // The slots must come back rather than being re-allocated from the system.
    auto *obj1 = pool.allocate();
    auto *obj2 = pool.allocate();
    expect(static_cast<bool>(obj1 != nullptr && obj2 != nullptr));
    expect(static_cast<bool>(obj1 != obj2));
    expect(std::binary_search(sorted.begin(), sorted.end(), obj1))
        << "ConcurrentPool must recycle a freed slot instead of allocating";
    expect(std::binary_search(sorted.begin(), sorted.end(), obj2))
        << "ConcurrentPool must recycle a freed slot instead of allocating";
    pool.deallocate(obj1);
    pool.deallocate(obj2);

    LUISA_INFO("ConcurrentPool basic allocate/deallocate test passed.");
}

// Test create/destroy: the destructor must run exactly once per object
void test_concurrent_pool_create_destroy() {
    LUISA_INFO("Testing ConcurrentPool create/destroy...");

    ComplexStruct::reset_counts();
    {
        ConcurrentPool<ComplexStruct> pool;
        std::vector<ComplexStruct *> objects;
        for (int i = 0; i < 50; ++i) {
            auto *obj = pool.create(i);
            expect(static_cast<bool>(obj->value == i));
            objects.push_back(obj);
        }
        expect(static_cast<bool>(ComplexStruct::constructor_count.load() == 50));
        for (auto *obj : objects) { pool.destroy(obj); }
        expect(static_cast<bool>(ComplexStruct::destructor_count.load() == 50));
    }
    LUISA_INFO("ConcurrentPool create/destroy test passed.");
}

// Test block growth: a 65th allocation must open a second block, and freeing
// everything must make the pool reuse both blocks instead of growing again.
void test_concurrent_pool_block_growth() {
    LUISA_INFO("Testing ConcurrentPool block growth...");

    ConcurrentPool<SimpleStruct> pool;
    constexpr auto block = ConcurrentPool<SimpleStruct>::block_size;

    std::vector<SimpleStruct *> first;
    for (auto i = 0u; i < block + 1u; i++) { first.push_back(pool.allocate()); }
    std::vector<SimpleStruct *> sorted{first};
    std::sort(sorted.begin(), sorted.end());
    expect(static_cast<bool>(std::adjacent_find(sorted.begin(), sorted.end()) == sorted.end()));


    expect(static_cast<bool>(pool.num_blocks() == 2u))
        << "65 allocations over 64-slot blocks must own exactly two blocks";

    for (auto *object : first) { pool.deallocate(object); }

    // A second round of the same size must fit in the two existing blocks: the pool
    // recycles slots, it does not keep growing with the number of allocations.
    std::vector<SimpleStruct *> second;
    for (auto i = 0u; i < block + 1u; i++) { second.push_back(pool.allocate()); }
    std::vector<SimpleStruct *> sorted_again{second};
    std::sort(sorted_again.begin(), sorted_again.end());
    expect(static_cast<bool>(std::adjacent_find(sorted_again.begin(), sorted_again.end()) == sorted_again.end()))
        << "ConcurrentPool handed out the same address twice on reuse";
    expect(static_cast<bool>(pool.num_blocks() == 2u))
        << "ConcurrentPool grew instead of reusing its blocks";
    for (auto *object : second) { pool.deallocate(object); }

    LUISA_INFO("ConcurrentPool block growth test passed.");
}

// Test that concurrent allocations never hand out the same address twice
void test_concurrent_pool_thread_safety() {
    LUISA_INFO("Testing ConcurrentPool thread safety...");

    ConcurrentPool<SimpleStruct> pool;
    constexpr auto num_threads = 8;
    constexpr auto per_thread = 2000;

    std::vector<std::vector<SimpleStruct *>> thread_objects(num_threads);
    std::vector<std::thread> threads;
    threads.reserve(num_threads);
    for (auto t = 0; t < num_threads; t++) {
        threads.emplace_back([&, t] {
            thread_objects[t].reserve(per_thread);
            for (auto i = 0; i < per_thread; i++) {
                auto *obj = pool.allocate();
                obj->x = t;
                obj->y = static_cast<float>(i);
                thread_objects[t].push_back(obj);
            }
        });
    }
    for (auto &thread : threads) { thread.join(); }

    std::vector<SimpleStruct *> all;
    all.reserve(num_threads * per_thread);
    for (auto t = 0; t < num_threads; t++) {
        for (auto *obj : thread_objects[t]) {
            expect(static_cast<bool>(obj->x == t)) << "object was overwritten by another thread";
            all.push_back(obj);
        }
    }
    std::sort(all.begin(), all.end());
    expect(static_cast<bool>(std::adjacent_find(all.begin(), all.end()) == all.end()))
        << "ConcurrentPool handed out the same slot to two live objects";
    expect(static_cast<bool>(all.size() == num_threads * per_thread));

    // Free everything from a single thread: a reference-count block is typically
    // released by whichever thread happens to drop the last handle.
    for (auto *obj : all) { pool.deallocate(obj); }

    LUISA_INFO("ConcurrentPool thread safety test passed.");
}

// Test churn: threads allocate and free at once, with objects migrating between
// threads, which is the pattern the resource reference counts produce.
void test_concurrent_pool_churn() {
    LUISA_INFO("Testing ConcurrentPool churn...");

    struct Counted {
        std::atomic<int> *live;
        explicit Counted(std::atomic<int> *counter) noexcept : live{counter} { live->fetch_add(1); }
        ~Counted() noexcept { live->fetch_sub(1); }
    };

    ConcurrentPool<Counted> pool;
    std::atomic<int> live{0};
    constexpr auto num_threads = 8;
    constexpr auto rounds = 20000;

    std::vector<std::thread> threads;
    threads.reserve(num_threads);
    for (auto t = 0; t < num_threads; t++) {
        threads.emplace_back([&, t] {
            // Each thread keeps a rotating set of objects alive and hands one of
            // them to another thread's slot every so often.
            std::array<Counted *, 4> held{};
            for (auto &slot : held) { slot = pool.create(&live); }
            for (auto i = 0; i < rounds; i++) {
                auto index = static_cast<size_t>(i) & 3u;
                auto *object = pool.create(&live);
                pool.destroy(held[index]);
                held[index] = object;
            }
            for (auto *object : held) { pool.destroy(object); }
            (void)t;
        });
    }
    for (auto &thread : threads) { thread.join(); }

    expect(static_cast<bool>(live.load() == 0))
        << "objects leaked through ConcurrentPool, live = " << live.load();

    LUISA_INFO("ConcurrentPool churn test passed.");
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
    };
    "concurrent_pool_basic"_test = [] { test_concurrent_pool_basic(); };
    "concurrent_pool_create_destroy"_test = [] { test_concurrent_pool_create_destroy(); };
    "concurrent_pool_block_growth"_test = [] { test_concurrent_pool_block_growth(); };
    "concurrent_pool_thread_safety"_test = [] { test_concurrent_pool_thread_safety(); };
    "concurrent_pool_churn"_test = [] {
        test_concurrent_pool_churn();
        LUISA_INFO("All Pool tests passed!");
    };
    return 0;
}();
int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));

}
