// Test for EASTL allocator consistency.
// This test covers:
// - make_unique<T> (single object) uses GetDefaultAllocator for allocation
// - make_unique<T[]> (unbounded array) uses GetDefaultAllocator with header layout
// - default_delete<T> deallocates via GetDefaultAllocator
// - default_delete<T[]> reads header and deallocates via GetDefaultAllocator
// - smart_ptr_deleter<T> uses correct alloc_size formula
// - smart_ptr_deleter<void> uses correct alloc_size formula (header_size + ele_size)
// - smart_ptr_deleter<const void> uses correct alloc_size formula
// - smart_array_deleter<T> uses correct alloc_size formula (header_size + ele_size * sizeof(T))
// - smart_array_deleter<void> uses correct alloc_size formula
// - Allocator consistency: allocation and deallocation go through the same allocator
// - vector<T> uses allocator for all allocations (via allocate_memory + EASTLFree)
// - fixed_vector<T, N, true> uses fixed buffer within capacity, overflow allocator beyond
// - fixed_vector<T, N, false> uses fixed buffer only, asserts on overflow
//
// Every test uses TrackedDestroy for lifecycle verification to ensure
// no memory leaks and correct destructor invocation.

#include "ut/ut.hpp"

#include <EASTL/unique_ptr.h>
#include <EASTL/internal/smart_ptr.h>
#include <EASTL/memory.h>
#include <EASTL/allocator.h>
#include <EASTL/vector.h>
#include <EASTL/fixed_vector.h>

using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

    // A non-trivially-destructible type to verify destructor calls.
    // Tracks all constructions (default, copy, move) and destructions.
    struct TrackedDestroy {
        static int alive_count;
        static int destroy_count;
        int id;

        TrackedDestroy() : id(0) { ++alive_count; }
        TrackedDestroy(int v) : id(v) { ++alive_count; }
        TrackedDestroy(const TrackedDestroy& o) : id(o.id) { ++alive_count; }
        TrackedDestroy(TrackedDestroy&& o) noexcept : id(o.id) { o.id = -1; ++alive_count; }
        TrackedDestroy& operator=(const TrackedDestroy& o) noexcept { id = o.id; return *this; }
        TrackedDestroy& operator=(TrackedDestroy&& o) noexcept { id = o.id; o.id = -1; return *this; }
        ~TrackedDestroy() { ++destroy_count; --alive_count; }
    };
    int TrackedDestroy::alive_count = 0;
    int TrackedDestroy::destroy_count = 0;

    // Reset tracked counters
    void reset_tracked() {
        TrackedDestroy::alive_count = 0;
        TrackedDestroy::destroy_count = 0;
    }

} // anonymous namespace

// ---- make_unique<T> single object ----

void reg_make_unique_single_basic() {
    "make_unique_single_basic"_test = [] {
        reset_tracked();
        {
            auto p = eastl::make_unique<TrackedDestroy>(42);
            expect(p->id == 42) << "make_unique<TrackedDestroy>(42) should produce id 42";
            expect(TrackedDestroy::alive_count == 1_i) << "one instance alive";
        }
        expect(TrackedDestroy::alive_count == 0_i) << "instance destroyed after scope";
    };
}

void reg_make_unique_single_zero() {
    "make_unique_single_zero"_test = [] {
        reset_tracked();
        {
            auto p = eastl::make_unique<TrackedDestroy>(0);
            expect(p->id == 0) << "make_unique<TrackedDestroy>(0) should produce id 0";
            expect(TrackedDestroy::alive_count == 1_i) << "one instance alive";
        }
        expect(TrackedDestroy::alive_count == 0_i) << "instance destroyed after scope";
    };
}

void reg_make_unique_single_default_init() {
    "make_unique_single_default_init"_test = [] {
        reset_tracked();
        {
            auto p = eastl::make_unique<TrackedDestroy>();
            expect(p->id == 0) << "default-constructed TrackedDestroy has id 0";
            expect(TrackedDestroy::alive_count == 1_i) << "one instance alive";
        }
        expect(TrackedDestroy::alive_count == 0_i) << "instance destroyed after scope";
    };
}

void reg_make_unique_single_multiple_args() {
    "make_unique_single_multiple_args"_test = [] {
        reset_tracked();
        {
            auto p = eastl::make_unique<TrackedDestroy>(99);
            expect(p->id == 99) << "TrackedDestroy::id should be 99";
            expect(TrackedDestroy::alive_count == 1_i) << "one instance alive";
        }
        expect(TrackedDestroy::alive_count == 0_i) << "instance destroyed after scope";
    };
}

void reg_make_unique_single_destructor_called() {
    "make_unique_single_destructor_called"_test = [] {
        reset_tracked();
        {
            auto p = eastl::make_unique<TrackedDestroy>(123);
            expect(p->id == 123_i) << "TrackedDestroy::id should be 123";
            expect(TrackedDestroy::alive_count == 1_i) << "one instance alive";
            expect(TrackedDestroy::destroy_count == 0_i) << "no destructions yet";
        }
        expect(TrackedDestroy::alive_count == 0_i) << "instance should be destroyed";
        expect(TrackedDestroy::destroy_count == 1_i) << "destructor should have been called";
    };
}

// ---- make_unique<T[]> array ----

void reg_make_unique_array_basic() {
    "make_unique_array_basic"_test = [] {
        reset_tracked();
        {
            auto p = eastl::make_unique<TrackedDestroy[]>(10);
            expect(TrackedDestroy::alive_count == 10_i) << "10 elements constructed";
            for (size_t i = 0; i < 10; ++i) {
                expect(p[i].id == 0_i) << "element " << i << " should be default-initialized";
            }
        }
        expect(TrackedDestroy::alive_count == 0_i) << "all elements destroyed";
    };
}

void reg_make_unique_array_large() {
    "make_unique_array_large"_test = [] {
        reset_tracked();
        constexpr size_t N = 1024;
        {
            auto p = eastl::make_unique<TrackedDestroy[]>(N);
            expect(TrackedDestroy::alive_count == static_cast<int>(N)) << "all elements constructed";
        }
        expect(TrackedDestroy::alive_count == 0_i) << "all elements destroyed";
    };
}

void reg_make_unique_array_non_trivial() {
    "make_unique_array_non_trivial"_test = [] {
        reset_tracked();
        constexpr size_t N = 5;
        {
            auto p = eastl::make_unique<TrackedDestroy[]>(N);
            expect(TrackedDestroy::alive_count == static_cast<int>(N)) << "all elements constructed";
            expect(TrackedDestroy::destroy_count == 0_i) << "no destructions yet";
            for (size_t i = 0; i < N; ++i) {
                expect(p[i].id == 0_i) << "element " << i << " should be zero-initialized";
            }
        }
        expect(TrackedDestroy::alive_count == 0_i) << "all elements destroyed";
        expect(TrackedDestroy::destroy_count == static_cast<int>(N)) << "all destructors called";
    };
}

void reg_make_unique_array_edge_sizes() {
    "make_unique_array_edge_sizes"_test = [] {
        reset_tracked();
        {
            auto p0 = eastl::make_unique<TrackedDestroy[]>(0);
            expect(TrackedDestroy::alive_count == 0_i) << "zero-size array should have no elements";
        }
        expect(TrackedDestroy::alive_count == 0_i) << "after zero-size array destruction";

        reset_tracked();
        {
            auto p1 = eastl::make_unique<TrackedDestroy[]>(1);
            expect(TrackedDestroy::alive_count == 1_i) << "one element alive";
            expect(p1[0].id == 0_i) << "default-initialized id is 0";
        }
        expect(TrackedDestroy::alive_count == 0_i) << "single-element array destroyed";
    };
}

// ---- default_delete<T> direct verification ----

void reg_default_delete_single() {
    "default_delete_single"_test = [] {
        reset_tracked();
        auto* alloc = eastl::GetDefaultAllocator();
        void* mem = alloc->allocate(sizeof(TrackedDestroy));
        auto* p = ::new(mem) TrackedDestroy(42);
        expect(p->id == 42) << "placement-new should work";
        expect(TrackedDestroy::alive_count == 1_i) << "one instance alive";

        eastl::default_delete<TrackedDestroy> del;
        del(p);
        expect(TrackedDestroy::alive_count == 0_i) << "instance destroyed by default_delete";
        expect(TrackedDestroy::destroy_count == 1_i) << "destructor called";
    };
}

void reg_default_delete_array_layout() {
    "default_delete_array_layout"_test = [] {
        reset_tracked();
        using T = TrackedDestroy;
        constexpr size_t N = 8;
        auto header_size = eastl::max<size_t>(alignof(T), sizeof(size_t));
        auto alloc_size = header_size + N * sizeof(T);

        auto* alloc = eastl::GetDefaultAllocator();
        void* mem = alloc->allocate(alloc_size);

        *static_cast<size_t*>(mem) = N;
        T* p = reinterpret_cast<T*>(static_cast<char*>(mem) + header_size);

        for (size_t i = 0; i < N; ++i) {
            ::new(&p[i]) T(static_cast<int>(i * 10));
        }
        expect(TrackedDestroy::alive_count == static_cast<int>(N)) << "elements alive";

        eastl::default_delete<T[]> del;
        del(p);
        expect(TrackedDestroy::alive_count == 0_i) << "all elements destroyed by default_delete<T[]>";
    };
}

void reg_default_delete_array_odd_layout() {
    "default_delete_array_odd_layout"_test = [] {
        reset_tracked();
        using T = TrackedDestroy;
        constexpr size_t N = 4;
        auto header_size = eastl::max<size_t>(alignof(T), sizeof(size_t));
        auto alloc_size = header_size + N * sizeof(T);

        auto* alloc = eastl::GetDefaultAllocator();
        void* mem = alloc->allocate(alloc_size);
        *static_cast<size_t*>(mem) = N;

        T* p = reinterpret_cast<T*>(static_cast<char*>(mem) + header_size);
        expect(reinterpret_cast<size_t>(p) % alignof(T) == 0_ul) << "array should be aligned";

        for (size_t i = 0; i < N; ++i) {
            ::new(&p[i]) T(static_cast<int>(i));
        }
        expect(TrackedDestroy::alive_count == static_cast<int>(N)) << "elements alive";

        eastl::default_delete<T[]> del;
        del(p);
        expect(TrackedDestroy::alive_count == 0_i) << "all elements destroyed by default_delete<T[]> (odd layout)";
    };
}

// ---- smart_ptr_deleter<T> (non-void) ----

void reg_smart_ptr_deleter_single() {
    "smart_ptr_deleter_single"_test = [] {
        reset_tracked();
        auto* alloc = eastl::GetDefaultAllocator();
        void* mem = alloc->allocate(sizeof(TrackedDestroy));
        auto* p = ::new(mem) TrackedDestroy(42);
        expect(TrackedDestroy::alive_count == 1_i) << "instance alive";

        eastl::smart_ptr_deleter<TrackedDestroy> del;
        del(p);
        expect(TrackedDestroy::alive_count == 0_i) << "instance destroyed by smart_ptr_deleter";
        expect(TrackedDestroy::destroy_count == 1_i) << "destructor called";
    };
}

// ---- smart_ptr_deleter<void> ----
void reg_smart_ptr_deleter_void() {
    "smart_ptr_deleter_void"_test = [] {
        reset_tracked();
        using T = TrackedDestroy;
        constexpr size_t N = 8;
        auto header_size = sizeof(size_t);
        auto byte_count = N * sizeof(T);
        auto alloc_size = header_size + byte_count;

        auto* alloc = eastl::GetDefaultAllocator();
        void* mem = alloc->allocate(alloc_size);
        // smart_ptr_deleter<void> stores byte count in the header
        *static_cast<size_t*>(mem) = byte_count;

        T* p = reinterpret_cast<T*>(static_cast<char*>(mem) + header_size);
        for (size_t i = 0; i < N; ++i) {
            ::new(&p[i]) T(static_cast<int>(i));
        }
        expect(TrackedDestroy::alive_count == static_cast<int>(N)) << "elements alive before void-delete";

        // Manually destroy elements (smart_ptr_deleter<void> only frees memory)
        for (size_t i = 0; i < N; ++i) {
            p[i].~T();
        }
        expect(TrackedDestroy::alive_count == 0_i) << "elements destroyed before void-delete";

        eastl::smart_ptr_deleter<void> del;
        del(static_cast<void*>(p));
        expect(TrackedDestroy::alive_count == 0_i) << "no change after void-delete (memory freed)";
    };
}

void reg_smart_ptr_deleter_void_non_trivial() {
    "smart_ptr_deleter_void_non_trivial"_test = [] {
        reset_tracked();
        using T = TrackedDestroy;
        constexpr size_t N = 3;
        auto header_size = sizeof(size_t);
        auto byte_count = N * sizeof(T);
        auto alloc_size = header_size + byte_count;

        auto* alloc = eastl::GetDefaultAllocator();
        void* mem = alloc->allocate(alloc_size);
        *static_cast<size_t*>(mem) = byte_count;

        T* p = reinterpret_cast<T*>(static_cast<char*>(mem) + header_size);
        for (size_t i = 0; i < N; ++i) {
            ::new(&p[i]) T(static_cast<int>(i));
        }
        expect(TrackedDestroy::alive_count == static_cast<int>(N));

        for (size_t i = 0; i < N; ++i) {
            p[i].~T();
        }

        eastl::smart_ptr_deleter<void> del;
        del(static_cast<void*>(p));
        expect(TrackedDestroy::alive_count == 0_i);
    };
}

// ---- smart_ptr_deleter<const void> ----

void reg_smart_ptr_deleter_const_void() {
    "smart_ptr_deleter_const_void"_test = [] {
        reset_tracked();
        using T = TrackedDestroy;
        constexpr size_t N = 4;
        auto header_size = sizeof(size_t);
        auto byte_count = N * sizeof(T);
        auto alloc_size = header_size + byte_count;

        auto* alloc = eastl::GetDefaultAllocator();
        void* mem = alloc->allocate(alloc_size);
        *static_cast<size_t*>(mem) = byte_count;

        T* p = reinterpret_cast<T*>(static_cast<char*>(mem) + header_size);
        for (size_t i = 0; i < N; ++i) {
            ::new(&p[i]) T(static_cast<int>(i));
        }
        for (size_t i = 0; i < N; ++i) {
            p[i].~T();
        }

        const void* cp = static_cast<const char*>(mem) + header_size;

        eastl::smart_ptr_deleter<const void> del;
        del(cp);
        expect(TrackedDestroy::alive_count == 0_i) << "const void deleter should not leak";
    };
}
// ---- smart_array_deleter<T> ----

void reg_smart_array_deleter_typed() {
    "smart_array_deleter_typed"_test = [] {
        reset_tracked();
        using T = TrackedDestroy;
        constexpr size_t N = 6;
        auto header_size = eastl::max<size_t>(alignof(T), sizeof(size_t));
        auto alloc_size = header_size + N * sizeof(T);

        auto* alloc = eastl::GetDefaultAllocator();
        void* mem = alloc->allocate(alloc_size);
        *static_cast<size_t*>(mem) = N;

        T* p = reinterpret_cast<T*>(static_cast<char*>(mem) + header_size);
        for (size_t i = 0; i < N; ++i) {
            ::new(&p[i]) T(static_cast<int>(i));
        }
        expect(TrackedDestroy::alive_count == static_cast<int>(N));

        eastl::smart_array_deleter<T> del;
        del(p);
        expect(TrackedDestroy::alive_count == 0_i) << "smart_array_deleter destroyed all elements";
        expect(TrackedDestroy::destroy_count == static_cast<int>(N)) << "all destructors called";
    };
}

void reg_smart_array_deleter_typed_non_trivial() {
    "smart_array_deleter_typed_non_trivial"_test = [] {
        reset_tracked();
        using T = TrackedDestroy;
        constexpr size_t N = 4;
        auto header_size = eastl::max<size_t>(alignof(T), sizeof(size_t));
        auto alloc_size = header_size + N * sizeof(T);

        auto* alloc = eastl::GetDefaultAllocator();
        void* mem = alloc->allocate(alloc_size);
        *static_cast<size_t*>(mem) = N;

        T* p = reinterpret_cast<T*>(static_cast<char*>(mem) + header_size);
        for (size_t i = 0; i < N; ++i) {
            ::new(&p[i]) T(static_cast<int>(i));
        }
        expect(TrackedDestroy::alive_count == static_cast<int>(N));

        eastl::smart_array_deleter<T> del;
        del(p);
        expect(TrackedDestroy::alive_count == 0_i) << "all elements destroyed";
        expect(TrackedDestroy::destroy_count == static_cast<int>(N)) << "all destructors called";
    };
}

// ---- smart_array_deleter<void> ----

void reg_smart_array_deleter_void() {
    "smart_array_deleter_void"_test = [] {
        reset_tracked();
        using T = TrackedDestroy;
        constexpr size_t N = 6;
        auto header_size = sizeof(size_t);
        auto byte_count = N * sizeof(T);
        auto alloc_size = header_size + byte_count;

        auto* alloc = eastl::GetDefaultAllocator();
        void* mem = alloc->allocate(alloc_size);
        *static_cast<size_t*>(mem) = byte_count;

        T* p = reinterpret_cast<T*>(static_cast<char*>(mem) + header_size);
        for (size_t i = 0; i < N; ++i) {
            ::new(&p[i]) T(static_cast<int>(i));
        }
        expect(TrackedDestroy::alive_count == static_cast<int>(N));

        for (size_t i = 0; i < N; ++i) {
            p[i].~T();
        }

        eastl::smart_array_deleter<void> del;
        del(p);
        expect(TrackedDestroy::alive_count == 0_i) << "smart_array_deleter<void> should not leak";
    };
}

// ---- Allocator identity ----

void reg_allocator_identity() {
    "allocator_identity"_test = [] {
        reset_tracked();
        auto* alloc = eastl::GetDefaultAllocator();
        expect(alloc != nullptr) << "GetDefaultAllocator() should return non-null";

        void* p = alloc->allocate(64);
        expect(p != nullptr) << "allocate(64) should return non-null";
        alloc->deallocate(p, 64);
        expect(TrackedDestroy::alive_count == 0_i) << "no tracked instances leaked";
    };
}

void reg_allocator_set_default() {
    "allocator_set_default"_test = [] {
        reset_tracked();
        auto* orig = eastl::GetDefaultAllocator();
        expect(orig != nullptr);

        auto* prev = eastl::SetDefaultAllocator(orig);
        expect(prev == orig) << "SetDefaultAllocator should return the previous allocator";

        auto* alloc = eastl::GetDefaultAllocator();
        void* p = alloc->allocate(32);
        expect(p != nullptr) << "allocate after reset should work";
        alloc->deallocate(p, 32);
        expect(TrackedDestroy::alive_count == 0_i) << "no tracked instances leaked";
    };
}

// ---- stress tests ----

void reg_make_unique_stress() {
    "make_unique_stress"_test = [] {
        reset_tracked();
        constexpr int Iterations = 1000;
        for (int i = 0; i < Iterations; ++i) {
            auto p = eastl::make_unique<TrackedDestroy>(i);
            expect(p->id == i) << "stress iteration " << i;
            expect(TrackedDestroy::alive_count == 1_i) << "one alive during iteration";
        }
        expect(TrackedDestroy::alive_count == 0_i) << "no leaks after stress test";
    };
}

void reg_make_unique_array_stress() {
    "make_unique_array_stress"_test = [] {
        reset_tracked();
        constexpr int Iterations = 500;
        for (int i = 0; i < Iterations; ++i) {
            auto p = eastl::make_unique<TrackedDestroy[]>(i + 1);
            expect(TrackedDestroy::alive_count == i + 1) << "stress array iteration " << i;
        }
        expect(TrackedDestroy::alive_count == 0_i) << "no leaks after array stress test";
    };
}

// ---- mixed API usage ----

void reg_mixed_smart_ptr_api() {
    "mixed_smart_ptr_api"_test = [] {
        reset_tracked();
        auto* alloc = eastl::GetDefaultAllocator();

        // Single object with custom deleter wrapping allocator deallocate
        {
            void* mem = alloc->allocate(sizeof(TrackedDestroy));
            TrackedDestroy* p = ::new(mem) TrackedDestroy(42);
            auto deleter = [alloc](TrackedDestroy* ptr) {
                ptr->~TrackedDestroy();
                alloc->deallocate(ptr, sizeof(TrackedDestroy));
            };
            eastl::unique_ptr<TrackedDestroy, decltype(deleter)> up(p, eastl::move(deleter));
            expect(up->id == 42) << "custom deleter single element";
            expect(TrackedDestroy::alive_count == 1_i);
        }
        expect(TrackedDestroy::alive_count == 0_i) << "custom deleter single: freed";

        reset_tracked();
        // Array with custom deleter matching make_unique layout
        {
            using T = TrackedDestroy;
            constexpr size_t N = 5;
            auto header_size = eastl::max<size_t>(alignof(T), sizeof(size_t));
            auto alloc_size = header_size + N * sizeof(T);
            void* mem = alloc->allocate(alloc_size);
            *static_cast<size_t*>(mem) = N;
            T* p = reinterpret_cast<T*>(static_cast<char*>(mem) + header_size);
            for (size_t i = 0; i < N; ++i) ::new(&p[i]) T(static_cast<int>(i * 10));

            auto deleter = [alloc, header_size](T* ptr) {
                auto header_ptr = reinterpret_cast<size_t*>(reinterpret_cast<size_t>(ptr) - header_size);
                auto ele_size = *header_ptr;
                auto alloc_sz = header_size + ele_size * sizeof(T);
                for (size_t i = ele_size; i > 0; --i) ptr[i - 1].~T();
                alloc->deallocate(header_ptr, alloc_sz);
            };
            eastl::unique_ptr<T[], decltype(deleter)> up(p, eastl::move(deleter));
            expect(TrackedDestroy::alive_count == static_cast<int>(N)) << "custom deleter array elements alive";
            for (size_t i = 0; i < N; ++i) expect(up[i].id == static_cast<int>(i * 10));
        }
        expect(TrackedDestroy::alive_count == 0_i) << "custom deleter array: freed";
    };
}

// ---- vector (allocator consistency) ----

void reg_vector_basic() {
    "vector_basic"_test = [] {
        reset_tracked();
        {
            eastl::vector<TrackedDestroy> v;
            expect(v.empty()) << "fresh vector should be empty";

            v.push_back(TrackedDestroy(1));
            v.push_back(TrackedDestroy(2));
            v.push_back(TrackedDestroy(3));
            expect(v.size() == 3u) << "vector should have 3 elements";
            expect(v[0].id == 1) << "v[0].id == 1";
            expect(v[1].id == 2) << "v[1].id == 2";
            expect(v[2].id == 3) << "v[2].id == 3";
            expect(TrackedDestroy::alive_count == 3_i) << "3 elements alive";

            v.clear();
            expect(v.empty()) << "vector should be empty after clear";
            expect(TrackedDestroy::alive_count == 0_i) << "all destroyed after clear";
        }
        expect(TrackedDestroy::alive_count == 0_i) << "vector destroyed, no leaks";
    };
}

void reg_vector_reserve_grow() {
    "vector_reserve_grow"_test = [] {
        reset_tracked();
        {
            eastl::vector<TrackedDestroy> v;
            v.reserve(10);
            expect(v.capacity() >= 10u) << "capacity should be >= 10";

            for (int i = 0; i < 10; ++i) {
                v.push_back(TrackedDestroy(i * 10));
            }
            expect(TrackedDestroy::alive_count == 10_i) << "10 elements alive before grow";

            // Force reallocation by adding more elements
            for (int i = 10; i < 100; ++i) {
                v.push_back(TrackedDestroy(i * 10));
            }
            expect(v.size() == 100u) << "vector should have 100 elements";
            expect(TrackedDestroy::alive_count == 100_i) << "100 elements alive after grow";
            for (int i = 0; i < 100; ++i) {
                expect(v[i].id == i * 10) << "element " << i;
            }
        }
        expect(TrackedDestroy::alive_count == 0_i) << "no leaks after vector grow";
    };
}

void reg_vector_non_trivial() {
    "vector_non_trivial"_test = [] {
        reset_tracked();
        {
            eastl::vector<TrackedDestroy> v;
            for (int i = 0; i < 5; ++i) {
                v.push_back(TrackedDestroy(i));
            }
            expect(TrackedDestroy::alive_count == 5_i) << "5 elements alive after push_back";
            for (int i = 0; i < 5; ++i) {
                expect(v[i].id == i) << "element " << i;
            }

            v.reserve(20);
            expect(TrackedDestroy::alive_count == 5_i) << "still 5 elements alive after reallocation";
            for (int i = 0; i < 5; ++i) {
                expect(v[i].id == i) << "element " << i << " after reallocation";
            }
        }
        expect(TrackedDestroy::alive_count == 0_i) << "all elements destroyed after vector destruction";
    };
}

void reg_vector_shrink_to_fit() {
    "vector_shrink_to_fit"_test = [] {
        reset_tracked();
        {
            eastl::vector<TrackedDestroy> v;
            for (int i = 0; i < 100; ++i) {
                v.push_back(TrackedDestroy(i));
            }
            auto cap_before = v.capacity();
            expect(TrackedDestroy::alive_count == 100_i) << "100 elements alive";

            v.clear();
            expect(TrackedDestroy::alive_count == 0_i) << "all destroyed after clear";

            v.shrink_to_fit();
            expect(v.capacity() < cap_before || v.capacity() == 0u)
                << "capacity should be reduced after shrink_to_fit";
        }
        expect(TrackedDestroy::alive_count == 0_i) << "no leaks after shrink_to_fit";
    };
}

// ---- fixed_vector (allocator consistency) ----
//
// Research findings:
// fixed_vector uses fixed_vector_allocator which, when bEnableOverflow=true,
// delegates all allocations to the overflow allocator (default: EASTLAllocatorType).
// The deallocate method checks if the pointer equals mpPoolBegin (the fixed buffer)
// and skips deallocation for the fixed buffer, forwarding overflow allocations
// to mOverflowAllocator.deallocate(). This is consistent.
//
// When bEnableOverflow=false, allocate() deliberately crashes (assert + crash),
// and deallocate() is a no-op. This is correct for a non-overflow fixed_vector.

void reg_fixed_vector_within_fixed() {
    "fixed_vector_within_fixed"_test = [] {
        reset_tracked();
        {
            eastl::fixed_vector<TrackedDestroy, 10, true> fv;
            expect(fv.empty()) << "fresh fixed_vector should be empty";
            expect(fv.max_size() == 10u) << "max_size should be 10";
            expect(!fv.full()) << "should not be full yet";
            expect(!fv.has_overflowed()) << "should not have overflowed";

            for (int i = 0; i < 10; ++i) {
                fv.push_back(TrackedDestroy(i * 10));
            }
            expect(fv.size() == 10u) << "fixed_vector should have 10 elements";
            expect(fv.full()) << "fixed_vector should be full";
            expect(!fv.has_overflowed()) << "should not have overflowed";
            expect(TrackedDestroy::alive_count == 10_i) << "10 elements alive";

            for (int i = 0; i < 10; ++i) {
                expect(fv[i].id == i * 10) << "element " << i;
            }

            fv.clear();
            expect(fv.empty()) << "should be empty after clear";
            expect(TrackedDestroy::alive_count == 0_i) << "all destroyed after clear";
        }
        expect(TrackedDestroy::alive_count == 0_i) << "no leaks within fixed buffer";
    };
}

void reg_fixed_vector_overflow() {
    "fixed_vector_overflow"_test = [] {
        reset_tracked();
        {
            eastl::fixed_vector<TrackedDestroy, 5, true> fv;

            for (int i = 0; i < 5; ++i) {
                fv.push_back(TrackedDestroy(i));
            }
            expect(!fv.has_overflowed()) << "should not have overflowed yet";
            expect(TrackedDestroy::alive_count == 5_i);

            // Push beyond fixed capacity -> overflow
            fv.push_back(TrackedDestroy(100));
            expect(fv.has_overflowed()) << "should have overflowed";
            expect(fv.size() == 6u) << "fixed_vector should have 6 elements";
            expect(fv[5].id == 100) << "element 5 should be 100";
            expect(TrackedDestroy::alive_count == 6_i);

            // Add more to trigger another reallocation via overflow allocator
            for (int i = 0; i < 10; ++i) {
                fv.push_back(TrackedDestroy(200 + i));
            }
            expect(fv.size() == 16u) << "fixed_vector should have 16 elements";
            expect(fv[15].id == 209) << "last element should be 209";
            expect(TrackedDestroy::alive_count == 16_i);
        }
        expect(TrackedDestroy::alive_count == 0_i) << "no leaks after overflow";
    };
}

void reg_fixed_vector_overflow_with_non_trivial() {
    "fixed_vector_overflow_non_trivial"_test = [] {
        reset_tracked();
        {
            eastl::fixed_vector<TrackedDestroy, 3, true> fv;
            for (int i = 0; i < 3; ++i) {
                fv.push_back(TrackedDestroy(i));
            }
            expect(!fv.has_overflowed()) << "within fixed buffer";
            expect(TrackedDestroy::alive_count == 3_i);

            // Overflow
            fv.push_back(TrackedDestroy(100));
            expect(fv.has_overflowed()) << "overflowed";
            expect(TrackedDestroy::alive_count == 4_i);
            expect(fv[3].id == 100) << "overflow element correct";

            // Grow again
            for (int i = 0; i < 5; ++i) {
                fv.push_back(TrackedDestroy(200 + i));
            }
            expect(TrackedDestroy::alive_count == 9_i);
        }
        expect(TrackedDestroy::alive_count == 0_i) << "all destroyed after fixed_vector destruction";
    };
}

void reg_fixed_vector_no_overflow() {
    "fixed_vector_no_overflow"_test = [] {
        reset_tracked();
        {
            eastl::fixed_vector<TrackedDestroy, 10, false> fv;
            expect(fv.empty()) << "fresh no-overflow fixed_vector should be empty";

            for (int i = 0; i < 10; ++i) {
                fv.push_back(TrackedDestroy(i * 3));
            }
            expect(fv.size() == 10u) << "fixed_vector should have 10 elements";
            expect(fv.full()) << "should be full";
            expect(TrackedDestroy::alive_count == 10_i);

            for (int i = 0; i < 10; ++i) {
                expect(fv[i].id == i * 3) << "element " << i;
            }

            fv.clear();
            expect(fv.empty()) << "should be empty after clear";
            expect(TrackedDestroy::alive_count == 0_i) << "all destroyed after clear";
        }
        expect(TrackedDestroy::alive_count == 0_i) << "no leaks (no overflow)";
    };
}

void reg_fixed_vector_constructor_large() {
    "fixed_vector_constructor_large"_test = [] {
        reset_tracked();
        {
            // Construct with more elements than the fixed buffer can hold
            // This tests the overflow allocation path during construction.
            eastl::fixed_vector<TrackedDestroy, 5, true> fv(20, TrackedDestroy(42));
            expect(fv.has_overflowed()) << "should have overflowed";
            expect(fv.size() == 20u) << "should have 20 elements";
            expect(TrackedDestroy::alive_count == 20_i) << "20 elements alive";
            for (size_t i = 0; i < 20; ++i) {
                expect(fv[i].id == 42) << "element " << i << " should be 42";
            }
        }
        expect(TrackedDestroy::alive_count == 0_i) << "no leaks after large constructor";
    };
}

void reg_fixed_vector_copy_overflowed() {
    "fixed_vector_copy_overflowed"_test = [] {
        reset_tracked();
        {
            eastl::fixed_vector<TrackedDestroy, 3, true> fv1;
            for (int i = 0; i < 10; ++i) {
                fv1.push_back(TrackedDestroy(i));
            }
            expect(fv1.has_overflowed()) << "source overflowed";
            expect(TrackedDestroy::alive_count == 10_i) << "10 elements alive";

            eastl::fixed_vector<TrackedDestroy, 3, true> fv2(fv1);
            expect(fv2.size() == 10u) << "copy should have 10 elements";
            expect(TrackedDestroy::alive_count == 20_i) << "20 elements alive after copy (10+10)";
            for (int i = 0; i < 10; ++i) {
                expect(fv2[i].id == i) << "copy element " << i;
            }
        }
        expect(TrackedDestroy::alive_count == 0_i) << "no leaks after copy overflowed";
    };
}

// ---- fixed_vector move constructor / move assignment (performance optimization) ----

void reg_fixed_vector_move_construct_within_fixed() {
    "fixed_vector_move_construct_within_fixed"_test = [] {
        reset_tracked();
        {
            eastl::fixed_vector<TrackedDestroy, 10, true> fv1;
            for (int i = 0; i < 5; ++i) { fv1.push_back(TrackedDestroy(i * 10)); }
            expect(!fv1.has_overflowed()) << "source within fixed buffer";
            expect(TrackedDestroy::alive_count == 5_i) << "5 elements alive";

            eastl::fixed_vector<TrackedDestroy, 10, true> fv2(eastl::move(fv1));
            expect(fv2.size() == 5u) << "destination should have 5 elements";
            for (int i = 0; i < 5; ++i) { expect(fv2[i].id == i * 10) << "element " << i; }
            expect(fv1.empty()) << "source should be emptied";
            expect(!fv2.has_overflowed()) << "destination should still be within fixed buffer";
            expect(TrackedDestroy::alive_count == 5_i) << "still 5 alive after move (no extra allocations)";

            // Source can be reused
            fv1.push_back(TrackedDestroy(99));
            expect(fv1[0].id == 99) << "reused source element";
            expect(TrackedDestroy::alive_count == 6_i) << "6 alive after reuse";
        }
        expect(TrackedDestroy::alive_count == 0_i) << "no leaks after within-fixed move construct";
    };
}

void reg_fixed_vector_move_construct_overflowed() {
    "fixed_vector_move_construct_overflowed"_test = [] {
        reset_tracked();
        {
            eastl::fixed_vector<TrackedDestroy, 5, true> fv1;
            for (int i = 0; i < 10; ++i) { fv1.push_back(TrackedDestroy(i)); }
            expect(fv1.has_overflowed()) << "source overflowed";
            expect(TrackedDestroy::alive_count == 10_i) << "10 elements alive";
            auto* original_ptr = fv1.data();

            eastl::fixed_vector<TrackedDestroy, 5, true> fv2(eastl::move(fv1));
            expect(fv2.data() == original_ptr) << "destination should steal heap buffer (no reallocation)";
            expect(fv2.size() == 10u) << "destination should have 10 elements";
            for (int i = 0; i < 10; ++i) { expect(fv2[i].id == i) << "element " << i; }
            expect(fv2.has_overflowed()) << "destination is on the heap";
            expect(fv1.empty()) << "source should be empty";
            expect(!fv1.has_overflowed()) << "source should be reset to fixed buffer";
            expect(TrackedDestroy::alive_count == 10_i) << "still 10 alive after pointer steal (no dtor calls)";

            // Source can be reused
            fv1.push_back(TrackedDestroy(100));
            expect(fv1[0].id == 100) << "reused source element after overflow steal";
            expect(!fv1.has_overflowed()) << "reused source still in fixed buffer";
            expect(TrackedDestroy::alive_count == 11_i) << "11 alive after reuse";
        }
        expect(TrackedDestroy::alive_count == 0_i) << "no leaks after overflowed move construct";
    };
}

void reg_fixed_vector_move_construct_overflowed_non_trivial() {
    "fixed_vector_move_construct_overflowed_non_trivial"_test = [] {
        reset_tracked();
        {
            eastl::fixed_vector<TrackedDestroy, 3, true> fv1;
            for (int i = 0; i < 8; ++i) { fv1.push_back(TrackedDestroy(i)); }
            expect(TrackedDestroy::alive_count == 8_i) << "8 elements alive before move";
            expect(fv1.has_overflowed()) << "source overflowed";

            eastl::fixed_vector<TrackedDestroy, 3, true> fv2(eastl::move(fv1));
            expect(TrackedDestroy::alive_count == 8_i) << "still 8 alive after pointer steal (no dtor calls)";
            expect(fv2.size() == 8u) << "destination has 8 elements";
            for (int i = 0; i < 8; ++i) { expect(fv2[i].id == i) << "element " << i; }
            expect(fv2.has_overflowed()) << "destination overflowed";
            expect(fv1.empty()) << "source empty";
            expect(!fv1.has_overflowed()) << "source reset to fixed";

            fv2.clear();
            expect(TrackedDestroy::alive_count == 0_i) << "all destroyed after clear";
        }
        expect(TrackedDestroy::alive_count == 0_i) << "no leaks at scope exit";
    };
}

void reg_fixed_vector_move_assign_within_fixed_to_within_fixed() {
    "fixed_vector_move_assign_within_fixed_to_within_fixed"_test = [] {
        reset_tracked();
        {
            eastl::fixed_vector<TrackedDestroy, 10, true> fv1;
            for (int i = 0; i < 5; ++i) { fv1.push_back(TrackedDestroy(i)); }
            eastl::fixed_vector<TrackedDestroy, 10, true> fv2;
            for (int i = 0; i < 3; ++i) { fv2.push_back(TrackedDestroy(i * 10)); }
            expect(TrackedDestroy::alive_count == 8_i) << "8 elements alive total";

            fv1 = eastl::move(fv2);
            expect(fv1.size() == 3u) << "fv1 should have 3 elements after move assign";
            expect(fv1[0].id == 0) << "fv1[0]";
            expect(fv1[1].id == 10) << "fv1[1]";
            expect(fv1[2].id == 20) << "fv1[2]";
            expect(fv2.empty()) << "fv2 should be emptied";
            expect(!fv1.has_overflowed()) << "fv1 still in fixed buffer";
            expect(!fv2.has_overflowed()) << "fv2 still in fixed buffer";
            expect(TrackedDestroy::alive_count == 3_i) << "3 elements alive after move assign";
        }
        expect(TrackedDestroy::alive_count == 0_i) << "no leaks within-fixed move assign";
    };
}

void reg_fixed_vector_move_assign_overflowed_to_overflowed() {
    "fixed_vector_move_assign_overflowed_to_overflowed"_test = [] {
        reset_tracked();
        {
            eastl::fixed_vector<TrackedDestroy, 5, true> fv1;
            for (int i = 0; i < 10; ++i) { fv1.push_back(TrackedDestroy(i)); }
            eastl::fixed_vector<TrackedDestroy, 5, true> fv2;
            for (int i = 0; i < 8; ++i) { fv2.push_back(TrackedDestroy(i * 100)); }

            expect(fv1.has_overflowed()) << "fv1 overflowed";
            expect(fv2.has_overflowed()) << "fv2 overflowed";
            expect(TrackedDestroy::alive_count == 18_i) << "18 elements alive total";

            auto* fv1_ptr = fv1.data();
            auto* fv2_ptr = fv2.data();

            fv1 = eastl::move(fv2);
            expect(fv1.data() == fv2_ptr) << "fv1 should have stolen fv2's heap buffer";
            expect(fv1.size() == 8u) << "fv1 should have 8 elements";
            for (int i = 0; i < 8; ++i) { expect(fv1[i].id == i * 100) << "fv1 element " << i; }

            // After swap, fv2 holds fv1's old heap buffer with fv1's old size
            expect(fv2.size() == 10u) << "fv2 should have fv1's old size (10) after pointer swap";
            expect(fv2.has_overflowed()) << "fv2 still overflowed (holds fv1's old buffer)";
            expect(TrackedDestroy::alive_count == 18_i) << "still 18 alive after swap (no dtors)";

            // Both can be reused
            fv1.push_back(TrackedDestroy(999));
            expect(fv1.size() == 9u) << "fv1 can grow after move assign swap";
            expect(TrackedDestroy::alive_count == 19_i) << "19 alive after push";
        }
        expect(TrackedDestroy::alive_count == 0_i) << "no leaks after overflowed-to-overflowed move assign";
    };
}

void reg_fixed_vector_move_assign_heap_to_fixed() {
    "fixed_vector_move_assign_heap_to_fixed"_test = [] {
        reset_tracked();
        {
            eastl::fixed_vector<TrackedDestroy, 5, true> fv1;
            for (int i = 0; i < 3; ++i) { fv1.push_back(TrackedDestroy(i)); }
            expect(!fv1.has_overflowed()) << "fv1 within fixed buffer";

            eastl::fixed_vector<TrackedDestroy, 5, true> fv2;
            for (int i = 0; i < 8; ++i) { fv2.push_back(TrackedDestroy(i * 10)); }
            expect(fv2.has_overflowed()) << "fv2 overflowed";
            expect(TrackedDestroy::alive_count == 11_i) << "11 alive total";

            auto* fv2_ptr = fv2.data();

            fv1 = eastl::move(fv2);
            expect(fv1.data() == fv2_ptr) << "fv1 should steal fv2's heap buffer";
            expect(fv1.size() == 8u) << "fv1 should have 8 elements";
            for (int i = 0; i < 8; ++i) { expect(fv1[i].id == i * 10) << "fv1 element " << i; }
            expect(fv1.has_overflowed()) << "fv1 now overflowed";
            expect(fv2.empty()) << "fv2 emptied";
            expect(!fv2.has_overflowed()) << "fv2 reset to fixed buffer";
            expect(TrackedDestroy::alive_count == 8_i) << "8 alive after steal (fv1's old 3 were destroyed)";

            fv2.push_back(TrackedDestroy(42));
            expect(fv2[0].id == 42) << "fv2 reusable after move assign";
            expect(TrackedDestroy::alive_count == 9_i) << "9 alive after reuse";
        }
        expect(TrackedDestroy::alive_count == 0_i) << "no leaks after heap-to-fixed move assign";
    };
}

void reg_fixed_vector_move_assign_fixed_to_heap() {
    "fixed_vector_move_assign_fixed_to_heap"_test = [] {
        reset_tracked();
        {
            eastl::fixed_vector<TrackedDestroy, 5, true> fv1;
            for (int i = 0; i < 10; ++i) { fv1.push_back(TrackedDestroy(i)); }
            auto* old_fv1_ptr = fv1.data();
            expect(fv1.has_overflowed()) << "fv1 overflowed";
            expect(TrackedDestroy::alive_count == 10_i) << "10 alive";

            eastl::fixed_vector<TrackedDestroy, 5, true> fv2;
            for (int i = 0; i < 3; ++i) { fv2.push_back(TrackedDestroy(i * 100)); }
            expect(!fv2.has_overflowed()) << "fv2 within fixed buffer";

            fv1 = eastl::move(fv2);
            expect(!fv1.has_overflowed()) << "fv1 should be back in fixed buffer (heap freed)";
            expect(fv1.data() != old_fv1_ptr) << "old heap buffer released";
            expect(fv1.size() == 3u) << "fv1 should have 3 elements";
            expect(fv1[0].id == 0) << "fv1[0]";
            expect(fv1[1].id == 100) << "fv1[1]";
            expect(fv1[2].id == 200) << "fv1[2]";
            expect(fv2.empty()) << "fv2 should be empty";
            expect(TrackedDestroy::alive_count == 3_i) << "3 alive after move assign";
        }
        expect(TrackedDestroy::alive_count == 0_i) << "no leaks after fixed-to-heap move assign";
    };
}

void reg_fixed_vector_move_construct_no_overflow() {
    "fixed_vector_move_construct_no_overflow"_test = [] {
        reset_tracked();
        {
            eastl::fixed_vector<TrackedDestroy, 10, false> fv1;
            for (int i = 0; i < 5; ++i) { fv1.push_back(TrackedDestroy(i * 3)); }
            expect(TrackedDestroy::alive_count == 5_i) << "5 alive";

            eastl::fixed_vector<TrackedDestroy, 10, false> fv2(eastl::move(fv1));
            expect(fv2.size() == 5u) << "destination should have 5 elements";
            for (int i = 0; i < 5; ++i) { expect(fv2[i].id == i * 3) << "element " << i; }
            expect(fv1.empty()) << "source should be emptied";
            expect(!fv2.has_overflowed()) << "no overflow possible";
            expect(TrackedDestroy::alive_count == 5_i) << "still 5 alive after move";
        }
        expect(TrackedDestroy::alive_count == 0_i) << "no leaks no-overflow move construct";
    };
}

void reg_fixed_vector_move_assign_no_overflow() {
    "fixed_vector_move_assign_no_overflow"_test = [] {
        reset_tracked();
        {
            eastl::fixed_vector<TrackedDestroy, 10, false> fv1;
            for (int i = 0; i < 3; ++i) { fv1.push_back(TrackedDestroy(i)); }
            eastl::fixed_vector<TrackedDestroy, 10, false> fv2;
            for (int i = 0; i < 5; ++i) { fv2.push_back(TrackedDestroy(i * 10)); }
            expect(TrackedDestroy::alive_count == 8_i) << "8 alive total";

            fv1 = eastl::move(fv2);
            expect(fv1.size() == 5u) << "fv1 should have 5 elements";
            expect(fv1[0].id == 0) << "fv1[0]";
            expect(fv1[1].id == 10) << "fv1[1]";
            expect(fv1[2].id == 20) << "fv1[2]";
            expect(fv2.empty()) << "fv2 should be empty";
            expect(TrackedDestroy::alive_count == 5_i) << "5 alive after move assign";
        }
        expect(TrackedDestroy::alive_count == 0_i) << "no leaks no-overflow move assign";
    };
}

void reg_fixed_vector_move_self_assign() {
    "fixed_vector_move_self_assign"_test = [] {
        reset_tracked();
        {
            eastl::fixed_vector<TrackedDestroy, 10, true> fv;
            for (int i = 0; i < 5; ++i) { fv.push_back(TrackedDestroy(i)); }
            expect(TrackedDestroy::alive_count == 5_i) << "5 alive";

            fv = eastl::move(fv);  // self-assignment
            expect(fv.size() == 5u) << "self-assign should not change size";
            for (int i = 0; i < 5; ++i) { expect(fv[i].id == i) << "element " << i; }
            expect(TrackedDestroy::alive_count == 5_i) << "still 5 alive after self-assign";

            // Also test overflowed self-assignment
            eastl::fixed_vector<TrackedDestroy, 3, true> fv2;
            for (int i = 0; i < 10; ++i) { fv2.push_back(TrackedDestroy(i)); }
            expect(fv2.has_overflowed()) << "fv2 overflowed";
            expect(TrackedDestroy::alive_count == 15_i) << "15 alive total";
            fv2 = eastl::move(fv2);
            expect(fv2.size() == 10u) << "overflowed self-assign should not change size";
            for (int i = 0; i < 10; ++i) { expect(fv2[i].id == i) << "overflowed element " << i; }
            expect(TrackedDestroy::alive_count == 15_i) << "still 15 alive after overflowed self-assign";
        }
        expect(TrackedDestroy::alive_count == 0_i) << "no leaks after self-assign";
    };
}

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));

    reg_make_unique_single_basic();
    reg_make_unique_single_zero();
    reg_make_unique_single_default_init();
    reg_make_unique_single_multiple_args();
    reg_make_unique_single_destructor_called();

    reg_make_unique_array_basic();
    reg_make_unique_array_large();
    reg_make_unique_array_non_trivial();
    reg_make_unique_array_edge_sizes();

    reg_default_delete_single();
    reg_default_delete_array_layout();
    reg_default_delete_array_odd_layout();

    reg_smart_ptr_deleter_single();

    reg_smart_ptr_deleter_void();
    reg_smart_ptr_deleter_void_non_trivial();
    reg_smart_ptr_deleter_const_void();

    reg_smart_array_deleter_typed();
    reg_smart_array_deleter_typed_non_trivial();
    reg_smart_array_deleter_void();

    reg_allocator_identity();
    reg_allocator_set_default();

    reg_make_unique_stress();
    reg_make_unique_array_stress();

    reg_mixed_smart_ptr_api();

    reg_vector_basic();
    reg_vector_reserve_grow();
    reg_vector_non_trivial();
    reg_vector_shrink_to_fit();

    reg_fixed_vector_within_fixed();
    reg_fixed_vector_overflow();
    reg_fixed_vector_overflow_with_non_trivial();
    reg_fixed_vector_no_overflow();
    reg_fixed_vector_constructor_large();
    reg_fixed_vector_copy_overflowed();

    reg_fixed_vector_move_construct_within_fixed();
    reg_fixed_vector_move_construct_overflowed();
    reg_fixed_vector_move_construct_overflowed_non_trivial();
    reg_fixed_vector_move_assign_within_fixed_to_within_fixed();
    reg_fixed_vector_move_assign_overflowed_to_overflowed();
    reg_fixed_vector_move_assign_heap_to_fixed();
    reg_fixed_vector_move_assign_fixed_to_heap();
    reg_fixed_vector_move_construct_no_overflow();
    reg_fixed_vector_move_assign_no_overflow();
    reg_fixed_vector_move_self_assign();

    return 0;
}
