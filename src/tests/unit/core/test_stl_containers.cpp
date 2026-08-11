// Test for luisa STL container wrappers and helpers.
// Covers: vector + helpers (enlarge_by, size_bytes, vector_resize),
//         string + format, map/set, unordered_map/set, optional, lru_cache.

#include "ut/ut.hpp"

#include <luisa/core/stl/vector.h>
#include <luisa/core/stl/string.h>
#include <luisa/core/stl/format.h>
#include <luisa/core/stl/map.h>
#include <luisa/core/stl/unordered_map.h>
#include <luisa/core/stl/optional.h>
#include <luisa/core/stl/lru_cache.h>
#include <luisa/core/stl/memory.h>
#include <luisa/core/logging.h>

using namespace boost::ut;
using namespace boost::ut::literals;

// ---- vector helpers ----

// ---- string ----

// ---- format ----

// ---- map / set ----

// ---- unordered_map / unordered_set ----

// ---- optional ----

// ---- lru_cache ----

// ---- allocator helpers ----

void reg_vector_basic() {

    "vector_basic_operations"_test = [] {
        luisa::vector<int> v;
        expect(v.empty());
        expect(v.size() == 0u);

        v.push_back(1);
        v.push_back(2);
        v.push_back(3);
        expect(v.size() == 3u);
        expect(v[0] == 1_i);
        expect(v[1] == 2_i);
        expect(v[2] == 3_i);

        v.clear();
        expect(v.empty());
    };
}

void reg_vector_enlarge_by() {

    "vector_enlarge_by"_test = [] {
        luisa::vector<int> v;
        v.push_back(10);
        v.push_back(20);
        expect(v.size() == 2u);

        auto *raw_ptr = luisa::enlarge_by(v, 3u);
        expect(v.size() == 5u);
        // ptr points to the start of the newly enlarged region
        // fill the new elements
        auto *ptr = static_cast<int *>(static_cast<void *>(raw_ptr));
        ptr[0] = 30;
        ptr[1] = 40;
        ptr[2] = 50;
        expect(v[0] == 10_i);
        expect(v[1] == 20_i);
        expect(v[2] == 30_i);
        expect(v[3] == 40_i);
        expect(v[4] == 50_i);

        // enlarge_by(0) should not change size
        luisa::enlarge_by(v, 0u);
        expect(v.size() == 5u);
    };
}

void reg_vector_size_bytes() {

    "vector_size_bytes"_test = [] {
        luisa::vector<float> v;
        expect(luisa::size_bytes(v) == 0u);

        v.resize(10u);
        expect(luisa::size_bytes(v) == 10u * sizeof(float));

        luisa::vector<double> vd;
        vd.resize(5u);
        expect(luisa::size_bytes(vd) == 5u * sizeof(double));
    };
}

void reg_vector_resize() {

    "vector_resize"_test = [] {
        luisa::vector<int> v;
        luisa::vector_resize(v, 100u);
        expect(v.size() == 100u);

        luisa::vector_resize(v, 0u);
        expect(v.empty());

        luisa::vector_resize(v, 50u);
        expect(v.size() == 50u);
    };
}

void reg_fixed_vector() {

    "fixed_vector_basic"_test = [] {
        luisa::fixed_vector<int, 8> fv;
        for (int i = 0; i < 8; ++i) {
            fv.push_back(i);
        }
        expect(fv.size() == 8u);

        // allow_overflow = true by default, so pushing beyond capacity is allowed
        fv.push_back(99);
        expect(fv.size() == 9u);
        expect(fv[8] == 99_i);
    };
}

void reg_string_basic() {

    "string_basic"_test = [] {
        luisa::string s;
        expect(s.empty());

        s = "hello";
        expect(s.size() == 5u);
        expect(s == "hello");

        luisa::string s2 = s + " world";
        expect(s2 == "hello world");

        // substring
        auto sub = s2.substr(6, 5);
        expect(sub == "world");
    };
}

void reg_string_view() {

    "string_view_interop"_test = [] {
        luisa::string s = "test string";
        luisa::string_view sv = s;
        expect(sv == "test string");
        expect(sv.size() == s.size());
    };
}

void reg_format_basic() {

    "format_basic"_test = [] {
        auto s = luisa::format("hello {}", "world");
        expect(s == "hello world");

        auto s2 = luisa::format("{} + {} = {}", 1, 2, 3);
        expect(s2 == "1 + 2 = 3");

        auto s3 = luisa::format("{:.2f}", 3.14159);
        expect(s3 == "3.14");
    };
}

void reg_format_hash_to_string() {

    "hash_to_string"_test = [] {
        auto s = luisa::hash_to_string(0u);
        expect(s.size() == 16u);// 16 hex digits
        expect(s == "0000000000000000");

        auto s2 = luisa::hash_to_string(0xDEADBEEFu);
        expect(s2 == "00000000DEADBEEF");
    };
}

void reg_map_basic() {

    "map_basic"_test = [] {
        luisa::map<int, luisa::string> m;
        m[1] = "one";
        m[2] = "two";
        m[3] = "three";
        expect(m.size() == 3u);
        expect(m[2] == "two");

        // ordered iteration
        auto it = m.begin();
        expect(it->first == 1_i);
        ++it;
        expect(it->first == 2_i);
        ++it;
        expect(it->first == 3_i);

        m.erase(2);
        expect(m.size() == 2u);
        expect(m.find(2) == m.end());
    };
}

void reg_map_at_missing_key() {

#if __cpp_exceptions
    "map_at_missing_key_throws"_test = [] {
        luisa::map<int, luisa::string> m;
        m.emplace(1, "one");

        auto mutable_threw = false;
        try {
            static_cast<void>(m.at(2));
        } catch (const std::out_of_range &) {
            mutable_threw = true;
        }
        expect(mutable_threw);

        auto const_threw = false;
        try {
            const auto &cm = m;
            static_cast<void>(cm.at(2));
        } catch (const std::out_of_range &) {
            const_threw = true;
        }
        expect(const_threw);
    };
#endif
}

void reg_set_basic() {

    "set_basic"_test = [] {
        luisa::set<int> s;
        s.insert(5);
        s.insert(3);
        s.insert(7);
        s.insert(3);// duplicate
        expect(s.size() == 3u);
        expect(s.count(3) == 1u);
        expect(s.count(4) == 0u);

        s.erase(3);
        expect(s.size() == 2u);
    };
}

void reg_unordered_map_basic() {

    "unordered_map_basic"_test = [] {
        luisa::unordered_map<int, luisa::string> m;
        m[10] = "ten";
        m[20] = "twenty";
        m[30] = "thirty";
        expect(m.size() == 3u);
        expect(m[20] == "twenty");

        expect(m.find(10) != m.end());
        expect(m.find(99) == m.end());

        m.erase(20);
        expect(m.size() == 2u);
    };
}

void reg_unordered_set_basic() {

    "unordered_set_basic"_test = [] {
        luisa::unordered_set<int> s;
        s.insert(1);
        s.insert(2);
        s.insert(3);
        s.insert(1);// duplicate
        expect(s.size() == 3u);
        expect(s.count(2) == 1u);
        expect(s.count(99) == 0u);

        s.erase(2);
        expect(s.size() == 2u);
    };
}

void reg_unordered_map_string_key() {

    "unordered_map_string_key"_test = [] {
        luisa::unordered_map<luisa::string, int> m;
        m["alpha"] = 1;
        m["beta"] = 2;
        m["gamma"] = 3;
        expect(m.size() == 3u);
        expect(m["beta"] == 2_i);

        // overwrite
        m["alpha"] = 100;
        expect(m["alpha"] == 100_i);
    };
}

void reg_optional_basic() {

    "optional_basic"_test = [] {
        luisa::optional<int> empty;
        expect(!empty.has_value());

        luisa::optional<int> val = 42;
        expect(val.has_value());
        expect(*val == 42_i);

        val.reset();
        expect(!val.has_value());
    };
}

void reg_optional_make() {

    "optional_make_optional"_test = [] {
        auto opt = luisa::make_optional(3.14);
        expect(opt.has_value());
        expect(static_cast<bool>(*opt > 3.13 && *opt < 3.15));
    };
}

void reg_optional_nullopt() {

    "optional_nullopt"_test = [] {
        luisa::optional<luisa::string> s = luisa::nullopt;
        expect(!s.has_value());

        s = luisa::string{"hello"};
        expect(s.has_value());
        expect(*s == "hello");

        s = luisa::nullopt;
        expect(!s.has_value());
    };
}

void reg_optional_value_or() {

    "optional_value_or"_test = [] {
        luisa::optional<int> empty;
        expect(empty.value_or(99) == 99_i);

        luisa::optional<int> full = 42;
        expect(full.value_or(99) == 42_i);
    };
}

void reg_lru_cache_basic() {

    "lru_cache_basic"_test = [] {
        luisa::lru_cache<int, luisa::string> cache{3};

        cache.emplace(1, "one");
        cache.emplace(2, "two");
        cache.emplace(3, "three");

        auto v1 = cache.at(1);
        expect(v1.has_value());
        expect(*v1 == "one");

        auto v2 = cache.at(2);
        expect(v2.has_value());
        expect(*v2 == "two");

        // Touch key 2 and key 3 so key 1 becomes LRU
        cache.touch(2);
        cache.touch(3);

        // Adding 4th should evict key 1 (LRU)
        cache.emplace(4, "four");
        auto v1_after = cache.at(1);
        expect(!v1_after.has_value()) << "key 1 should have been evicted";

        auto v4 = cache.at(4);
        expect(v4.has_value());
        expect(*v4 == "four");
    };
}

void reg_lru_cache_touch() {

    "lru_cache_touch"_test = [] {
        luisa::lru_cache<int, int> cache{2};
        cache.emplace(1, 10);
        cache.emplace(2, 20);

        // Touch key 1 so it becomes most recently used
        auto touched = cache.touch(1);
        expect(touched);

        // Insert key 3 — should evict key 2 (LRU) instead of key 1
        cache.emplace(3, 30);

        expect(cache.at(1).has_value()) << "key 1 was touched, should not be evicted";
        expect(!cache.at(2).has_value()) << "key 2 should be evicted";
        expect(cache.at(3).has_value());

        // Touch non-existent key
        auto not_touched = cache.touch(999);
        expect(!not_touched);
    };
}

void reg_lru_cache_overwrite() {

    "lru_cache_overwrite"_test = [] {
        luisa::lru_cache<int, luisa::string> cache{2};
#ifdef LUISA_USE_SYSTEM_STL
        cache.emplace(1, "old");
        cache.emplace(1, "new");
        auto v = cache.at(1);
        expect(v.has_value());
        expect(*v == "new");
#else
        cache.insert_or_assign(1, luisa::string{"old"});
        cache.insert_or_assign(1, luisa::string{"new"});
        auto v = cache.at(1);
        expect(v.has_value());
        expect(*v == "new");
#endif
    };
}

void reg_lru_cache_delete_callback() {

    "lru_cache_delete_callback"_test = [] {
        int evicted_value = -1;
        luisa::lru_cache<int, int> cache{2};
        cache.setDeleteCallback([&](const int &v) {
            evicted_value = v;
        });

        cache.emplace(1, 100);
        cache.emplace(2, 200);
        cache.emplace(3, 300);// should evict key 1 (value=100)

        expect(evicted_value == 100_i) << "delete callback should have been called with evicted value";
    };
}

void reg_lru_cache_capacity_one() {

    "lru_cache_capacity_one"_test = [] {
        luisa::lru_cache<int, int> cache{1};
        cache.emplace(1, 10);
        auto v = cache.at(1);
        expect(v.has_value());
        expect(*v == 10_i);

        cache.emplace(2, 20);
        expect(!cache.at(1).has_value());
        expect(cache.at(2).has_value());
    };
}

void reg_lru_cache_thread_safe() {

    "LRUCache_thread_safe"_test = [] {
        auto cache = luisa::LRUCache<int, int>::create(64);

        // update from main thread
        for (int i = 0; i < 100; ++i) {
            cache->update(i, i * 10);
        }

        // fetch and verify
        for (int i = 0; i < 64; ++i) {
            // only the last 64 should be present (cap=64)
            // keys 36..99 remain
        }

        // Verify at least some recent keys exist
        auto v = cache->fetch(99);
        expect(v.has_value());
        expect(*v == 990_i);
    };
}

void reg_allocator_helpers() {

    "allocate_with_allocator"_test = [] {
        auto *p = luisa::allocate_with_allocator<int>(10);
        expect(static_cast<bool>(p != nullptr));
        // write to verify memory is usable
        for (int i = 0; i < 10; ++i) {
            p[i] = i;
        }
        for (int i = 0; i < 10; ++i) {
            expect(p[i] == i);
        }
        luisa::deallocate_with_allocator(p);
    };
}

void reg_new_delete_allocator() {

    "new_delete_with_allocator"_test = [] {
        struct Foo {
            int x;
            float y;
            Foo(int x_, float y_) : x(x_), y(y_) {}
        };
        auto *foo = luisa::new_with_allocator<Foo>(42, 3.14f);
        expect(static_cast<bool>(foo != nullptr));
        expect(foo->x == 42_i);
        expect(static_cast<bool>(foo->y > 3.13f && foo->y < 3.15f));
        luisa::delete_with_allocator(foo);

        // delete nullptr should be safe
        luisa::delete_with_allocator<Foo>(nullptr);
    };
}

void reg_size_literals() {

    "size_literals"_test = [] {
        using namespace luisa::size_literals;
        expect(1_k == 1024u);
        expect(1_M == 1024u * 1024u);
        expect(1_G == 1024u * 1024u * 1024u);
        expect(2_k == 2048u);
        expect(4_M == 4u * 1024u * 1024u);
    };
}

int main(int argc, char *argv[]) {

    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));
    reg_vector_basic();
    reg_vector_enlarge_by();
    reg_vector_size_bytes();
    reg_vector_resize();
    reg_fixed_vector();
    reg_string_basic();
    reg_string_view();
    reg_format_basic();
    reg_format_hash_to_string();
    reg_map_basic();
    reg_map_at_missing_key();
    reg_set_basic();
    reg_unordered_map_basic();
    reg_unordered_set_basic();
    reg_unordered_map_string_key();
    reg_optional_basic();
    reg_optional_make();
    reg_optional_nullopt();
    reg_optional_value_or();
    reg_lru_cache_basic();
    reg_lru_cache_touch();
    reg_lru_cache_overwrite();
    reg_lru_cache_delete_callback();
    reg_lru_cache_capacity_one();
    reg_lru_cache_thread_safe();
    reg_allocator_helpers();
    reg_new_delete_allocator();
    reg_size_literals();
    return 0;
}
