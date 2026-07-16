// Test for the type system and core utility classes.
// This comprehensive test covers:
// - Type introspection (scalars, vectors, matrices, arrays, structures)
// - Vector and matrix operations
// - Managed pointer system with reference counting
// - Intrusive linked list containers
// - Serialization support

#include <string>
#include <fstream>
#include <memory>
#include <variant>
#include <atomic>
#include <iostream>

#include <luisa/luisa-compute.h>
#include "ut/ut.hpp"

// Simple struct types for testing
struct S1 {
    float x;
};

struct S2 {
    float x;
    float y;
};

struct S3 {
    float x;
    float y;
    float z;
};

struct S4 {
    float x;
    float y;
    float z;
    float w;
};

// Test struct with serialization support
struct Test {
    std::string s;
    int a;

    template<typename Archive>
    void serialize(Archive &&ar) noexcept {
        ar(s, a);
    }
};

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;
using namespace boost::ut::literals;

// Test struct with 16-byte alignment for layout testing
struct alignas(16) AA {
    float4 x;
    float ba[16];
    float a;
};

// Nested test struct with matrix member
struct BB {
    AA a;
    float b;
    float3x3 m;
};

LUISA_STRUCT_REFLECT(AA, x, ba, a)
LUISA_STRUCT_REFLECT(BB, a, b, m)

// Non-copyable interface base class
struct Interface : public luisa::concepts::Noncopyable {
    Interface() noexcept = default;
    Interface(Interface &&) noexcept = default;
    Interface &operator=(Interface &&) noexcept = default;
    ~Interface() noexcept = default;
};

// Concept-constrained function for container types
template<typename T>
    requires luisa::concepts::container<T>
void foo(T &&) noexcept {}

// Implementation of interface
struct Impl : public Interface {};

// Managed object for reference counting tests
class Something : public luisa::Managed<Something> {};

// Value type for intrusive list nodes
struct SomeValue : Something {
    int value;
    explicit SomeValue(int v = -1) noexcept : value{v} {}
    ~SomeValue() noexcept override { LUISA_INFO("SomeValue destroyed with value: {}", value); }
};

// Doubly-linked intrusive list node
struct SomeNode : public luisa::ManagedIntrusiveNode<SomeNode, SomeValue> {
    using Super::Super;
};

// Singly-linked intrusive list node
struct SomeForwardNode : public luisa::ManagedIntrusiveForwardNode<SomeForwardNode, SomeValue> {
    using Super::Super;
};

// Convert type tag to human-readable string
std::string_view tag_name(Type::Tag tag) noexcept {
    using namespace std::string_view_literals;
    if (tag == Type::Tag::BOOL) { return "bool"sv; }
    if (tag == Type::Tag::FLOAT32) { return "float"sv; }
    if (tag == Type::Tag::INT32) { return "int"sv; }
    if (tag == Type::Tag::UINT32) { return "uint"sv; }
    if (tag == Type::Tag::VECTOR) { return "vector"sv; }
    if (tag == Type::Tag::MATRIX) { return "matrix"sv; }
    if (tag == Type::Tag::ARRAY) { return "array"sv; }
    if (tag == Type::Tag::STRUCTURE) { return "struct"sv; }
    return "unknown"sv;
}

// Print type information recursively
template<int max_level = -1>
void print(const Type *info, int level = 0) {
    std::string indent_string;
    for (auto i = 0; i < level; i++) { indent_string.append("  "); }
    if (max_level >= 0 && level > max_level) {
        std::cout << indent_string << info->description() << "\n";
        return;
    }

    std::cout << indent_string << tag_name(info->tag()) << ": {\n"
              << indent_string << "  size:        " << info->size() << "\n"
              << indent_string << "  alignment:   " << info->alignment() << "\n"
              << indent_string << "  hash:        " << info->hash() << "\n"
              << indent_string << "  description: " << info->description() << "\n";

    if (info->is_structure()) {
        std::cout << indent_string << "  members:\n";
        for (auto m : info->members()) { print<max_level>(m, level + 2); }
    } else if (info->is_vector() || info->is_array() || info->is_matrix()) {
        std::cout << indent_string << "  dimension:   " << info->dimension() << "\n";
        std::cout << indent_string << "  element:\n";
        print<max_level>(info->element(), level + 2);
    }
    std::cout << indent_string << "}\n";
}

static auto test_type_registration = [] {
    "test_type"_test = [] {
        using namespace luisa;
        log_level_verbose();

        // Test logging macros
        LUISA_VERBOSE("verbose...");
        LUISA_VERBOSE_WITH_LOCATION("verbose with {}...", "location");
        LUISA_INFO("info...");
        LUISA_INFO_WITH_LOCATION("info with location...");
        LUISA_WARNING("warning...");
        LUISA_WARNING_WITH_LOCATION("warning with location...");

        // Test struct sizes and alignments
        LUISA_INFO("size = {}, alignment = {}", sizeof(AA), alignof(AA));
        LUISA_INFO("size = {}, alignment = {}", sizeof(BB), alignof(BB));
        LUISA_INFO("trivially destructible: {}", std::is_trivially_destructible_v<Impl>);

        // Test type parsing from string
        print(Type::from("array<array<vector<float,3>,5>,9>"));

        LUISA_INFO("{}", Type::of<std::array<float, 5>>()->description());

        // Test array type deduction
        int aa[1024];
        print(Type::of(aa));

        // Test struct type introspection
        BB bb;
        print(Type::of(bb));

        // Verify vector alignment
        static_assert(alignof(float3) == 16);

        // Test vector construction and operations
        auto u = make_float2(1.0f, 2.0f);
        auto v = make_float3(1.0f, 2.0f, 3.0f);
        auto w = make_float3(u, 1.0f);

        auto vv = v + w;
        auto bvv = v == w;
        static_assert(std::is_same_v<decltype(bvv), bool3>);
        v += w;
        v *= 10.0f;

        v = 2.0f * v;
        auto ff = v[2];
        ff = 1.0f;
        auto tt = make_float2(v);

        // Test matrix type
        print(Type::of<float3x3>());

        // Test container concept
        foo<std::initializer_list<int>>({1, 2, 3, 4});

        // Test structured bindings
        auto [m, n] = std::array{1, 2};

        // Test managed pointer system
        auto sth = luisa::make_managed<Something>();
        sth = sth;
        sth = std::move(sth);
        {
            auto another = sth;
            luisa::ManagedPtr<const Something> good = std::move(another);
            expect(static_cast<bool>(nullptr == another));
            auto gg = good.get();
            expect(static_cast<bool>(gg == sth));
            auto ggg = gg->lock();
            auto more = good->lock();
            expect(static_cast<bool>(sth == more));
            more = std::move(ggg);
            expect(static_cast<bool>(more != nullptr));
            good = more;
            expect(static_cast<bool>(good));
            another = sth;
        }

        {
            luisa::ManagedPtr<const luisa::detail::ManagedObject> bad = std::move(sth);
            auto worse = std::move(bad).into<Something>();
        }

        luisa::unordered_set<luisa::ManagedPtr<Something>> set;

        // Test doubly-linked intrusive list
        LUISA_INFO("Begin managed intrusive list test...");
        {
            luisa::ManagedIntrusiveList<SomeNode> list;
            auto n1 = list.push_front(make_managed<SomeNode>(1));// [1]
            auto n2 = list.push_back(make_managed<SomeNode>(2)); // [1, 2]
            auto n3 = list.push_front(make_managed<SomeNode>(3));// [3, 1, 2]
            auto n4 = list.push_back(make_managed<SomeNode>(4)); // [3, 1, 2, 4]
            {
                auto rm_n2 = n2->remove_self();// [3, 1, 4]
                expect(static_cast<bool>(!rm_n2->is_linked()));
            }
            auto n5 = n3->insert_after_self(make_managed<SomeNode>(5)); // [3, 5, 1, 4]
            auto n6 = n5->insert_before_self(make_managed<SomeNode>(6));// [3, 6, 5, 1, 4]
            {
                auto rm_n3 = n3->remove_self();// [6, 5, 1, 4]
                expect(static_cast<bool>(!rm_n3->is_linked()));
                auto rm_n4 = n4->remove_self();// [6, 5, 1]
                expect(static_cast<bool>(!rm_n4->is_linked()));
            }
            for (auto node : list) {
                LUISA_INFO("Node value: {}", node->value);
            }
            for (auto iter = list.crbegin(); iter != list.crend(); ++iter) {
                LUISA_INFO("Reverse Node value: {}", (*iter)->value);
            }
        }
        LUISA_INFO("End managed intrusive list test...");

        // Test singly-linked intrusive list
        LUISA_INFO("Begin managed intrusive forward list test...");
        {
            luisa::ManagedIntrusiveForwardList<SomeForwardNode> list;
            auto n1 = list.push_front(make_managed<SomeForwardNode>(1)->lock_into<SomeForwardNode>());// [1]
            auto n2 = list.push_front(make_managed<SomeForwardNode>(2));                              // [2, 1]
            auto n3 = list.push_front(make_managed<SomeForwardNode>(3));                              // [3, 2, 1]
            auto n4 = list.push_front(make_managed<SomeForwardNode>(4));                              // [4, 3, 2, 1]
            {
                auto rm_n2 = n2->remove_self();// [4, 3, 1]
                expect(static_cast<bool>(!rm_n2->is_linked()));
            }
            auto n5 = list.push_front(make_managed<SomeForwardNode>(5));// [5, 4, 3, 1]
            auto n6 = list.push_front(make_managed<SomeForwardNode>(6));// [6, 5, 4, 3, 1]
            {
                auto rm_n1 = n1->remove_self();// [6, 5, 4, 3]
                expect(static_cast<bool>(!rm_n1->is_linked()));
                auto rm_n6 = n6->remove_self();// [5, 4, 3]
                expect(static_cast<bool>(!rm_n6->is_linked()));
            }
            auto n7 = list.push_front(make_managed<SomeForwardNode>(7));// [7, 5, 4, 3]
            auto n8 = list.push_front(make_managed<SomeForwardNode>(8));// [8, 7, 5, 4, 3]
            {
                auto rm_n5 = n5->remove_self();// [8, 7, 4, 3]
                expect(static_cast<bool>(!rm_n5->is_linked()));
            }
            for (auto node : list) {
                LUISA_INFO("Node value: {}", node->value);
            }
        }
        LUISA_INFO("End managed intrusive forward list test...");
    };
    return 0;
}();
int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));

}
