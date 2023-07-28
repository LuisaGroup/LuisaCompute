#include <string>
#include <fstream>
#include <memory>
#include <variant>
#include <atomic>
#include <iostream>

#include <spdlog/spdlog.h>

#include <luisa/core/logging.h>
#include <luisa/core/stl/hash.h>
#include <luisa/ast/type.h>
#include <luisa/ast/type_registry.h>

#include <luisa/ast/expression.h>
#include <luisa/ast/statement.h>
#include <luisa/ast/variable.h>

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

struct alignas(16) AA {
    float4 x;
    float ba[16];
    float a;
};

struct BB {
    AA a;
    float b;
    float3x3 m;
};

LUISA_STRUCT_REFLECT(AA, x, ba, a)
LUISA_STRUCT_REFLECT(BB, a, b, m)

struct Interface : public concepts::Noncopyable {
    Interface() noexcept = default;
    Interface(Interface &&) noexcept = default;
    Interface &operator=(Interface &&) noexcept = default;
    ~Interface() noexcept = default;
};

template<typename T>
requires concepts::container<T> void foo(T &&) noexcept {}

struct Impl : public Interface {};

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

int main() {

    using namespace luisa;
    log_level_verbose();

    LUISA_VERBOSE("verbose...");
    LUISA_VERBOSE_WITH_LOCATION("verbose with {}...", "location");
    LUISA_INFO("info...");
    LUISA_INFO_WITH_LOCATION("info with location...");
    LUISA_WARNING("warning...");
    LUISA_WARNING_WITH_LOCATION("warning with location...");

    LUISA_INFO("size = {}, alignment = {}", sizeof(AA), alignof(AA));
    LUISA_INFO("size = {}, alignment = {}", sizeof(BB), alignof(BB));

    auto &&type_aa = typeid(AA);
    auto &&type_bb = typeid(BB);
    LUISA_INFO("{}", type_aa.before(type_bb));

    LUISA_INFO("trivially destructible: {}", std::is_trivially_destructible_v<Impl>);

    print(Type::from("array<array<vector<float,3>,5>,9>"));

    auto hash_aa = luisa::hash_value(type_aa.name());
    auto hash_bb = luisa::hash_value(type_bb.name());
    LUISA_INFO("{} {}", hash_aa, hash_bb);

    LUISA_INFO("{}", Type::of<std::array<float, 5>>()->description());

    int aa[1024];
    print(Type::of(aa));

    BB bb;
    print(Type::of(bb));

    static_assert(alignof(float3) == 16);

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

    print(Type::of<float3x3>());

    foo<std::initializer_list<int>>({1, 2, 3, 4});

    auto [m, n] = std::array{1, 2};
}

