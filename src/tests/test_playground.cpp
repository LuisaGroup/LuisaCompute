//
// Created by Mike Smith on 2021/9/3.
//

#include <iostream>
#include <luisa-compute.h>

using namespace luisa;
using namespace luisa::compute;

struct Good {
    float3 f;
    bool x;
    uint y;
};

struct Test {
    Good good;
    float4 bad;
    bool4 not_bad;
};

// align: 8B, size: 16B
struct StructInCpp {
    float2 a;// align: 8B, size: 8B
    bool2 b; // align: 2B, size: 2B
};

// align: 16B, size: 32B
struct StructInISPC {
    float2 a;// align: 16B, size: 16B
    bool2 b; // align: 16B, size: 16B
};

LUISA_STRUCT(Good, f, x, y){};
LUISA_STRUCT(Test, good, bad, not_bad){};

void structured_buffer_read_impl(luisa::string &s, size_t offset, std::string_view self, const Type *t) noexcept {
    auto offset_uint = offset / 4u;
    if (t->description() == "bool") {
        s.append(luisa::format(
            "  {} = *({} *)(ptr + o + {}) & (0xffu << {}u);\n",
            self, t->description(), offset_uint, (offset - offset_uint * 4u) * 8u));
    } else if (t->is_scalar()) {
        s.append(luisa::format(
            "  {} = *({} *)(ptr + o + {});\n",
            self, t->description(), offset_uint));
    } else if (t->is_vector()) {
        for (auto i = 0u; i < t->dimension(); i++) {
            structured_buffer_read_impl(
                s, offset + t->element()->size() * i,
                luisa::format("{}.{}", self, "xyzw"[i]),
                t->element());
        }
    } else if (t->is_array()) {
        for (auto i = 0u; i < t->dimension(); i++) {
            structured_buffer_read_impl(
                s, offset + t->element()->size() * i,
                luisa::format("{}[{}]", self, i),
                t->element());
        }
    } else if (t->is_matrix()) {
        auto v = Type::from(luisa::format(
            "vector<{},{}>",
            t->element()->description(),
            t->dimension()));
        for (auto i = 0u; i < t->dimension(); i++) {
            structured_buffer_read_impl(
                s, offset + v->size() * i,
                luisa::format("{}[{}]", self, i), v);
        }
    } else if (t->is_structure()) {
        for (auto i = 0u; i < t->members().size(); i++) {
            auto m = t->members()[i];
            auto ma = m->alignment();
            offset = (offset + ma - 1u) / ma * ma;
            structured_buffer_read_impl(
                s, offset,
                luisa::format("{}.m{}", self, i), m);
            offset += m->size();
        }
    } else {
        LUISA_ERROR_WITH_LOCATION(
            "Invalid type: {}.",
            t->description());
    }
}

[[nodiscard]] auto structured_buffer_read(const Type *type) noexcept {
    auto s = luisa::format(
        "{} lc_buffer_read_{:016x}(const uint *ptr, int index) {{\n"
        "  {} v;\n"
        "  uint o = index * {};\n",
        type->description(), type->hash(), type->description(), type->size() / 4u);
    structured_buffer_read_impl(s, 0u, "v", type);
    s.append("  return v;\n}\n");
    return s;
}

int main() {

    LUISA_INFO("{}", luisa::format("Hello: {}!", 42));

    CallOpSet ops;
    ops.mark(CallOp::BINDLESS_BUFFER_READ);
    ops.mark(CallOp::SELECT);
    ops.mark(CallOp::ACOS);

    auto properties = nlohmann::json::object();
    LUISA_INFO("is_object: {}", properties.is_object());
    LUISA_INFO("Properties: {}", properties.dump());

    auto json = nlohmann::json::parse("{}");
    LUISA_INFO("Index: {}", json.value("index", 0));
    for (auto op : ops) {
        LUISA_INFO("Op: {}", to_underlying(op));
    }

    FirstFit ff{1024u, 16u};
    LUISA_INFO("free_list = {}.", ff.dump_free_list());
    auto n0 = ff.allocate(13);
    LUISA_INFO("n0 = [{}, {}), free_list = {}.", n0->offset(), n0->offset() + n0->size(), ff.dump_free_list());
    auto n1 = ff.allocate(256u);
    LUISA_INFO("n1 = [{}, {}), free_list = {}.", n1->offset(), n1->offset() + n1->size(), ff.dump_free_list());
    auto n2 = ff.allocate(32u);
    LUISA_INFO("n2 = [{}, {}), free_list = {}.", n2->offset(), n2->offset() + n2->size(), ff.dump_free_list());
    auto n3 = ff.allocate(64u);
    LUISA_INFO("n3 = [{}, {}), free_list = {}.", n3->offset(), n3->offset() + n3->size(), ff.dump_free_list());
    ff.free(n1);
    LUISA_INFO("free_list = {}.", ff.dump_free_list());
    ff.free(n0);
    LUISA_INFO("free_list = {}.", ff.dump_free_list());
    ff.free(n3);
    LUISA_INFO("free_list = {}.", ff.dump_free_list());
    ff.free(n2);
    LUISA_INFO("free_list = {}.", ff.dump_free_list());

    auto &&thread_pool = ThreadPool::global();
    auto f1 = thread_pool.dispatch([] {
        LUISA_INFO("Hello!");
        return 1234;
    });
    thread_pool.barrier();
    thread_pool.dispatch([&thread_pool, f = std::move(f1)]() mutable noexcept {
        LUISA_INFO("Hello: {}!", f.get());
        thread_pool.dispatch([] { LUISA_INFO("Sub-hello!"); });
    });
    thread_pool.parallel(2, 2, [](auto x, auto y) noexcept {
        std::ostringstream oss;
        oss << std::this_thread::get_id();
        LUISA_INFO("Hello from thread {}: ({}, {}).", oss.str(), x, y);
    });
    thread_pool.barrier();
    thread_pool.parallel(4, 4, [](auto x, auto y) noexcept {
        std::ostringstream oss;
        oss << std::this_thread::get_id();
        LUISA_INFO("Bye from thread {}: ({}, {}).", oss.str(), x, y);
    });
    thread_pool.synchronize();

    eastl::vector<int> a;
    a.emplace_back(0);

    LUISA_INFO(
        "Generated structured buffer read function:\n{}",
        structured_buffer_read(Type::of<Test>()));
}
