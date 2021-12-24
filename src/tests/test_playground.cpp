//
// Created by Mike Smith on 2021/9/3.
//

#include <iostream>
#include <EASTL/vector.h>
#include <nlohmann/json.hpp>
#include <ast/op.h>
#include <core/logging.h>
#include <core/first_fit.h>
#include <core/thread_pool.h>

int main() {

    using namespace luisa;
    using namespace luisa::compute;

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
        thread_pool.dispatch([]{ LUISA_INFO("Sub-hello!"); });
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
}
