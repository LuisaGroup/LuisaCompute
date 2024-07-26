#pragma once
#include <luisa/vstl/common.h>
#include <luisa/vstl/stack_allocator.h>
#include <luisa/vstl/lockfree_array_queue.h>
#include <luisa/core/basic_types.h>
#include <luisa/ast/expression.h>
namespace lc::toy_c {
class LCStream;
class LCDevice;
}// namespace lc::toy_c
struct MemoryManager {
    struct Context {
        vstd::VEngineMallocVisitor alloc;
        vstd::StackAllocator temp_alloc;
        lc::toy_c::LCDevice* device;
        lc::toy_c::LCStream* stream;
        luisa::string_view print_format;
        luisa::vector<luisa::compute::detail::LiteralValueVariant> print_values;
        Context() : temp_alloc(1024ull * 1024ull, &alloc, 2) {}
    };
    MemoryManager();
    ~MemoryManager();
    vstd::Pool<Context, false> pool;
    luisa::spin_mutex alloc_mtx;
    vstd::vector<Context *> ctx;
    static Context *get_tlocal_ctx();
    void alloc_tlocal_ctx();
    void dealloc_tlocal_ctx();
};