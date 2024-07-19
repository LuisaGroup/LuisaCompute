#include "memory_manager.h"
static thread_local MemoryManager::Context *manager_ctx = nullptr;
auto MemoryManager::get_tlocal_ctx() -> Context * {
    return manager_ctx;
};
MemoryManager::MemoryManager() : pool(std::thread::hardware_concurrency(), true) {}
MemoryManager::~MemoryManager() {}

void MemoryManager::alloc_tlocal_ctx() {
    if (manager_ctx) {
        return;
    }
    {
        std::lock_guard lck{alloc_mtx};
        if (!ctx.empty()) {
            auto v = ctx.back();
            ctx.pop_back();
            manager_ctx = v;
            return;
        }
    }
    manager_ctx = pool.create_lock(alloc_mtx);
}
void MemoryManager::dealloc_tlocal_ctx() {
    if (!manager_ctx) [[unlikely]] {
        return;
    }
    manager_ctx->temp_alloc.clear();
    {
        std::lock_guard lck{alloc_mtx};
        ctx.emplace_back(manager_ctx);
    }
    manager_ctx = nullptr;
}