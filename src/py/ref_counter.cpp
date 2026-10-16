#include "ref_counter.h"

namespace luisa::compute {

RefCounter::~RefCounter() noexcept {
    for (auto &&i : refCounts) {
        auto &disp = i.second.second;
        disp.first(disp.second, i.first);
    }
}

RefCounter::RefCounter() noexcept {}

void RefCounter::AddObject(Handle handle, Disposer disposer) noexcept {
    std::lock_guard lck(mtx);
    refCounts[handle] = {1, disposer};
}

void RefCounter::InRef(Handle handle) noexcept {
    std::lock_guard lck(mtx);
    auto ite = refCounts.find(handle);
    if (ite != refCounts.end()) ite->second.first++;
}

void RefCounter::DeRef(Handle handle) noexcept {
    Disposer disposer;
    {
        std::lock_guard lck(mtx);
        auto ite = refCounts.find(handle);
        if (--ite->second.first > 0) { return; }
        disposer = ite->second.second;
        refCounts.erase(ite);
    }
    // Synchronizing a disposer can release other completed references.
    // Remove this entry and unlock before allowing that reentrant work.
    disposer.first(disposer.second, handle);
}

vstd::unique_ptr<RefCounter> RefCounter::current;

}// namespace luisa::compute
