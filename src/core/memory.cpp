//
// Created by Mike Smith on 2021/3/17.
//

#include <core/memory.h>

namespace luisa {

luisa::Arena::~Arena() noexcept {
    auto p = _head;
    while (p != nullptr) {
        auto next = p->next;
        aligned_free(p->data);
        p = next;
    }
}

Arena &Arena::global(bool is_thread_local) noexcept {
    if (is_thread_local) {
        static thread_local auto &arena = []() -> Arena & {
            static thread_local Arena arena{false};
            LUISA_VERBOSE_WITH_LOCATION("Created thread-local global arena.");
            return arena;
        }();
        return arena;
    }
    static auto &arena = []() -> Arena & {
        static Arena arena{true};
        LUISA_VERBOSE_WITH_LOCATION("Created thread-safe global arena.");
        return arena;
    }();
    return arena;
}

}// namespace luisa
