//
// Created by Mike Smith on 2021/3/17.
//

#include <core/arena.h>

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
        static thread_local Arena arena;
        return arena;
    }
    static Arena arena;
    return arena;
}

}// namespace luisa
