//
// Created by Mike on 2021/12/11.
//

#include <core/observer.h>

namespace luisa {

void Subject::add(luisa::weak_ptr<Observer> observer) noexcept {
    if (auto [iter, first] = _observers.try_emplace(std::move(observer), 1u); !first) {
        iter->second++;
    }
}

void Subject::remove(const Observer *observer) noexcept {
    if (auto iter = _observers.find(observer); iter != _observers.end()) {
        if (--iter->second == 0) {
            _observers.erase(iter);
        }
    }
}

void Subject::notify_all() noexcept {
    std::erase_if(_observers, [](auto &&w) noexcept {
        if (auto o = w.first.lock()) {
            o->notify();
            return false;
        }
        return true;
    });
}

}
