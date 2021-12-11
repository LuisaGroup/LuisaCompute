//
// Created by Mike on 2021/12/11.
//

#include <mutex>
#include <core/observer.h>

namespace luisa {

void Subject::_add(Observer *observer) noexcept {
    std::scoped_lock lock{_mutex};
    if (auto [iter, first] = _observers.try_emplace(observer, 1u); !first) {
        iter->second++;
    }
}

void Subject::_remove(const Observer *observer) noexcept {
    std::scoped_lock lock{_mutex};
    if (auto iter = _observers.find(observer); iter != _observers.end()) {
        if (--iter->second == 0) {
            _observers.erase(iter);
        }
    }
}

void Subject::notify_all() noexcept {
    std::scoped_lock lock{_mutex};
    for (auto o : _observers) {
        o.first->notify();
    }
}

void Observer::pop_back() noexcept {
    _subjects.back()->_remove(this);
    _subjects.pop_back();
}

void Observer::emplace_back(luisa::shared_ptr<Subject> subject) noexcept {
    subject->_add(this);
    _subjects.emplace_back(std::move(subject));
}

void Observer::set(size_t index, luisa::shared_ptr<Subject> subject) noexcept {
    _subjects[index]->_remove(this);
    subject->_add(this);
    _subjects.emplace_back(std::move(subject));
}

Observer::~Observer() noexcept {
    for (auto &&s : _subjects) {
        s->_remove(this);
    }
}

}
