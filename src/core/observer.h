//
// Created by Mike on 2021/12/11.
//

#pragma once

#include <core/stl/vector.h>
#include <core/stl/memory.h>
#include <core/stl/unordered_map.h>
#include <core/spin_mutex.h>

namespace luisa {

class Subject;

class LC_CORE_API Observer {

private:
    luisa::vector<luisa::shared_ptr<Subject>> _subjects;

public:
    Observer() noexcept = default;
    virtual ~Observer() noexcept;
    Observer(Observer &&) noexcept = delete;
    Observer(const Observer &) noexcept = delete;
    Observer &operator=(Observer &&) noexcept = delete;
    Observer &operator=(const Observer &) noexcept = delete;
    virtual void notify() noexcept = 0;
    void pop_back() noexcept;
    void emplace_back(luisa::shared_ptr<Subject> subject) noexcept;
    void set(size_t index, luisa::shared_ptr<Subject> subject) noexcept;
    [[nodiscard]] auto size() const noexcept { return _subjects.size(); }
};

class Subject final : public luisa::enable_shared_from_this<Subject> {

private:
    spin_mutex _mutex;
    luisa::unordered_map<Observer *, size_t, pointer_hash<Observer>> _observers;

private:
    friend class Observer;
    void _add(Observer *observer) noexcept;
    void _remove(Observer *observer) noexcept;

public:
    Subject() noexcept = default;
    ~Subject() noexcept = default;
    Subject(Subject &&) noexcept = delete;
    Subject(const Subject &) noexcept = delete;
    Subject &operator=(Subject &&) noexcept = delete;
    Subject &operator=(const Subject &) noexcept = delete;
    void notify_all() noexcept;
};

}// namespace luisa
