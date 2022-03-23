//
// Created by Mike on 2021/12/11.
//

#pragma once

#include <core/hash.h>
#include <core/stl.h>
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

public:
    struct ObserverHash {
        using is_transparent = void;
        [[nodiscard]] auto operator()(const Observer *p) const noexcept -> uint64_t {
            return hash64(reinterpret_cast<uint64_t>(p));
        }
    };

private:
    spin_mutex _mutex;
    luisa::unordered_map<Observer *, size_t, ObserverHash> _observers;

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
