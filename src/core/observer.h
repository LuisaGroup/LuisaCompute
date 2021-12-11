//
// Created by Mike on 2021/12/11.
//

#pragma once

#include <core/hash.h>
#include <core/allocator.h>
#include <core/spin_mutex.h>

namespace luisa {

class Subject;

class Observer {

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

class Subject final : public std::enable_shared_from_this<Subject> {

public:
    struct PointerHash {
        using is_transparent = void;
        template<typename T>
        [[nodiscard]] auto operator()(T *p) const noexcept {
            return hash64(reinterpret_cast<uint64_t>(p));
        }
    };

private:
    spin_mutex _mutex;
    luisa::unordered_map<Observer *, size_t, PointerHash> _observers;

private:
    friend class Observer;
    void _add(Observer *observer) noexcept;
    void _remove(const Observer *observer) noexcept;

public:
    Subject() noexcept = default;
    ~Subject() noexcept = default;
    Subject(Subject &&) noexcept = delete;
    Subject(const Subject &) noexcept = delete;
    Subject &operator=(Subject &&) noexcept = delete;
    Subject &operator=(const Subject &) noexcept = delete;
    void notify_all() noexcept;
};

}
