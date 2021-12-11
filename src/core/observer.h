//
// Created by Mike on 2021/12/11.
//

#pragma once

#include <core/hash.h>
#include <core/allocator.h>

namespace luisa {

class Subject;

struct Observer : public std::enable_shared_from_this<Observer> {
    Observer() noexcept = default;
    virtual ~Observer() noexcept = default;
    Observer(Observer &&) noexcept = delete;
    Observer(const Observer &) noexcept = delete;
    Observer &operator=(Observer &&) noexcept = delete;
    Observer &operator=(const Observer &) noexcept = delete;
    virtual void notify() noexcept = 0;
};

class Subject final {

public:
    struct PointerHash {
        using is_transparent = void;
        template<typename T>
        [[nodiscard]] auto operator()(const T *p) const noexcept {
            return hash64(reinterpret_cast<uint64_t>(p));
        }
        template<typename T>
        [[nodiscard]] auto operator()(const luisa::shared_ptr<T> &p) const noexcept {
            return (*this)(p.get());
        }
        template<typename T>
        [[nodiscard]] auto operator()(const luisa::weak_ptr<T> &p) const noexcept {
            return (*this)(p.lock().get());
        }
    };

    struct PointerEqual {
        using is_transparent = void;
        template<typename T>
        [[nodiscard]] auto extract(const T *p) const noexcept { return p; }
        template<typename T>
        [[nodiscard]] auto extract(const luisa::shared_ptr<T> &p) const noexcept { return p.get(); }
        template<typename T>
        [[nodiscard]] auto extract(const luisa::weak_ptr<T> &p) const noexcept { return p.lock().get(); }
        template<typename Lhs, typename Rhs>
        [[nodiscard]] auto operator()(Lhs &&lhs, Rhs &&rhs) const noexcept {
            return extract(std::forward<Lhs>(lhs)) == extract(std::forward<Rhs>(rhs));
        }
    };

private:
    luisa::unordered_map<luisa::weak_ptr<Observer>, size_t, PointerHash, PointerEqual> _observers;

public:
    Subject() noexcept = default;
    ~Subject() noexcept = default;
    Subject(Subject &&) noexcept = delete;
    Subject(const Subject &) noexcept = delete;
    Subject &operator=(Subject &&) noexcept = delete;
    Subject &operator=(const Subject &) noexcept = delete;
    void add(luisa::weak_ptr<Observer> observer) noexcept;
    void remove(const Observer *observer) noexcept;
    void notify_all() noexcept;
};

}
