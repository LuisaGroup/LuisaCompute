//
// Created by Mike Smith on 2021/10/19.
//

#pragma once

#include <memory>
#include <utility>

#include <core/logging.h>
#include <core/allocator.h>

namespace luisa {

template<typename T>
class RC;

template<typename T>
class RC {

private:
    T _object;
    size_t _ref_count;

private:
    explicit RC(T object) noexcept
        : _object{std::move(object)},
          _ref_count{1u} {}

public:
    template<typename... Args>
    [[nodiscard]] static auto create(Args &&...args) noexcept {
        return new_with_allocator<RC>(RC{T{std::forward<Args>(args)...}});
    }
    RC(RC &&)
    noexcept = default;
    RC(const RC &)
    noexcept = delete;
    RC &operator=(RC &&) noexcept = delete;
    RC &operator=(const RC &) noexcept = delete;
    [[nodiscard]] auto retain() noexcept {
        _ref_count++;
        return &_object;
    }
    [[nodiscard]] auto object() noexcept { return &_object; }
    void release() noexcept {
        if (_ref_count == 0u) [[unlikely]] {
            LUISA_ERROR_WITH_LOCATION(
                "Releasing RC object with zero reference count.");
        }
        if (--_ref_count == 0u) { delete_with_allocator(this); }
    }
};

}// namespace luisa
