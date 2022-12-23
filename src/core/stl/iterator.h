//
// Created by Mike Smith on 2022/12/23.
//

#pragma once

namespace luisa {

struct default_sentinel_t {};
inline constexpr default_sentinel_t default_sentinel{};

template<typename T>
class range {

public:
    struct iterator_type {

    private:
        T _value;
        T _step;
        T _end;

    public:
        iterator_type(T value, T end, T step) noexcept
            : _value{value}, _end{end}, _step{step} {}
        [[nodiscard]] auto &operator++() noexcept {
            _value += _step;
            return *this;
        }
        [[nodiscard]] auto operator++(int) noexcept {
            auto copy = *this;
            _value += _step;
            return copy;
        }
        [[nodiscard]] constexpr auto operator*() const noexcept { return _value; }
        [[nodiscard]] constexpr auto operator==(default_sentinel_t) const noexcept -> bool {
            return _step > static_cast<T>(0) ?
                       _value >= _end :
                       _value <= _end;
        }
    };

private:
    T _begin;
    T _end;
    T _step;

public:
    explicit constexpr range(T end) noexcept
        : _begin{}, _end{end}, _step{1} {}
    constexpr range(T begin, T end, T step = static_cast<T>(1)) noexcept
        : _begin{begin}, _end{end}, _step{step} {}
    [[nodiscard]] auto begin() const noexcept { return iterator_type{_begin, _end, _step}; }
    [[nodiscard]] auto end() const noexcept { return default_sentinel; }
};

template<typename T>
range(T) -> range<T>;

template<typename T>
range(T, T) -> range<T>;

template<typename T>
range(T, T, T) -> range<T>;

template<typename T>
[[nodiscard]] constexpr auto make_range(T end) noexcept {
    return range{end};
}

template<typename T>
[[nodiscard]] constexpr auto make_range(T begin, T end, T step = static_cast<T>(1)) noexcept {
    return range{begin, end};
}

}// namespace luisa
