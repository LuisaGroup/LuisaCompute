// clang-format off

[[nodiscard, access]] constexpr auto access_(uint32 idx) const noexcept {
    if constexpr (N == 3)
        return vec<T, 3>(_v[0][0], _v[0][1], _v[0][2]);
    else
        return _v[idx];
}

[[nodiscard, access]] constexpr auto operator[](uint32 idx) const noexcept {
    if constexpr (N == 3)
        return vec<T, 3>(_v[0][0], _v[0][1], _v[0][2]);
    else
        return _v[idx];
}

[[nodiscard, noignore]] constexpr T get(uint32 row, uint32 col) const noexcept {
    return access_(row).access_(col);
}

template<typename X>
static constexpr bool operatable = is_same_v<X, ThisType> || is_same_v<X, vec<T, N>>;

[[unaop("PLUS")]] ThisType operator+() const;
[[unaop("MINUS")]] ThisType operator-() const;

template <typename U> requires(operatable<U>)
[[binop("MUL")]] ThisType operator*(const U&) const;

// clang-format on