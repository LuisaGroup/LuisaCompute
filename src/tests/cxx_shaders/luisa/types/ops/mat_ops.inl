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


[[unaop("PLUS")]] ThisType operator+() const;
[[unaop("MINUS")]] ThisType operator-() const;

template <typename U> requires(is_same_v<U, ThisType>)
[[binop("MUL")]] ThisType operator*(const U&) const;
template <typename U> requires(is_same_v<U, vec<T, N>>)
[[binop("MUL")]] vec<T, N> operator*(const U&) const;

// clang-format on