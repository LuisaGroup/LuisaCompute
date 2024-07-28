// clang-format off
using ElementType = T;

[[nodiscard, access]] constexpr T access_(uint32 idx) const noexcept {
    return _v[idx];
}

[[nodiscard, access]] constexpr T operator[](uint32 idx) const noexcept {
    return _v[idx];
}
[[nodiscard, access]] constexpr T &access_(uint32 idx) noexcept {
    return _v[idx];
}

[[nodiscard, access]] constexpr T &operator[](uint32 idx) noexcept {
    return _v[idx];
}

template<uint32 i>
constexpr void set() {}

template<uint32 i, concepts::arithmetic_scalar U, typename...Args>
constexpr void set(U v, Args...args) { _v[i] = v; set<i + 1>(args...); }

template<uint32 i, concepts::arithmetic_vec U, typename...Args>
constexpr void set(U v, Args...args) {
    constexpr auto dim = vec_dim_v<U>;
    if constexpr (dim == 2)
    {
        _v[i] = v[0]; _v[i + 1] = v[1];
    }
    else if constexpr (dim == 3)
    {
        _v[i] = v[0]; _v[i + 1] = v[1]; _v[i + 2] = v[2];
    }
    else if constexpr (dim == 4)
    {
        _v[i] = v[0]; _v[i + 1] = v[1]; _v[i + 2] = v[2]; _v[i + 3] = v[3];
    }
    set<i + dim>(args...);
}

template<typename X>
static constexpr bool operatable = is_same_v<X, ThisType> || is_same_v<X, ElementType>;

[[unaop("PLUS")]] ThisType operator+() const;
[[unaop("MINUS")]] ThisType operator-() const;

template <typename U> requires(is_same_v<U, matrix<dim>>)
[[binop("MUL")]] ThisType operator*(const U&) const;
template <typename U> requires(operatable<U>)
[[binop("ADD")]] ThisType operator+(const U&) const;
template <typename U> requires(operatable<U>)
[[binop("SUB")]] ThisType operator-(const U&) const;
template <typename U> requires(operatable<U>)
[[binop("MUL")]] ThisType operator*(const U&) const;
template <typename U> requires(operatable<U>)
[[binop("DIV")]] ThisType operator/(const U&) const;
template <typename U> requires(operatable<U>)
[[binop("MOD")]] ThisType operator%(const U&) const requires(is_int_family_v<ThisType>);
template <typename U> requires(operatable<U>)
[[binop("BIT_AND")]] ThisType operator&(const U&) const requires(is_int_family_v<ThisType>);
template <typename U> requires(operatable<U>)
[[binop("BIT_OR")]] ThisType operator|(const U&) const requires(is_int_family_v<ThisType>);
template <typename U> requires(operatable<U>)
[[binop("BIT_XOR")]] ThisType operator^(const U&) const requires(is_int_family_v<ThisType>);
template <typename U> requires(operatable<U>)
[[binop("SHL")]] ThisType operator<<(const U&) const requires(is_int_family_v<ThisType>);
template <typename U> requires(operatable<U>)
[[binop("SHR")]] ThisType operator>>(const U&) const requires(is_int_family_v<ThisType>);
template <typename U> requires(operatable<U>)
[[binop("AND")]] ThisType operator&&(const U&) const requires(is_bool_family_v<ThisType>);
template <typename U> requires(operatable<U>)
[[binop("OR")]] ThisType operator||(const U&) const requires(is_bool_family_v<ThisType>);
template <typename U> requires(operatable<U>)
[[binop("LESS")]] vec<bool, dim> operator<(const U&) const requires(!is_bool_family_v<ThisType>);
template <typename U> requires(operatable<U>)
[[binop("GREATER")]] vec<bool, dim> operator>(const U&) const requires(!is_bool_family_v<ThisType>);
template <typename U> requires(operatable<U>)
[[binop("LESS_EQUAL")]] vec<bool, dim> operator<=(const U&) const requires(!is_bool_family_v<ThisType>);
template <typename U> requires(operatable<U>)
[[binop("GREATER_EQUAL")]] vec<bool, dim> operator>=(const U&) const requires(!is_bool_family_v<ThisType>);
template <typename U> requires(operatable<U>)
[[binop("EQUAL")]] vec<bool, dim> operator==(const U&) const;
template <typename U> requires(operatable<U>)
[[binop("NOT_EQUAL")]] vec<bool, dim> operator!=(const U&) const;

template <typename U> requires(operatable<U>)
[[noignore]] ThisType operator+=(const U& rhs) { return *this = *this + rhs; }
template <typename U> requires(operatable<U>)
[[noignore]] ThisType operator-=(const U& rhs) { return *this = *this - rhs; }
template <typename U> requires(operatable<U>)
[[noignore]] ThisType operator*=(const U& rhs) { return *this = *this * rhs; }
template <typename U> requires(operatable<U>)
[[noignore]] ThisType operator/=(const U& rhs) { return *this = *this / rhs; }

template <typename U> requires(operatable<U>)
[[noignore]] ThisType operator%=(const U& rhs) requires(is_int_family_v<ThisType>) { return *this = (*this % rhs); }
template <typename U> requires(operatable<U>)
[[noignore]] ThisType operator&=(const U& rhs) requires(is_int_family_v<ThisType>)  { return *this = (*this & rhs); }
template <typename U> requires(operatable<U>)
[[noignore]] ThisType operator|=(const U& rhs) requires(is_int_family_v<ThisType>) { return *this = (*this | rhs); }
template <typename U> requires(operatable<U>)
[[noignore]] ThisType operator^=(const U& rhs) requires(is_int_family_v<ThisType>) { return *this = (*this ^ rhs); }
template <typename U> requires(operatable<U>)
[[noignore]] ThisType operator<<=(const U& rhs) requires(is_int_family_v<ThisType>) { return *this = (*this << rhs); }
template <typename U> requires(operatable<U>)
[[noignore]] ThisType operator>>=(const U& rhs) requires(is_int_family_v<ThisType>) { return *this = (*this >> rhs); }
// clang-format on