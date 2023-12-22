// clang-format off
[[unaop("PLUS")]] ThisType operator+() const;
[[unaop("MINUS")]] ThisType operator-() const;

using ElementType = T;

template<typename X>
static constexpr bool operatable = is_same_v<X, ThisType> || is_same_v<X, ElementType>;

template <typename U> requires(operatable<U>)
[[binop("ADD")]] ThisType operator+(U) const;
template <typename U> requires(operatable<U>)
[[binop("SUB")]] ThisType operator-(U) const;
template <typename U> requires(operatable<U>)
[[binop("MUL")]] ThisType operator*(U) const;
template <typename U> requires(operatable<U>)
[[binop("DIV")]] ThisType operator/(U) const;
template <typename U> requires(operatable<U>)
[[binop("MOD")]] ThisType operator%(U) const requires(is_int_family_v<ThisType>);
template <typename U> requires(operatable<U>)
[[binop("BIT_AND")]] ThisType operator&(U) const requires(is_int_family_v<ThisType>);
template <typename U> requires(operatable<U>)
[[binop("BIT_OR")]] ThisType operator|(U) const requires(is_int_family_v<ThisType>);
template <typename U> requires(operatable<U>)
[[binop("BIT_XOR")]] ThisType operator^(U) const requires(is_int_family_v<ThisType>);
template <typename U> requires(operatable<U>)
[[binop("SHL")]] ThisType operator<<(U) const requires(is_int_family_v<ThisType>);
template <typename U> requires(operatable<U>)
[[binop("SHR")]] ThisType operator>>(U) const requires(is_int_family_v<ThisType>);
template <typename U> requires(operatable<U>)
[[binop("AND")]] ThisType operator&&(U) const requires(is_bool_family_v<ThisType>);
template <typename U> requires(operatable<U>)
[[binop("OR")]] ThisType operator||(U) const requires(is_bool_family_v<ThisType>);
template <typename U> requires(operatable<U>)
[[binop("LESS")]] vec<bool, dim> operator<(U) const requires(!is_bool_family_v<ThisType>);
template <typename U> requires(operatable<U>)
[[binop("GREATER")]] vec<bool, dim> operator>(U) const requires(!is_bool_family_v<ThisType>);
template <typename U> requires(operatable<U>)
[[binop("LESS_EQUAL")]] vec<bool, dim> operator<=(U) const requires(!is_bool_family_v<ThisType>);
template <typename U> requires(operatable<U>)
[[binop("GREATER_EQUAL")]] vec<bool, dim> operator>=(U) const requires(!is_bool_family_v<ThisType>);
template <typename U> requires(operatable<U>)
[[binop("EQUAL")]] vec<bool, dim> operator==(U) const;
template <typename U> requires(operatable<U>)
[[binop("NOT_EQUAL")]] vec<bool, dim> operator!=(U) const;

template <typename U> requires(operatable<U>)
[[noignore]] ThisType operator+=(U rhs) { return *this = *this + rhs; }
template <typename U> requires(operatable<U>)
[[noignore]] ThisType operator-=(U rhs) { return *this = *this - rhs; }
template <typename U> requires(operatable<U>)
[[noignore]] ThisType operator*=(U rhs) { return *this = *this * rhs; }
template <typename U> requires(operatable<U>)
[[noignore]] ThisType operator/=(U rhs) { return *this = *this / rhs; }

template <typename U> requires(operatable<U>)
[[noignore]] ThisType operator%=(U rhs) const requires(is_int_family_v<ThisType>) { return *this = *this % rhs; }
template <typename U> requires(operatable<U>)
[[noignore]] ThisType operator&=(U rhs) const requires(is_int_family_v<ThisType>)  { return *this = *this % rhs; }
template <typename U> requires(operatable<U>)
[[noignore]] ThisType operator|=(U rhs) const requires(is_int_family_v<ThisType>) { return *this = *this % rhs; }
template <typename U> requires(operatable<U>)
[[noignore]] ThisType operator^=(U rhs) const requires(is_int_family_v<ThisType>) { return *this = *this % rhs; }
template <typename U> requires(operatable<U>)
[[noignore]] ThisType operator<<=(U rhs) const requires(is_int_family_v<ThisType>) { return *this = *this % rhs; }
template <typename U> requires(operatable<U>)
[[noignore]] ThisType operator>>=(U rhs) const requires(is_int_family_v<ThisType>) { return *this = *this % rhs; }
// clang-format on