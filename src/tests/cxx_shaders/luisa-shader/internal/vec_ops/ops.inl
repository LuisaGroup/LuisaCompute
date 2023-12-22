[[unaop("PLUS")]] ThisType operator+() const;
[[unaop("MINUS")]] ThisType operator-() const;

using ElementType = T;

template <typename U>
    requires(is_same_v<U, ThisType> || is_same_v<U, ElementType>)
[[binop("ADD")]] ThisType operator+(U) const;
template <typename U>
    requires(is_same_v<U, ThisType> || is_same_v<U, ElementType>)
[[binop("SUB")]] ThisType operator-(U) const;
template <typename U>
    requires(is_same_v<U, ThisType> || is_same_v<U, ElementType>)
[[binop("MUL")]] ThisType operator*(U) const;
template <typename U>
    requires(is_same_v<U, ThisType> || is_same_v<U, ElementType>)
[[binop("DIV")]] ThisType operator/(U) const;
template <typename U>
    requires(is_same_v<U, ThisType> || is_same_v<U, ElementType>)
[[binop("MOD")]] ThisType operator%(U) const requires(is_int_family_v<ThisType>);
template <typename U>
    requires(is_same_v<U, ThisType> || is_same_v<U, ElementType>)
[[binop("BIT_AND")]] ThisType operator&(U) const requires(is_int_family_v<ThisType>);
template <typename U>
    requires(is_same_v<U, ThisType> || is_same_v<U, ElementType>)
[[binop("BIT_OR")]] ThisType operator|(U) const requires(is_int_family_v<ThisType>);
template <typename U>
    requires(is_same_v<U, ThisType> || is_same_v<U, ElementType>)
[[binop("BIT_XOR")]] ThisType operator^(U) const requires(is_int_family_v<ThisType>);
template <typename U>
    requires(is_same_v<U, ThisType> || is_same_v<U, ElementType>)
[[binop("SHL")]] ThisType operator<<(U) const requires(is_int_family_v<ThisType>);
template <typename U>
    requires(is_same_v<U, ThisType> || is_same_v<U, ElementType>)
[[binop("SHR")]] ThisType operator>>(U) const requires(is_int_family_v<ThisType>);
template <typename U>
    requires(is_same_v<U, ThisType> || is_same_v<U, ElementType>)
[[binop("AND")]] ThisType operator&&(U) const requires(is_bool_family_v<ThisType>);
template <typename U>
    requires(is_same_v<U, ThisType> || is_same_v<U, ElementType>)
[[binop("OR")]] ThisType operator||(U) const requires(is_bool_family_v<ThisType>);
template <typename U>
    requires(is_same_v<U, ThisType> || is_same_v<U, ElementType>)
[[binop("LESS")]] vec<bool, dim> operator<(U) const requires(!is_bool_family_v<ThisType>);
template <typename U>
    requires(is_same_v<U, ThisType> || is_same_v<U, ElementType>)
[[binop("GREATER")]] vec<bool, dim> operator>(U) const requires(!is_bool_family_v<ThisType>);
template <typename U>
    requires(is_same_v<U, ThisType> || is_same_v<U, ElementType>)
[[binop("LESS_EQUAL")]] vec<bool, dim> operator<=(U) const requires(!is_bool_family_v<ThisType>);
template <typename U>
    requires(is_same_v<U, ThisType> || is_same_v<U, ElementType>)
[[binop("GREATER_EQUAL")]] vec<bool, dim> operator>=(U) const requires(!is_bool_family_v<ThisType>);
template <typename U>
    requires(is_same_v<U, ThisType> || is_same_v<U, ElementType>)
[[binop("EQUAL")]] vec<bool, dim> operator==(U) const;
template <typename U>
    requires(is_same_v<U, ThisType> || is_same_v<U, ElementType>)
[[binop("NOT_EQUAL")]] vec<bool, dim> operator!=(U) const;

template <typename U>
    requires(is_same_v<U, ThisType> || is_same_v<U, ElementType>)
[[noignore]] ThisType operator+=(U rhs) { return *this = *this + rhs; }
template <typename U>
    requires(is_same_v<U, ThisType> || is_same_v<U, ElementType>)
[[noignore]] ThisType operator-=(U rhs) { return *this = *this - rhs; }
template <typename U>
    requires(is_same_v<U, ThisType> || is_same_v<U, ElementType>)
[[noignore]] ThisType operator*=(U rhs) { return *this = *this * rhs; }
template <typename U>
    requires(is_same_v<U, ThisType> || is_same_v<U, ElementType>)
[[noignore]] ThisType operator/=(U rhs) { return *this = *this / rhs; }

template <typename U>
    requires(is_same_v<U, ThisType> || is_same_v<U, ElementType>)
[[noignore]] ThisType operator%=(U rhs) const requires(is_int_family_v<ThisType>)
{
    return *this = *this % rhs;
}
template <typename U>
    requires(is_same_v<U, ThisType> || is_same_v<U, ElementType>)
[[noignore]] ThisType operator&=(U rhs) const requires(is_int_family_v<ThisType>)
{
    return *this = *this & rhs;
}
template <typename U>
    requires(is_same_v<U, ThisType> || is_same_v<U, ElementType>)
[[noignore]] ThisType operator|=(U rhs) const requires(is_int_family_v<ThisType>)
{
    return *this = *this | rhs;
}
template <typename U>
    requires(is_same_v<U, ThisType> || is_same_v<U, ElementType>)
[[noignore]] ThisType operator^=(U rhs) const requires(is_int_family_v<ThisType>)
{
    return *this = *this ^ rhs;
}
template <typename U>
    requires(is_same_v<U, ThisType> || is_same_v<U, ElementType>)
[[noignore]] ThisType operator<<=(U rhs) const requires(is_int_family_v<ThisType>)
{
    return *this = *this << rhs;
}
template <typename U>
    requires(is_same_v<U, ThisType> || is_same_v<U, ElementType>)
[[noignore]] ThisType operator>>=(U rhs) const requires(is_int_family_v<ThisType>)
{
    return *this = *this >> rhs;
}