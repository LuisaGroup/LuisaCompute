[[unaop("PLUS")]] ThisType operator+() const;
[[unaop("MINUS")]] ThisType operator-() const;

[[binop("ADD")]] ThisType operator+(ThisType) const;
[[binop("SUB")]] ThisType operator-(ThisType) const;
[[binop("MUL")]] ThisType operator*(ThisType) const;
[[binop("DIV")]] ThisType operator/(ThisType) const;
[[binop("MOD")]] ThisType operator%(ThisType) const requires(is_int_family_v<ThisType>);
[[binop("BIT_AND")]] ThisType operator&(ThisType) const requires(is_int_family_v<ThisType>);
[[binop("BIT_OR")]] ThisType operator|(ThisType) const requires(is_int_family_v<ThisType>);
[[binop("BIT_XOR")]] ThisType operator^(ThisType) const requires(is_int_family_v<ThisType>);
[[binop("SHL")]] ThisType operator<<(ThisType) const requires(is_int_family_v<ThisType>);
[[binop("SHR")]] ThisType operator>>(ThisType) const requires(is_int_family_v<ThisType>);
[[binop("AND")]] ThisType operator&&(ThisType) const requires(is_bool_family_v<ThisType>);
[[binop("OR")]] ThisType operator||(ThisType) const requires(is_bool_family_v<ThisType>);
[[binop("LESS")]] vec<bool, dim> operator<(ThisType) const requires(!is_bool_family_v<ThisType>);
[[binop("GREATER")]] vec<bool, dim> operator>(ThisType) const requires(!is_bool_family_v<ThisType>);
[[binop("LESS_EQUAL")]] vec<bool, dim> operator<=(ThisType) const requires(!is_bool_family_v<ThisType>);
[[binop("GREATER_EQUAL")]] vec<bool, dim> operator>=(ThisType) const requires(!is_bool_family_v<ThisType>);
[[binop("EQUAL")]] vec<bool, dim> operator==(ThisType) const;
[[binop("NOT_EQUAL")]] vec<bool, dim> operator!=(ThisType) const;

[[binop("ADD")]] ThisType operator+(T) const;
[[binop("SUB")]] ThisType operator-(T) const;
[[binop("MUL")]] ThisType operator*(T) const;
[[binop("DIV")]] ThisType operator/(T) const;
[[binop("MOD")]] ThisType operator%(T) const requires(is_int_family_v<ThisType>);
[[binop("BIT_AND")]] ThisType operator&(T) const requires(is_int_family_v<ThisType>);
[[binop("BIT_OR")]] ThisType operator|(T) const requires(is_int_family_v<ThisType>);
[[binop("BIT_XOR")]] ThisType operator^(T) const requires(is_int_family_v<ThisType>);
[[binop("SHL")]] ThisType operator<<(T) const requires(is_int_family_v<ThisType>);
[[binop("SHR")]] ThisType operator>>(T) const requires(is_int_family_v<ThisType>);
[[binop("AND")]] ThisType operator&&(T) const requires(is_bool_family_v<ThisType>);
[[binop("OR")]] ThisType operator||(T) const requires(is_bool_family_v<ThisType>);
[[binop("LESS")]] vec<bool, dim> operator<(T) const requires(!is_bool_family_v<ThisType>);
[[binop("GREATER")]] vec<bool, dim> operator>(T) const requires(!is_bool_family_v<ThisType>);
[[binop("LESS_EQUAL")]] vec<bool, dim> operator<=(T) const requires(!is_bool_family_v<ThisType>);
[[binop("GREATER_EQUAL")]] vec<bool, dim> operator>=(T) const requires(!is_bool_family_v<ThisType>);
[[binop("EQUAL")]] vec<bool, dim> operator==(T) const;
[[binop("NOT_EQUAL")]] vec<bool, dim> operator!=(T) const;

[[noignore]] ThisType operator+=(ThisType rhs) { return *this = *this + rhs; }
[[noignore]] ThisType operator-=(ThisType rhs) { return *this = *this - rhs; }
[[noignore]] ThisType operator*=(ThisType rhs) { return *this = *this * rhs; }
[[noignore]] ThisType operator/=(ThisType rhs) { return *this = *this * rhs; }

[[noignore]] ThisType operator+=(T rhs) { return *this = *this + rhs; }
[[noignore]] ThisType operator-=(T rhs) { return *this = *this - rhs; }
[[noignore]] ThisType operator*=(T rhs) { return *this = *this * rhs; }
[[noignore]] ThisType operator/=(T rhs) { return *this = *this * rhs; }

[[noignore]] ThisType operator%=(ThisType) const requires(is_int_family_v<ThisType>)
{
    return *this = *this % rhs;
}
[[noignore]] ThisType operator&=(ThisType) const requires(is_int_family_v<ThisType>)
{
    return *this = *this & rhs;
}
[[noignore]] ThisType operator|=(ThisType) const requires(is_int_family_v<ThisType>)
{
    return *this = *this | rhs;
}
[[noignore]] ThisType operator^=(ThisType) const requires(is_int_family_v<ThisType>)
{
    return *this = *this ^ rhs;
}
[[noignore]] ThisType operator<<=(ThisType) const requires(is_int_family_v<ThisType>)
{
    return *this = *this << rhs;
}
[[noignore]] ThisType operator>>=(ThisType) const requires(is_int_family_v<ThisType>)
{
    return *this = *this >> rhs;
}

[[noignore]] ThisType operator%=(T) const requires(is_int_family_v<ThisType>)
{
    return *this = *this % rhs;
}
[[noignore]] ThisType operator&=(T) const requires(is_int_family_v<ThisType>)
{
    return *this = *this & rhs;
}
[[noignore]] ThisType operator|=(T) const requires(is_int_family_v<ThisType>)
{
    return *this = *this | rhs;
}
[[noignore]] ThisType operator^=(T) const requires(is_int_family_v<ThisType>)
{
    return *this = *this ^ rhs;
}
[[noignore]] ThisType operator<<=(T) const requires(is_int_family_v<ThisType>)
{
    return *this = *this << rhs;
}
[[noignore]] ThisType operator>>=(T) const requires(is_int_family_v<ThisType>)
{
    return *this = *this >> rhs;
}