[[unaop("PLUS")]] ThisType operator+() const;
[[unaop("MINUS")]] ThisType operator-() const;
[[binop("ADD")]] ThisType operator+(ThisType) const;
[[binop("SUB")]] ThisType operator-(ThisType) const;
[[binop("MUL")]] ThisType operator*(ThisType) const;
[[binop("DIV")]] ThisType operator/(ThisType) const;
[[binop("MOD")]] ThisType operator%(ThisType) const;

[[binop("ADD")]] ThisType operator+(T) const;
[[binop("SUB")]] ThisType operator-(T) const;
[[binop("MUL")]] ThisType operator*(T) const;
[[binop("DIV")]] ThisType operator/(T) const;
[[binop("MOD")]] ThisType operator%(T) const;