[[nodiscard]] constexpr auto xx() const noexcept { return Vector<T, 2>{x, x}; }
[[nodiscard]] constexpr auto xy() const noexcept { return Vector<T, 2>{x, y}; }
[[nodiscard]] constexpr auto xz() const noexcept { return Vector<T, 2>{x, z}; }
[[nodiscard]] constexpr auto yx() const noexcept { return Vector<T, 2>{y, x}; }
[[nodiscard]] constexpr auto yy() const noexcept { return Vector<T, 2>{y, y}; }
[[nodiscard]] constexpr auto yz() const noexcept { return Vector<T, 2>{y, z}; }
[[nodiscard]] constexpr auto zx() const noexcept { return Vector<T, 2>{z, x}; }
[[nodiscard]] constexpr auto zy() const noexcept { return Vector<T, 2>{z, y}; }
[[nodiscard]] constexpr auto zz() const noexcept { return Vector<T, 2>{z, z}; }
[[nodiscard]] constexpr auto xxx() const noexcept { return Vector<T, 3>{x, x, x}; }
[[nodiscard]] constexpr auto xxy() const noexcept { return Vector<T, 3>{x, x, y}; }
[[nodiscard]] constexpr auto xxz() const noexcept { return Vector<T, 3>{x, x, z}; }
[[nodiscard]] constexpr auto xyx() const noexcept { return Vector<T, 3>{x, y, x}; }
[[nodiscard]] constexpr auto xyy() const noexcept { return Vector<T, 3>{x, y, y}; }
[[nodiscard]] constexpr auto xyz() const noexcept { return Vector<T, 3>{x, y, z}; }
[[nodiscard]] constexpr auto xzx() const noexcept { return Vector<T, 3>{x, z, x}; }
[[nodiscard]] constexpr auto xzy() const noexcept { return Vector<T, 3>{x, z, y}; }
[[nodiscard]] constexpr auto xzz() const noexcept { return Vector<T, 3>{x, z, z}; }
[[nodiscard]] constexpr auto yxx() const noexcept { return Vector<T, 3>{y, x, x}; }
[[nodiscard]] constexpr auto yxy() const noexcept { return Vector<T, 3>{y, x, y}; }
[[nodiscard]] constexpr auto yxz() const noexcept { return Vector<T, 3>{y, x, z}; }
[[nodiscard]] constexpr auto yyx() const noexcept { return Vector<T, 3>{y, y, x}; }
[[nodiscard]] constexpr auto yyy() const noexcept { return Vector<T, 3>{y, y, y}; }
[[nodiscard]] constexpr auto yyz() const noexcept { return Vector<T, 3>{y, y, z}; }
[[nodiscard]] constexpr auto yzx() const noexcept { return Vector<T, 3>{y, z, x}; }
[[nodiscard]] constexpr auto yzy() const noexcept { return Vector<T, 3>{y, z, y}; }
[[nodiscard]] constexpr auto yzz() const noexcept { return Vector<T, 3>{y, z, z}; }
[[nodiscard]] constexpr auto zxx() const noexcept { return Vector<T, 3>{z, x, x}; }
[[nodiscard]] constexpr auto zxy() const noexcept { return Vector<T, 3>{z, x, y}; }
[[nodiscard]] constexpr auto zxz() const noexcept { return Vector<T, 3>{z, x, z}; }
[[nodiscard]] constexpr auto zyx() const noexcept { return Vector<T, 3>{z, y, x}; }
[[nodiscard]] constexpr auto zyy() const noexcept { return Vector<T, 3>{z, y, y}; }
[[nodiscard]] constexpr auto zyz() const noexcept { return Vector<T, 3>{z, y, z}; }
[[nodiscard]] constexpr auto zzx() const noexcept { return Vector<T, 3>{z, z, x}; }
[[nodiscard]] constexpr auto zzy() const noexcept { return Vector<T, 3>{z, z, y}; }
[[nodiscard]] constexpr auto zzz() const noexcept { return Vector<T, 3>{z, z, z}; }
[[nodiscard]] constexpr auto xxxx() const noexcept { return Vector<T, 4>{x, x, x, x}; }
[[nodiscard]] constexpr auto xxxy() const noexcept { return Vector<T, 4>{x, x, x, y}; }
[[nodiscard]] constexpr auto xxxz() const noexcept { return Vector<T, 4>{x, x, x, z}; }
[[nodiscard]] constexpr auto xxyx() const noexcept { return Vector<T, 4>{x, x, y, x}; }
[[nodiscard]] constexpr auto xxyy() const noexcept { return Vector<T, 4>{x, x, y, y}; }
[[nodiscard]] constexpr auto xxyz() const noexcept { return Vector<T, 4>{x, x, y, z}; }
[[nodiscard]] constexpr auto xxzx() const noexcept { return Vector<T, 4>{x, x, z, x}; }
[[nodiscard]] constexpr auto xxzy() const noexcept { return Vector<T, 4>{x, x, z, y}; }
[[nodiscard]] constexpr auto xxzz() const noexcept { return Vector<T, 4>{x, x, z, z}; }
[[nodiscard]] constexpr auto xyxx() const noexcept { return Vector<T, 4>{x, y, x, x}; }
[[nodiscard]] constexpr auto xyxy() const noexcept { return Vector<T, 4>{x, y, x, y}; }
[[nodiscard]] constexpr auto xyxz() const noexcept { return Vector<T, 4>{x, y, x, z}; }
[[nodiscard]] constexpr auto xyyx() const noexcept { return Vector<T, 4>{x, y, y, x}; }
[[nodiscard]] constexpr auto xyyy() const noexcept { return Vector<T, 4>{x, y, y, y}; }
[[nodiscard]] constexpr auto xyyz() const noexcept { return Vector<T, 4>{x, y, y, z}; }
[[nodiscard]] constexpr auto xyzx() const noexcept { return Vector<T, 4>{x, y, z, x}; }
[[nodiscard]] constexpr auto xyzy() const noexcept { return Vector<T, 4>{x, y, z, y}; }
[[nodiscard]] constexpr auto xyzz() const noexcept { return Vector<T, 4>{x, y, z, z}; }
[[nodiscard]] constexpr auto xzxx() const noexcept { return Vector<T, 4>{x, z, x, x}; }
[[nodiscard]] constexpr auto xzxy() const noexcept { return Vector<T, 4>{x, z, x, y}; }
[[nodiscard]] constexpr auto xzxz() const noexcept { return Vector<T, 4>{x, z, x, z}; }
[[nodiscard]] constexpr auto xzyx() const noexcept { return Vector<T, 4>{x, z, y, x}; }
[[nodiscard]] constexpr auto xzyy() const noexcept { return Vector<T, 4>{x, z, y, y}; }
[[nodiscard]] constexpr auto xzyz() const noexcept { return Vector<T, 4>{x, z, y, z}; }
[[nodiscard]] constexpr auto xzzx() const noexcept { return Vector<T, 4>{x, z, z, x}; }
[[nodiscard]] constexpr auto xzzy() const noexcept { return Vector<T, 4>{x, z, z, y}; }
[[nodiscard]] constexpr auto xzzz() const noexcept { return Vector<T, 4>{x, z, z, z}; }
[[nodiscard]] constexpr auto yxxx() const noexcept { return Vector<T, 4>{y, x, x, x}; }
[[nodiscard]] constexpr auto yxxy() const noexcept { return Vector<T, 4>{y, x, x, y}; }
[[nodiscard]] constexpr auto yxxz() const noexcept { return Vector<T, 4>{y, x, x, z}; }
[[nodiscard]] constexpr auto yxyx() const noexcept { return Vector<T, 4>{y, x, y, x}; }
[[nodiscard]] constexpr auto yxyy() const noexcept { return Vector<T, 4>{y, x, y, y}; }
[[nodiscard]] constexpr auto yxyz() const noexcept { return Vector<T, 4>{y, x, y, z}; }
[[nodiscard]] constexpr auto yxzx() const noexcept { return Vector<T, 4>{y, x, z, x}; }
[[nodiscard]] constexpr auto yxzy() const noexcept { return Vector<T, 4>{y, x, z, y}; }
[[nodiscard]] constexpr auto yxzz() const noexcept { return Vector<T, 4>{y, x, z, z}; }
[[nodiscard]] constexpr auto yyxx() const noexcept { return Vector<T, 4>{y, y, x, x}; }
[[nodiscard]] constexpr auto yyxy() const noexcept { return Vector<T, 4>{y, y, x, y}; }
[[nodiscard]] constexpr auto yyxz() const noexcept { return Vector<T, 4>{y, y, x, z}; }
[[nodiscard]] constexpr auto yyyx() const noexcept { return Vector<T, 4>{y, y, y, x}; }
[[nodiscard]] constexpr auto yyyy() const noexcept { return Vector<T, 4>{y, y, y, y}; }
[[nodiscard]] constexpr auto yyyz() const noexcept { return Vector<T, 4>{y, y, y, z}; }
[[nodiscard]] constexpr auto yyzx() const noexcept { return Vector<T, 4>{y, y, z, x}; }
[[nodiscard]] constexpr auto yyzy() const noexcept { return Vector<T, 4>{y, y, z, y}; }
[[nodiscard]] constexpr auto yyzz() const noexcept { return Vector<T, 4>{y, y, z, z}; }
[[nodiscard]] constexpr auto yzxx() const noexcept { return Vector<T, 4>{y, z, x, x}; }
[[nodiscard]] constexpr auto yzxy() const noexcept { return Vector<T, 4>{y, z, x, y}; }
[[nodiscard]] constexpr auto yzxz() const noexcept { return Vector<T, 4>{y, z, x, z}; }
[[nodiscard]] constexpr auto yzyx() const noexcept { return Vector<T, 4>{y, z, y, x}; }
[[nodiscard]] constexpr auto yzyy() const noexcept { return Vector<T, 4>{y, z, y, y}; }
[[nodiscard]] constexpr auto yzyz() const noexcept { return Vector<T, 4>{y, z, y, z}; }
[[nodiscard]] constexpr auto yzzx() const noexcept { return Vector<T, 4>{y, z, z, x}; }
[[nodiscard]] constexpr auto yzzy() const noexcept { return Vector<T, 4>{y, z, z, y}; }
[[nodiscard]] constexpr auto yzzz() const noexcept { return Vector<T, 4>{y, z, z, z}; }
[[nodiscard]] constexpr auto zxxx() const noexcept { return Vector<T, 4>{z, x, x, x}; }
[[nodiscard]] constexpr auto zxxy() const noexcept { return Vector<T, 4>{z, x, x, y}; }
[[nodiscard]] constexpr auto zxxz() const noexcept { return Vector<T, 4>{z, x, x, z}; }
[[nodiscard]] constexpr auto zxyx() const noexcept { return Vector<T, 4>{z, x, y, x}; }
[[nodiscard]] constexpr auto zxyy() const noexcept { return Vector<T, 4>{z, x, y, y}; }
[[nodiscard]] constexpr auto zxyz() const noexcept { return Vector<T, 4>{z, x, y, z}; }
[[nodiscard]] constexpr auto zxzx() const noexcept { return Vector<T, 4>{z, x, z, x}; }
[[nodiscard]] constexpr auto zxzy() const noexcept { return Vector<T, 4>{z, x, z, y}; }
[[nodiscard]] constexpr auto zxzz() const noexcept { return Vector<T, 4>{z, x, z, z}; }
[[nodiscard]] constexpr auto zyxx() const noexcept { return Vector<T, 4>{z, y, x, x}; }
[[nodiscard]] constexpr auto zyxy() const noexcept { return Vector<T, 4>{z, y, x, y}; }
[[nodiscard]] constexpr auto zyxz() const noexcept { return Vector<T, 4>{z, y, x, z}; }
[[nodiscard]] constexpr auto zyyx() const noexcept { return Vector<T, 4>{z, y, y, x}; }
[[nodiscard]] constexpr auto zyyy() const noexcept { return Vector<T, 4>{z, y, y, y}; }
[[nodiscard]] constexpr auto zyyz() const noexcept { return Vector<T, 4>{z, y, y, z}; }
[[nodiscard]] constexpr auto zyzx() const noexcept { return Vector<T, 4>{z, y, z, x}; }
[[nodiscard]] constexpr auto zyzy() const noexcept { return Vector<T, 4>{z, y, z, y}; }
[[nodiscard]] constexpr auto zyzz() const noexcept { return Vector<T, 4>{z, y, z, z}; }
[[nodiscard]] constexpr auto zzxx() const noexcept { return Vector<T, 4>{z, z, x, x}; }
[[nodiscard]] constexpr auto zzxy() const noexcept { return Vector<T, 4>{z, z, x, y}; }
[[nodiscard]] constexpr auto zzxz() const noexcept { return Vector<T, 4>{z, z, x, z}; }
[[nodiscard]] constexpr auto zzyx() const noexcept { return Vector<T, 4>{z, z, y, x}; }
[[nodiscard]] constexpr auto zzyy() const noexcept { return Vector<T, 4>{z, z, y, y}; }
[[nodiscard]] constexpr auto zzyz() const noexcept { return Vector<T, 4>{z, z, y, z}; }
[[nodiscard]] constexpr auto zzzx() const noexcept { return Vector<T, 4>{z, z, z, x}; }
[[nodiscard]] constexpr auto zzzy() const noexcept { return Vector<T, 4>{z, z, z, y}; }
[[nodiscard]] constexpr auto zzzz() const noexcept { return Vector<T, 4>{z, z, z, z}; }
