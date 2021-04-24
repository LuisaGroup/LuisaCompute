#pragma once
[[nodiscard]] auto xx() const noexcept { return Vector<T, 2>{x, x}; }
[[nodiscard]] auto xy() const noexcept { return Vector<T, 2>{x, y}; }
[[nodiscard]] auto xz() const noexcept { return Vector<T, 2>{x, z}; }
[[nodiscard]] auto yx() const noexcept { return Vector<T, 2>{y, x}; }
[[nodiscard]] auto yy() const noexcept { return Vector<T, 2>{y, y}; }
[[nodiscard]] auto yz() const noexcept { return Vector<T, 2>{y, z}; }
[[nodiscard]] auto zx() const noexcept { return Vector<T, 2>{z, x}; }
[[nodiscard]] auto zy() const noexcept { return Vector<T, 2>{z, y}; }
[[nodiscard]] auto zz() const noexcept { return Vector<T, 2>{z, z}; }
[[nodiscard]] auto xxx() const noexcept { return Vector<T, 3>{x, x, x}; }
[[nodiscard]] auto xxy() const noexcept { return Vector<T, 3>{x, x, y}; }
[[nodiscard]] auto xxz() const noexcept { return Vector<T, 3>{x, x, z}; }
[[nodiscard]] auto xyx() const noexcept { return Vector<T, 3>{x, y, x}; }
[[nodiscard]] auto xyy() const noexcept { return Vector<T, 3>{x, y, y}; }
[[nodiscard]] auto xyz() const noexcept { return Vector<T, 3>{x, y, z}; }
[[nodiscard]] auto xzx() const noexcept { return Vector<T, 3>{x, z, x}; }
[[nodiscard]] auto xzy() const noexcept { return Vector<T, 3>{x, z, y}; }
[[nodiscard]] auto xzz() const noexcept { return Vector<T, 3>{x, z, z}; }
[[nodiscard]] auto yxx() const noexcept { return Vector<T, 3>{y, x, x}; }
[[nodiscard]] auto yxy() const noexcept { return Vector<T, 3>{y, x, y}; }
[[nodiscard]] auto yxz() const noexcept { return Vector<T, 3>{y, x, z}; }
[[nodiscard]] auto yyx() const noexcept { return Vector<T, 3>{y, y, x}; }
[[nodiscard]] auto yyy() const noexcept { return Vector<T, 3>{y, y, y}; }
[[nodiscard]] auto yyz() const noexcept { return Vector<T, 3>{y, y, z}; }
[[nodiscard]] auto yzx() const noexcept { return Vector<T, 3>{y, z, x}; }
[[nodiscard]] auto yzy() const noexcept { return Vector<T, 3>{y, z, y}; }
[[nodiscard]] auto yzz() const noexcept { return Vector<T, 3>{y, z, z}; }
[[nodiscard]] auto zxx() const noexcept { return Vector<T, 3>{z, x, x}; }
[[nodiscard]] auto zxy() const noexcept { return Vector<T, 3>{z, x, y}; }
[[nodiscard]] auto zxz() const noexcept { return Vector<T, 3>{z, x, z}; }
[[nodiscard]] auto zyx() const noexcept { return Vector<T, 3>{z, y, x}; }
[[nodiscard]] auto zyy() const noexcept { return Vector<T, 3>{z, y, y}; }
[[nodiscard]] auto zyz() const noexcept { return Vector<T, 3>{z, y, z}; }
[[nodiscard]] auto zzx() const noexcept { return Vector<T, 3>{z, z, x}; }
[[nodiscard]] auto zzy() const noexcept { return Vector<T, 3>{z, z, y}; }
[[nodiscard]] auto zzz() const noexcept { return Vector<T, 3>{z, z, z}; }
[[nodiscard]] auto xxxx() const noexcept { return Vector<T, 4>{x, x, x, x}; }
[[nodiscard]] auto xxxy() const noexcept { return Vector<T, 4>{x, x, x, y}; }
[[nodiscard]] auto xxxz() const noexcept { return Vector<T, 4>{x, x, x, z}; }
[[nodiscard]] auto xxyx() const noexcept { return Vector<T, 4>{x, x, y, x}; }
[[nodiscard]] auto xxyy() const noexcept { return Vector<T, 4>{x, x, y, y}; }
[[nodiscard]] auto xxyz() const noexcept { return Vector<T, 4>{x, x, y, z}; }
[[nodiscard]] auto xxzx() const noexcept { return Vector<T, 4>{x, x, z, x}; }
[[nodiscard]] auto xxzy() const noexcept { return Vector<T, 4>{x, x, z, y}; }
[[nodiscard]] auto xxzz() const noexcept { return Vector<T, 4>{x, x, z, z}; }
[[nodiscard]] auto xyxx() const noexcept { return Vector<T, 4>{x, y, x, x}; }
[[nodiscard]] auto xyxy() const noexcept { return Vector<T, 4>{x, y, x, y}; }
[[nodiscard]] auto xyxz() const noexcept { return Vector<T, 4>{x, y, x, z}; }
[[nodiscard]] auto xyyx() const noexcept { return Vector<T, 4>{x, y, y, x}; }
[[nodiscard]] auto xyyy() const noexcept { return Vector<T, 4>{x, y, y, y}; }
[[nodiscard]] auto xyyz() const noexcept { return Vector<T, 4>{x, y, y, z}; }
[[nodiscard]] auto xyzx() const noexcept { return Vector<T, 4>{x, y, z, x}; }
[[nodiscard]] auto xyzy() const noexcept { return Vector<T, 4>{x, y, z, y}; }
[[nodiscard]] auto xyzz() const noexcept { return Vector<T, 4>{x, y, z, z}; }
[[nodiscard]] auto xzxx() const noexcept { return Vector<T, 4>{x, z, x, x}; }
[[nodiscard]] auto xzxy() const noexcept { return Vector<T, 4>{x, z, x, y}; }
[[nodiscard]] auto xzxz() const noexcept { return Vector<T, 4>{x, z, x, z}; }
[[nodiscard]] auto xzyx() const noexcept { return Vector<T, 4>{x, z, y, x}; }
[[nodiscard]] auto xzyy() const noexcept { return Vector<T, 4>{x, z, y, y}; }
[[nodiscard]] auto xzyz() const noexcept { return Vector<T, 4>{x, z, y, z}; }
[[nodiscard]] auto xzzx() const noexcept { return Vector<T, 4>{x, z, z, x}; }
[[nodiscard]] auto xzzy() const noexcept { return Vector<T, 4>{x, z, z, y}; }
[[nodiscard]] auto xzzz() const noexcept { return Vector<T, 4>{x, z, z, z}; }
[[nodiscard]] auto yxxx() const noexcept { return Vector<T, 4>{y, x, x, x}; }
[[nodiscard]] auto yxxy() const noexcept { return Vector<T, 4>{y, x, x, y}; }
[[nodiscard]] auto yxxz() const noexcept { return Vector<T, 4>{y, x, x, z}; }
[[nodiscard]] auto yxyx() const noexcept { return Vector<T, 4>{y, x, y, x}; }
[[nodiscard]] auto yxyy() const noexcept { return Vector<T, 4>{y, x, y, y}; }
[[nodiscard]] auto yxyz() const noexcept { return Vector<T, 4>{y, x, y, z}; }
[[nodiscard]] auto yxzx() const noexcept { return Vector<T, 4>{y, x, z, x}; }
[[nodiscard]] auto yxzy() const noexcept { return Vector<T, 4>{y, x, z, y}; }
[[nodiscard]] auto yxzz() const noexcept { return Vector<T, 4>{y, x, z, z}; }
[[nodiscard]] auto yyxx() const noexcept { return Vector<T, 4>{y, y, x, x}; }
[[nodiscard]] auto yyxy() const noexcept { return Vector<T, 4>{y, y, x, y}; }
[[nodiscard]] auto yyxz() const noexcept { return Vector<T, 4>{y, y, x, z}; }
[[nodiscard]] auto yyyx() const noexcept { return Vector<T, 4>{y, y, y, x}; }
[[nodiscard]] auto yyyy() const noexcept { return Vector<T, 4>{y, y, y, y}; }
[[nodiscard]] auto yyyz() const noexcept { return Vector<T, 4>{y, y, y, z}; }
[[nodiscard]] auto yyzx() const noexcept { return Vector<T, 4>{y, y, z, x}; }
[[nodiscard]] auto yyzy() const noexcept { return Vector<T, 4>{y, y, z, y}; }
[[nodiscard]] auto yyzz() const noexcept { return Vector<T, 4>{y, y, z, z}; }
[[nodiscard]] auto yzxx() const noexcept { return Vector<T, 4>{y, z, x, x}; }
[[nodiscard]] auto yzxy() const noexcept { return Vector<T, 4>{y, z, x, y}; }
[[nodiscard]] auto yzxz() const noexcept { return Vector<T, 4>{y, z, x, z}; }
[[nodiscard]] auto yzyx() const noexcept { return Vector<T, 4>{y, z, y, x}; }
[[nodiscard]] auto yzyy() const noexcept { return Vector<T, 4>{y, z, y, y}; }
[[nodiscard]] auto yzyz() const noexcept { return Vector<T, 4>{y, z, y, z}; }
[[nodiscard]] auto yzzx() const noexcept { return Vector<T, 4>{y, z, z, x}; }
[[nodiscard]] auto yzzy() const noexcept { return Vector<T, 4>{y, z, z, y}; }
[[nodiscard]] auto yzzz() const noexcept { return Vector<T, 4>{y, z, z, z}; }
[[nodiscard]] auto zxxx() const noexcept { return Vector<T, 4>{z, x, x, x}; }
[[nodiscard]] auto zxxy() const noexcept { return Vector<T, 4>{z, x, x, y}; }
[[nodiscard]] auto zxxz() const noexcept { return Vector<T, 4>{z, x, x, z}; }
[[nodiscard]] auto zxyx() const noexcept { return Vector<T, 4>{z, x, y, x}; }
[[nodiscard]] auto zxyy() const noexcept { return Vector<T, 4>{z, x, y, y}; }
[[nodiscard]] auto zxyz() const noexcept { return Vector<T, 4>{z, x, y, z}; }
[[nodiscard]] auto zxzx() const noexcept { return Vector<T, 4>{z, x, z, x}; }
[[nodiscard]] auto zxzy() const noexcept { return Vector<T, 4>{z, x, z, y}; }
[[nodiscard]] auto zxzz() const noexcept { return Vector<T, 4>{z, x, z, z}; }
[[nodiscard]] auto zyxx() const noexcept { return Vector<T, 4>{z, y, x, x}; }
[[nodiscard]] auto zyxy() const noexcept { return Vector<T, 4>{z, y, x, y}; }
[[nodiscard]] auto zyxz() const noexcept { return Vector<T, 4>{z, y, x, z}; }
[[nodiscard]] auto zyyx() const noexcept { return Vector<T, 4>{z, y, y, x}; }
[[nodiscard]] auto zyyy() const noexcept { return Vector<T, 4>{z, y, y, y}; }
[[nodiscard]] auto zyyz() const noexcept { return Vector<T, 4>{z, y, y, z}; }
[[nodiscard]] auto zyzx() const noexcept { return Vector<T, 4>{z, y, z, x}; }
[[nodiscard]] auto zyzy() const noexcept { return Vector<T, 4>{z, y, z, y}; }
[[nodiscard]] auto zyzz() const noexcept { return Vector<T, 4>{z, y, z, z}; }
[[nodiscard]] auto zzxx() const noexcept { return Vector<T, 4>{z, z, x, x}; }
[[nodiscard]] auto zzxy() const noexcept { return Vector<T, 4>{z, z, x, y}; }
[[nodiscard]] auto zzxz() const noexcept { return Vector<T, 4>{z, z, x, z}; }
[[nodiscard]] auto zzyx() const noexcept { return Vector<T, 4>{z, z, y, x}; }
[[nodiscard]] auto zzyy() const noexcept { return Vector<T, 4>{z, z, y, y}; }
[[nodiscard]] auto zzyz() const noexcept { return Vector<T, 4>{z, z, y, z}; }
[[nodiscard]] auto zzzx() const noexcept { return Vector<T, 4>{z, z, z, x}; }
[[nodiscard]] auto zzzy() const noexcept { return Vector<T, 4>{z, z, z, y}; }
[[nodiscard]] auto zzzz() const noexcept { return Vector<T, 4>{z, z, z, z}; }
