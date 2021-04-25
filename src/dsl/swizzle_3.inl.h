#pragma once
[[nodiscard]] auto xx() const noexcept { return make_vector(x, x); }
[[nodiscard]] auto xy() const noexcept { return make_vector(x, y); }
[[nodiscard]] auto xz() const noexcept { return make_vector(x, z); }
[[nodiscard]] auto yx() const noexcept { return make_vector(y, x); }
[[nodiscard]] auto yy() const noexcept { return make_vector(y, y); }
[[nodiscard]] auto yz() const noexcept { return make_vector(y, z); }
[[nodiscard]] auto zx() const noexcept { return make_vector(z, x); }
[[nodiscard]] auto zy() const noexcept { return make_vector(z, y); }
[[nodiscard]] auto zz() const noexcept { return make_vector(z, z); }
[[nodiscard]] auto xxx() const noexcept { return make_vector(x, x, x); }
[[nodiscard]] auto xxy() const noexcept { return make_vector(x, x, y); }
[[nodiscard]] auto xxz() const noexcept { return make_vector(x, x, z); }
[[nodiscard]] auto xyx() const noexcept { return make_vector(x, y, x); }
[[nodiscard]] auto xyy() const noexcept { return make_vector(x, y, y); }
[[nodiscard]] auto xyz() const noexcept { return make_vector(x, y, z); }
[[nodiscard]] auto xzx() const noexcept { return make_vector(x, z, x); }
[[nodiscard]] auto xzy() const noexcept { return make_vector(x, z, y); }
[[nodiscard]] auto xzz() const noexcept { return make_vector(x, z, z); }
[[nodiscard]] auto yxx() const noexcept { return make_vector(y, x, x); }
[[nodiscard]] auto yxy() const noexcept { return make_vector(y, x, y); }
[[nodiscard]] auto yxz() const noexcept { return make_vector(y, x, z); }
[[nodiscard]] auto yyx() const noexcept { return make_vector(y, y, x); }
[[nodiscard]] auto yyy() const noexcept { return make_vector(y, y, y); }
[[nodiscard]] auto yyz() const noexcept { return make_vector(y, y, z); }
[[nodiscard]] auto yzx() const noexcept { return make_vector(y, z, x); }
[[nodiscard]] auto yzy() const noexcept { return make_vector(y, z, y); }
[[nodiscard]] auto yzz() const noexcept { return make_vector(y, z, z); }
[[nodiscard]] auto zxx() const noexcept { return make_vector(z, x, x); }
[[nodiscard]] auto zxy() const noexcept { return make_vector(z, x, y); }
[[nodiscard]] auto zxz() const noexcept { return make_vector(z, x, z); }
[[nodiscard]] auto zyx() const noexcept { return make_vector(z, y, x); }
[[nodiscard]] auto zyy() const noexcept { return make_vector(z, y, y); }
[[nodiscard]] auto zyz() const noexcept { return make_vector(z, y, z); }
[[nodiscard]] auto zzx() const noexcept { return make_vector(z, z, x); }
[[nodiscard]] auto zzy() const noexcept { return make_vector(z, z, y); }
[[nodiscard]] auto zzz() const noexcept { return make_vector(z, z, z); }
[[nodiscard]] auto xxxx() const noexcept { return make_vector(x, x, x, x); }
[[nodiscard]] auto xxxy() const noexcept { return make_vector(x, x, x, y); }
[[nodiscard]] auto xxxz() const noexcept { return make_vector(x, x, x, z); }
[[nodiscard]] auto xxyx() const noexcept { return make_vector(x, x, y, x); }
[[nodiscard]] auto xxyy() const noexcept { return make_vector(x, x, y, y); }
[[nodiscard]] auto xxyz() const noexcept { return make_vector(x, x, y, z); }
[[nodiscard]] auto xxzx() const noexcept { return make_vector(x, x, z, x); }
[[nodiscard]] auto xxzy() const noexcept { return make_vector(x, x, z, y); }
[[nodiscard]] auto xxzz() const noexcept { return make_vector(x, x, z, z); }
[[nodiscard]] auto xyxx() const noexcept { return make_vector(x, y, x, x); }
[[nodiscard]] auto xyxy() const noexcept { return make_vector(x, y, x, y); }
[[nodiscard]] auto xyxz() const noexcept { return make_vector(x, y, x, z); }
[[nodiscard]] auto xyyx() const noexcept { return make_vector(x, y, y, x); }
[[nodiscard]] auto xyyy() const noexcept { return make_vector(x, y, y, y); }
[[nodiscard]] auto xyyz() const noexcept { return make_vector(x, y, y, z); }
[[nodiscard]] auto xyzx() const noexcept { return make_vector(x, y, z, x); }
[[nodiscard]] auto xyzy() const noexcept { return make_vector(x, y, z, y); }
[[nodiscard]] auto xyzz() const noexcept { return make_vector(x, y, z, z); }
[[nodiscard]] auto xzxx() const noexcept { return make_vector(x, z, x, x); }
[[nodiscard]] auto xzxy() const noexcept { return make_vector(x, z, x, y); }
[[nodiscard]] auto xzxz() const noexcept { return make_vector(x, z, x, z); }
[[nodiscard]] auto xzyx() const noexcept { return make_vector(x, z, y, x); }
[[nodiscard]] auto xzyy() const noexcept { return make_vector(x, z, y, y); }
[[nodiscard]] auto xzyz() const noexcept { return make_vector(x, z, y, z); }
[[nodiscard]] auto xzzx() const noexcept { return make_vector(x, z, z, x); }
[[nodiscard]] auto xzzy() const noexcept { return make_vector(x, z, z, y); }
[[nodiscard]] auto xzzz() const noexcept { return make_vector(x, z, z, z); }
[[nodiscard]] auto yxxx() const noexcept { return make_vector(y, x, x, x); }
[[nodiscard]] auto yxxy() const noexcept { return make_vector(y, x, x, y); }
[[nodiscard]] auto yxxz() const noexcept { return make_vector(y, x, x, z); }
[[nodiscard]] auto yxyx() const noexcept { return make_vector(y, x, y, x); }
[[nodiscard]] auto yxyy() const noexcept { return make_vector(y, x, y, y); }
[[nodiscard]] auto yxyz() const noexcept { return make_vector(y, x, y, z); }
[[nodiscard]] auto yxzx() const noexcept { return make_vector(y, x, z, x); }
[[nodiscard]] auto yxzy() const noexcept { return make_vector(y, x, z, y); }
[[nodiscard]] auto yxzz() const noexcept { return make_vector(y, x, z, z); }
[[nodiscard]] auto yyxx() const noexcept { return make_vector(y, y, x, x); }
[[nodiscard]] auto yyxy() const noexcept { return make_vector(y, y, x, y); }
[[nodiscard]] auto yyxz() const noexcept { return make_vector(y, y, x, z); }
[[nodiscard]] auto yyyx() const noexcept { return make_vector(y, y, y, x); }
[[nodiscard]] auto yyyy() const noexcept { return make_vector(y, y, y, y); }
[[nodiscard]] auto yyyz() const noexcept { return make_vector(y, y, y, z); }
[[nodiscard]] auto yyzx() const noexcept { return make_vector(y, y, z, x); }
[[nodiscard]] auto yyzy() const noexcept { return make_vector(y, y, z, y); }
[[nodiscard]] auto yyzz() const noexcept { return make_vector(y, y, z, z); }
[[nodiscard]] auto yzxx() const noexcept { return make_vector(y, z, x, x); }
[[nodiscard]] auto yzxy() const noexcept { return make_vector(y, z, x, y); }
[[nodiscard]] auto yzxz() const noexcept { return make_vector(y, z, x, z); }
[[nodiscard]] auto yzyx() const noexcept { return make_vector(y, z, y, x); }
[[nodiscard]] auto yzyy() const noexcept { return make_vector(y, z, y, y); }
[[nodiscard]] auto yzyz() const noexcept { return make_vector(y, z, y, z); }
[[nodiscard]] auto yzzx() const noexcept { return make_vector(y, z, z, x); }
[[nodiscard]] auto yzzy() const noexcept { return make_vector(y, z, z, y); }
[[nodiscard]] auto yzzz() const noexcept { return make_vector(y, z, z, z); }
[[nodiscard]] auto zxxx() const noexcept { return make_vector(z, x, x, x); }
[[nodiscard]] auto zxxy() const noexcept { return make_vector(z, x, x, y); }
[[nodiscard]] auto zxxz() const noexcept { return make_vector(z, x, x, z); }
[[nodiscard]] auto zxyx() const noexcept { return make_vector(z, x, y, x); }
[[nodiscard]] auto zxyy() const noexcept { return make_vector(z, x, y, y); }
[[nodiscard]] auto zxyz() const noexcept { return make_vector(z, x, y, z); }
[[nodiscard]] auto zxzx() const noexcept { return make_vector(z, x, z, x); }
[[nodiscard]] auto zxzy() const noexcept { return make_vector(z, x, z, y); }
[[nodiscard]] auto zxzz() const noexcept { return make_vector(z, x, z, z); }
[[nodiscard]] auto zyxx() const noexcept { return make_vector(z, y, x, x); }
[[nodiscard]] auto zyxy() const noexcept { return make_vector(z, y, x, y); }
[[nodiscard]] auto zyxz() const noexcept { return make_vector(z, y, x, z); }
[[nodiscard]] auto zyyx() const noexcept { return make_vector(z, y, y, x); }
[[nodiscard]] auto zyyy() const noexcept { return make_vector(z, y, y, y); }
[[nodiscard]] auto zyyz() const noexcept { return make_vector(z, y, y, z); }
[[nodiscard]] auto zyzx() const noexcept { return make_vector(z, y, z, x); }
[[nodiscard]] auto zyzy() const noexcept { return make_vector(z, y, z, y); }
[[nodiscard]] auto zyzz() const noexcept { return make_vector(z, y, z, z); }
[[nodiscard]] auto zzxx() const noexcept { return make_vector(z, z, x, x); }
[[nodiscard]] auto zzxy() const noexcept { return make_vector(z, z, x, y); }
[[nodiscard]] auto zzxz() const noexcept { return make_vector(z, z, x, z); }
[[nodiscard]] auto zzyx() const noexcept { return make_vector(z, z, y, x); }
[[nodiscard]] auto zzyy() const noexcept { return make_vector(z, z, y, y); }
[[nodiscard]] auto zzyz() const noexcept { return make_vector(z, z, y, z); }
[[nodiscard]] auto zzzx() const noexcept { return make_vector(z, z, z, x); }
[[nodiscard]] auto zzzy() const noexcept { return make_vector(z, z, z, y); }
[[nodiscard]] auto zzzz() const noexcept { return make_vector(z, z, z, z); }
