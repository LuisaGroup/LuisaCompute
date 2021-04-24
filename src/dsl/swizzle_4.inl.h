#pragma once
[[nodiscard]] auto xx() const noexcept { return make_vector(x, x); }
[[nodiscard]] auto xy() const noexcept { return make_vector(x, y); }
[[nodiscard]] auto xz() const noexcept { return make_vector(x, z); }
[[nodiscard]] auto xw() const noexcept { return make_vector(x, w); }
[[nodiscard]] auto yx() const noexcept { return make_vector(y, x); }
[[nodiscard]] auto yy() const noexcept { return make_vector(y, y); }
[[nodiscard]] auto yz() const noexcept { return make_vector(y, z); }
[[nodiscard]] auto yw() const noexcept { return make_vector(y, w); }
[[nodiscard]] auto zx() const noexcept { return make_vector(z, x); }
[[nodiscard]] auto zy() const noexcept { return make_vector(z, y); }
[[nodiscard]] auto zz() const noexcept { return make_vector(z, z); }
[[nodiscard]] auto zw() const noexcept { return make_vector(z, w); }
[[nodiscard]] auto wx() const noexcept { return make_vector(w, x); }
[[nodiscard]] auto wy() const noexcept { return make_vector(w, y); }
[[nodiscard]] auto wz() const noexcept { return make_vector(w, z); }
[[nodiscard]] auto ww() const noexcept { return make_vector(w, w); }
[[nodiscard]] auto xxx() const noexcept { return make_vector(x, x, x); }
[[nodiscard]] auto xxy() const noexcept { return make_vector(x, x, y); }
[[nodiscard]] auto xxz() const noexcept { return make_vector(x, x, z); }
[[nodiscard]] auto xxw() const noexcept { return make_vector(x, x, w); }
[[nodiscard]] auto xyx() const noexcept { return make_vector(x, y, x); }
[[nodiscard]] auto xyy() const noexcept { return make_vector(x, y, y); }
[[nodiscard]] auto xyz() const noexcept { return make_vector(x, y, z); }
[[nodiscard]] auto xyw() const noexcept { return make_vector(x, y, w); }
[[nodiscard]] auto xzx() const noexcept { return make_vector(x, z, x); }
[[nodiscard]] auto xzy() const noexcept { return make_vector(x, z, y); }
[[nodiscard]] auto xzz() const noexcept { return make_vector(x, z, z); }
[[nodiscard]] auto xzw() const noexcept { return make_vector(x, z, w); }
[[nodiscard]] auto xwx() const noexcept { return make_vector(x, w, x); }
[[nodiscard]] auto xwy() const noexcept { return make_vector(x, w, y); }
[[nodiscard]] auto xwz() const noexcept { return make_vector(x, w, z); }
[[nodiscard]] auto xww() const noexcept { return make_vector(x, w, w); }
[[nodiscard]] auto yxx() const noexcept { return make_vector(y, x, x); }
[[nodiscard]] auto yxy() const noexcept { return make_vector(y, x, y); }
[[nodiscard]] auto yxz() const noexcept { return make_vector(y, x, z); }
[[nodiscard]] auto yxw() const noexcept { return make_vector(y, x, w); }
[[nodiscard]] auto yyx() const noexcept { return make_vector(y, y, x); }
[[nodiscard]] auto yyy() const noexcept { return make_vector(y, y, y); }
[[nodiscard]] auto yyz() const noexcept { return make_vector(y, y, z); }
[[nodiscard]] auto yyw() const noexcept { return make_vector(y, y, w); }
[[nodiscard]] auto yzx() const noexcept { return make_vector(y, z, x); }
[[nodiscard]] auto yzy() const noexcept { return make_vector(y, z, y); }
[[nodiscard]] auto yzz() const noexcept { return make_vector(y, z, z); }
[[nodiscard]] auto yzw() const noexcept { return make_vector(y, z, w); }
[[nodiscard]] auto ywx() const noexcept { return make_vector(y, w, x); }
[[nodiscard]] auto ywy() const noexcept { return make_vector(y, w, y); }
[[nodiscard]] auto ywz() const noexcept { return make_vector(y, w, z); }
[[nodiscard]] auto yww() const noexcept { return make_vector(y, w, w); }
[[nodiscard]] auto zxx() const noexcept { return make_vector(z, x, x); }
[[nodiscard]] auto zxy() const noexcept { return make_vector(z, x, y); }
[[nodiscard]] auto zxz() const noexcept { return make_vector(z, x, z); }
[[nodiscard]] auto zxw() const noexcept { return make_vector(z, x, w); }
[[nodiscard]] auto zyx() const noexcept { return make_vector(z, y, x); }
[[nodiscard]] auto zyy() const noexcept { return make_vector(z, y, y); }
[[nodiscard]] auto zyz() const noexcept { return make_vector(z, y, z); }
[[nodiscard]] auto zyw() const noexcept { return make_vector(z, y, w); }
[[nodiscard]] auto zzx() const noexcept { return make_vector(z, z, x); }
[[nodiscard]] auto zzy() const noexcept { return make_vector(z, z, y); }
[[nodiscard]] auto zzz() const noexcept { return make_vector(z, z, z); }
[[nodiscard]] auto zzw() const noexcept { return make_vector(z, z, w); }
[[nodiscard]] auto zwx() const noexcept { return make_vector(z, w, x); }
[[nodiscard]] auto zwy() const noexcept { return make_vector(z, w, y); }
[[nodiscard]] auto zwz() const noexcept { return make_vector(z, w, z); }
[[nodiscard]] auto zww() const noexcept { return make_vector(z, w, w); }
[[nodiscard]] auto wxx() const noexcept { return make_vector(w, x, x); }
[[nodiscard]] auto wxy() const noexcept { return make_vector(w, x, y); }
[[nodiscard]] auto wxz() const noexcept { return make_vector(w, x, z); }
[[nodiscard]] auto wxw() const noexcept { return make_vector(w, x, w); }
[[nodiscard]] auto wyx() const noexcept { return make_vector(w, y, x); }
[[nodiscard]] auto wyy() const noexcept { return make_vector(w, y, y); }
[[nodiscard]] auto wyz() const noexcept { return make_vector(w, y, z); }
[[nodiscard]] auto wyw() const noexcept { return make_vector(w, y, w); }
[[nodiscard]] auto wzx() const noexcept { return make_vector(w, z, x); }
[[nodiscard]] auto wzy() const noexcept { return make_vector(w, z, y); }
[[nodiscard]] auto wzz() const noexcept { return make_vector(w, z, z); }
[[nodiscard]] auto wzw() const noexcept { return make_vector(w, z, w); }
[[nodiscard]] auto wwx() const noexcept { return make_vector(w, w, x); }
[[nodiscard]] auto wwy() const noexcept { return make_vector(w, w, y); }
[[nodiscard]] auto wwz() const noexcept { return make_vector(w, w, z); }
[[nodiscard]] auto www() const noexcept { return make_vector(w, w, w); }
[[nodiscard]] auto xxxx() const noexcept { return make_vector(x, x, x, x); }
[[nodiscard]] auto xxxy() const noexcept { return make_vector(x, x, x, y); }
[[nodiscard]] auto xxxz() const noexcept { return make_vector(x, x, x, z); }
[[nodiscard]] auto xxxw() const noexcept { return make_vector(x, x, x, w); }
[[nodiscard]] auto xxyx() const noexcept { return make_vector(x, x, y, x); }
[[nodiscard]] auto xxyy() const noexcept { return make_vector(x, x, y, y); }
[[nodiscard]] auto xxyz() const noexcept { return make_vector(x, x, y, z); }
[[nodiscard]] auto xxyw() const noexcept { return make_vector(x, x, y, w); }
[[nodiscard]] auto xxzx() const noexcept { return make_vector(x, x, z, x); }
[[nodiscard]] auto xxzy() const noexcept { return make_vector(x, x, z, y); }
[[nodiscard]] auto xxzz() const noexcept { return make_vector(x, x, z, z); }
[[nodiscard]] auto xxzw() const noexcept { return make_vector(x, x, z, w); }
[[nodiscard]] auto xxwx() const noexcept { return make_vector(x, x, w, x); }
[[nodiscard]] auto xxwy() const noexcept { return make_vector(x, x, w, y); }
[[nodiscard]] auto xxwz() const noexcept { return make_vector(x, x, w, z); }
[[nodiscard]] auto xxww() const noexcept { return make_vector(x, x, w, w); }
[[nodiscard]] auto xyxx() const noexcept { return make_vector(x, y, x, x); }
[[nodiscard]] auto xyxy() const noexcept { return make_vector(x, y, x, y); }
[[nodiscard]] auto xyxz() const noexcept { return make_vector(x, y, x, z); }
[[nodiscard]] auto xyxw() const noexcept { return make_vector(x, y, x, w); }
[[nodiscard]] auto xyyx() const noexcept { return make_vector(x, y, y, x); }
[[nodiscard]] auto xyyy() const noexcept { return make_vector(x, y, y, y); }
[[nodiscard]] auto xyyz() const noexcept { return make_vector(x, y, y, z); }
[[nodiscard]] auto xyyw() const noexcept { return make_vector(x, y, y, w); }
[[nodiscard]] auto xyzx() const noexcept { return make_vector(x, y, z, x); }
[[nodiscard]] auto xyzy() const noexcept { return make_vector(x, y, z, y); }
[[nodiscard]] auto xyzz() const noexcept { return make_vector(x, y, z, z); }
[[nodiscard]] auto xyzw() const noexcept { return make_vector(x, y, z, w); }
[[nodiscard]] auto xywx() const noexcept { return make_vector(x, y, w, x); }
[[nodiscard]] auto xywy() const noexcept { return make_vector(x, y, w, y); }
[[nodiscard]] auto xywz() const noexcept { return make_vector(x, y, w, z); }
[[nodiscard]] auto xyww() const noexcept { return make_vector(x, y, w, w); }
[[nodiscard]] auto xzxx() const noexcept { return make_vector(x, z, x, x); }
[[nodiscard]] auto xzxy() const noexcept { return make_vector(x, z, x, y); }
[[nodiscard]] auto xzxz() const noexcept { return make_vector(x, z, x, z); }
[[nodiscard]] auto xzxw() const noexcept { return make_vector(x, z, x, w); }
[[nodiscard]] auto xzyx() const noexcept { return make_vector(x, z, y, x); }
[[nodiscard]] auto xzyy() const noexcept { return make_vector(x, z, y, y); }
[[nodiscard]] auto xzyz() const noexcept { return make_vector(x, z, y, z); }
[[nodiscard]] auto xzyw() const noexcept { return make_vector(x, z, y, w); }
[[nodiscard]] auto xzzx() const noexcept { return make_vector(x, z, z, x); }
[[nodiscard]] auto xzzy() const noexcept { return make_vector(x, z, z, y); }
[[nodiscard]] auto xzzz() const noexcept { return make_vector(x, z, z, z); }
[[nodiscard]] auto xzzw() const noexcept { return make_vector(x, z, z, w); }
[[nodiscard]] auto xzwx() const noexcept { return make_vector(x, z, w, x); }
[[nodiscard]] auto xzwy() const noexcept { return make_vector(x, z, w, y); }
[[nodiscard]] auto xzwz() const noexcept { return make_vector(x, z, w, z); }
[[nodiscard]] auto xzww() const noexcept { return make_vector(x, z, w, w); }
[[nodiscard]] auto xwxx() const noexcept { return make_vector(x, w, x, x); }
[[nodiscard]] auto xwxy() const noexcept { return make_vector(x, w, x, y); }
[[nodiscard]] auto xwxz() const noexcept { return make_vector(x, w, x, z); }
[[nodiscard]] auto xwxw() const noexcept { return make_vector(x, w, x, w); }
[[nodiscard]] auto xwyx() const noexcept { return make_vector(x, w, y, x); }
[[nodiscard]] auto xwyy() const noexcept { return make_vector(x, w, y, y); }
[[nodiscard]] auto xwyz() const noexcept { return make_vector(x, w, y, z); }
[[nodiscard]] auto xwyw() const noexcept { return make_vector(x, w, y, w); }
[[nodiscard]] auto xwzx() const noexcept { return make_vector(x, w, z, x); }
[[nodiscard]] auto xwzy() const noexcept { return make_vector(x, w, z, y); }
[[nodiscard]] auto xwzz() const noexcept { return make_vector(x, w, z, z); }
[[nodiscard]] auto xwzw() const noexcept { return make_vector(x, w, z, w); }
[[nodiscard]] auto xwwx() const noexcept { return make_vector(x, w, w, x); }
[[nodiscard]] auto xwwy() const noexcept { return make_vector(x, w, w, y); }
[[nodiscard]] auto xwwz() const noexcept { return make_vector(x, w, w, z); }
[[nodiscard]] auto xwww() const noexcept { return make_vector(x, w, w, w); }
[[nodiscard]] auto yxxx() const noexcept { return make_vector(y, x, x, x); }
[[nodiscard]] auto yxxy() const noexcept { return make_vector(y, x, x, y); }
[[nodiscard]] auto yxxz() const noexcept { return make_vector(y, x, x, z); }
[[nodiscard]] auto yxxw() const noexcept { return make_vector(y, x, x, w); }
[[nodiscard]] auto yxyx() const noexcept { return make_vector(y, x, y, x); }
[[nodiscard]] auto yxyy() const noexcept { return make_vector(y, x, y, y); }
[[nodiscard]] auto yxyz() const noexcept { return make_vector(y, x, y, z); }
[[nodiscard]] auto yxyw() const noexcept { return make_vector(y, x, y, w); }
[[nodiscard]] auto yxzx() const noexcept { return make_vector(y, x, z, x); }
[[nodiscard]] auto yxzy() const noexcept { return make_vector(y, x, z, y); }
[[nodiscard]] auto yxzz() const noexcept { return make_vector(y, x, z, z); }
[[nodiscard]] auto yxzw() const noexcept { return make_vector(y, x, z, w); }
[[nodiscard]] auto yxwx() const noexcept { return make_vector(y, x, w, x); }
[[nodiscard]] auto yxwy() const noexcept { return make_vector(y, x, w, y); }
[[nodiscard]] auto yxwz() const noexcept { return make_vector(y, x, w, z); }
[[nodiscard]] auto yxww() const noexcept { return make_vector(y, x, w, w); }
[[nodiscard]] auto yyxx() const noexcept { return make_vector(y, y, x, x); }
[[nodiscard]] auto yyxy() const noexcept { return make_vector(y, y, x, y); }
[[nodiscard]] auto yyxz() const noexcept { return make_vector(y, y, x, z); }
[[nodiscard]] auto yyxw() const noexcept { return make_vector(y, y, x, w); }
[[nodiscard]] auto yyyx() const noexcept { return make_vector(y, y, y, x); }
[[nodiscard]] auto yyyy() const noexcept { return make_vector(y, y, y, y); }
[[nodiscard]] auto yyyz() const noexcept { return make_vector(y, y, y, z); }
[[nodiscard]] auto yyyw() const noexcept { return make_vector(y, y, y, w); }
[[nodiscard]] auto yyzx() const noexcept { return make_vector(y, y, z, x); }
[[nodiscard]] auto yyzy() const noexcept { return make_vector(y, y, z, y); }
[[nodiscard]] auto yyzz() const noexcept { return make_vector(y, y, z, z); }
[[nodiscard]] auto yyzw() const noexcept { return make_vector(y, y, z, w); }
[[nodiscard]] auto yywx() const noexcept { return make_vector(y, y, w, x); }
[[nodiscard]] auto yywy() const noexcept { return make_vector(y, y, w, y); }
[[nodiscard]] auto yywz() const noexcept { return make_vector(y, y, w, z); }
[[nodiscard]] auto yyww() const noexcept { return make_vector(y, y, w, w); }
[[nodiscard]] auto yzxx() const noexcept { return make_vector(y, z, x, x); }
[[nodiscard]] auto yzxy() const noexcept { return make_vector(y, z, x, y); }
[[nodiscard]] auto yzxz() const noexcept { return make_vector(y, z, x, z); }
[[nodiscard]] auto yzxw() const noexcept { return make_vector(y, z, x, w); }
[[nodiscard]] auto yzyx() const noexcept { return make_vector(y, z, y, x); }
[[nodiscard]] auto yzyy() const noexcept { return make_vector(y, z, y, y); }
[[nodiscard]] auto yzyz() const noexcept { return make_vector(y, z, y, z); }
[[nodiscard]] auto yzyw() const noexcept { return make_vector(y, z, y, w); }
[[nodiscard]] auto yzzx() const noexcept { return make_vector(y, z, z, x); }
[[nodiscard]] auto yzzy() const noexcept { return make_vector(y, z, z, y); }
[[nodiscard]] auto yzzz() const noexcept { return make_vector(y, z, z, z); }
[[nodiscard]] auto yzzw() const noexcept { return make_vector(y, z, z, w); }
[[nodiscard]] auto yzwx() const noexcept { return make_vector(y, z, w, x); }
[[nodiscard]] auto yzwy() const noexcept { return make_vector(y, z, w, y); }
[[nodiscard]] auto yzwz() const noexcept { return make_vector(y, z, w, z); }
[[nodiscard]] auto yzww() const noexcept { return make_vector(y, z, w, w); }
[[nodiscard]] auto ywxx() const noexcept { return make_vector(y, w, x, x); }
[[nodiscard]] auto ywxy() const noexcept { return make_vector(y, w, x, y); }
[[nodiscard]] auto ywxz() const noexcept { return make_vector(y, w, x, z); }
[[nodiscard]] auto ywxw() const noexcept { return make_vector(y, w, x, w); }
[[nodiscard]] auto ywyx() const noexcept { return make_vector(y, w, y, x); }
[[nodiscard]] auto ywyy() const noexcept { return make_vector(y, w, y, y); }
[[nodiscard]] auto ywyz() const noexcept { return make_vector(y, w, y, z); }
[[nodiscard]] auto ywyw() const noexcept { return make_vector(y, w, y, w); }
[[nodiscard]] auto ywzx() const noexcept { return make_vector(y, w, z, x); }
[[nodiscard]] auto ywzy() const noexcept { return make_vector(y, w, z, y); }
[[nodiscard]] auto ywzz() const noexcept { return make_vector(y, w, z, z); }
[[nodiscard]] auto ywzw() const noexcept { return make_vector(y, w, z, w); }
[[nodiscard]] auto ywwx() const noexcept { return make_vector(y, w, w, x); }
[[nodiscard]] auto ywwy() const noexcept { return make_vector(y, w, w, y); }
[[nodiscard]] auto ywwz() const noexcept { return make_vector(y, w, w, z); }
[[nodiscard]] auto ywww() const noexcept { return make_vector(y, w, w, w); }
[[nodiscard]] auto zxxx() const noexcept { return make_vector(z, x, x, x); }
[[nodiscard]] auto zxxy() const noexcept { return make_vector(z, x, x, y); }
[[nodiscard]] auto zxxz() const noexcept { return make_vector(z, x, x, z); }
[[nodiscard]] auto zxxw() const noexcept { return make_vector(z, x, x, w); }
[[nodiscard]] auto zxyx() const noexcept { return make_vector(z, x, y, x); }
[[nodiscard]] auto zxyy() const noexcept { return make_vector(z, x, y, y); }
[[nodiscard]] auto zxyz() const noexcept { return make_vector(z, x, y, z); }
[[nodiscard]] auto zxyw() const noexcept { return make_vector(z, x, y, w); }
[[nodiscard]] auto zxzx() const noexcept { return make_vector(z, x, z, x); }
[[nodiscard]] auto zxzy() const noexcept { return make_vector(z, x, z, y); }
[[nodiscard]] auto zxzz() const noexcept { return make_vector(z, x, z, z); }
[[nodiscard]] auto zxzw() const noexcept { return make_vector(z, x, z, w); }
[[nodiscard]] auto zxwx() const noexcept { return make_vector(z, x, w, x); }
[[nodiscard]] auto zxwy() const noexcept { return make_vector(z, x, w, y); }
[[nodiscard]] auto zxwz() const noexcept { return make_vector(z, x, w, z); }
[[nodiscard]] auto zxww() const noexcept { return make_vector(z, x, w, w); }
[[nodiscard]] auto zyxx() const noexcept { return make_vector(z, y, x, x); }
[[nodiscard]] auto zyxy() const noexcept { return make_vector(z, y, x, y); }
[[nodiscard]] auto zyxz() const noexcept { return make_vector(z, y, x, z); }
[[nodiscard]] auto zyxw() const noexcept { return make_vector(z, y, x, w); }
[[nodiscard]] auto zyyx() const noexcept { return make_vector(z, y, y, x); }
[[nodiscard]] auto zyyy() const noexcept { return make_vector(z, y, y, y); }
[[nodiscard]] auto zyyz() const noexcept { return make_vector(z, y, y, z); }
[[nodiscard]] auto zyyw() const noexcept { return make_vector(z, y, y, w); }
[[nodiscard]] auto zyzx() const noexcept { return make_vector(z, y, z, x); }
[[nodiscard]] auto zyzy() const noexcept { return make_vector(z, y, z, y); }
[[nodiscard]] auto zyzz() const noexcept { return make_vector(z, y, z, z); }
[[nodiscard]] auto zyzw() const noexcept { return make_vector(z, y, z, w); }
[[nodiscard]] auto zywx() const noexcept { return make_vector(z, y, w, x); }
[[nodiscard]] auto zywy() const noexcept { return make_vector(z, y, w, y); }
[[nodiscard]] auto zywz() const noexcept { return make_vector(z, y, w, z); }
[[nodiscard]] auto zyww() const noexcept { return make_vector(z, y, w, w); }
[[nodiscard]] auto zzxx() const noexcept { return make_vector(z, z, x, x); }
[[nodiscard]] auto zzxy() const noexcept { return make_vector(z, z, x, y); }
[[nodiscard]] auto zzxz() const noexcept { return make_vector(z, z, x, z); }
[[nodiscard]] auto zzxw() const noexcept { return make_vector(z, z, x, w); }
[[nodiscard]] auto zzyx() const noexcept { return make_vector(z, z, y, x); }
[[nodiscard]] auto zzyy() const noexcept { return make_vector(z, z, y, y); }
[[nodiscard]] auto zzyz() const noexcept { return make_vector(z, z, y, z); }
[[nodiscard]] auto zzyw() const noexcept { return make_vector(z, z, y, w); }
[[nodiscard]] auto zzzx() const noexcept { return make_vector(z, z, z, x); }
[[nodiscard]] auto zzzy() const noexcept { return make_vector(z, z, z, y); }
[[nodiscard]] auto zzzz() const noexcept { return make_vector(z, z, z, z); }
[[nodiscard]] auto zzzw() const noexcept { return make_vector(z, z, z, w); }
[[nodiscard]] auto zzwx() const noexcept { return make_vector(z, z, w, x); }
[[nodiscard]] auto zzwy() const noexcept { return make_vector(z, z, w, y); }
[[nodiscard]] auto zzwz() const noexcept { return make_vector(z, z, w, z); }
[[nodiscard]] auto zzww() const noexcept { return make_vector(z, z, w, w); }
[[nodiscard]] auto zwxx() const noexcept { return make_vector(z, w, x, x); }
[[nodiscard]] auto zwxy() const noexcept { return make_vector(z, w, x, y); }
[[nodiscard]] auto zwxz() const noexcept { return make_vector(z, w, x, z); }
[[nodiscard]] auto zwxw() const noexcept { return make_vector(z, w, x, w); }
[[nodiscard]] auto zwyx() const noexcept { return make_vector(z, w, y, x); }
[[nodiscard]] auto zwyy() const noexcept { return make_vector(z, w, y, y); }
[[nodiscard]] auto zwyz() const noexcept { return make_vector(z, w, y, z); }
[[nodiscard]] auto zwyw() const noexcept { return make_vector(z, w, y, w); }
[[nodiscard]] auto zwzx() const noexcept { return make_vector(z, w, z, x); }
[[nodiscard]] auto zwzy() const noexcept { return make_vector(z, w, z, y); }
[[nodiscard]] auto zwzz() const noexcept { return make_vector(z, w, z, z); }
[[nodiscard]] auto zwzw() const noexcept { return make_vector(z, w, z, w); }
[[nodiscard]] auto zwwx() const noexcept { return make_vector(z, w, w, x); }
[[nodiscard]] auto zwwy() const noexcept { return make_vector(z, w, w, y); }
[[nodiscard]] auto zwwz() const noexcept { return make_vector(z, w, w, z); }
[[nodiscard]] auto zwww() const noexcept { return make_vector(z, w, w, w); }
[[nodiscard]] auto wxxx() const noexcept { return make_vector(w, x, x, x); }
[[nodiscard]] auto wxxy() const noexcept { return make_vector(w, x, x, y); }
[[nodiscard]] auto wxxz() const noexcept { return make_vector(w, x, x, z); }
[[nodiscard]] auto wxxw() const noexcept { return make_vector(w, x, x, w); }
[[nodiscard]] auto wxyx() const noexcept { return make_vector(w, x, y, x); }
[[nodiscard]] auto wxyy() const noexcept { return make_vector(w, x, y, y); }
[[nodiscard]] auto wxyz() const noexcept { return make_vector(w, x, y, z); }
[[nodiscard]] auto wxyw() const noexcept { return make_vector(w, x, y, w); }
[[nodiscard]] auto wxzx() const noexcept { return make_vector(w, x, z, x); }
[[nodiscard]] auto wxzy() const noexcept { return make_vector(w, x, z, y); }
[[nodiscard]] auto wxzz() const noexcept { return make_vector(w, x, z, z); }
[[nodiscard]] auto wxzw() const noexcept { return make_vector(w, x, z, w); }
[[nodiscard]] auto wxwx() const noexcept { return make_vector(w, x, w, x); }
[[nodiscard]] auto wxwy() const noexcept { return make_vector(w, x, w, y); }
[[nodiscard]] auto wxwz() const noexcept { return make_vector(w, x, w, z); }
[[nodiscard]] auto wxww() const noexcept { return make_vector(w, x, w, w); }
[[nodiscard]] auto wyxx() const noexcept { return make_vector(w, y, x, x); }
[[nodiscard]] auto wyxy() const noexcept { return make_vector(w, y, x, y); }
[[nodiscard]] auto wyxz() const noexcept { return make_vector(w, y, x, z); }
[[nodiscard]] auto wyxw() const noexcept { return make_vector(w, y, x, w); }
[[nodiscard]] auto wyyx() const noexcept { return make_vector(w, y, y, x); }
[[nodiscard]] auto wyyy() const noexcept { return make_vector(w, y, y, y); }
[[nodiscard]] auto wyyz() const noexcept { return make_vector(w, y, y, z); }
[[nodiscard]] auto wyyw() const noexcept { return make_vector(w, y, y, w); }
[[nodiscard]] auto wyzx() const noexcept { return make_vector(w, y, z, x); }
[[nodiscard]] auto wyzy() const noexcept { return make_vector(w, y, z, y); }
[[nodiscard]] auto wyzz() const noexcept { return make_vector(w, y, z, z); }
[[nodiscard]] auto wyzw() const noexcept { return make_vector(w, y, z, w); }
[[nodiscard]] auto wywx() const noexcept { return make_vector(w, y, w, x); }
[[nodiscard]] auto wywy() const noexcept { return make_vector(w, y, w, y); }
[[nodiscard]] auto wywz() const noexcept { return make_vector(w, y, w, z); }
[[nodiscard]] auto wyww() const noexcept { return make_vector(w, y, w, w); }
[[nodiscard]] auto wzxx() const noexcept { return make_vector(w, z, x, x); }
[[nodiscard]] auto wzxy() const noexcept { return make_vector(w, z, x, y); }
[[nodiscard]] auto wzxz() const noexcept { return make_vector(w, z, x, z); }
[[nodiscard]] auto wzxw() const noexcept { return make_vector(w, z, x, w); }
[[nodiscard]] auto wzyx() const noexcept { return make_vector(w, z, y, x); }
[[nodiscard]] auto wzyy() const noexcept { return make_vector(w, z, y, y); }
[[nodiscard]] auto wzyz() const noexcept { return make_vector(w, z, y, z); }
[[nodiscard]] auto wzyw() const noexcept { return make_vector(w, z, y, w); }
[[nodiscard]] auto wzzx() const noexcept { return make_vector(w, z, z, x); }
[[nodiscard]] auto wzzy() const noexcept { return make_vector(w, z, z, y); }
[[nodiscard]] auto wzzz() const noexcept { return make_vector(w, z, z, z); }
[[nodiscard]] auto wzzw() const noexcept { return make_vector(w, z, z, w); }
[[nodiscard]] auto wzwx() const noexcept { return make_vector(w, z, w, x); }
[[nodiscard]] auto wzwy() const noexcept { return make_vector(w, z, w, y); }
[[nodiscard]] auto wzwz() const noexcept { return make_vector(w, z, w, z); }
[[nodiscard]] auto wzww() const noexcept { return make_vector(w, z, w, w); }
[[nodiscard]] auto wwxx() const noexcept { return make_vector(w, w, x, x); }
[[nodiscard]] auto wwxy() const noexcept { return make_vector(w, w, x, y); }
[[nodiscard]] auto wwxz() const noexcept { return make_vector(w, w, x, z); }
[[nodiscard]] auto wwxw() const noexcept { return make_vector(w, w, x, w); }
[[nodiscard]] auto wwyx() const noexcept { return make_vector(w, w, y, x); }
[[nodiscard]] auto wwyy() const noexcept { return make_vector(w, w, y, y); }
[[nodiscard]] auto wwyz() const noexcept { return make_vector(w, w, y, z); }
[[nodiscard]] auto wwyw() const noexcept { return make_vector(w, w, y, w); }
[[nodiscard]] auto wwzx() const noexcept { return make_vector(w, w, z, x); }
[[nodiscard]] auto wwzy() const noexcept { return make_vector(w, w, z, y); }
[[nodiscard]] auto wwzz() const noexcept { return make_vector(w, w, z, z); }
[[nodiscard]] auto wwzw() const noexcept { return make_vector(w, w, z, w); }
[[nodiscard]] auto wwwx() const noexcept { return make_vector(w, w, w, x); }
[[nodiscard]] auto wwwy() const noexcept { return make_vector(w, w, w, y); }
[[nodiscard]] auto wwwz() const noexcept { return make_vector(w, w, w, z); }
[[nodiscard]] auto wwww() const noexcept { return make_vector(w, w, w, w); }
