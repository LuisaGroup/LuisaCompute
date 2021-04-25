#pragma once
[[nodiscard]] auto xx() const noexcept { return Vector<T, 2>{x, x}; }
[[nodiscard]] auto xy() const noexcept { return Vector<T, 2>{x, y}; }
[[nodiscard]] auto xz() const noexcept { return Vector<T, 2>{x, z}; }
[[nodiscard]] auto xw() const noexcept { return Vector<T, 2>{x, w}; }
[[nodiscard]] auto yx() const noexcept { return Vector<T, 2>{y, x}; }
[[nodiscard]] auto yy() const noexcept { return Vector<T, 2>{y, y}; }
[[nodiscard]] auto yz() const noexcept { return Vector<T, 2>{y, z}; }
[[nodiscard]] auto yw() const noexcept { return Vector<T, 2>{y, w}; }
[[nodiscard]] auto zx() const noexcept { return Vector<T, 2>{z, x}; }
[[nodiscard]] auto zy() const noexcept { return Vector<T, 2>{z, y}; }
[[nodiscard]] auto zz() const noexcept { return Vector<T, 2>{z, z}; }
[[nodiscard]] auto zw() const noexcept { return Vector<T, 2>{z, w}; }
[[nodiscard]] auto wx() const noexcept { return Vector<T, 2>{w, x}; }
[[nodiscard]] auto wy() const noexcept { return Vector<T, 2>{w, y}; }
[[nodiscard]] auto wz() const noexcept { return Vector<T, 2>{w, z}; }
[[nodiscard]] auto ww() const noexcept { return Vector<T, 2>{w, w}; }
[[nodiscard]] auto xxx() const noexcept { return Vector<T, 3>{x, x, x}; }
[[nodiscard]] auto xxy() const noexcept { return Vector<T, 3>{x, x, y}; }
[[nodiscard]] auto xxz() const noexcept { return Vector<T, 3>{x, x, z}; }
[[nodiscard]] auto xxw() const noexcept { return Vector<T, 3>{x, x, w}; }
[[nodiscard]] auto xyx() const noexcept { return Vector<T, 3>{x, y, x}; }
[[nodiscard]] auto xyy() const noexcept { return Vector<T, 3>{x, y, y}; }
[[nodiscard]] auto xyz() const noexcept { return Vector<T, 3>{x, y, z}; }
[[nodiscard]] auto xyw() const noexcept { return Vector<T, 3>{x, y, w}; }
[[nodiscard]] auto xzx() const noexcept { return Vector<T, 3>{x, z, x}; }
[[nodiscard]] auto xzy() const noexcept { return Vector<T, 3>{x, z, y}; }
[[nodiscard]] auto xzz() const noexcept { return Vector<T, 3>{x, z, z}; }
[[nodiscard]] auto xzw() const noexcept { return Vector<T, 3>{x, z, w}; }
[[nodiscard]] auto xwx() const noexcept { return Vector<T, 3>{x, w, x}; }
[[nodiscard]] auto xwy() const noexcept { return Vector<T, 3>{x, w, y}; }
[[nodiscard]] auto xwz() const noexcept { return Vector<T, 3>{x, w, z}; }
[[nodiscard]] auto xww() const noexcept { return Vector<T, 3>{x, w, w}; }
[[nodiscard]] auto yxx() const noexcept { return Vector<T, 3>{y, x, x}; }
[[nodiscard]] auto yxy() const noexcept { return Vector<T, 3>{y, x, y}; }
[[nodiscard]] auto yxz() const noexcept { return Vector<T, 3>{y, x, z}; }
[[nodiscard]] auto yxw() const noexcept { return Vector<T, 3>{y, x, w}; }
[[nodiscard]] auto yyx() const noexcept { return Vector<T, 3>{y, y, x}; }
[[nodiscard]] auto yyy() const noexcept { return Vector<T, 3>{y, y, y}; }
[[nodiscard]] auto yyz() const noexcept { return Vector<T, 3>{y, y, z}; }
[[nodiscard]] auto yyw() const noexcept { return Vector<T, 3>{y, y, w}; }
[[nodiscard]] auto yzx() const noexcept { return Vector<T, 3>{y, z, x}; }
[[nodiscard]] auto yzy() const noexcept { return Vector<T, 3>{y, z, y}; }
[[nodiscard]] auto yzz() const noexcept { return Vector<T, 3>{y, z, z}; }
[[nodiscard]] auto yzw() const noexcept { return Vector<T, 3>{y, z, w}; }
[[nodiscard]] auto ywx() const noexcept { return Vector<T, 3>{y, w, x}; }
[[nodiscard]] auto ywy() const noexcept { return Vector<T, 3>{y, w, y}; }
[[nodiscard]] auto ywz() const noexcept { return Vector<T, 3>{y, w, z}; }
[[nodiscard]] auto yww() const noexcept { return Vector<T, 3>{y, w, w}; }
[[nodiscard]] auto zxx() const noexcept { return Vector<T, 3>{z, x, x}; }
[[nodiscard]] auto zxy() const noexcept { return Vector<T, 3>{z, x, y}; }
[[nodiscard]] auto zxz() const noexcept { return Vector<T, 3>{z, x, z}; }
[[nodiscard]] auto zxw() const noexcept { return Vector<T, 3>{z, x, w}; }
[[nodiscard]] auto zyx() const noexcept { return Vector<T, 3>{z, y, x}; }
[[nodiscard]] auto zyy() const noexcept { return Vector<T, 3>{z, y, y}; }
[[nodiscard]] auto zyz() const noexcept { return Vector<T, 3>{z, y, z}; }
[[nodiscard]] auto zyw() const noexcept { return Vector<T, 3>{z, y, w}; }
[[nodiscard]] auto zzx() const noexcept { return Vector<T, 3>{z, z, x}; }
[[nodiscard]] auto zzy() const noexcept { return Vector<T, 3>{z, z, y}; }
[[nodiscard]] auto zzz() const noexcept { return Vector<T, 3>{z, z, z}; }
[[nodiscard]] auto zzw() const noexcept { return Vector<T, 3>{z, z, w}; }
[[nodiscard]] auto zwx() const noexcept { return Vector<T, 3>{z, w, x}; }
[[nodiscard]] auto zwy() const noexcept { return Vector<T, 3>{z, w, y}; }
[[nodiscard]] auto zwz() const noexcept { return Vector<T, 3>{z, w, z}; }
[[nodiscard]] auto zww() const noexcept { return Vector<T, 3>{z, w, w}; }
[[nodiscard]] auto wxx() const noexcept { return Vector<T, 3>{w, x, x}; }
[[nodiscard]] auto wxy() const noexcept { return Vector<T, 3>{w, x, y}; }
[[nodiscard]] auto wxz() const noexcept { return Vector<T, 3>{w, x, z}; }
[[nodiscard]] auto wxw() const noexcept { return Vector<T, 3>{w, x, w}; }
[[nodiscard]] auto wyx() const noexcept { return Vector<T, 3>{w, y, x}; }
[[nodiscard]] auto wyy() const noexcept { return Vector<T, 3>{w, y, y}; }
[[nodiscard]] auto wyz() const noexcept { return Vector<T, 3>{w, y, z}; }
[[nodiscard]] auto wyw() const noexcept { return Vector<T, 3>{w, y, w}; }
[[nodiscard]] auto wzx() const noexcept { return Vector<T, 3>{w, z, x}; }
[[nodiscard]] auto wzy() const noexcept { return Vector<T, 3>{w, z, y}; }
[[nodiscard]] auto wzz() const noexcept { return Vector<T, 3>{w, z, z}; }
[[nodiscard]] auto wzw() const noexcept { return Vector<T, 3>{w, z, w}; }
[[nodiscard]] auto wwx() const noexcept { return Vector<T, 3>{w, w, x}; }
[[nodiscard]] auto wwy() const noexcept { return Vector<T, 3>{w, w, y}; }
[[nodiscard]] auto wwz() const noexcept { return Vector<T, 3>{w, w, z}; }
[[nodiscard]] auto www() const noexcept { return Vector<T, 3>{w, w, w}; }
[[nodiscard]] auto xxxx() const noexcept { return Vector<T, 4>{x, x, x, x}; }
[[nodiscard]] auto xxxy() const noexcept { return Vector<T, 4>{x, x, x, y}; }
[[nodiscard]] auto xxxz() const noexcept { return Vector<T, 4>{x, x, x, z}; }
[[nodiscard]] auto xxxw() const noexcept { return Vector<T, 4>{x, x, x, w}; }
[[nodiscard]] auto xxyx() const noexcept { return Vector<T, 4>{x, x, y, x}; }
[[nodiscard]] auto xxyy() const noexcept { return Vector<T, 4>{x, x, y, y}; }
[[nodiscard]] auto xxyz() const noexcept { return Vector<T, 4>{x, x, y, z}; }
[[nodiscard]] auto xxyw() const noexcept { return Vector<T, 4>{x, x, y, w}; }
[[nodiscard]] auto xxzx() const noexcept { return Vector<T, 4>{x, x, z, x}; }
[[nodiscard]] auto xxzy() const noexcept { return Vector<T, 4>{x, x, z, y}; }
[[nodiscard]] auto xxzz() const noexcept { return Vector<T, 4>{x, x, z, z}; }
[[nodiscard]] auto xxzw() const noexcept { return Vector<T, 4>{x, x, z, w}; }
[[nodiscard]] auto xxwx() const noexcept { return Vector<T, 4>{x, x, w, x}; }
[[nodiscard]] auto xxwy() const noexcept { return Vector<T, 4>{x, x, w, y}; }
[[nodiscard]] auto xxwz() const noexcept { return Vector<T, 4>{x, x, w, z}; }
[[nodiscard]] auto xxww() const noexcept { return Vector<T, 4>{x, x, w, w}; }
[[nodiscard]] auto xyxx() const noexcept { return Vector<T, 4>{x, y, x, x}; }
[[nodiscard]] auto xyxy() const noexcept { return Vector<T, 4>{x, y, x, y}; }
[[nodiscard]] auto xyxz() const noexcept { return Vector<T, 4>{x, y, x, z}; }
[[nodiscard]] auto xyxw() const noexcept { return Vector<T, 4>{x, y, x, w}; }
[[nodiscard]] auto xyyx() const noexcept { return Vector<T, 4>{x, y, y, x}; }
[[nodiscard]] auto xyyy() const noexcept { return Vector<T, 4>{x, y, y, y}; }
[[nodiscard]] auto xyyz() const noexcept { return Vector<T, 4>{x, y, y, z}; }
[[nodiscard]] auto xyyw() const noexcept { return Vector<T, 4>{x, y, y, w}; }
[[nodiscard]] auto xyzx() const noexcept { return Vector<T, 4>{x, y, z, x}; }
[[nodiscard]] auto xyzy() const noexcept { return Vector<T, 4>{x, y, z, y}; }
[[nodiscard]] auto xyzz() const noexcept { return Vector<T, 4>{x, y, z, z}; }
[[nodiscard]] auto xyzw() const noexcept { return Vector<T, 4>{x, y, z, w}; }
[[nodiscard]] auto xywx() const noexcept { return Vector<T, 4>{x, y, w, x}; }
[[nodiscard]] auto xywy() const noexcept { return Vector<T, 4>{x, y, w, y}; }
[[nodiscard]] auto xywz() const noexcept { return Vector<T, 4>{x, y, w, z}; }
[[nodiscard]] auto xyww() const noexcept { return Vector<T, 4>{x, y, w, w}; }
[[nodiscard]] auto xzxx() const noexcept { return Vector<T, 4>{x, z, x, x}; }
[[nodiscard]] auto xzxy() const noexcept { return Vector<T, 4>{x, z, x, y}; }
[[nodiscard]] auto xzxz() const noexcept { return Vector<T, 4>{x, z, x, z}; }
[[nodiscard]] auto xzxw() const noexcept { return Vector<T, 4>{x, z, x, w}; }
[[nodiscard]] auto xzyx() const noexcept { return Vector<T, 4>{x, z, y, x}; }
[[nodiscard]] auto xzyy() const noexcept { return Vector<T, 4>{x, z, y, y}; }
[[nodiscard]] auto xzyz() const noexcept { return Vector<T, 4>{x, z, y, z}; }
[[nodiscard]] auto xzyw() const noexcept { return Vector<T, 4>{x, z, y, w}; }
[[nodiscard]] auto xzzx() const noexcept { return Vector<T, 4>{x, z, z, x}; }
[[nodiscard]] auto xzzy() const noexcept { return Vector<T, 4>{x, z, z, y}; }
[[nodiscard]] auto xzzz() const noexcept { return Vector<T, 4>{x, z, z, z}; }
[[nodiscard]] auto xzzw() const noexcept { return Vector<T, 4>{x, z, z, w}; }
[[nodiscard]] auto xzwx() const noexcept { return Vector<T, 4>{x, z, w, x}; }
[[nodiscard]] auto xzwy() const noexcept { return Vector<T, 4>{x, z, w, y}; }
[[nodiscard]] auto xzwz() const noexcept { return Vector<T, 4>{x, z, w, z}; }
[[nodiscard]] auto xzww() const noexcept { return Vector<T, 4>{x, z, w, w}; }
[[nodiscard]] auto xwxx() const noexcept { return Vector<T, 4>{x, w, x, x}; }
[[nodiscard]] auto xwxy() const noexcept { return Vector<T, 4>{x, w, x, y}; }
[[nodiscard]] auto xwxz() const noexcept { return Vector<T, 4>{x, w, x, z}; }
[[nodiscard]] auto xwxw() const noexcept { return Vector<T, 4>{x, w, x, w}; }
[[nodiscard]] auto xwyx() const noexcept { return Vector<T, 4>{x, w, y, x}; }
[[nodiscard]] auto xwyy() const noexcept { return Vector<T, 4>{x, w, y, y}; }
[[nodiscard]] auto xwyz() const noexcept { return Vector<T, 4>{x, w, y, z}; }
[[nodiscard]] auto xwyw() const noexcept { return Vector<T, 4>{x, w, y, w}; }
[[nodiscard]] auto xwzx() const noexcept { return Vector<T, 4>{x, w, z, x}; }
[[nodiscard]] auto xwzy() const noexcept { return Vector<T, 4>{x, w, z, y}; }
[[nodiscard]] auto xwzz() const noexcept { return Vector<T, 4>{x, w, z, z}; }
[[nodiscard]] auto xwzw() const noexcept { return Vector<T, 4>{x, w, z, w}; }
[[nodiscard]] auto xwwx() const noexcept { return Vector<T, 4>{x, w, w, x}; }
[[nodiscard]] auto xwwy() const noexcept { return Vector<T, 4>{x, w, w, y}; }
[[nodiscard]] auto xwwz() const noexcept { return Vector<T, 4>{x, w, w, z}; }
[[nodiscard]] auto xwww() const noexcept { return Vector<T, 4>{x, w, w, w}; }
[[nodiscard]] auto yxxx() const noexcept { return Vector<T, 4>{y, x, x, x}; }
[[nodiscard]] auto yxxy() const noexcept { return Vector<T, 4>{y, x, x, y}; }
[[nodiscard]] auto yxxz() const noexcept { return Vector<T, 4>{y, x, x, z}; }
[[nodiscard]] auto yxxw() const noexcept { return Vector<T, 4>{y, x, x, w}; }
[[nodiscard]] auto yxyx() const noexcept { return Vector<T, 4>{y, x, y, x}; }
[[nodiscard]] auto yxyy() const noexcept { return Vector<T, 4>{y, x, y, y}; }
[[nodiscard]] auto yxyz() const noexcept { return Vector<T, 4>{y, x, y, z}; }
[[nodiscard]] auto yxyw() const noexcept { return Vector<T, 4>{y, x, y, w}; }
[[nodiscard]] auto yxzx() const noexcept { return Vector<T, 4>{y, x, z, x}; }
[[nodiscard]] auto yxzy() const noexcept { return Vector<T, 4>{y, x, z, y}; }
[[nodiscard]] auto yxzz() const noexcept { return Vector<T, 4>{y, x, z, z}; }
[[nodiscard]] auto yxzw() const noexcept { return Vector<T, 4>{y, x, z, w}; }
[[nodiscard]] auto yxwx() const noexcept { return Vector<T, 4>{y, x, w, x}; }
[[nodiscard]] auto yxwy() const noexcept { return Vector<T, 4>{y, x, w, y}; }
[[nodiscard]] auto yxwz() const noexcept { return Vector<T, 4>{y, x, w, z}; }
[[nodiscard]] auto yxww() const noexcept { return Vector<T, 4>{y, x, w, w}; }
[[nodiscard]] auto yyxx() const noexcept { return Vector<T, 4>{y, y, x, x}; }
[[nodiscard]] auto yyxy() const noexcept { return Vector<T, 4>{y, y, x, y}; }
[[nodiscard]] auto yyxz() const noexcept { return Vector<T, 4>{y, y, x, z}; }
[[nodiscard]] auto yyxw() const noexcept { return Vector<T, 4>{y, y, x, w}; }
[[nodiscard]] auto yyyx() const noexcept { return Vector<T, 4>{y, y, y, x}; }
[[nodiscard]] auto yyyy() const noexcept { return Vector<T, 4>{y, y, y, y}; }
[[nodiscard]] auto yyyz() const noexcept { return Vector<T, 4>{y, y, y, z}; }
[[nodiscard]] auto yyyw() const noexcept { return Vector<T, 4>{y, y, y, w}; }
[[nodiscard]] auto yyzx() const noexcept { return Vector<T, 4>{y, y, z, x}; }
[[nodiscard]] auto yyzy() const noexcept { return Vector<T, 4>{y, y, z, y}; }
[[nodiscard]] auto yyzz() const noexcept { return Vector<T, 4>{y, y, z, z}; }
[[nodiscard]] auto yyzw() const noexcept { return Vector<T, 4>{y, y, z, w}; }
[[nodiscard]] auto yywx() const noexcept { return Vector<T, 4>{y, y, w, x}; }
[[nodiscard]] auto yywy() const noexcept { return Vector<T, 4>{y, y, w, y}; }
[[nodiscard]] auto yywz() const noexcept { return Vector<T, 4>{y, y, w, z}; }
[[nodiscard]] auto yyww() const noexcept { return Vector<T, 4>{y, y, w, w}; }
[[nodiscard]] auto yzxx() const noexcept { return Vector<T, 4>{y, z, x, x}; }
[[nodiscard]] auto yzxy() const noexcept { return Vector<T, 4>{y, z, x, y}; }
[[nodiscard]] auto yzxz() const noexcept { return Vector<T, 4>{y, z, x, z}; }
[[nodiscard]] auto yzxw() const noexcept { return Vector<T, 4>{y, z, x, w}; }
[[nodiscard]] auto yzyx() const noexcept { return Vector<T, 4>{y, z, y, x}; }
[[nodiscard]] auto yzyy() const noexcept { return Vector<T, 4>{y, z, y, y}; }
[[nodiscard]] auto yzyz() const noexcept { return Vector<T, 4>{y, z, y, z}; }
[[nodiscard]] auto yzyw() const noexcept { return Vector<T, 4>{y, z, y, w}; }
[[nodiscard]] auto yzzx() const noexcept { return Vector<T, 4>{y, z, z, x}; }
[[nodiscard]] auto yzzy() const noexcept { return Vector<T, 4>{y, z, z, y}; }
[[nodiscard]] auto yzzz() const noexcept { return Vector<T, 4>{y, z, z, z}; }
[[nodiscard]] auto yzzw() const noexcept { return Vector<T, 4>{y, z, z, w}; }
[[nodiscard]] auto yzwx() const noexcept { return Vector<T, 4>{y, z, w, x}; }
[[nodiscard]] auto yzwy() const noexcept { return Vector<T, 4>{y, z, w, y}; }
[[nodiscard]] auto yzwz() const noexcept { return Vector<T, 4>{y, z, w, z}; }
[[nodiscard]] auto yzww() const noexcept { return Vector<T, 4>{y, z, w, w}; }
[[nodiscard]] auto ywxx() const noexcept { return Vector<T, 4>{y, w, x, x}; }
[[nodiscard]] auto ywxy() const noexcept { return Vector<T, 4>{y, w, x, y}; }
[[nodiscard]] auto ywxz() const noexcept { return Vector<T, 4>{y, w, x, z}; }
[[nodiscard]] auto ywxw() const noexcept { return Vector<T, 4>{y, w, x, w}; }
[[nodiscard]] auto ywyx() const noexcept { return Vector<T, 4>{y, w, y, x}; }
[[nodiscard]] auto ywyy() const noexcept { return Vector<T, 4>{y, w, y, y}; }
[[nodiscard]] auto ywyz() const noexcept { return Vector<T, 4>{y, w, y, z}; }
[[nodiscard]] auto ywyw() const noexcept { return Vector<T, 4>{y, w, y, w}; }
[[nodiscard]] auto ywzx() const noexcept { return Vector<T, 4>{y, w, z, x}; }
[[nodiscard]] auto ywzy() const noexcept { return Vector<T, 4>{y, w, z, y}; }
[[nodiscard]] auto ywzz() const noexcept { return Vector<T, 4>{y, w, z, z}; }
[[nodiscard]] auto ywzw() const noexcept { return Vector<T, 4>{y, w, z, w}; }
[[nodiscard]] auto ywwx() const noexcept { return Vector<T, 4>{y, w, w, x}; }
[[nodiscard]] auto ywwy() const noexcept { return Vector<T, 4>{y, w, w, y}; }
[[nodiscard]] auto ywwz() const noexcept { return Vector<T, 4>{y, w, w, z}; }
[[nodiscard]] auto ywww() const noexcept { return Vector<T, 4>{y, w, w, w}; }
[[nodiscard]] auto zxxx() const noexcept { return Vector<T, 4>{z, x, x, x}; }
[[nodiscard]] auto zxxy() const noexcept { return Vector<T, 4>{z, x, x, y}; }
[[nodiscard]] auto zxxz() const noexcept { return Vector<T, 4>{z, x, x, z}; }
[[nodiscard]] auto zxxw() const noexcept { return Vector<T, 4>{z, x, x, w}; }
[[nodiscard]] auto zxyx() const noexcept { return Vector<T, 4>{z, x, y, x}; }
[[nodiscard]] auto zxyy() const noexcept { return Vector<T, 4>{z, x, y, y}; }
[[nodiscard]] auto zxyz() const noexcept { return Vector<T, 4>{z, x, y, z}; }
[[nodiscard]] auto zxyw() const noexcept { return Vector<T, 4>{z, x, y, w}; }
[[nodiscard]] auto zxzx() const noexcept { return Vector<T, 4>{z, x, z, x}; }
[[nodiscard]] auto zxzy() const noexcept { return Vector<T, 4>{z, x, z, y}; }
[[nodiscard]] auto zxzz() const noexcept { return Vector<T, 4>{z, x, z, z}; }
[[nodiscard]] auto zxzw() const noexcept { return Vector<T, 4>{z, x, z, w}; }
[[nodiscard]] auto zxwx() const noexcept { return Vector<T, 4>{z, x, w, x}; }
[[nodiscard]] auto zxwy() const noexcept { return Vector<T, 4>{z, x, w, y}; }
[[nodiscard]] auto zxwz() const noexcept { return Vector<T, 4>{z, x, w, z}; }
[[nodiscard]] auto zxww() const noexcept { return Vector<T, 4>{z, x, w, w}; }
[[nodiscard]] auto zyxx() const noexcept { return Vector<T, 4>{z, y, x, x}; }
[[nodiscard]] auto zyxy() const noexcept { return Vector<T, 4>{z, y, x, y}; }
[[nodiscard]] auto zyxz() const noexcept { return Vector<T, 4>{z, y, x, z}; }
[[nodiscard]] auto zyxw() const noexcept { return Vector<T, 4>{z, y, x, w}; }
[[nodiscard]] auto zyyx() const noexcept { return Vector<T, 4>{z, y, y, x}; }
[[nodiscard]] auto zyyy() const noexcept { return Vector<T, 4>{z, y, y, y}; }
[[nodiscard]] auto zyyz() const noexcept { return Vector<T, 4>{z, y, y, z}; }
[[nodiscard]] auto zyyw() const noexcept { return Vector<T, 4>{z, y, y, w}; }
[[nodiscard]] auto zyzx() const noexcept { return Vector<T, 4>{z, y, z, x}; }
[[nodiscard]] auto zyzy() const noexcept { return Vector<T, 4>{z, y, z, y}; }
[[nodiscard]] auto zyzz() const noexcept { return Vector<T, 4>{z, y, z, z}; }
[[nodiscard]] auto zyzw() const noexcept { return Vector<T, 4>{z, y, z, w}; }
[[nodiscard]] auto zywx() const noexcept { return Vector<T, 4>{z, y, w, x}; }
[[nodiscard]] auto zywy() const noexcept { return Vector<T, 4>{z, y, w, y}; }
[[nodiscard]] auto zywz() const noexcept { return Vector<T, 4>{z, y, w, z}; }
[[nodiscard]] auto zyww() const noexcept { return Vector<T, 4>{z, y, w, w}; }
[[nodiscard]] auto zzxx() const noexcept { return Vector<T, 4>{z, z, x, x}; }
[[nodiscard]] auto zzxy() const noexcept { return Vector<T, 4>{z, z, x, y}; }
[[nodiscard]] auto zzxz() const noexcept { return Vector<T, 4>{z, z, x, z}; }
[[nodiscard]] auto zzxw() const noexcept { return Vector<T, 4>{z, z, x, w}; }
[[nodiscard]] auto zzyx() const noexcept { return Vector<T, 4>{z, z, y, x}; }
[[nodiscard]] auto zzyy() const noexcept { return Vector<T, 4>{z, z, y, y}; }
[[nodiscard]] auto zzyz() const noexcept { return Vector<T, 4>{z, z, y, z}; }
[[nodiscard]] auto zzyw() const noexcept { return Vector<T, 4>{z, z, y, w}; }
[[nodiscard]] auto zzzx() const noexcept { return Vector<T, 4>{z, z, z, x}; }
[[nodiscard]] auto zzzy() const noexcept { return Vector<T, 4>{z, z, z, y}; }
[[nodiscard]] auto zzzz() const noexcept { return Vector<T, 4>{z, z, z, z}; }
[[nodiscard]] auto zzzw() const noexcept { return Vector<T, 4>{z, z, z, w}; }
[[nodiscard]] auto zzwx() const noexcept { return Vector<T, 4>{z, z, w, x}; }
[[nodiscard]] auto zzwy() const noexcept { return Vector<T, 4>{z, z, w, y}; }
[[nodiscard]] auto zzwz() const noexcept { return Vector<T, 4>{z, z, w, z}; }
[[nodiscard]] auto zzww() const noexcept { return Vector<T, 4>{z, z, w, w}; }
[[nodiscard]] auto zwxx() const noexcept { return Vector<T, 4>{z, w, x, x}; }
[[nodiscard]] auto zwxy() const noexcept { return Vector<T, 4>{z, w, x, y}; }
[[nodiscard]] auto zwxz() const noexcept { return Vector<T, 4>{z, w, x, z}; }
[[nodiscard]] auto zwxw() const noexcept { return Vector<T, 4>{z, w, x, w}; }
[[nodiscard]] auto zwyx() const noexcept { return Vector<T, 4>{z, w, y, x}; }
[[nodiscard]] auto zwyy() const noexcept { return Vector<T, 4>{z, w, y, y}; }
[[nodiscard]] auto zwyz() const noexcept { return Vector<T, 4>{z, w, y, z}; }
[[nodiscard]] auto zwyw() const noexcept { return Vector<T, 4>{z, w, y, w}; }
[[nodiscard]] auto zwzx() const noexcept { return Vector<T, 4>{z, w, z, x}; }
[[nodiscard]] auto zwzy() const noexcept { return Vector<T, 4>{z, w, z, y}; }
[[nodiscard]] auto zwzz() const noexcept { return Vector<T, 4>{z, w, z, z}; }
[[nodiscard]] auto zwzw() const noexcept { return Vector<T, 4>{z, w, z, w}; }
[[nodiscard]] auto zwwx() const noexcept { return Vector<T, 4>{z, w, w, x}; }
[[nodiscard]] auto zwwy() const noexcept { return Vector<T, 4>{z, w, w, y}; }
[[nodiscard]] auto zwwz() const noexcept { return Vector<T, 4>{z, w, w, z}; }
[[nodiscard]] auto zwww() const noexcept { return Vector<T, 4>{z, w, w, w}; }
[[nodiscard]] auto wxxx() const noexcept { return Vector<T, 4>{w, x, x, x}; }
[[nodiscard]] auto wxxy() const noexcept { return Vector<T, 4>{w, x, x, y}; }
[[nodiscard]] auto wxxz() const noexcept { return Vector<T, 4>{w, x, x, z}; }
[[nodiscard]] auto wxxw() const noexcept { return Vector<T, 4>{w, x, x, w}; }
[[nodiscard]] auto wxyx() const noexcept { return Vector<T, 4>{w, x, y, x}; }
[[nodiscard]] auto wxyy() const noexcept { return Vector<T, 4>{w, x, y, y}; }
[[nodiscard]] auto wxyz() const noexcept { return Vector<T, 4>{w, x, y, z}; }
[[nodiscard]] auto wxyw() const noexcept { return Vector<T, 4>{w, x, y, w}; }
[[nodiscard]] auto wxzx() const noexcept { return Vector<T, 4>{w, x, z, x}; }
[[nodiscard]] auto wxzy() const noexcept { return Vector<T, 4>{w, x, z, y}; }
[[nodiscard]] auto wxzz() const noexcept { return Vector<T, 4>{w, x, z, z}; }
[[nodiscard]] auto wxzw() const noexcept { return Vector<T, 4>{w, x, z, w}; }
[[nodiscard]] auto wxwx() const noexcept { return Vector<T, 4>{w, x, w, x}; }
[[nodiscard]] auto wxwy() const noexcept { return Vector<T, 4>{w, x, w, y}; }
[[nodiscard]] auto wxwz() const noexcept { return Vector<T, 4>{w, x, w, z}; }
[[nodiscard]] auto wxww() const noexcept { return Vector<T, 4>{w, x, w, w}; }
[[nodiscard]] auto wyxx() const noexcept { return Vector<T, 4>{w, y, x, x}; }
[[nodiscard]] auto wyxy() const noexcept { return Vector<T, 4>{w, y, x, y}; }
[[nodiscard]] auto wyxz() const noexcept { return Vector<T, 4>{w, y, x, z}; }
[[nodiscard]] auto wyxw() const noexcept { return Vector<T, 4>{w, y, x, w}; }
[[nodiscard]] auto wyyx() const noexcept { return Vector<T, 4>{w, y, y, x}; }
[[nodiscard]] auto wyyy() const noexcept { return Vector<T, 4>{w, y, y, y}; }
[[nodiscard]] auto wyyz() const noexcept { return Vector<T, 4>{w, y, y, z}; }
[[nodiscard]] auto wyyw() const noexcept { return Vector<T, 4>{w, y, y, w}; }
[[nodiscard]] auto wyzx() const noexcept { return Vector<T, 4>{w, y, z, x}; }
[[nodiscard]] auto wyzy() const noexcept { return Vector<T, 4>{w, y, z, y}; }
[[nodiscard]] auto wyzz() const noexcept { return Vector<T, 4>{w, y, z, z}; }
[[nodiscard]] auto wyzw() const noexcept { return Vector<T, 4>{w, y, z, w}; }
[[nodiscard]] auto wywx() const noexcept { return Vector<T, 4>{w, y, w, x}; }
[[nodiscard]] auto wywy() const noexcept { return Vector<T, 4>{w, y, w, y}; }
[[nodiscard]] auto wywz() const noexcept { return Vector<T, 4>{w, y, w, z}; }
[[nodiscard]] auto wyww() const noexcept { return Vector<T, 4>{w, y, w, w}; }
[[nodiscard]] auto wzxx() const noexcept { return Vector<T, 4>{w, z, x, x}; }
[[nodiscard]] auto wzxy() const noexcept { return Vector<T, 4>{w, z, x, y}; }
[[nodiscard]] auto wzxz() const noexcept { return Vector<T, 4>{w, z, x, z}; }
[[nodiscard]] auto wzxw() const noexcept { return Vector<T, 4>{w, z, x, w}; }
[[nodiscard]] auto wzyx() const noexcept { return Vector<T, 4>{w, z, y, x}; }
[[nodiscard]] auto wzyy() const noexcept { return Vector<T, 4>{w, z, y, y}; }
[[nodiscard]] auto wzyz() const noexcept { return Vector<T, 4>{w, z, y, z}; }
[[nodiscard]] auto wzyw() const noexcept { return Vector<T, 4>{w, z, y, w}; }
[[nodiscard]] auto wzzx() const noexcept { return Vector<T, 4>{w, z, z, x}; }
[[nodiscard]] auto wzzy() const noexcept { return Vector<T, 4>{w, z, z, y}; }
[[nodiscard]] auto wzzz() const noexcept { return Vector<T, 4>{w, z, z, z}; }
[[nodiscard]] auto wzzw() const noexcept { return Vector<T, 4>{w, z, z, w}; }
[[nodiscard]] auto wzwx() const noexcept { return Vector<T, 4>{w, z, w, x}; }
[[nodiscard]] auto wzwy() const noexcept { return Vector<T, 4>{w, z, w, y}; }
[[nodiscard]] auto wzwz() const noexcept { return Vector<T, 4>{w, z, w, z}; }
[[nodiscard]] auto wzww() const noexcept { return Vector<T, 4>{w, z, w, w}; }
[[nodiscard]] auto wwxx() const noexcept { return Vector<T, 4>{w, w, x, x}; }
[[nodiscard]] auto wwxy() const noexcept { return Vector<T, 4>{w, w, x, y}; }
[[nodiscard]] auto wwxz() const noexcept { return Vector<T, 4>{w, w, x, z}; }
[[nodiscard]] auto wwxw() const noexcept { return Vector<T, 4>{w, w, x, w}; }
[[nodiscard]] auto wwyx() const noexcept { return Vector<T, 4>{w, w, y, x}; }
[[nodiscard]] auto wwyy() const noexcept { return Vector<T, 4>{w, w, y, y}; }
[[nodiscard]] auto wwyz() const noexcept { return Vector<T, 4>{w, w, y, z}; }
[[nodiscard]] auto wwyw() const noexcept { return Vector<T, 4>{w, w, y, w}; }
[[nodiscard]] auto wwzx() const noexcept { return Vector<T, 4>{w, w, z, x}; }
[[nodiscard]] auto wwzy() const noexcept { return Vector<T, 4>{w, w, z, y}; }
[[nodiscard]] auto wwzz() const noexcept { return Vector<T, 4>{w, w, z, z}; }
[[nodiscard]] auto wwzw() const noexcept { return Vector<T, 4>{w, w, z, w}; }
[[nodiscard]] auto wwwx() const noexcept { return Vector<T, 4>{w, w, w, x}; }
[[nodiscard]] auto wwwy() const noexcept { return Vector<T, 4>{w, w, w, y}; }
[[nodiscard]] auto wwwz() const noexcept { return Vector<T, 4>{w, w, w, z}; }
[[nodiscard]] auto wwww() const noexcept { return Vector<T, 4>{w, w, w, w}; }
