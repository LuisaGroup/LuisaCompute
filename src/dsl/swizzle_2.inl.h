#pragma once
[[nodiscard]] auto xx() const noexcept { return make_vector2(x, x); }
[[nodiscard]] auto xy() const noexcept { return make_vector2(x, y); }
[[nodiscard]] auto yx() const noexcept { return make_vector2(y, x); }
[[nodiscard]] auto yy() const noexcept { return make_vector2(y, y); }
[[nodiscard]] auto xxx() const noexcept { return make_vector3(x, x, x); }
[[nodiscard]] auto xxy() const noexcept { return make_vector3(x, x, y); }
[[nodiscard]] auto xyx() const noexcept { return make_vector3(x, y, x); }
[[nodiscard]] auto xyy() const noexcept { return make_vector3(x, y, y); }
[[nodiscard]] auto yxx() const noexcept { return make_vector3(y, x, x); }
[[nodiscard]] auto yxy() const noexcept { return make_vector3(y, x, y); }
[[nodiscard]] auto yyx() const noexcept { return make_vector3(y, y, x); }
[[nodiscard]] auto yyy() const noexcept { return make_vector3(y, y, y); }
[[nodiscard]] auto xxxx() const noexcept { return make_vector4(x, x, x, x); }
[[nodiscard]] auto xxxy() const noexcept { return make_vector4(x, x, x, y); }
[[nodiscard]] auto xxyx() const noexcept { return make_vector4(x, x, y, x); }
[[nodiscard]] auto xxyy() const noexcept { return make_vector4(x, x, y, y); }
[[nodiscard]] auto xyxx() const noexcept { return make_vector4(x, y, x, x); }
[[nodiscard]] auto xyxy() const noexcept { return make_vector4(x, y, x, y); }
[[nodiscard]] auto xyyx() const noexcept { return make_vector4(x, y, y, x); }
[[nodiscard]] auto xyyy() const noexcept { return make_vector4(x, y, y, y); }
[[nodiscard]] auto yxxx() const noexcept { return make_vector4(y, x, x, x); }
[[nodiscard]] auto yxxy() const noexcept { return make_vector4(y, x, x, y); }
[[nodiscard]] auto yxyx() const noexcept { return make_vector4(y, x, y, x); }
[[nodiscard]] auto yxyy() const noexcept { return make_vector4(y, x, y, y); }
[[nodiscard]] auto yyxx() const noexcept { return make_vector4(y, y, x, x); }
[[nodiscard]] auto yyxy() const noexcept { return make_vector4(y, y, x, y); }
[[nodiscard]] auto yyyx() const noexcept { return make_vector4(y, y, y, x); }
[[nodiscard]] auto yyyy() const noexcept { return make_vector4(y, y, y, y); }
