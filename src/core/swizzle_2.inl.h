[[nodiscard]] constexpr auto xx() const noexcept { return Vector<T, 2>{x, x}; }
[[nodiscard]] constexpr auto xy() const noexcept { return Vector<T, 2>{x, y}; }
[[nodiscard]] constexpr auto yx() const noexcept { return Vector<T, 2>{y, x}; }
[[nodiscard]] constexpr auto yy() const noexcept { return Vector<T, 2>{y, y}; }
[[nodiscard]] constexpr auto xxx() const noexcept { return Vector<T, 3>{x, x, x}; }
[[nodiscard]] constexpr auto xxy() const noexcept { return Vector<T, 3>{x, x, y}; }
[[nodiscard]] constexpr auto xyx() const noexcept { return Vector<T, 3>{x, y, x}; }
[[nodiscard]] constexpr auto xyy() const noexcept { return Vector<T, 3>{x, y, y}; }
[[nodiscard]] constexpr auto yxx() const noexcept { return Vector<T, 3>{y, x, x}; }
[[nodiscard]] constexpr auto yxy() const noexcept { return Vector<T, 3>{y, x, y}; }
[[nodiscard]] constexpr auto yyx() const noexcept { return Vector<T, 3>{y, y, x}; }
[[nodiscard]] constexpr auto yyy() const noexcept { return Vector<T, 3>{y, y, y}; }
[[nodiscard]] constexpr auto xxxx() const noexcept { return Vector<T, 4>{x, x, x, x}; }
[[nodiscard]] constexpr auto xxxy() const noexcept { return Vector<T, 4>{x, x, x, y}; }
[[nodiscard]] constexpr auto xxyx() const noexcept { return Vector<T, 4>{x, x, y, x}; }
[[nodiscard]] constexpr auto xxyy() const noexcept { return Vector<T, 4>{x, x, y, y}; }
[[nodiscard]] constexpr auto xyxx() const noexcept { return Vector<T, 4>{x, y, x, x}; }
[[nodiscard]] constexpr auto xyxy() const noexcept { return Vector<T, 4>{x, y, x, y}; }
[[nodiscard]] constexpr auto xyyx() const noexcept { return Vector<T, 4>{x, y, y, x}; }
[[nodiscard]] constexpr auto xyyy() const noexcept { return Vector<T, 4>{x, y, y, y}; }
[[nodiscard]] constexpr auto yxxx() const noexcept { return Vector<T, 4>{y, x, x, x}; }
[[nodiscard]] constexpr auto yxxy() const noexcept { return Vector<T, 4>{y, x, x, y}; }
[[nodiscard]] constexpr auto yxyx() const noexcept { return Vector<T, 4>{y, x, y, x}; }
[[nodiscard]] constexpr auto yxyy() const noexcept { return Vector<T, 4>{y, x, y, y}; }
[[nodiscard]] constexpr auto yyxx() const noexcept { return Vector<T, 4>{y, y, x, x}; }
[[nodiscard]] constexpr auto yyxy() const noexcept { return Vector<T, 4>{y, y, x, y}; }
[[nodiscard]] constexpr auto yyyx() const noexcept { return Vector<T, 4>{y, y, y, x}; }
[[nodiscard]] constexpr auto yyyy() const noexcept { return Vector<T, 4>{y, y, y, y}; }
