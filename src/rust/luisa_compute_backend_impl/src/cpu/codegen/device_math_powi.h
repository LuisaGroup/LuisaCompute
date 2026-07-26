template<typename T, typename I>
[[nodiscard]] __device__ inline T powi_impl(T x, I y) noexcept {
    T r = static_cast<T>(1.0f);
    auto is_y_neg = y < static_cast<I>(0);
    // Convert first, then negate in the unsigned domain. This preserves the
    // magnitude of every signed minimum value and the complete width of every
    // unsigned exponent.
    using exponent_magnitude_type = unsigned long long;
    auto y_abs = static_cast<exponent_magnitude_type>(y);
    if (is_y_neg) y_abs = exponent_magnitude_type{0} - y_abs;

    while (y_abs != 0u) {
        if ((y_abs & 1u) != 0u) r *= x;
        x *= x;
        y_abs >>= 1;
    }
    return is_y_neg ? static_cast<T>(1.0f) / r : r;
}
