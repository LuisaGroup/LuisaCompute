#pragma once
#include <luisa/core/mathematics.h>

namespace luisa::compute::tensor {

class dense_vector_indexing {
public:
    static constexpr auto index(auto inc, auto i) noexcept { return inc * i; }
    static constexpr auto min_buffer_size(auto inc, auto n) noexcept { return inc * n; }
};

class general_dense_matrix_indexing {
public:
    static constexpr auto index(auto lda, auto i, auto j) noexcept { return lda * j + i; }
    static constexpr auto min_buffer_size(auto lda, auto col) noexcept { return lda * col; }
};

class triangular_matrix_lower_mode_indexing {
public:
    static constexpr auto index(auto lda, auto i, auto j) noexcept { return lda * j + i; }

    static constexpr auto i_min(auto j) noexcept { return j; }
    static constexpr auto i_max(auto col) noexcept { return col - 1; }
    static constexpr auto min_lda(auto row) noexcept { return row; }
    static constexpr auto min_buffer_size(auto lda, auto col) noexcept { return lda * col; }
};

class triangular_matrix_upper_mode_indexing {
public:
    static constexpr auto index(auto lda, auto i, auto j) noexcept { return lda * j + i; }

    static constexpr auto i_min() noexcept { return 0; }
    static constexpr auto i_max(auto j) noexcept { return j; }
    static constexpr auto min_lda(auto row) noexcept { return row; }
    static constexpr auto min_buffer_size(auto lda, auto col) noexcept { return lda * col; }
};

class general_band_matrix_indexing {
public:
    static constexpr auto index(auto lda, auto i, auto j, auto ku) noexcept {
        return general_dense_matrix_indexing::index(lda, ku + i - j, j);
    }

    static constexpr auto i_min(auto j, auto ku) noexcept { return max(0, j - ku); }

    static constexpr auto i_max(auto row, auto j, auto kl) { return min(row, j + kl); }

    static constexpr auto min_lda(auto kl, auto ku) noexcept { return kl + ku + 1; }
};

class symmetric_band_matrix_lower_mode_indexing {
public:
    static constexpr auto index(auto lda, auto i, auto j) noexcept {
        return general_dense_matrix_indexing::index(lda, i - j, j);
    }

    static constexpr auto i_min(auto j) noexcept { return j; }

    static constexpr auto i_max(auto row, auto j, auto k) noexcept {
        return min(row - 1, j + k);
    }

    static constexpr auto min_lda(auto row) noexcept { return row; }
    static constexpr auto min_buffer_size(auto lda, auto col) noexcept { return lda * col; }
};

class symmetric_band_matrix_upper_mode_indexing {
public:
    static constexpr auto index(auto lda, auto i, auto j, auto k) noexcept {
        return general_dense_matrix_indexing::index(lda, k + i - j, j);
    }

    static constexpr auto i_min(auto j, auto k) noexcept { return max(0, j - k); }

    static constexpr auto i_max(auto j) noexcept { return j; }

    static constexpr auto min_lda(auto row) noexcept { return row; }
    static constexpr auto min_buffer_size(auto lda, auto col) noexcept { return lda * col; }
};

using triangular_band_matrix_upper_mode_indexing = symmetric_band_matrix_upper_mode_indexing;

using triangular_band_matrix_lower_mode_indexing = symmetric_band_matrix_lower_mode_indexing;

class packed_symmetric_matrix_upper_mode_indexing {
public:
    static constexpr auto index(auto i, auto j) noexcept {
        return i + (j * (j + 1)) / 2;
    }
    static constexpr auto i_min(auto j) noexcept { return j; }
    static constexpr auto buffer_size(auto n) noexcept { return n * (n + 1) / 2; }
};

class packed_symmetric_matrix_lower_mode_indexing {
public:
    static constexpr auto index(auto n, auto i, auto j) noexcept {
        return i + ((2 * n - j + 1) * j) / 2;
    }
    static constexpr auto i_max(auto j) noexcept { return j; }
    static constexpr auto buffer_size(auto n) noexcept { return n * (n + 1) / 2; }
};

}// namespace luisa::compute::tensor