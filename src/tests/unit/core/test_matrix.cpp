// Test for matrix layout and basic matrix operations.
//
// This test demonstrates Luisa's column-major matrix storage format
// and verifies matrix-matrix and matrix-vector multiplication.
//
// In Luisa, float4x4 stores columns contiguously in memory:
//   cols[0] = column 0 = (m00, m10, m20, m30)
//   cols[1] = column 1 = (m01, m11, m21, m31)
//   ...
// This is the standard OpenGL/GLSL convention.

#include <luisa/core/mathematics.h>
#include <luisa/core/stl/string.h>
#include <luisa/core/logging.h>
#include "ut/ut.hpp"
#include <random>

using namespace luisa;
using namespace boost::ut;
using namespace boost::ut::literals;

// Print matrix in row-major order for human readability
void print_matrix_in_row(float4x4 matrix) {
    luisa::string result;
    for (int row = 0; row < 4; ++row) {
        for (int col = 0; col < 4; ++col) {
            // Access element at (row, col) via column-major storage
            result += luisa::format("{}, ", matrix.cols[col][row]);
        }
        result += "\n";
    }
    LUISA_INFO("Print Matrix in row-major:\n{}", result);
}

// Print matrix in its native column-major storage order
void print_matrix_in_col(float4x4 matrix) {
    luisa::string result;
    for (int col = 0; col < 4; ++col) {
        for (int row = 0; row < 4; ++row) {
            result += luisa::format("{}, ", matrix.cols[col][row]);
        }
        result += "\n";
    }
    LUISA_INFO("Print Matrix in column-major:\n{}", result);
}

// Extract a row from a column-major matrix
// Returns the row as a float4 vector
float4 get_matrix_row(float4x4 matrix, uint32_t row_index) {
    return float4(matrix.cols[0][row_index], matrix.cols[1][row_index], matrix.cols[2][row_index], matrix.cols[3][row_index]);
}

static auto test_matrix_registration = [] {
    "test_matrix"_test = [] {
        //////////////////////////////////// Layout Demonstration
        // Luisa's matrix is column-major. For float4x4, the memory layout is:
        // cols[0] = first column vector, cols[1] = second column vector, etc.
        float4x4 my_matrix = make_float4x4(
            float4(11, 12, 13, 14),// column 0 (rows 0-3)
            float4(21, 22, 23, 24),// column 1 (rows 0-3)
            float4(31, 32, 33, 34),// column 2 (rows 0-3)
            float4(41, 42, 43, 44) // column 3 (rows 0-3)
        );
        print_matrix_in_row(my_matrix);
        print_matrix_in_col(my_matrix);
        /*
    When printed in row-order, it appears transposed from the input:
        11, 21, 31, 41,  
        12, 22, 32, 42,  
        13, 23, 33, 43, 
        14, 24, 34, 44,
    */

        //////////////////////////////////// Matrix-Matrix Multiplication Test
        // Initialize random number generator for test matrices
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dist(0, 1);

        // Create two random 4x4 matrices
        float4x4 lhs_matrix;
        float4x4 rhs_matrix;
        for (int row = 0; row < 4; ++row) {
            for (int col = 0; col < 4; ++col) {
                lhs_matrix.cols[col][row] = dist(gen);
                rhs_matrix.cols[col][row] = dist(gen);
            }
        }

        // Compute C = A * B using built-in operator
        float4x4 result = lhs_matrix * rhs_matrix;

        // Verify result by manual multiplication
        // For column-major: result[row][col] = dot(A[row], B[col])
        for (int row = 0; row < 4; ++row) {
            for (int col = 0; col < 4; ++col) {
                // Extract row from LHS and column from RHS for dot product
                float multiply_result = dot(get_matrix_row(lhs_matrix, row), rhs_matrix.cols[col]);
                // Assert that built-in multiplication matches manual calculation
                expect(static_cast<bool>(abs(result.cols[col][row] - multiply_result) <= 1e-5f));
            }
        }

        //////////////////////////////////// Matrix-Vector Multiplication Test
        // Create random 4D vector
        float4 rhs_vector{
            dist(gen),
            dist(gen),
            dist(gen),
            dist(gen)};

        // Compute v' = M * v using built-in operator
        float4 result_vector = lhs_matrix * rhs_vector;

        // Verify by manual calculation
        for (int row = 0; row < 4; ++row) {
            // Each output element is dot product of matrix row with input vector
            float multiply_result = dot(get_matrix_row(lhs_matrix, row), rhs_vector);
            expect(static_cast<bool>(abs(result_vector[row] - multiply_result) <= 1e-5f));
        }
    };
    return 0;
}();
int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));

}
