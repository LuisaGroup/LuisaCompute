#pragma once

namespace luisa::compute::tensor {
enum class MatrixASide {
    NONE = 0,
    LEFT = 1,
    RIGHT = 2
};

class MatrixMulOptions {
public:
    MatrixASide side;
};
}