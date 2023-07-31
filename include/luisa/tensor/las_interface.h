#pragma once
#include "view.h"
namespace luisa::compute::tensor {
// linear algebric subroutine
class DTensor;
class LASInterface {
public:
    // BLAS
    // level-1
    virtual void Iamax(DTensor &result, const DTensor &vec_x) noexcept = 0;
    virtual void Iamin(DTensor &result, const DTensor &vec_x) noexcept = 0;
    virtual void dot(DTensor &result, const DTensor &vec_x, const DTensor &vec_y) noexcept = 0;
    virtual void nrm2(DTensor &result, const DTensor &vec_x) noexcept = 0;

    // level-2
    virtual void mv(DTensor &y, const DTensor &alpha, const DTensor &A, const DTensor &x, const DTensor &beta) noexcept = 0;
    virtual void sv(DTensor &x, const DTensor &A) noexcept = 0;

    // level-3
    virtual void mm(DTensor &C, const DTensor &alpha, const DTensor &A, const DTensor &B, const DTensor &beta, MatrixMulOptions options) noexcept = 0;
    virtual void sm(DTensor &X, const DTensor &alpha, const DTensor &A, MatrixMulOptions options) noexcept = 0;

    
    // SPARSE

};
}// namespace luisa::compute::tensor