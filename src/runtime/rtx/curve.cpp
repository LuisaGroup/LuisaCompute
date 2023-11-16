#include <luisa/runtime/rtx/curve.h>

namespace luisa::compute {

Curve::~Curve() noexcept {
    if (*this) { device()->destroy_curve(handle()); }
}

CurveBasis Curve::basis() const noexcept {
    _check_is_valid();
    return _basis;
}

size_t Curve::control_point_count() const noexcept {
    _check_is_valid();
    return _cp_count;
}

size_t Curve::segment_count() const noexcept {
    _check_is_valid();
    return _seg_count;
}

luisa::unique_ptr<Command> Curve::build(BuildRequest request) noexcept {
    _check_is_valid();
    return luisa::make_unique<CurveBuildCommand>(
        handle(), request,
        _basis, _cp_count, _seg_count,
        _cp_buffer, _cp_buffer_offset, _cp_stride,
        _seg_buffer, _seg_buffer_offset);
}

}// namespace luisa::compute
