//
// Created by Mike on 2021/12/1.
//

#include <utility>
#include <cmath>

#include <core/dirty_range.h>

namespace luisa::compute {

void DirtyRange::clear() noexcept {
    _offset = 0u;
    _size = 0u;
}

void DirtyRange::mark(size_t index) noexcept {
    if (empty()) {
        _offset = index;
        _size = 1u;
    } else {
        auto end = std::max(_offset + _size, index + 1u);
        _offset = std::min(_offset, index);
        _size = end - _offset;
    }
}

DirtyRange DirtyRange::merge(DirtyRange r1, DirtyRange r2) noexcept {
    if (r1.empty()) { return r2; }
    r2.mark(r1.offset());
    r2.mark(r1.offset() + r1.size() - 1u);
    return r2;
}

}// namespace luisa::compute
