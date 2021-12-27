//
// Created by Mike on 2021/12/1.
//

#pragma once

#include <cstddef>
#include <cstdint>

namespace luisa::compute {

class DirtyRange {

private:
    size_t _offset{};
    size_t _size{};

public:
    void clear() noexcept;
    void mark(size_t index) noexcept;
    [[nodiscard]] static DirtyRange merge(DirtyRange r1, DirtyRange r2) noexcept;
    [[nodiscard]] auto empty() const noexcept { return _size == 0u; }
    [[nodiscard]] auto offset() const noexcept { return _offset; }
    [[nodiscard]] auto size() const noexcept { return _size; }
};

}
