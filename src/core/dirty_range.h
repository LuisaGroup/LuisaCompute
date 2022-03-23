//
// Created by Mike on 2021/12/1.
//

#pragma once

#include <cstddef>
#include <cstdint>

namespace luisa::compute {

/**
 * @brief A class to mark dirty range.
 * Dirty range = [offset, offset + size)
 */
class LC_CORE_API DirtyRange {

private:
    size_t _offset{};
    size_t _size{};

public:
    /**
     * @brief Clear all dirty ranges
     * 
     */
    void clear() noexcept;
    /**
     * @brief Mark index as dirty
     * 
     * @param index
     */
    void mark(size_t index) noexcept;
    /**
     * @brief Merge 2 dirty ranges
     * 
     * @param r1 dirty range 1
     * @param r2 dirty range 2
     * @return DirtyRange 
     */
    [[nodiscard]] static DirtyRange merge(DirtyRange r1, DirtyRange r2) noexcept;
    /**
     * @brief If dirty range is empty(size == 0)
     * 
     * @return true / false
     */
    [[nodiscard]] auto empty() const noexcept { return _size == 0u; }
    /**
     * @brief Return start index of dirty range
     * 
     * @return size_t
     */
    [[nodiscard]] auto offset() const noexcept { return _offset; }
    /**
     * @brief Return size of dirty range
     * 
     * @return size_t
     */
    [[nodiscard]] auto size() const noexcept { return _size; }
};

}
