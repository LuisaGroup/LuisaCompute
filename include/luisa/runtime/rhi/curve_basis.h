#pragma once

#include <cstdint>
#include <bitset>

#include <luisa/core/basic_traits.h>
#include <luisa/core/stl/hash.h>

namespace luisa::compute {

enum class CurveBasis : uint32_t {
    PIECEWISE_LINEAR,
    QUADRATIC_BSPLINE,
    CUBIC_BSPLINE,
    CATMULL_ROM,
    BEZIER
};

static constexpr auto curve_basis_count =
    luisa::to_underlying(CurveBasis::BEZIER) + 1u;

class CurveBasisSet {

private:
    std::bitset<curve_basis_count> _set;

public:
    CurveBasisSet() noexcept : _set{0} {}
    ~CurveBasisSet() noexcept = default;
    void mark(CurveBasis basis) noexcept { _set.set(luisa::to_underlying(basis)); }
    void clear(CurveBasis basis) noexcept { _set.reset(luisa::to_underlying(basis)); }
    void clear() noexcept { _set.reset(); }
    [[nodiscard]] auto any() const noexcept { return _set.any(); }
    [[nodiscard]] auto all() const noexcept { return _set.all(); }
    [[nodiscard]] auto none() const noexcept { return _set.none(); }
    [[nodiscard]] auto test(CurveBasis basis) const noexcept { return _set.test(luisa::to_underlying(basis)); }
    [[nodiscard]] auto propagate(CurveBasisSet s) noexcept { _set |= s._set; }

public:
    [[nodiscard]] auto hash() const noexcept {
        static_assert(sizeof(_set) <= sizeof(uint64_t));
        return luisa::hash_value(_set.to_ullong());
    }

public:
    template<typename... T>
        requires std::conjunction_v<std::is_same<T, CurveBasis>...>
    [[nodiscard]] static auto make(T... bases) noexcept {
        CurveBasisSet bs;
        (bs.mark(bases), ...);
        return bs;
    }

    [[nodiscard]] static auto make_all() noexcept {
        CurveBasisSet bs;
        bs._set.set();
        return bs;
    }
};

}// namespace luisa::compute
