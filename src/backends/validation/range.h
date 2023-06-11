#pragma once
#include <stdint.h>
#include <numeric>
namespace lc::validation {
struct Range {
    uint64_t min;
    uint64_t max;
    Range() : min{0}, max{std::numeric_limits<uint64_t>::max()} {}
    Range(uint64_t min, uint64_t size) : min{min}, max{min + size} {}
    static bool collide(Range const& l, Range const &r) {
        return l.min < r.max && r.min < l.max;
    }
    bool operator==(Range const &r) const {
        return min == r.min && max == r.max;
    }
    bool operator!=(Range const &r) const { return !operator==(r); }
};
}// namespace lc::validation
