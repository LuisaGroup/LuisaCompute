#pragma once

#include <ast/type.h>
#include <core/stl/unordered_map.h>

namespace  luisa::compute {

class LC_RUNTIME_API DynamicStruct {

private:
    Type const *_type;
    luisa::unordered_map<luisa::string, size_t> _idx_map;

public:
    explicit DynamicStruct(luisa::span<std::pair<luisa::string_view, const Type*> const> types) noexcept;
    ~DynamicStruct() noexcept;
    [[nodiscard]] auto type() const noexcept { return _type; }
    [[nodiscard]] auto begin() const noexcept{ return _idx_map.begin(); }
    [[nodiscard]] auto end() const noexcept{ return _idx_map.end(); }
    [[nodiscard]] const Type* member(luisa::string_view name) const noexcept;
    [[nodiscard]] size_t member_index(luisa::string_view name) const noexcept;
};

}// namespace luisa::compute
