//
// Created by Mike on 2024/5/10.
//

#pragma once

#include <luisa/core/dll_export.h>
#include <luisa/core/stl/string.h>
#include <luisa/core/stl/unordered_map.h>
#include <luisa/ast/type.h>

namespace luisa::compute::coroutine {

class LUISA_CORO_API CoroFrameDesc : public luisa::enable_shared_from_this<CoroFrameDesc> {

public:
    using DesignatedFieldDict = luisa::unordered_map<luisa::string, uint>;

private:
    const Type *_type{nullptr};
    DesignatedFieldDict _designated_fields;

private:
    CoroFrameDesc(const Type *type, DesignatedFieldDict m) noexcept;

public:
    [[nodiscard]] static luisa::shared_ptr<CoroFrameDesc> create(const Type *type, DesignatedFieldDict m) noexcept;
    [[nodiscard]] auto type() const noexcept { return _type; }
    [[nodiscard]] const Type *field(uint index) const noexcept;
    [[nodiscard]] auto &designated_fields() const noexcept { return _designated_fields; }
    [[nodiscard]] uint designated_field(luisa::string_view name) const noexcept;
    [[nodiscard]] luisa::string dump() const noexcept;
};

}// namespace luisa::compute::coroutine
