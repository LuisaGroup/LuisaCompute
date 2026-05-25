//
// Created by Mike on 2024/5/10.
//

#include <luisa/core/logging.h>
#include <luisa/coro/coro_frame_desc.h>

namespace luisa::compute::coroutine {

CoroFrameDesc::CoroFrameDesc(const Type *type, DesignatedFieldDict m) noexcept
    : _type{type}, _designated_fields{std::move(m)} {
    LUISA_ASSERT(_type != nullptr, "CoroFrame underlying type must not be null.");
    LUISA_ASSERT(_type->is_structure(), "CoroFrame underlying type must be a structure.");
    LUISA_ASSERT(_type->members().size() >= 2u, "CoroFrame underlying type must have at least 2 members (coro_id and target_token).");
    LUISA_ASSERT(_type->members()[0]->is_uint32_vector() && _type->members()[0]->dimension() == 3u, "CoroFrame member 0 (coro_id) must be uint3.");
    LUISA_ASSERT(_type->members()[1]->is_uint32(), "CoroFrame member 1 (target_token) must be uint.");
    auto member_count = _type->members().size();
    for (auto &&[name, index] : _designated_fields) {
        LUISA_ASSERT(name != "coro_id", "CoroFrame designated member name 'coro_id' is reserved.");
        LUISA_ASSERT(name != "target_token", "CoroFrame designated member name 'target_token' is reserved.");
        LUISA_ASSERT(index != 0, "CoroFrame designated member index 0 is reserved for coro_id.");
        LUISA_ASSERT(index != 1, "CoroFrame designated member index 1 is reserved for target_token.");
        LUISA_ASSERT(index < member_count, "CoroFrame designated member index out of range.");
    }
}

luisa::shared_ptr<CoroFrameDesc> CoroFrameDesc::create(const Type *type, DesignatedFieldDict m) noexcept {
    return luisa::make_shared<CoroFrameDesc>(CoroFrameDesc{type, std::move(m)});
}

const Type *CoroFrameDesc::field(uint index) const noexcept {
    auto fields = _type->members();
    LUISA_ASSERT(index < fields.size(), "CoroFrame field index out of range.");
    return fields[index];
}

uint CoroFrameDesc::designated_field(luisa::string_view name) const noexcept {
    if (name == "coro_id") { return 0u; }
    if (name == "target_token") { return 1u; }
    auto iter = _designated_fields.find(name);
    LUISA_ASSERT(iter != _designated_fields.end(), "CoroFrame designated member not found.");
    return iter->second;
}

luisa::string CoroFrameDesc::dump() const noexcept {
    luisa::string s;
    for (auto i = 0u; i < _type->members().size(); i++) {
        s.append(luisa::format("  Field {}: {}\n", i, _type->members()[i]->description()));
    }
    if (!_designated_fields.empty()) {
        s.append("Designated Fields:\n");
        for (auto &&[name, index] : _designated_fields) {
            s.append(luisa::format("  {} -> \"{}\"\n", index, name));
        }
    }
    return s;
}

}// namespace luisa::compute::coroutine
