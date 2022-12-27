#include <runtime/dynamic_buffer.h>
#include <runtime/dynamic_struct.h>
#include <core/stl/format.h>

namespace luisa::compute {

Buffer<DynamicStruct> Device::create_buffer(const DynamicStruct &type, size_t size) noexcept {
    return _create<Buffer<DynamicStruct>>(type.type(), size);
}

DynamicStruct::DynamicStruct(luisa::span<std::pair<luisa::string_view, const Type *> const> types) noexcept {
    using namespace std::literals;
    size_t align = 4;
    for (auto &&i : types) {
        align = std::max(align, i.second->alignment());
    }
    auto name = luisa::format("struct<{}", align);
    for (auto &&i : types) {
        name += ',';
        name += i.second->description();
    }
    name += '>';
    _type = Type::from(name);
    _idx_map.reserve(types.size());
    size_t idx = 0;
    for (auto &&i : types) {
        _idx_map.try_emplace(i.first, idx);
        ++idx;
    }
}

const Type* DynamicStruct::member(luisa::string_view name) const noexcept{
    auto ite = _idx_map.find(name);
    if(ite == _idx_map.end()) return nullptr;
    return _type->members()[ite->second];
}

size_t DynamicStruct::member_index(luisa::string_view name) const noexcept{
    auto ite = _idx_map.find(name);
    if(ite == _idx_map.end()) return ~0ull;
    return ite->second;
}

DynamicStruct::~DynamicStruct() noexcept {}

}// namespace luisa::compute
