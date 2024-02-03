#pragma once
#include <luisa/core/stl/string.h>
#include <luisa/core/stl/unordered_map.h>
#include <luisa/core/stl/memory.h>
namespace luisa::compute {
struct Attribute {
    luisa::string key;
    luisa::string value;
};
struct TypeAttribute : public Attribute {
    luisa::vector<TypeAttribute> elements;
};
struct VarAttribute : public Attribute {
    luisa::shared_ptr<TypeAttribute> type;
};
struct FuncAttribute : public Attribute {
    luisa::shared_ptr<TypeAttribute> type;
};
};// namespace luisa::compute