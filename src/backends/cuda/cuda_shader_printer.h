#pragma once

#include <luisa/ast/type.h>

namespace luisa::compute::cuda {

class CUDAShaderPrinter {

public:
    struct Binding {
        // TODO
    };

private:

public:
    [[nodiscard]] static auto create(luisa::span<const std::pair<luisa::string, const Type *>> arg_types) noexcept {
        if (arg_types.empty()) { return luisa::unique_ptr<CUDAShaderPrinter>{}; }
        return luisa::make_unique<CUDAShaderPrinter>();// TODO
    }
    [[nodiscard]] static auto create(luisa::span<const std::pair<luisa::string, luisa::string>> arg_types) noexcept {
        luisa::vector<std::pair<luisa::string, const Type *>> types;
        types.reserve(arg_types.size());
        for (auto &&[name, type] : arg_types) {
            types.emplace_back(name, Type::from(type));
        }
        return create(types);
    }
};

}// namespace luisa::compute::cuda
