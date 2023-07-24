//
// Created by Mike Smith on 2023/7/24.
//

#pragma once

#include <luisa/core/stl/string.h>
#include <luisa/core/stl/vector.h>

namespace luisa::compute::osl {

class Type;
class Symbol;

class Instruction {

private:
    luisa::string _opcode;
    luisa::vector<const Symbol *> _args;
    luisa::vector<int> _jump_targets;
    luisa::vector<luisa::string> _hints;

public:
    Instruction(luisa::string opcode,
                luisa::vector<const Symbol *> args,
                luisa::vector<int> jump_targets,
                luisa::vector<luisa::string> hints) noexcept
        : _opcode{std::move(opcode)},
          _args{std::move(args)},
          _jump_targets{std::move(jump_targets)},
          _hints{std::move(hints)} {}
    ~Instruction() noexcept = default;
    Instruction(Instruction &&) noexcept = default;
    Instruction(const Instruction &) noexcept = delete;
    Instruction &operator=(Instruction &&) noexcept = default;
    Instruction &operator=(const Instruction &) noexcept = delete;
    [[nodiscard]] auto opcode() const noexcept { return luisa::string_view{_opcode}; }
    [[nodiscard]] auto args() const noexcept { return luisa::span{_args}; }
    [[nodiscard]] auto jump_targets() const noexcept { return luisa::span{_jump_targets}; }
    [[nodiscard]] auto hints() const noexcept { return luisa::span{_hints}; }
};

}// namespace luisa::compute::osl
