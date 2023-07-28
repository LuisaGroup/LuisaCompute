#pragma once

#include <luisa/osl/hint.h>

namespace luisa::compute::osl {

class Type;
class Symbol;

class LC_OSL_API Instruction {

private:
    luisa::string _opcode;
    luisa::vector<const Symbol *> _args;
    luisa::vector<int> _jump_targets;
    luisa::vector<Hint> _hints;

public:
    Instruction(luisa::string opcode,
                luisa::vector<const Symbol *> args,
                luisa::vector<int> jump_targets,
                luisa::vector<Hint> hints) noexcept
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

    // for debugging
    [[nodiscard]] luisa::string dump() const noexcept;
};

}// namespace luisa::compute::osl
