#pragma once

#include <luisa/xir/basic_block.h>
#include <luisa/xir/variable.h>

namespace luisa::compute::xir {

enum struct FunctionTag {
    KERNEL,
    CALLABLE,
};

class LC_XIR_API Function : public Value {

public:

private:
    FunctionTag _function_tag;
    FunctionBodyBlock *_body = nullptr;
    ArgumentList _arguments;
    SharedVariableList _shared_variables;
    LocalVariableList _local_variables;

public:
    explicit Function(Pool *pool, FunctionTag tag,
                      const Type *type = nullptr,
                      const Name *name = nullptr) noexcept;

    [[nodiscard]] auto function_tag() const noexcept { return _function_tag; }
    [[nodiscard]] DerivedValueTag derived_value_tag() const noexcept final {
        return DerivedValueTag::FUNCTION;
    }

    void add_argument(Argument *argument) noexcept;
    void add_shared_variable(SharedVariable *shared) noexcept;
    void add_local_variable(LocalVariable *local) noexcept;

    [[nodiscard]] FunctionBodyBlock *body() noexcept { return _body; }
    [[nodiscard]] const FunctionBodyBlock *body() const noexcept { return _body; }

    [[nodiscard]] auto &arguments() noexcept { return _arguments; }
    [[nodiscard]] auto &arguments() const noexcept { return _arguments; }

    [[nodiscard]] auto &shared_variables() noexcept { return _shared_variables; }
    [[nodiscard]] auto &shared_variables() const noexcept { return _shared_variables; }

    [[nodiscard]] auto &local_variables() noexcept { return _local_variables; }
    [[nodiscard]] auto &local_variables() const noexcept { return _local_variables; }
};

}// namespace luisa::compute::xir
