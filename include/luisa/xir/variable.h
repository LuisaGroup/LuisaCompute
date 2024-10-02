#pragma once

#include <luisa/xir/value.h>

namespace luisa::compute::xir {

class Function;

class LC_XIR_API Variable : public Value {

private:
    Function *_parent_function = nullptr;

public:
    explicit Variable(Pool *pool,
                      Function *parent_function = nullptr,
                      const Type *type = nullptr,
                      const Name *name = nullptr) noexcept;

    [[nodiscard]] DerivedValueTag derived_value_tag() const noexcept final {
        return DerivedValueTag::VARIABLE;
    }

    void set_parent_function(Function *func) noexcept;
    [[nodiscard]] Function *parent_function() noexcept { return _parent_function; }
    [[nodiscard]] const Function *parent_function() const noexcept { return _parent_function; }
};

class LC_XIR_API Argument : public IntrusiveNode<Argument, Variable> {

private:
    bool _by_ref = false;

public:
    explicit Argument(Pool *pool, bool by_ref = false,
                      Function *parent_function = nullptr,
                      const Type *type = nullptr,
                      const Name *name = nullptr) noexcept;
    void set_by_ref(bool by_ref) noexcept { _by_ref = by_ref; }
    [[nodiscard]] auto by_ref() const noexcept { return _by_ref; }

    void remove_self() noexcept override;
    void insert_before_self(Argument *node) noexcept override;
    void insert_after_self(Argument *node) noexcept override;
};

class LC_XIR_API LocalVariable : public IntrusiveNode<LocalVariable, Variable> {
public:
    using Super::Super;
    void remove_self() noexcept override;
    void insert_before_self(LocalVariable *node) noexcept override;
    void insert_after_self(LocalVariable *node) noexcept override;
};

class LC_XIR_API SharedVariable : public IntrusiveNode<SharedVariable, Variable> {
public:
    using Super::Super;
    void remove_self() noexcept override;
    void insert_before_self(SharedVariable *node) noexcept override;
    void insert_after_self(SharedVariable *node) noexcept override;
};

using ArgumentList = InlineIntrusiveList<Argument>;
using LocalVariableList = InlineIntrusiveList<LocalVariable>;
using SharedVariableList = InlineIntrusiveList<SharedVariable>;

}// namespace luisa::compute::xir
