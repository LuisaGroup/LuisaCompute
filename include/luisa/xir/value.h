#pragma once

#include <luisa/xir/use.h>
#include <luisa/xir/metadata.h>

namespace luisa::compute {
class Type;
}// namespace luisa::compute

namespace luisa::compute::xir {

enum struct DerivedValueTag {
    UNDEFINED,
    FUNCTION,
    BASIC_BLOCK,
    INSTRUCTION,
    CONSTANT,
    ARGUMENT,
    SPECIAL_REGISTER,
};

[[nodiscard]] inline luisa::string_view to_string(DerivedValueTag tag) noexcept {
    using namespace std::string_view_literals;
    switch (tag) {
        case DerivedValueTag::UNDEFINED: return "undef"sv;
        case DerivedValueTag::FUNCTION: return "func"sv;
        case DerivedValueTag::BASIC_BLOCK: return "block"sv;
        case DerivedValueTag::INSTRUCTION: return "inst"sv;
        case DerivedValueTag::CONSTANT: return "const"sv;
        case DerivedValueTag::ARGUMENT: return "arg"sv;
        case DerivedValueTag::SPECIAL_REGISTER: return "sreg"sv;
    }
    return "unknown"sv;
}

class LUISA_XIR_API Value : public Managed<Value>,
                         public MetadataListMixin {

private:
    const Type *_type;
    UseList _use_list;

public:
    explicit Value(const Type *type) noexcept;
    [[nodiscard]] virtual DerivedValueTag derived_value_tag() const noexcept = 0;
    [[nodiscard]] virtual bool is_user() const noexcept { return false; }
    [[nodiscard]] virtual bool is_lvalue() const noexcept { return false; }
    [[nodiscard]] virtual bool is_global() const noexcept { return false; }
    [[nodiscard]] auto type() const noexcept { return _type; }
    [[nodiscard]] auto &use_list() noexcept { return _use_list; }
    [[nodiscard]] auto &use_list() const noexcept { return _use_list; }

    void replace_all_uses_with(Value *value) noexcept;
    LUISA_XIR_DEFINED_ISA_METHOD(Value, value)
};

template<typename Derived, DerivedValueTag Tag, typename Base = Value>
    requires std::derived_from<Base, Value>
class DerivedValue : public Base {
public:
    using derived_value_type = Derived;
    using Super = DerivedValue;
    using Base::Base;

    [[nodiscard]] static constexpr DerivedValueTag
    static_derived_value_tag() noexcept { return Tag; }

    [[nodiscard]] DerivedValueTag
    derived_value_tag() const noexcept final {
        Base::validate_canary();
        return static_derived_value_tag();
    }
};

class Module;
class Function;
class FunctionDefinition;
class BasicBlock;

class LUISA_XIR_API GlobalValueModuleMixin {

private:
    Module *_parent_module;

protected:
    explicit GlobalValueModuleMixin(Module *module) noexcept;
    ~GlobalValueModuleMixin() noexcept = default;

public:
    [[nodiscard]] Module *parent_module() noexcept { return _parent_module; }
    [[nodiscard]] const Module *parent_module() const noexcept { return _parent_module; }
};

template<typename Derived, DerivedValueTag Tag, typename Base = Value>
    requires std::derived_from<Base, Value>
class DerivedGlobalValue : public ManagedIntrusiveNode<Derived, DerivedValue<Derived, Tag, Base>>,
                           public GlobalValueModuleMixin {
public:
    using Super = DerivedGlobalValue;
    template<typename... Args>
    explicit DerivedGlobalValue(Module *module, Args &&...args) noexcept
        : ManagedIntrusiveNode<Derived, DerivedValue<Derived, Tag, Base>>{std::forward<Args>(args)...},
          GlobalValueModuleMixin{module} {}
    [[nodiscard]] bool is_global() const noexcept final { return true; }
};

class LUISA_XIR_API LocalValueFunctionMixin {

private:
    friend class Function;
    friend class FunctionDefinition;
    Function *_parent_function;

protected:
    explicit LocalValueFunctionMixin(Function *function) noexcept;
    ~LocalValueFunctionMixin() noexcept = default;
    void _set_parent_function(Function *function) noexcept;

public:
    [[nodiscard]] Function *parent_function() noexcept { return _parent_function; }
    [[nodiscard]] const Function *parent_function() const noexcept { return _parent_function; }
    [[nodiscard]] Module *parent_module() noexcept;
    [[nodiscard]] const Module *parent_module() const noexcept;
};

template<typename Derived, DerivedValueTag Tag, typename Base = Value>
    requires std::derived_from<Base, Value>
class DerivedFunctionScopeValue : public ManagedIntrusiveNode<Derived, DerivedValue<Derived, Tag, Base>>,
                                  public LocalValueFunctionMixin {
public:
    using Super = DerivedFunctionScopeValue;
    template<typename... Args>
    explicit DerivedFunctionScopeValue(Function *function, Args &&...args) noexcept
        : ManagedIntrusiveNode<Derived, DerivedValue<Derived, Tag, Base>>{std::forward<Args>(args)...},
          LocalValueFunctionMixin{function} {}
};

class LUISA_XIR_API LocalValueBlockMixin {

private:
    friend class BasicBlock;
    BasicBlock *_parent_block;

protected:
    explicit LocalValueBlockMixin(BasicBlock *block) noexcept;
    ~LocalValueBlockMixin() noexcept = default;
    void _set_parent_block(BasicBlock *block) noexcept;

public:
    [[nodiscard]] BasicBlock *parent_block() noexcept { return _parent_block; }
    [[nodiscard]] const BasicBlock *parent_block() const noexcept { return _parent_block; }
    [[nodiscard]] Function *parent_function() noexcept;
    [[nodiscard]] const Function *parent_function() const noexcept;
    [[nodiscard]] Module *parent_module() noexcept;
    [[nodiscard]] const Module *parent_module() const noexcept;
};

template<typename Derived, DerivedValueTag Tag, typename Base = Value>
    requires std::derived_from<Base, Value>
class DerivedBlockScopeValue : public ManagedIntrusiveNode<Derived, DerivedValue<Derived, Tag, Base>>,
                               public LocalValueBlockMixin {
public:
    using Super = DerivedBlockScopeValue;
    template<typename... Args>
    explicit DerivedBlockScopeValue(BasicBlock *block, Args &&...args) noexcept
        : ManagedIntrusiveNode<Derived, DerivedValue<Derived, Tag, Base>>{std::forward<Args>(args)...},
          LocalValueBlockMixin{block} {}
};

}// namespace luisa::compute::xir
