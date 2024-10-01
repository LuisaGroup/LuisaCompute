#pragma once

#include <luisa/xir/basic_block.h>
#include <luisa/xir/argument.h>
#include <luisa/xir/shared.h>

namespace luisa::compute::xir {

class LC_XIR_API Function : public Value {

public:
    enum struct Tag {
        KERNEL,
        CALLABLE,
    };

private:
    Tag _tag;
    BasicBlock *_body = nullptr;
    luisa::vector<Argument *> _arguments;
    luisa::vector<Shared *> _shared_variables;

public:
    explicit Function(Pool *pool, Tag tag,
                      const Type *type = nullptr,
                      const Name *name = nullptr) noexcept;
    void add_argument(Argument *argument) noexcept;
    [[nodiscard]] auto tag() const noexcept { return _tag; }
    [[nodiscard]] auto body() noexcept { return _body; }
    [[nodiscard]] auto body() const noexcept { return const_cast<const BasicBlock *>(_body); }
    [[nodiscard]] auto arguments() noexcept { return luisa::span{_arguments}; }
    [[nodiscard]] auto arguments() const noexcept { return luisa::span<const Argument *const>{_arguments}; }
};

}// namespace luisa::compute::xir
