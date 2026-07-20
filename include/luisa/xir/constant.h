#pragma once

#include <luisa/xir/value.h>

namespace luisa::compute::xir {

class LUISA_XIR_API Constant : public DerivedGlobalValue<Constant, DerivedValueTag::CONSTANT> {

private:
    union {
        std::byte _small[sizeof(void *)] = {};
        void *_large;
    };
    uint64_t _hash = {};

    [[nodiscard]] bool _is_small() const noexcept;
    [[nodiscard]] void *_data() noexcept;
    void _update_hash(luisa::optional<uint64_t> hash) noexcept;
    void _check_reinterpret_cast_type_size(size_t size) const noexcept;

protected:
    Constant(Module *module, const Type *type) noexcept;

protected:
    friend class Module;
    struct ctor_tag_zero {};
    struct ctor_tag_one {};
    struct ctor_tag_sentinel {};

public:
    Constant(Module *parent_module, const Type *type, const void *data,
             luisa::optional<uint64_t> hash = luisa::nullopt) noexcept;
    Constant(Module *parent_module, const Type *type, ctor_tag_zero,
             luisa::optional<uint64_t> hash = luisa::nullopt) noexcept;
    Constant(Module *parent_module, const Type *type, ctor_tag_one,
             luisa::optional<uint64_t> hash = luisa::nullopt) noexcept;
    Constant(Module *parent_module, ctor_tag_sentinel) noexcept;// for sentinel constant
    ~Constant() noexcept override;

    [[nodiscard]] const void *data() const noexcept;
    [[nodiscard]] auto hash() const noexcept { return _hash; }

    template<typename T>
    [[nodiscard]] const T &as() const noexcept {
        _check_reinterpret_cast_type_size(sizeof(T));
        return *static_cast<const T *>(data());
    }
};

class SentinelConstant final : public Constant {
public:
    explicit SentinelConstant(Module *parent_module) noexcept
        : Constant{parent_module, ctor_tag_sentinel{}} {}
};

using ConstantList = ManagedIntrusiveList<Constant, SentinelConstant>;

// Decodes a scalar integer constant without reinterpreting its storage. Returns
// false for non-constants, non-integer types, and negative signed values.
[[nodiscard]] LUISA_XIR_API bool try_decode_constant_nonnegative_integer(
    const Value *value, uint64_t &result) noexcept;

}// namespace luisa::compute::xir
