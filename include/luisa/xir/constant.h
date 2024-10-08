#pragma once

#include <luisa/xir/value.h>

namespace luisa::compute::xir {

class LC_XIR_API Constant : public Value {

private:
    union {
        std::byte _small[sizeof(void *)] = {};
        void *_large;
    };

    [[nodiscard]] bool _is_small() const noexcept;

public:
    explicit Constant(Pool *pool, const Type *type,
                      const void *data = nullptr,
                      const Name *name = nullptr) noexcept;
    ~Constant() noexcept override;
    void set_type(const Type *type) noexcept override;
    void set_data(const void *data) noexcept;
    [[nodiscard]] void *data() noexcept;
    [[nodiscard]] const void *data() const noexcept;
    [[nodiscard]] DerivedValueTag derived_value_tag() const noexcept final {
        return DerivedValueTag::CONSTANT;
    }

    template<typename T>
    [[nodiscard]] T &as() noexcept {
        assert(type()->size() == sizeof(T) && "Type size mismatch.");
        return *reinterpret_cast<T *>(data());
    }

    template<typename T>
    [[nodiscard]] const T &as() const noexcept {
        return const_cast<Constant *>(this)->as<T>();
    }
};

}// namespace luisa::compute::xir
