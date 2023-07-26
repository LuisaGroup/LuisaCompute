#pragma once

#include <luisa/core/dll_export.h>
#include <luisa/ast/atomic_ref_node.h>
#include <luisa/dsl/expr.h>

namespace luisa::compute::detail {

template<typename>
class AtomicRef;

class AtomicRefBase {

private:
    const AtomicRefNode *_access_chain{nullptr};

protected:
    explicit AtomicRefBase(const AtomicRefNode *access_chain) noexcept
        : _access_chain{access_chain} {}

public:
    AtomicRefBase(AtomicRefBase &&) noexcept = default;
    AtomicRefBase(const AtomicRefBase &) noexcept = delete;
    AtomicRefBase &operator=(AtomicRefBase &&) noexcept = delete;
    AtomicRefBase &operator=(const AtomicRefBase &) noexcept = delete;

protected:
    [[nodiscard]] auto access_chain() const noexcept { return _access_chain; }

    template<typename T, typename I>
    [[nodiscard]] auto access(I &&index) const noexcept {
        auto i = def(std::forward<I>(index));
        return AtomicRef<T>{_access_chain->access(i.expression())};
    }

    template<typename T>
    [[nodiscard]] auto member(size_t i) const noexcept {
        return AtomicRef<T>{_access_chain->access(i)};
    }
};

#define LUISA_ATOMIC_REF_COMMON()                                  \
public:                                                            \
    explicit AtomicRef(const AtomicRefNode *access_chain) noexcept \
        : AtomicRefBase{access_chain} {}

template<typename T>
class AtomicRef : private AtomicRefBase {
public:
    LUISA_ATOMIC_REF_COMMON()
};

template<typename T>
    requires std::same_as<T, int> || std::same_as<T, uint>
class AtomicRef<T> : private AtomicRefBase {

public:
    LUISA_ATOMIC_REF_COMMON()

public:
    /// Atomic exchange. Stores desired, returns old. See also CallOp::ATOMIC_EXCHANGE.
    auto exchange(Expr<T> desired) noexcept {
        return def<T>(access_chain()->operate(
            CallOp::ATOMIC_EXCHANGE,
            {desired.expression()}));
    }

    /// Atomic compare exchange. Stores old == expected ? desired : old, returns old. See also CallOp::ATOMIC_COMPARE_EXCHANGE.
    auto compare_exchange(Expr<T> expected, Expr<T> desired) noexcept {
        return def<T>(access_chain()->operate(
            CallOp::ATOMIC_COMPARE_EXCHANGE,
            {expected.expression(), desired.expression()}));
    }

    /// Atomic fetch add. Stores old + val, returns old. See also CallOp::ATOMIC_FETCH_ADD.
    auto fetch_add(Expr<T> val) noexcept {
        return def<T>(access_chain()->operate(
            CallOp::ATOMIC_FETCH_ADD,
            {val.expression()}));
    };

    /// Atomic fetch sub. Stores old - val, returns old. See also CallOp::ATOMIC_FETCH_SUB.
    auto fetch_sub(Expr<T> val) noexcept {
        return def<T>(access_chain()->operate(
            CallOp::ATOMIC_FETCH_SUB,
            {val.expression()}));
    };

    /// Atomic fetch and. Stores old & val, returns old. See also CallOp::ATOMIC_FETCH_AND.
    auto fetch_and(Expr<T> val) noexcept {
        return def<T>(access_chain()->operate(
            CallOp::ATOMIC_FETCH_AND,
            {val.expression()}));
    };

    /// Atomic fetch or. Stores old | val, returns old. See also CallOp::ATOMIC_FETCH_OR.
    auto fetch_or(Expr<T> val) noexcept {
        return def<T>(access_chain()->operate(
            CallOp::ATOMIC_FETCH_OR,
            {val.expression()}));
    };

    /// Atomic fetch xor. Stores old ^ val, returns old. See also CallOp::ATOMIC_FETCH_XOR.
    auto fetch_xor(Expr<T> val) noexcept {
        return def<T>(access_chain()->operate(
            CallOp::ATOMIC_FETCH_XOR,
            {val.expression()}));
    };

    /// Atomic fetch min. Stores min(old, val), returns old. See also CallOp::ATOMIC_FETCH_MIN.
    auto fetch_min(Expr<T> val) noexcept {
        return def<T>(access_chain()->operate(
            CallOp::ATOMIC_FETCH_MIN,
            {val.expression()}));
    };

    /// Atomic fetch max. Stores max(old, val), returns old. See also CallOp::ATOMIC_FETCH_MAX.
    auto fetch_max(Expr<T> val) noexcept {
        return def<T>(access_chain()->operate(
            CallOp::ATOMIC_FETCH_MAX,
            {val.expression()}));
    };
};

template<typename T>
    requires std::same_as<T, float>
class AtomicRef<T> : private AtomicRefBase {

public:
    LUISA_ATOMIC_REF_COMMON()

public:
    /// Atomic exchange. Stores desired, returns old. See also CallOp::ATOMIC_EXCHANGE.
    auto exchange(Expr<T> desired) noexcept {
        return def<T>(access_chain()->operate(
            CallOp::ATOMIC_EXCHANGE,
            {desired.expression()}));
    }

    /// Atomic compare exchange. Stores old == expected ? desired : old, returns old. See also CallOp::ATOMIC_COMPARE_EXCHANGE.
    auto compare_exchange(Expr<T> expected, Expr<T> desired) noexcept {
        return def<T>(access_chain()->operate(
            CallOp::ATOMIC_COMPARE_EXCHANGE,
            {expected.expression(), desired.expression()}));
    }

    /// Atomic fetch add. Stores old + val, returns old. See also CallOp::ATOMIC_FETCH_ADD.
    auto fetch_add(Expr<T> val) noexcept {
        return def<T>(access_chain()->operate(
            CallOp::ATOMIC_FETCH_ADD,
            {val.expression()}));
    };

    /// Atomic fetch sub. Stores old - val, returns old. See also CallOp::ATOMIC_FETCH_SUB.
    auto fetch_sub(Expr<T> val) noexcept {
        return def<T>(access_chain()->operate(
            CallOp::ATOMIC_FETCH_SUB,
            {val.expression()}));
    };

    /// Atomic fetch min. Stores min(old, val), returns old. See also CallOp::ATOMIC_FETCH_MIN.
    auto fetch_min(Expr<T> val) noexcept {
        return def<T>(access_chain()->operate(
            CallOp::ATOMIC_FETCH_MIN,
            {val.expression()}));
    };

    /// Atomic fetch max. Stores max(old, val), returns old. See also CallOp::ATOMIC_FETCH_MAX.
    auto fetch_max(Expr<T> val) noexcept {
        return def<T>(access_chain()->operate(
            CallOp::ATOMIC_FETCH_MAX,
            {val.expression()}));
    };
};

/*
 * specialize for built-in aggregate types
 */

// arrays
template<typename T, size_t N>
class AtomicRef<std::array<T, N>> : private AtomicRefBase {
public:
    LUISA_ATOMIC_REF_COMMON()
    template<typename I>
    [[nodiscard]] auto operator[](I &&index) const noexcept {
        return this->access<T>(std::forward<I>(index));
    }
};

template<typename T, size_t N>
class AtomicRef<std::array<T, N> &> : private AtomicRefBase {
public:
    LUISA_ATOMIC_REF_COMMON()
    template<typename I>
    [[nodiscard]] auto operator[](I &&index) const noexcept {
        return this->access<T>(std::forward<I>(index));
    }
};

// vectors
template<typename T>
class AtomicRef<Vector<T, 2>> : private AtomicRefBase {

public:
    AtomicRef<T> x{this->member<T>(0u)};
    AtomicRef<T> y{this->member<T>(1u)};

public:
    LUISA_ATOMIC_REF_COMMON()
    template<typename I>
    [[nodiscard]] auto operator[](I &&index) const noexcept {
        return this->access<T>(std::forward<I>(index));
    }
};

template<typename T>
class AtomicRef<Vector<T, 3>> : private AtomicRefBase {

public:
    AtomicRef<T> x{this->member<T>(0u)};
    AtomicRef<T> y{this->member<T>(1u)};
    AtomicRef<T> z{this->member<T>(2u)};

public:
    LUISA_ATOMIC_REF_COMMON()
    template<typename I>
    [[nodiscard]] auto operator[](I &&index) const noexcept {
        return this->access<T>(std::forward<I>(index));
    }
};

template<typename T>
class AtomicRef<Vector<T, 4>> : private AtomicRefBase {

public:
    AtomicRef<T> x{this->member<T>(0u)};
    AtomicRef<T> y{this->member<T>(1u)};
    AtomicRef<T> z{this->member<T>(2u)};
    AtomicRef<T> w{this->member<T>(3u)};

public:
    LUISA_ATOMIC_REF_COMMON()
    template<typename I>
    [[nodiscard]] auto operator[](I &&index) const noexcept {
        return this->access<T>(std::forward<I>(index));
    }
};

// matrices
template<size_t N>
class AtomicRef<Matrix<N>> : private AtomicRefBase {
public:
    LUISA_ATOMIC_REF_COMMON()
    template<typename I>
    [[nodiscard]] auto operator[](I &&index) const noexcept {
        return this->access<Vector<float, N>>(std::forward<I>(index));
    }
};

// tuples
template<typename... Ts>
class AtomicRef<std::tuple<Ts...>> : private AtomicRefBase {
public:
    LUISA_ATOMIC_REF_COMMON()
    template<size_t i>
    [[nodiscard]] auto get() const noexcept {
        static_assert(i < sizeof...(Ts));
        using T = std::tuple_element_t<i, std::tuple<Ts...>>;
        return this->member<T>(i);
    }
};

#undef LUISA_ATOMIC_REF_COMMON

}// namespace luisa::compute::detail

