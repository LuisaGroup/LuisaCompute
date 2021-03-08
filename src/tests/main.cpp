//
// Created by Mike Smith on 2020/11/21.
//

#include <iostream>

#include <core/data_types.h>
#include <core/logging.h>

template<typename T>
struct DSLVar {

    using Type = DSLVar<T>;

    T value;

    constexpr DSLVar(T x) noexcept : value{x} {}

    [[nodiscard]] auto operator*(DSLVar rhs) const noexcept {
        return DSLVar{value * rhs.value};
    }

    friend std::ostream &operator<<(std::ostream &os, const DSLVar &v) noexcept {
        os << "DSL(" << v.value << ")";
        return os;
    }
};

template<typename>
struct IsDSLVar : std::false_type {};

template<typename T>
struct IsDSLVar<DSLVar<T>> : std::true_type {};

template<typename T>
DSLVar(T) -> DSLVar<T>;

template<typename T>
struct CPUVar {
    using Type = T;
    T value;
    constexpr CPUVar(T x) noexcept : value{x} {}
    constexpr operator T() const noexcept { return value; }
};

template<typename T>
CPUVar(T) -> CPUVar<T>;

template<typename T>
using CPU = T;

template<typename T>
using DSL = DSLVar<T>;

#define LUISA_XPU_TEMPLATE(...) template<template<typename> typename Var, ##__VA_ARGS__>
#define LUISA_XPU LUISA_XPU_TEMPLATE()

#define LUISA_DEVICE_FUNCTION(f)                                                            \
    []<typename... Args>(Args && ...args) noexcept->decltype(auto) {                        \
        static constexpr auto is_dsl = std::disjunction_v<IsDSLVar<std::decay_t<Args>>...>; \
        if constexpr (is_dsl) {                                                             \
            static std::once_flag flag;                                                     \
            std::call_once(flag, [] {                                                       \
                LUISA_INFO("Compiling device function: {}", #f);                            \
            });                                                                             \
            return f<DSL>(std::forward<Args>(args)...);                                     \
        } else {                                                                            \
            return f<CPU>(std::forward<Args>(args)...);                                     \
        }                                                                                   \
    }

LUISA_XPU_TEMPLATE(typename T)
auto foo(Var<T> f) { return f * f; }

int main() {

    auto f = LUISA_DEVICE_FUNCTION(foo);

    DSLVar a = 1.0f;
    std::cout << f(a) << std::endl;

    DSLVar another = 1.5f;
    std::cout << f(another) << std::endl;

    CPUVar b = 2.0f;
    std::cout << f(b) << std::endl;

    auto c = luisa::float3{1.5f};
    auto d = f(c);
    std::cout << "(" << d.x << ", " << d.y << ", " << d.z << ")" << std::endl;
}
