//
// Created by Mike Smith on 2021/2/27.
//

#include <iostream>
#include <dsl/syntax.h>

using namespace luisa;
using namespace luisa::compute;

namespace internal {

template<typename T>
struct DeductionGuide;

template<typename T>
struct Something {};

template<typename T>
struct DeductionGuideBase {
    constexpr DeductionGuideBase(T s) noexcept { LUISA_INFO("From scalar: {}", Type::of(s)->description()); }
    constexpr DeductionGuideBase(Something<T>) noexcept { LUISA_INFO("From something..."); }
    constexpr DeductionGuideBase(DeductionGuideBase &&) noexcept { LUISA_INFO("From another guide..."); }
    constexpr DeductionGuideBase(const DeductionGuideBase &) noexcept { LUISA_INFO("From another guide..."); }
    constexpr DeductionGuideBase(DeductionGuide<T> &&) noexcept { LUISA_INFO("From another guide..."); }
    constexpr DeductionGuideBase(const DeductionGuide<T> &) noexcept { LUISA_INFO("From another guide..."); }

    void operator=(const DeductionGuideBase &) const noexcept { LUISA_INFO("operator=(const &)"); }
    void operator=(DeductionGuideBase &&) const noexcept { LUISA_INFO("operator=(&&)"); }
    void operator+=(const DeductionGuideBase &) const noexcept { LUISA_INFO("operator+=(const &)"); }
    void operator+=(DeductionGuideBase &&) const noexcept { LUISA_INFO("operator+=(&&)"); }

    template<typename U>
    void operator+(DeductionGuide<U>) const noexcept { LUISA_INFO("operator+(DeductionGuide)"); }

    template<typename U>
    void operator+(const Something<U> &) const noexcept { LUISA_INFO("operator+(const Something &)"); }

//    template<typename U>
//    void operator+(Something<U> &&) const noexcept { LUISA_INFO("operator+(Something &&)"); }

    template<concepts::Native U>
    void operator+(U &&) const noexcept { LUISA_INFO("operator+(U)"); }

    template<concepts::Native U>
    friend void operator+(U &&, DeductionGuideBase) noexcept { LUISA_INFO("friend operator+(U)"); }
};

template<typename T>
struct DeductionGuide : public DeductionGuideBase<T> {
    using DeductionGuideBase<T>::DeductionGuideBase;
    //    using DeductionGuideBase<T>::operator=;
};

template<typename T>
DeductionGuide(const Something<T> &) -> DeductionGuide<T>;

template<typename T>
DeductionGuide(Something<T> &&) -> DeductionGuide<T>;

template<typename T>
DeductionGuide(DeductionGuide<T>) -> DeductionGuide<T>;

template<typename T>
DeductionGuide(T &&) -> DeductionGuide<std::remove_cvref_t<T>>;

}

int main() {
    internal::DeductionGuide g1 = 1.5f;
    internal::DeductionGuide g2{internal::Something<int>{}};
    internal::DeductionGuide g3{g1};
    internal::DeductionGuide g4{internal::DeductionGuide{float2{1.0f}}};
    
    internal::Something<float> f;
    g3 = f;
    g3 = 1.2f;
    g3 + f;
    g3 + internal::DeductionGuide{1.5f};
    g3 + internal::Something<float>{};
    g3 + 1.0f;
    1.0f + g3;
    g3 += 1;
    
    using namespace luisa::compute::dsl::detail;
    
    FunctionBuilder func{Function::Tag::KERNEL};
    
    func.define([] {
        Expr x = 1.0f;
        Expr y = 2.0f;
        x = 2.1f * 1 / x + y * 2;
        Expr z = x * y;
        auto b = !(x < y);
        Expr c = false;
        auto result = b + c;

        Expr i = 10u;
        LUISA_INFO("bad");
        i += 12;
        
        Expr v_int = 1;
        Expr v_float = v_int * 1.5f;
        static_assert(std::is_same_v<decltype(v_float), Expr<float>>);
    });
}
