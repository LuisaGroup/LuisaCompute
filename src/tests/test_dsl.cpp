//
// Created by Mike Smith on 2021/2/27.
//

#include <iostream>
#include <dsl/syntax.h>

using namespace luisa;
using namespace luisa::compute;

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
    
    template<typename U>
    void operator+(DeductionGuide<U>) const noexcept { LUISA_INFO("operator+(DeductionGuide)"); }
    
    template<typename U>
    void operator+(const Something<U> &) const noexcept { LUISA_INFO("operator+(const Something &)"); }
    
    template<typename U>
    void operator+(Something<U> &&) const noexcept { LUISA_INFO("operator+(Something &&)"); }
    
    template<concepts::core_data_type U>
    void operator+(U &&) const noexcept { LUISA_INFO("operator+(U)"); }
};

template<typename T>
struct DeductionGuide : public DeductionGuideBase<T> {
    using DeductionGuideBase<T>::DeductionGuideBase;
    using DeductionGuideBase<T>::operator=;
};

template<typename T>
DeductionGuide(const Something<T> &) -> DeductionGuide<T>;

template<typename T>
DeductionGuide(Something<T> &&) -> DeductionGuide<T>;

template<typename T>
DeductionGuide(DeductionGuide<T>) -> DeductionGuide<T>;

template<typename T>
DeductionGuide(T &&) -> DeductionGuide<std::remove_cvref_t<T>>;


int main() {
    DeductionGuide g1{1.5f};
    DeductionGuide g2{Something<int>{}};
    DeductionGuide g3{g1};
    DeductionGuide g4{DeductionGuide{float2{1.0f}}};
    
    Something<float> f;
    g3 = f;
    g3 = 1.2f;
    g3 + f;
    g3 + DeductionGuide{1.5f};
    g3 + 1;
}
