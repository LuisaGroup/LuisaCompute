//
// Created by Mike Smith on 2021/9/3.
//

#include <iostream>
#include <functional>
#include <memory>
#include <map>
#include <string>
#include <string_view>
#include <unordered_map>

#include <backends/cuda/cuda_device_math_embedded.inl.h>

template<typename T>
struct S {
    T x;
    [[nodiscard]] constexpr auto operator-() const noexcept
        requires requires { -x; }
    {
        return -x;
    }
};

int main() {
    auto a = [](auto x) { std::cout << x << std::endl; };
    std::cout << sizeof(std::function<void(int)>{a}) << std::endl;
    std::cout << sizeof(std::string) << std::endl;

    std::cout << cuda_device_math_source << std::endl;

    S<int> s{-1};
    auto x = -s;
}
