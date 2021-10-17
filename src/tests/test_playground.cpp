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

    S<int> s{-1};
    auto x = -s;
}
