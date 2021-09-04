//
// Created by Mike Smith on 2021/9/3.
//

#include <iostream>

struct A {
    A()
    noexcept { std::cout << "default constructor" << std::endl; }
    A(A &&)
    noexcept { std::cout << "move constructor" << std::endl; }
    A(const A &)
    noexcept { std::cout << "copy constructor" << std::endl; }
    A &operator=(A &&) noexcept {
        std::cout << "move operator=" << std::endl;
        return *this;
    }
    A &operator=(const A &) noexcept {
        std::cout << "copy operator=" << std::endl;
        return *this;
    }
    ~A() noexcept { std::cout << "destructor" << std::endl; }
};

void foo(const A &a, A b, A &c) noexcept {}

template<typename T, size_t index>
struct function_arg {
    using type = typename function_arg<
        std::remove_cvref_t<decltype(std::function{std::declval<T>()})>,
        index>::type;
};

template<typename R, typename... Args, size_t index>
struct function_arg<std::function<R(Args...)>, index> {
    using type = std::tuple_element_t<index, std::tuple<Args...>>;
};

template<typename T, size_t index>
using function_arg_t = typename function_arg<T, index>::type;

template<typename F, typename... Args>
decltype(auto) my_apply(F &&f, std::tuple<Args...> &t) {
    return [&]<size_t... i>(std::index_sequence<i...>) noexcept {
        return f(static_cast<function_arg_t<F, i> &&>(std::get<i>(t))...);
    }
    (std::index_sequence_for<Args...>{});
}

int main() {

    std::cout << "making tuple..." << std::endl;
    std::tuple t{A{}, A{}, A{}};

    std::cout << "\napplying by copy..." << std::endl;
    std::apply(foo, t);

    std::cout << "\napplying by my_apply..." << std::endl;
    my_apply(foo, t);

    std::cout << "\ndone" << std::endl;
}
