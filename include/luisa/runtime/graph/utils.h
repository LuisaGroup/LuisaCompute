#pragma once
#include <luisa/vstl/unique_ptr.h>
namespace luisa::compute::graph::detail {
//template<typename... Args>
//auto print_types_with_index() {
//    auto print = []<size_t... I>(std::index_sequence<I...>) {
//        ((std::cout << "[" << I << "] " << typeid(Args).name() << ";\n"),
//         ...);
//    };
//    std::cout << "{\n";
//    print(std::index_sequence_for<Args...>{});
//    std::cout << "}\n";
//}
//
//template<typename... Args>
//auto print_args_with_index(Args &&...args) {
//    auto print = []<size_t... I>(std::index_sequence<I...>, Args &&...args) {
//        ((std::cout << "[" << I << "] " << typeid(args).name() << ";\n"),
//         ...);
//    };
//    std::cout << "{\n";
//    print(std::index_sequence_for<Args...>{}, std::forward<Args>(args)...);
//    std::cout << "}\n";
//}

template<typename F, typename... Args>
auto for_each_type_with_index(F &&func) {
    []<typename Fn, size_t... I>(Fn &&f, std::index_sequence<I...>) {
        (f(I), ...);
    }(std::forward<F>(func), std::index_sequence_for<Args...>{});
}

template<typename F, typename... Args>
auto for_each_arg_with_index(F &&func, Args &&...args) {
    auto impl = []<typename Fn, typename... Args_, size_t... I>(Fn &&f, std::index_sequence<I...>, Args_ &&...args_) {
        (f(I, std::forward<Args_>(args_)), ...);
    };
    impl(std::forward<F>(func), std::index_sequence_for<Args...>{}, std::forward<Args>(args)...);
}
}// namespace luisa::compute::graph::detail