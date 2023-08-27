#pragma once
#include <luisa/ast/callable_library.h>
#include <luisa/core/logging.h>
#include <luisa/dsl/func.h>
namespace luisa::compute {
namespace detail {
template<typename T>
struct CallableTypeChecker;
template<typename Ret, typename... Args>
struct CallableTypeChecker<Ret(Args...)> {
    static void check(Type const *ret_type, luisa::span<const Variable> args) {
        if (ret_type != Type::of<Ret>()) [[unlikely]] {
            LUISA_ERROR("Return type not match.");
        }
        auto types = {Type::of<Args>()...};
        size_t arg_count = 0;
        for (auto &&i : args) {
            if (!i.is_shared() && !i.is_builtin()) {
                arg_count += 1;
            }
        }
        if (types.size() != arg_count) [[unlikely]] {
            LUISA_ERROR("Argument size not match, required: {}, contained: {}", types.size(), args.size());
        }
        auto arg = args.begin();
        for (size_t i = 0; i < types.size(); ++i) {
            while (arg->is_shared() || arg->is_builtin()) {
                arg++;
            }
            if (types.begin()[i] != arg->type()) [[unlikely]] {
                LUISA_ERROR("Argument {} type mismatch, required: {}, contained: {}", i, types.begin()[i]->description(), arg->type()->description());
            }
            arg++;
        }
    }
};
}// namespace detail
template<typename T>
Callable<T> CallableLibrary::get_callable(luisa::string_view name) const noexcept {
    auto iter = _callables.find(name);
    if (iter == _callables.end()) [[unlikely]] {
        LUISA_ERROR("Callable {} not found", name);
    }
    auto &func = iter->second;
    detail::CallableTypeChecker<T>::check(func->return_type(), func->arguments());
    return Callable<T>{func};
}
}// namespace luisa::compute