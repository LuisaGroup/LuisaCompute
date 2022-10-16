#pragma once

// #include <ir/autodiff.h>
// #include <dsl/var.h>
// namespace luisa::compute {
//     struct _with_grad {
//         _with_grad() noexcept {
//             detail::begin_autodiff();
//         }
//         ~_with_grad() noexcept {
//             detail::end_autodiff();
//         }
//     };
//     template<class F>
//     auto with_grad(F &&f) noexcept {
//         _with_grad _;
//         return f();
//     }
//     template<class T>
//     void backward(const Var<T>& out, const Var<T>& out_grad) noexcept {
//         detail::backward(out.expression(), out_grad.expression());
//     }
//     template<class T>
//     void requires_grad(const Var<T>& var) noexcept {
//         detail::requires_grad(var.expression());
//     }
//     template<class T>
//     Var<T> grad(const Var<T>& var) noexcept {
//         return Var<T>{detail::grad(var.expression())};
//     }
// }