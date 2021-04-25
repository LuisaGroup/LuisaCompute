//
// Created by Mike Smith on 2021/2/28.
//

#pragma once

#include <runtime/command.h>
#include <runtime/device.h>
#include <dsl/arg.h>
#include <dsl/expr.h>

namespace luisa::compute {

namespace detail {

template<typename T>
struct definition_to_prototype {
    static_assert(always_false_v<T>, "Invalid type in function definition.");
};

template<typename T>
struct definition_to_prototype<Var<T>> {
    using type = T;
};

template<typename T>
struct prototype_to_creation {
    using type = Var<T>;
};

template<typename T>
struct prototype_to_kernel_invocation {
    using type = T;
};

template<typename T>
struct prototype_to_kernel_invocation<Buffer<T>> {
    using type = BufferView<T>;
};

template<typename T>
struct prototype_to_kernel_invocation<Image<T>> {
    using type = ImageView<T>;
};

template<typename T>
struct prototype_to_kernel_invocation<Volume<T>> {
    using type = VolumeView<T>;
};

template<typename T>
struct prototype_to_callable_invocation {
    using type = Expr<T>;
};

template<typename T>
using definition_to_prototype_t = typename definition_to_prototype<T>::type;

template<typename T>
using prototype_to_creation_t = typename prototype_to_creation<T>::type;

template<typename T>
using prototype_to_kernel_invocation_t = typename prototype_to_kernel_invocation<T>::type;

template<typename T>
using prototype_to_callable_invocation_t = typename prototype_to_callable_invocation<T>::type;

}// namespace detail

namespace detail {

class KernelInvoke {

private:
    CommandHandle _command;
    Function _function;
    size_t _argument_index{0u};

private:
    [[nodiscard]] auto _launch_command() noexcept {
        return static_cast<KernelLaunchCommand *>(_command.get());
    }

public:
    explicit KernelInvoke(uint32_t function_uid) noexcept
        : _command{KernelLaunchCommand::create(function_uid)},
          _function{Function::kernel(function_uid)} {

        for (auto buffer : _function.captured_buffers()) {
            _launch_command()->encode_buffer(
                buffer.variable.uid(), buffer.handle, buffer.offset_bytes,
                static_cast<Command::Resource::Usage>(_function.variable_usage(buffer.variable.uid())));
        }

        for (auto texture : _function.captured_textures()) {
            _launch_command()->encode_texture(
                texture.variable.uid(), texture.handle, texture.offset,
                static_cast<Command::Resource::Usage>(_function.variable_usage(texture.variable.uid())));
        }
    }

    template<typename T>
    KernelInvoke &operator<<(BufferView<T> buffer) noexcept {
        auto variable_uid = _function.arguments()[_argument_index++].uid();
        auto usage = _function.variable_usage(variable_uid);
        _launch_command()->encode_buffer(
            variable_uid, buffer.handle(), buffer.offset_bytes(),
            static_cast<Command::Resource::Usage>(usage));
        return *this;
    }

    template<typename T>
    KernelInvoke &operator<<(ImageView<T> image) noexcept {
        auto variable_uid = _function.arguments()[_argument_index++].uid();
        auto usage = _function.variable_usage(variable_uid);
        _launch_command()->encode_texture(
            variable_uid, image.handle(), uint3{image.offset(), 0u},
            static_cast<Command::Resource::Usage>(usage));
        return *this;
    }

    template<typename T>
    KernelInvoke &operator<<(VolumeView<T> volume) noexcept {
        auto variable_uid = _function.arguments()[_argument_index++].uid();
        auto usage = _function.variable_usage(variable_uid);
        _launch_command()->encode_texture(
            variable_uid, volume.handle(), volume.offset(),
            static_cast<Command::Resource::Usage>(usage));
        return *this;
    }

    template<typename T>
    KernelInvoke &operator<<(T data) noexcept {
        auto variable_uid = _function.arguments()[_argument_index++].uid();
        _launch_command()->encode_uniform(variable_uid, &data, sizeof(T), alignof(T));
        return *this;
    }

    [[nodiscard]] auto parallelize(uint3 launch_size) noexcept {
        _launch_command()->set_launch_size(launch_size);
        auto command = std::move(_command);
        _command = nullptr;
        return command;
    }
};

struct KernelInvoke1D : public KernelInvoke {
    explicit KernelInvoke1D(uint32_t uid) noexcept : KernelInvoke{uid} {}
    [[nodiscard]] auto launch(uint size_x) noexcept {
        return parallelize(uint3{size_x, 1u, 1u});
    }
};

struct KernelInvoke2D : public KernelInvoke {
    explicit KernelInvoke2D(uint32_t uid) noexcept : KernelInvoke{uid} {}
    [[nodiscard]] auto launch(uint size_x, uint size_y) noexcept {
        return parallelize(uint3{size_x, size_y, 1u});
    }
};

struct KernelInvoke3D : public KernelInvoke {
    explicit KernelInvoke3D(uint32_t uid) noexcept : KernelInvoke{uid} {}
    [[nodiscard]] auto launch(uint size_x, uint size_y, uint size_z) noexcept {
        return parallelize(uint3{size_x, size_y, size_z});
    }
};

template<size_t N>
[[nodiscard]] constexpr auto kernel_default_block_size() {
    if constexpr (N == 1) {
        return uint3{256u, 1u, 1u};
    } else if constexpr (N == 2) {
        return uint3{16u, 16u, 1u};
    } else if constexpr (N == 3) {
        return uint3{8u, 8u, 8u};
    } else {
        static_assert(always_false_v<std::integral_constant<size_t, N>>);
    }
}

}// namespace detail

#define LUISA_MAKE_KERNEL_ND(N)                                                                                \
    template<typename T>                                                                                       \
    class Kernel##N##D {                                                                                       \
        static_assert(always_false_v<T>);                                                                      \
    };                                                                                                         \
    template<typename... Args>                                                                                 \
    class Kernel##N##D<void(Args...)> {                                                                        \
                                                                                                               \
    private:                                                                                                   \
        Function _function;                                                                                    \
                                                                                                               \
    public:                                                                                                    \
        Kernel##N##D(Kernel##N##D &&) noexcept = default;                                                      \
        Kernel##N##D(const Kernel##N##D &) noexcept = default;                                                 \
                                                                                                               \
        [[nodiscard]] auto function_uid() const noexcept { return _function.uid(); }                           \
                                                                                                               \
        template<typename Def>                                                                                 \
        requires concepts::invocable_with_return<void, Def, detail::prototype_to_creation_t<Args>...>          \
            Kernel##N##D(Def &&def) noexcept                                                                   \
            : _function{FunctionBuilder::define_kernel([&def] {                                                \
                  FunctionBuilder::current()->set_block_size(detail::kernel_default_block_size<N>());          \
                  def(detail::prototype_to_creation_t<Args>{detail::ArgumentCreation{}}...);                   \
              })} {}                                                                                           \
                                                                                                               \
        [[nodiscard]] auto operator()(detail::prototype_to_kernel_invocation_t<Args>... args) const noexcept { \
            detail::KernelInvoke##N##D invoke{_function.uid()};                                                \
            (invoke << ... << args);                                                                           \
            return invoke;                                                                                     \
        }                                                                                                      \
                                                                                                               \
        void wait_for_compilation(Device &device) const noexcept {                                             \
            device.impl()->compile_kernel(_function.uid());                                                    \
        }                                                                                                      \
    };

LUISA_MAKE_KERNEL_ND(1)
LUISA_MAKE_KERNEL_ND(2)
LUISA_MAKE_KERNEL_ND(3)
#undef LUISA_MAKE_KERNEL_ND

namespace detail {

template<typename T>
struct is_tuple : std::false_type {};

template<typename... T>
struct is_tuple<std::tuple<T...>> : std::true_type {};

template<typename T>
constexpr auto is_tuple_v = is_tuple<T>::value;

template<typename... T, size_t... i>
[[nodiscard]] inline auto tuple_to_var_impl(std::tuple<T...> tuple, std::index_sequence<i...>) noexcept {
    return Var<std::tuple<expr_value_t<T>...>>{std::get<i>(tuple)...};
}

template<typename... T>
[[nodiscard]] inline auto tuple_to_var(std::tuple<T...> tuple) noexcept {
    return tuple_to_var_impl(std::move(tuple), std::index_sequence_for<T...>{});
}

template<typename... T, size_t... i>
[[nodiscard]] inline auto var_to_tuple_impl(Expr<std::tuple<T...>> v, std::index_sequence<i...>) noexcept {
    return std::tuple<Expr<T>...>{v.template member<i>()...};
}

template<typename... T>
[[nodiscard]] inline auto var_to_tuple(Expr<std::tuple<T...>> v) noexcept {
    return var_to_tuple_impl(v, std::index_sequence_for<T...>{});
}

}// namespace detail

template<typename T>
class Callable {
    static_assert(always_false_v<T>);
};

template<typename Ret, typename... Args>
class Callable<Ret(Args...)> {

    static_assert(
        std::negation_v<std::disjunction<
            is_buffer_or_view<Ret>,
            is_image_or_view<Ret>,
            is_volume_or_view<Ret>>>,
        "Callables may not return buffers, "
        "images or volumes (or their views).");

private:
    Function _function;

public:
    Callable(Callable &&) noexcept = default;
    Callable(const Callable &) noexcept = default;

    template<typename Def>
    requires concepts::invocable<Def, detail::prototype_to_creation_t<Args>...>
    Callable(Def &&def) noexcept
        : _function{FunctionBuilder::define_callable([&def] {
              if constexpr (std::is_same_v<Ret, void>) {
                  def(detail::prototype_to_creation_t<Args>{detail::ArgumentCreation{}}...);
              } else if constexpr (detail::is_tuple_v<Ret>) {
                  auto ret = detail::tuple_to_var(def(detail::prototype_to_creation_t<Args>{detail::ArgumentCreation{}}...));
                  FunctionBuilder::current()->return_(ret.expression());
              } else {
                  auto ret = def(detail::prototype_to_creation_t<Args>{detail::ArgumentCreation{}}...);
                  FunctionBuilder::current()->return_(ret.expression());
              }
          })} {}

    auto operator()(detail::prototype_to_callable_invocation_t<Args>... args) const noexcept {
        if constexpr (std::is_same_v<Ret, void>) {
            FunctionBuilder::current()->call(
                _function.uid(),
                {args.expression()...});
        } else if constexpr (detail::is_tuple_v<Ret>) {
            Var ret = detail::Expr<Ret>{FunctionBuilder::current()->call(
                Type::of<Ret>(),
                _function.uid(),
                {args.expression()...})};
            return detail::var_to_tuple(ret);
        } else {
            return detail::Expr<Ret>{FunctionBuilder::current()->call(
                Type::of<Ret>(),
                _function.uid(),
                {args.expression()...})};
        }
    }
};

namespace detail {

template<typename T>
struct function {
    using type = typename function<
        decltype(std::function{std::declval<T>()})>::type;
};

template<typename R, typename... A>
using function_signature_t = R(A...);

template<typename... Args>
struct function<std::function<void(Args...)>> {
    using type = function_signature_t<
        void,
        definition_to_prototype_t<Args>...>;
};

template<typename Ret, typename... Args>
struct function<std::function<Ret(Args...)>> {
    using type = function_signature_t<
        expr_value_t<Ret>,
        definition_to_prototype_t<Args>...>;
};

template<typename... Ret, typename... Args>
struct function<std::function<std::tuple<Ret...>(Args...)>> {
    using type = function_signature_t<
        std::tuple<expr_value_t<Ret>...>,
        definition_to_prototype_t<Args>...>;
};

template<typename RA, typename RB, typename... Args>
struct function<std::function<std::pair<RA, RB>(Args...)>> {
    using type = function_signature_t<
        std::tuple<expr_value_t<RA>, expr_value_t<RB>>,
        definition_to_prototype_t<Args>...>;
};

template<typename T>
struct function<Kernel1D<T>> {
    using type = T;
};

template<typename T>
struct function<Kernel2D<T>> {
    using type = T;
};

template<typename T>
struct function<Kernel3D<T>> {
    using type = T;
};

template<typename T>
struct function<Callable<T>> {
    using type = T;
};

template<typename T>
using function_t = typename function<T>::type;

}// namespace detail

template<typename T>
Kernel1D(T &&) -> Kernel1D<detail::function_t<T>>;

template<typename T>
Kernel2D(T &&) -> Kernel2D<detail::function_t<T>>;

template<typename T>
Kernel3D(T &&) -> Kernel3D<detail::function_t<T>>;

template<typename T>
Callable(T &&) -> Callable<detail::function_t<T>>;

}// namespace luisa::compute

namespace luisa::compute::detail {

struct CallableBuilder {
    template<typename F>
    [[nodiscard]] auto operator%(F &&def) const noexcept { return Callable{std::forward<F>(def)}; }
};

struct Kernel1DBuilder {
    template<typename F>
    [[nodiscard]] auto operator%(F &&def) const noexcept { return Kernel1D{std::forward<F>(def)}; }
};

struct Kernel2DBuilder {
    template<typename F>
    [[nodiscard]] auto operator%(F &&def) const noexcept { return Kernel2D{std::forward<F>(def)}; }
};

struct Kernel3DBuilder {
    template<typename F>
    [[nodiscard]] auto operator%(F &&def) const noexcept { return Kernel3D{std::forward<F>(def)}; }
};

}// namespace luisa::compute::detail

#define LUISA_KERNEL1D ::luisa::compute::detail::Kernel1DBuilder{} % [&]
#define LUISA_KERNEL2D ::luisa::compute::detail::Kernel2DBuilder{} % [&]
#define LUISA_KERNEL3D ::luisa::compute::detail::Kernel3DBuilder{} % [&]
#define LUISA_CALLABLE ::luisa::compute::detail::CallableBuilder{} % [&]
