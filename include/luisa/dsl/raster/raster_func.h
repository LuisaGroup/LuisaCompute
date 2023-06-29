#pragma once
#include <luisa/dsl/func.h>
namespace luisa::compute {
/// RasterStageKernel class. RasterStageKernel<T> is not allowed, unless T is a function type.
template<typename T>
class RasterStageKernel {
    static_assert(always_false_v<T>);
};

template<typename T>
struct is_callable<RasterStageKernel<T>> : std::true_type {};
/// RasterStageKernel class with a function type as template parameter.
template<typename Ret, typename... Args>
class RasterStageKernel<Ret(Args...)> {

    static_assert(
        std::negation_v<std::disjunction<
            is_buffer_or_view<Ret>,
            is_image_or_view<Ret>,
            is_volume_or_view<Ret>>>,
        "Callables may not return buffers, "
        "images or volumes (or their views).");
    static_assert(std::negation_v<std::disjunction<std::is_pointer<Args>...>>);

private:
    luisa::shared_ptr<const detail::FunctionBuilder> _builder;

public:
    /**
     * @brief Construct a RasterStageKernel object.
     * 
     * The function provided will be called and recorded during construction.
     * 
     * @param f the function of callable.
     */
    template<typename Def>
        requires std::negation_v<is_callable<std::remove_cvref_t<Def>>> &&
                 std::negation_v<is_kernel<std::remove_cvref_t<Def>>>
    RasterStageKernel(Def &&f) noexcept {
        static_assert(std::is_invocable_r_v<void, Def, detail::prototype_to_creation_t<Args>...>);
        _builder = detail::FunctionBuilder::define_raster_stage([&f] {
            auto create = []<size_t... i>(auto &&def, std::index_sequence<i...>) noexcept {
                using arg_tuple = std::tuple<Args...>;
                using var_tuple = std::tuple<Var<std::remove_cvref_t<Args>>...>;
                using tag_tuple = std::tuple<detail::prototype_to_creation_tag_t<Args>...>;
                auto args = detail::create_argument_definitions<var_tuple, tag_tuple>(std::tuple<>{});
                static_assert(std::tuple_size_v<decltype(args)> == sizeof...(Args));
                return luisa::invoke(std::forward<decltype(def)>(def),
                                   static_cast<detail::prototype_to_creation_t<
                                       std::tuple_element_t<i, arg_tuple>> &&>(std::get<i>(args))...);
            };
            if constexpr (std::is_same_v<Ret, void>) {
                create(std::forward<Def>(f), std::index_sequence_for<Args...>{});
                detail::FunctionBuilder::current()->return_(nullptr);// to check if any previous $return called with non-void types
            } else {
                auto ret = def<Ret>(create(std::forward<Def>(f), std::index_sequence_for<Args...>{}));
                detail::FunctionBuilder::current()->return_(ret.expression());
            }
        });
    }
    /// Get the underlying AST
    [[nodiscard]] auto function() const noexcept { return Function{_builder.get()}; }
    [[nodiscard]] auto const &function_builder() const & noexcept { return _builder; }
    [[nodiscard]] auto &&function_builder() && noexcept { return std::move(_builder); }
};
namespace detail {
template<typename T>
struct dsl_function<RasterStageKernel<T>> {
    using type = T;
};
}// namespace detail
template<typename T>
RasterStageKernel(T &&) -> RasterStageKernel<detail::dsl_function_t<std::remove_cvref_t<T>>>;
}// namespace luisa::compute
