#pragma once
#include <luisa/ast/function.h>
#include <luisa/ast/function_builder.h>
#include <luisa/dsl/func.h>
#include <luisa/runtime/shader.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/buffer.h>
#include <luisa/runtime/image.h>
#include <luisa/runtime/volume.h>

namespace luisa::compute {

class LUISA_RUNTIME_API BuiltinKernel {
    Device _device;

public:
    explicit BuiltinKernel(Device const &device) noexcept;

    // Static methods return shared_ptr so the builder stays alive
    [[nodiscard]] static luisa::shared_ptr<const detail::FunctionBuilder> fill_buffer_from_first() noexcept;
    [[nodiscard]] static luisa::shared_ptr<const detail::FunctionBuilder> fill_buffer() noexcept;
    [[nodiscard]] static luisa::shared_ptr<const detail::FunctionBuilder> fill_image_uint() noexcept;
    [[nodiscard]] static luisa::shared_ptr<const detail::FunctionBuilder> fill_image_int() noexcept;
    [[nodiscard]] static luisa::shared_ptr<const detail::FunctionBuilder> fill_image_float() noexcept;
    [[nodiscard]] static luisa::shared_ptr<const detail::FunctionBuilder> fill_volume_uint() noexcept;
    [[nodiscard]] static luisa::shared_ptr<const detail::FunctionBuilder> fill_volume_int() noexcept;
    [[nodiscard]] static luisa::shared_ptr<const detail::FunctionBuilder> fill_volume_float() noexcept;

    template<size_t N, typename... Args>
    [[nodiscard]] auto compile(luisa::shared_ptr<const detail::FunctionBuilder> builder) noexcept -> Shader<N, Args...> {
        Kernel<N, Args...> kernel{builder};
        return _device.compile(kernel);
    }

    // Cache for each fill operation
    Shader<1, Buffer<uint>, uint> _fill_buffer_from_first;
    Shader<1, Buffer<uint>, uint> _fill_buffer_uint;
    Shader<2, Image<uint>, uint> _fill_image_uint;
    Shader<2, Image<int>, int> _fill_image_int;
    Shader<2, Image<float>, float> _fill_image_float;
    Shader<3, Volume<uint>, uint> _fill_volume_uint;
    Shader<3, Volume<int>, int> _fill_volume_int;
    Shader<3, Volume<float>, float> _fill_volume_float;
    void compile_all(Device &device);
    template<typename U>
        requires((!std::is_reference_v<U>) && (std::is_copy_constructible_v<U>) && (alignof(U) >= 4))
    void fill_buffer(
        CommandList &cmdlist,
        BufferView<U> buffer_view,
        U const &value) {
        // use temporal value
        if constexpr (sizeof(U) > 4) {
            auto ptr = luisa::new_with_allocator<U>(value);
            size_t element_size = sizeof(U) / sizeof(uint);
            cmdlist << buffer_view.subview(0, 1).copy_from(luisa::span<U>{ptr, 1});
            if (buffer_view.size() > 1)
                cmdlist << _fill_buffer_from_first(buffer_view.template as<uint>(), static_cast<uint>(element_size)).dispatch(static_cast<uint>(element_size * (buffer_view.size() - 1)));

            cmdlist.add_dtor_callback([ptr] {
                luisa::delete_with_allocator(ptr);
            });
        }
        // cast to uint
        else {
            cmdlist << _fill_buffer_uint(buffer_view.template as<uint>(), reinterpret_cast<uint const &>(value)).dispatch(buffer_view.size());
        }
    }

    template<typename T>
        requires(is_legal_image_element<T>)
    void fill_image(
        CommandList &cmdlist,
        ImageView<T> image_view,
        T const &value) noexcept {
        if constexpr (std::is_same_v<T, uint>) {
            cmdlist << _fill_image_uint(image_view, value).dispatch(image_view.size());
        } else if constexpr (std::is_same_v<T, int>) {
            cmdlist << _fill_image_int(image_view, value).dispatch(image_view.size());
        } else if constexpr (std::is_same_v<T, float>) {
            cmdlist << _fill_image_float(image_view, value).dispatch(image_view.size());
        }
    }

    template<typename T>
        requires(is_legal_image_element<T>)
    void fill_volume(
        CommandList &cmdlist,
        VolumeView<T> volume_view,
        T const &value) noexcept {
        if constexpr (std::is_same_v<T, uint>) {
            cmdlist << _fill_volume_uint(volume_view, value).dispatch(volume_view.size());
        } else if constexpr (std::is_same_v<T, int>) {
            cmdlist << _fill_volume_int(volume_view, value).dispatch(volume_view.size());
        } else if constexpr (std::is_same_v<T, float>) {
            cmdlist << _fill_volume_float(volume_view, value).dispatch(volume_view.size());
        }
    }
};

}// namespace luisa::compute
