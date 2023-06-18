#include "binding_to_arg.h"
#include <luisa/core/logging.h>
namespace lc::hlsl {
vstd::vector<Argument> binding_to_arg(vstd::span<const Function::Binding> bindings) {
    vstd::vector<Argument> r;
    vstd::push_back_func(
        r, bindings.size(),
        [&](size_t i) {
            return luisa::visit(
                [&]<typename T>(T const &a) -> Argument {
                    Argument arg;
                    if constexpr (std::is_same_v<T, Function::BufferBinding>) {
                        arg.tag = Argument::Tag::BUFFER;
                        arg.buffer = a;
                    } else if constexpr (std::is_same_v<T, Function::TextureBinding>) {
                        arg.tag = Argument::Tag::TEXTURE;
                        arg.texture = a;
                    } else if constexpr (std::is_same_v<T, Function::BindlessArrayBinding>) {
                        arg.tag = Argument::Tag::BINDLESS_ARRAY;
                        arg.bindless_array = a;
                    } else if constexpr (std::is_same_v<T, Function::AccelBinding>) {
                        arg.tag = Argument::Tag::ACCEL;
                        arg.accel = a;
                    } else {
                        LUISA_ERROR("Binding Contain unwanted variable.");
                    }
                    return arg;
                },
                bindings[i]);
        });
    return r;
}
}// namespace lc::hlsl
