//
// Created by Mike Smith on 2021/4/6.
//

#include <iostream>
#include <unordered_set>

#include <runtime/context.h>
#include <runtime/device.h>
#include <runtime/stream.h>
#include <runtime/event.h>
#include <dsl/syntax.h>
#include <tests/fake_device.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <tests/stb_image_write.h>

using namespace luisa;
using namespace luisa::compute;

struct SomeSOA {
    Buffer<float> a;
    Buffer<float2> b;
};

struct Some {
    float a;
    float2 b;
};

template<typename T>
struct soa {
    using type = typename soa<struct_member_tuple_t<T>>::type;
};

template<typename... T>
struct soa<std::tuple<T...>> {
    using type = std::tuple<typename soa<T>::type...>;
};

template<typename T, size_t N>
struct soa<Vector<T, N>> {
    using type = Buffer<Vector<T, N>>;
};

template<size_t N>
struct soa<Matrix<N>> {
    using type = Buffer<Matrix<N>>;
};

template<>
struct soa<float> {
    using type = Buffer<float>;
};

template<>
struct soa<bool> {
    using type = Buffer<bool>;
};

template<>
struct soa<int> {
    using type = Buffer<int>;
};

template<>
struct soa<uint> {
    using type = Buffer<uint>;
};

template<typename T, size_t N>
struct soa<T[N]> {
    using type = Buffer<std::array<T, N>>;
};

template<typename T, size_t N>
struct soa<std::array<T, N>> {
    using type = Buffer<std::array<T, N>>;
};

template<typename T>
using soa_t = typename soa<T>::type;

LUISA_BINDING_GROUP(SomeSOA, a, b)
LUISA_STRUCT(Some, a, b)

struct Complicated {
    float a;
    std::tuple<int, bool, Some> b;
    std::tuple<Some> c;
};

LUISA_STRUCT(Complicated, a, b, c)

using complicated_tuple = canonical_layout_t<Complicated>;

template<typename T, size_t level = 0u>
struct tuple_printer {
    auto operator()() const noexcept {
        for (auto i = 0u; i < level; i++) {
            std::cout << "  ";
        }
        std::cout << Type::of<T>()->description() << "\n";
        return 0u;
    }
};

template<typename... T, size_t level>
struct tuple_printer<std::tuple<T...>, level> {
    auto operator()() const noexcept {
        for (auto i = 0u; i < level; i++) {
            std::cout << "  ";
        }
        std::cout << "tuple\n";
        static_cast<void>(std::array{tuple_printer<T, level + 1u>{}()...});
        return 0u;
    }
};

template<typename S, typename T = S>
struct BufferSOA : BufferSOA<S, struct_member_tuple_t<S>> {
};

template<typename T>
struct BasicBufferSOA {
    Buffer<T> buffer;
    template<typename I>
    [[nodiscard]] auto read(I &&i) const noexcept {
        return buffer[std::forward<I>(i)];
    }
};

template<>
struct BufferSOA<float> : BasicBufferSOA<float> {};

template<>
struct BufferSOA<bool> : BasicBufferSOA<bool> {};

template<>
struct BufferSOA<int> : BasicBufferSOA<int> {};

template<>
struct BufferSOA<uint> : BasicBufferSOA<uint> {};

template<typename S, typename... T>
struct BufferSOA<S, std::tuple<T...>> {
    std::tuple<BufferSOA<T>...> soa;
    template<typename I, size_t... m>
    [[nodiscard]] auto read_impl(I &&i, std::index_sequence<m...>) const noexcept {
        return make_tuple(std::get<m>(soa).template read(std::forward<I>(i))...);
    }
    template<typename I>
    [[nodiscard]] auto read(I &&i) const noexcept {
        Var<S> s{read_impl(std::forward<I>(i), std::index_sequence_for<T...>{})};
    }
};

int main(int argc, char *argv[]) {

    tuple_printer<complicated_tuple>{}();

    log_level_verbose();

    Context context{argv[0]};

#if defined(LUISA_BACKEND_METAL_ENABLED)
    auto device = context.create_device("metal");
#elif defined(LUISA_BACKEND_DX_ENABLED)
    auto device = context.create_device("dx");
#else
    auto device = FakeDevice::create(context);
#endif

    Callable multi_ret = [] {
        return make_tuple(0u, 1u);
    };

    Kernel1D useless = [&](BufferVar<float4> buffer, Var<SomeSOA> soa) noexcept {
        Var i = dispatch_id().x;
        Var x = buffer[i];

        Var v0 = all(x == 0.0f);
        Var v1 = saturate(x);

        Var s = soa_read<Some>(0u, soa.a, soa.b);
        soa_write(0u, s, soa.a, soa.b);

        Var v = soa_read<float2>(0u, soa.a, soa.a);
        Var t = soa_read(0u, soa.a, soa.a);
        soa_write(0u, t, soa.a, soa.a);

        Var u = multi_ret();
    };
    [[maybe_unused]] auto shader = device.compile(useless);

    using namespace std::string_literals;
    using namespace std::string_view_literals;
    std::unordered_set<std::string, Hash64, std::equal_to<>> s{"hello"s, "world"s};
    LUISA_INFO("Present: {}", s.contains("hello"));
    LUISA_INFO("Present: {}", s.contains("world"sv));
}
