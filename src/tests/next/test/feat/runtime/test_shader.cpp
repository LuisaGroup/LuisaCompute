/**
 * @file test/feat/runtime/test_shader.cpp
 * @author sailing-innocent
 * @date 2024-04-14
 * @brief The Shader 
*/
#include "common/config.h"

#include <luisa/runtime/context.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>
#include <luisa/dsl/sugar.h>
#include <luisa/runtime/command_list.h>
#include <luisa/core/stl/unordered_map.h>

#include <type_traits>

using namespace luisa;
using namespace luisa::compute;

namespace luisa::test {

template<typename T>
static constexpr bool is_numeric_v = std::is_integral_v<T> || std::is_floating_point_v<T>;
template<typename T>
concept NumericT = is_numeric_v<T>;

class TypedShaderClass {
    using Device = luisa::compute::Device;
    template<typename T>
    using BufferView = luisa::compute::BufferView<T>;
    template<typename T>
    using Buffer = luisa::compute::Buffer<T>;
    template<typename T>
    using BufferVar = luisa::compute::BufferVar<T>;

public:
    template<NumericT T>
    void compile(Device &device) {
        luisa::string_view key = Type::of<T>()->description();
        if (!key.empty()) {
            auto shad_ptr = luisa::make_unique<Shader<1, Buffer<T>, int>>(device.compile<1>([](BufferVar<T> buf, Int N) {
                auto idx = dispatch_id().x;
                $if (idx < N) {
                    buf->write(idx, buf->read(idx) + static_cast<T>(1));
                };
            }));

            shad_map.try_emplace(key, std::move(shad_ptr));
        }
    }

    template<NumericT T>
    void run(CommandList &cmdlist, BufferView<T> buf) {
        luisa::string_view key = Type::of<T>()->description();
        if (!key.empty()) {
            size_t N = buf.size();
            auto *shad_it = shad_map.find(key);
            if (shad_it != shad_map.end()) {
                LUISA_INFO("shader fetched! {}", N);
                auto &shad_ptr = shad_it->second;
                cmdlist << (*reinterpret_cast<Shader<1, Buffer<T>, int> *>(&(*shad_ptr)))(buf, N).dispatch(N);
            } else {
                LUISA_INFO("shader NOT fetched!");
            }
        }
    }

private:
    luisa::unordered_map<luisa::string, luisa::unique_ptr<Resource>> shad_map;
};

template<NumericT T>
int test_typed_shader_class(Device &device) {
    constexpr int N = 100;
    auto d_arr = device.create_buffer<T>(N);
    luisa::vector<T> h_arr;
    h_arr.resize(N);
    for (auto i = 0; i < N; i++) {
        h_arr[i] = static_cast<T>(i);
    }

    luisa::test::TypedShaderClass dummy{};
    dummy.compile<T>(device);

    auto stream = device.create_stream();
    CommandList cmdlist;
    cmdlist << d_arr.copy_from(h_arr.begin());
    dummy.run<T>(cmdlist, d_arr.view());        // increment
    cmdlist << d_arr.copy_to(h_arr.begin());    // copy back
    stream << cmdlist.commit() << synchronize();// sync

    for (auto i = 0; i < N; i++) {
        CHECK(h_arr[i] == doctest::Approx(static_cast<T>(i + 1)));
    }
    return 0;
}

}// namespace luisa::test

TEST_SUITE("runtime") {
    LUISA_TEST_CASE_WITH_DEVICE("typed_shader_class_int", luisa::test::test_typed_shader_class<int>(device) == 0);
    LUISA_TEST_CASE_WITH_DEVICE("typed_shader_class_uint", luisa::test::test_typed_shader_class<uint>(device) == 0);
    LUISA_TEST_CASE_WITH_DEVICE("typed_shader_class_float", luisa::test::test_typed_shader_class<float>(device) == 0);
}