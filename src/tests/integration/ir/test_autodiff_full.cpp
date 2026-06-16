// Comprehensive test suite for automatic differentiation.
//
// This test validates autodiff across various mathematical operations including:
// - Trigonometric functions: sin, cos, tan, asin, acos, atan
// - Hyperbolic functions: sinh, cosh, tanh, asinh, acosh, atanh
// - Exponential and logarithmic: exp, exp2, log
// - Vector operations: length, dot, cross, reduce_sum, reduce_prod
// - Control flow: if-else statements
// - Data structures: arrays and custom structs
//
// Each test compares autodiff gradients with finite differences (FD) to ensure correctness.

#include "ut/ut.hpp"
#include "test_device.h"

#include <fstream>
#include <random>
#include <luisa/luisa-compute.h>

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;
using namespace boost::ut::literals;

// Configuration options for AD testing
struct AdCheckOptions {
    uint32_t repeats = 1024 * 1024;// Number of random test samples
    float rel_tol = 5e-2f;         // Relative tolerance for comparison
    float fd_eps = 1e-3f;          // Epsilon for finite differences
    float max_precent_bad = 0.003f;// Maximum allowed percentage of failures
    float min_value = -1.0f;       // Minimum random value
    float max_value = 1.0f;        // Maximum random value
};
LUISA_STRUCT(AdCheckOptions, repeats, rel_tol, fd_eps, max_precent_bad, min_value, max_value) {};

using B = Buffer<float>;

// Helper template for testing N-argument functions with autodiff
// Compares autodiff gradients against central finite differences
template<int N, typename F>
void test_ad_helper(luisa::string_view name, Device &device, F &&f_, AdCheckOptions options = AdCheckOptions{}) {
    auto stream = device.create_stream(StreamTag::GRAPHICS);
    auto rng = std::mt19937{std::random_device{}()};

    // Generate random input data
    const auto input_data = [&] {
        auto input_data = luisa::vector<luisa::vector<float>>();
        for (auto i = 0; i < N; i++) {
            auto tmp = luisa::vector<float>();
            tmp.resize(options.repeats);
            std::uniform_real_distribution<float> dist{options.min_value, options.max_value};
            for (auto j = 0; j < options.repeats; j++) {
                tmp[j] = dist(rng);
            }
            input_data.emplace_back(std::move(tmp));
        }
        return input_data;
    }();

    // Create GPU buffers for inputs
    const auto inputs = [&] {
        auto inputs = luisa::vector<B>();
        for (auto i = 0; i < N; i++) {
            auto tmp = device.create_buffer<float>(options.repeats);
            stream << tmp.copy_from(luisa::span{input_data[i]}) << synchronize();
            inputs.emplace_back(std::move(tmp));
        }
        return inputs;
    }();

    // Buffers for finite difference gradients
    const auto dinputs_fd = [&] {
        auto dinputs_fd = luisa::vector<B>();
        for (auto i = 0; i < N; i++) {
            dinputs_fd.emplace_back(device.create_buffer<float>(options.repeats));
        }
        return dinputs_fd;
    }();

    // Buffers for autodiff gradients
    const auto dinputs_ad = [&] {
        auto dinputs_ad = luisa::vector<B>();
        for (auto i = 0; i < N; i++) {
            dinputs_ad.emplace_back(device.create_buffer<float>(options.repeats));
        }
        return dinputs_ad;
    }();

    // Wrapper to call the function with N arguments
    auto f = [&](luisa::span<Var<float>> x) {
        auto impl = [&]<size_t... i>(std::index_sequence<i...>) noexcept {
            return f_(x[i]...);
        };
        return impl(std::make_index_sequence<N>{});
    };

    // Finite differences kernel using central difference formula:
    // df/dx ≈ (f(x+ε) - f(x-ε)) / (2ε)
    Kernel1D fd_kernel = [&](Var<AdCheckOptions> options) {
        const auto i = dispatch_x();
        auto x = luisa::vector<Var<float>>();
        for (auto j = 0; j < N; j++) {
            x.emplace_back(def(inputs[j]->read(i)));
        }

        // Helper to evaluate f with perturbed input
        auto eval_f = [&](int comp, Expr<float> dx) {
            auto x_copy = x;
            x_copy[comp] += dx;
            auto y = f(x_copy);
            return y;
        };

        // Compute gradients using central differences
        auto dx = luisa::vector<Var<float>>();
        for (auto j = 0; j < N; j++) {
            auto f_plus_xi = eval_f(j, options.fd_eps);
            auto f_minus_xi = eval_f(j, -options.fd_eps);
            dx.emplace_back(def((f_plus_xi - f_minus_xi) / (2 * options.fd_eps)));
        }

        // Write FD gradients to buffer
        for (auto j = 0; j < N; j++) {
            dinputs_fd[j]->write(i, dx[j]);
        }
    };

    // Autodiff kernel using reverse-mode automatic differentiation
    Kernel1D ad_kernel = [&](Var<AdCheckOptions> options) {
        const auto i = dispatch_x();
        auto x = luisa::vector<Var<float>>();
        for (auto j = 0; j < N; j++) {
            x.emplace_back(def(inputs[j]->read(i)));
        }

        $autodiff {
            // Mark all inputs as requiring gradients
            for (auto j = 0; j < N; j++) {
                requires_grad(x[j]);
            }
            // Forward pass
            auto y = f(x);
            // Backward pass: compute gradients
            backward(y);
            // Write autodiff gradients to buffer
            for (auto j = 0; j < N; j++) {
                dinputs_ad[j]->write(i, grad(x[j]));
            }
        };
    };

    // Compile and execute both kernels
    auto o = luisa::compute::ShaderOption{.enable_fast_math = false};
    stream
        << device.compile(fd_kernel, o)(options).dispatch(options.repeats)
        << device.compile(ad_kernel, o)(options).dispatch(options.repeats)
        << synchronize();

    // Copy results back to host
    const auto fd_data = [&] {
        auto fd_data = luisa::vector<luisa::vector<float>>();
        for (auto i = 0; i < N; i++) {
            luisa::vector<float> tmp;
            tmp.resize(options.repeats);
            stream << dinputs_fd[i].copy_to(luisa::span{tmp}) << synchronize();
            fd_data.emplace_back(std::move(tmp));
        }
        return fd_data;
    }();
    const auto ad_data = [&] {
        auto ad_data = luisa::vector<luisa::vector<float>>();
        for (auto i = 0; i < N; i++) {
            luisa::vector<float> tmp;
            tmp.resize(options.repeats);
            stream << dinputs_ad[i].copy_to(luisa::span{tmp}) << synchronize();
            ad_data.emplace_back(std::move(tmp));
        }
        return ad_data;
    }();

    // Compare results and count failures
    size_t bad_count = 0;
    luisa::string error_msg;
    for (size_t i = 0; i < options.repeats; i++) {
        for (size_t j = 0; j < N; j++) {
            const auto fd = fd_data[j][i];
            const auto ad = ad_data[j][i];
            const auto diff = std::abs(fd - ad);
            const auto rel_diff = diff / std::abs(fd);
            if (rel_diff > options.rel_tol) {
                if (bad_count <= 20)
                    error_msg.append(luisa::format("x[{}] = {}, fd = {}, ad = {}, diff = {}, rel_diff = {}\n", j, input_data[j][i], fd, ad, diff, rel_diff));
                bad_count++;
            }
        }
    }

    // Report results
    const auto bad_percent = static_cast<float>(bad_count) / (options.repeats * N);
    if (bad_percent > options.max_precent_bad) {
        LUISA_ERROR("Test `{}` First 20 errors:\n{}\nTest `{}`: Bad percent {}% is greater than max percent {}%.\n", name, error_msg, name, bad_percent * 100, options.max_precent_bad * 100);
    }
    LUISA_INFO("Test `{}` passed.", name);
}

// Macro for simple 1-argument function tests
#define TEST_AD_1(f, min, max) [&] {                        \
    auto options = AdCheckOptions{};                        \
    options.min_value = min;                                \
    options.max_value = max;                                \
    test_ad_helper<1>(                                      \
        #f, device, [&](auto x) { return f(x); }, options); \
}()

// Custom struct for testing struct member access in autodiff
struct Foo {
    float3 v;
    float f;
    uint z[2];
};
LUISA_STRUCT(Foo, v, f, z) {};

void test_autodiff_full(Device &device) {

    // luisa::log_level_info();

    // Test trigonometric functions
    TEST_AD_1(sin, -1.0, 1.0);
    TEST_AD_1(cos, -1.0, 1.0);
    TEST_AD_1(tan, -1.0, 1.0);
    TEST_AD_1(asin, -1.0, 1.0);
    TEST_AD_1(acos, -1.0, 1.0);
    TEST_AD_1(atan, -1.0, 1.0);

    // Test hyperbolic functions
    TEST_AD_1(sinh, -1.0, 1.0);
    TEST_AD_1(cosh, -1.0, 1.0);
    TEST_AD_1(tanh, -1.0, 1.0);
    TEST_AD_1(asinh, -1.0, 1.0);
    TEST_AD_1(acosh, -1.0, 1.0);
    TEST_AD_1(atanh, -1.0, 1.0);

    // Test exponential and logarithmic functions
    TEST_AD_1(exp, -1.0, 1.0);
    TEST_AD_1(exp2, -1.0, 1.0);
    [&] {
        auto options = AdCheckOptions{};
        options.min_value = 0.001f;
        options.max_value = 10.0f;
        test_ad_helper<1>(
            "log", device, [&](auto x) { return luisa::compute::log(x); }, options);
    }();

    // Test float2 vector operations
    {
        test_ad_helper<2>("float2_length", device, [](auto x, auto y) { return length(make_float2(x, y)); });
        test_ad_helper<2>("float2_dot2", device, [](auto x, auto y) { return dot(make_float2(x, y), make_float2(x, y)); });
        test_ad_helper<4>("float2_dot", device, [](auto x, auto y, auto z, auto w) { return dot(make_float2(x, y), make_float2(z, w)); });
    }

    // Test float3 vector operations
    {
        test_ad_helper<3>("float3_sum", device, [](auto x, auto y, auto z) { return reduce_sum(make_float3(x, y, z)); });
        test_ad_helper<3>("float3_prod", device, [](auto x, auto y, auto z) { return reduce_prod(make_float3(x, y, z)); });
        test_ad_helper<3>("float3_length", device, [](auto x, auto y, auto z) { return length(make_float3(x, y, z)); });
        test_ad_helper<3>("float3_dot2", device, [](auto x, auto y, auto z) { return dot(make_float3(x, y, z), make_float3(x, y, z)); });
        test_ad_helper<6>("float3_dot", device, [](auto vx, auto vy, auto vz, auto wx, auto wy, auto wz) { return dot(make_float3(vx, vy, vz), make_float3(wx, wy, wz)); });
        test_ad_helper<6>("float3_cross_length", device, [](auto vx, auto vy, auto vz, auto wx, auto wy, auto wz) { return length(cross(make_float3(vx, vy, vz), make_float3(wx, wy, wz))); });
        test_ad_helper<6>("float3_cross_sum", device, [](auto vx, auto vy, auto vz, auto wx, auto wy, auto wz) {
             auto n = (cross(make_float3(vx, vy, vz), make_float3(wx, wy, wz)));
             return n.x + n.y + n.z; });
        test_ad_helper<6>("float3_cross_x", device, [](auto vx, auto vy, auto vz, auto wx, auto wy, auto wz) { return cross(make_float3(vx, vy, vz), make_float3(wx, wy, wz)).x; });
        test_ad_helper<6>("float3_cross_y", device, [](auto vx, auto vy, auto vz, auto wx, auto wy, auto wz) { return cross(make_float3(vx, vy, vz), make_float3(wx, wy, wz)).y; });
        test_ad_helper<6>("float3_cross_z", device, [](auto vx, auto vy, auto vz, auto wx, auto wy, auto wz) { return cross(make_float3(vx, vy, vz), make_float3(wx, wy, wz)).z; });
    }

    // Test struct member access
    {
        test_ad_helper<3>("struct", device, [](auto a, auto b, auto c) {
            Var<Foo> foo{make_float3(a, b, c), a + b + c};
            return foo.v.x * foo.v.y + foo.v.z * foo.f;
        });
    }

    // Test if-else control flow (dynamic indexing branch)
    {
        test_ad_helper<3>("if", device, [](auto a, auto b, auto c) {
            Var<Foo> foo{make_float3(a, b, c), a + b + c};
            auto zero = def(0u);
            $if (foo.v.x > 3.0f) {
                foo.v[zero] -= 1.0f;
            }
            $else {
                foo.v[zero + 1u] -= foo.f;
            };
            return foo.v.x * foo.v.y + foo.v.z * foo.f;
        });
    }

    // Test if-else with static member access
    {
        test_ad_helper<3>("if2", device, [](auto a, auto b, auto c) {
            Var<Foo> foo{make_float3(a, b, c), a + b + c};
            $if (foo.v.x > 3.0f) {
                foo.v.x -= 1.0f;
            }
            $else {
                foo.v.x -= foo.f;
            };
            return foo.v.x * foo.v.y + foo.v.z * foo.f;
        });
    }

    // Test select operand gradient routing
    {
        auto options = AdCheckOptions{};
        options.min_value = -2.0f;
        options.max_value = 2.0f;
        test_ad_helper<2>("select", device, [](auto a, auto b) { return select(a, b, a > 0.0f); }, options);
    }

    // Test array operations
    {
        test_ad_helper<3>("array_sum", device, [](auto a, auto b, auto c) {
            ArrayFloat<3> arr{a, b, c};
            return arr[0] + arr[1] + arr[2];
        });
    }
    {
        test_ad_helper<2>("array_sum2", device, [](auto a, auto b) {
            ArrayFloat<2> arr{a, b};
            return arr[0] + arr[1];
        });
    }
}

int main(int argc, char *argv[]) {
    auto dc = luisa::test::create_device_from_ut(argc, argv);
    if (!dc) {
        return 0;
    }
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));
    auto &device = dc->device;
    test_autodiff_full(device);
}
