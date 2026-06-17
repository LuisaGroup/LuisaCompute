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

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <random>
#include <utility>
#include <luisa/luisa-compute.h>

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;
using namespace boost::ut::literals;

// Configuration options for AD testing
struct AdCheckOptions {
    uint32_t repeats = 1024 * 1024;// Number of random test samples
    float rel_tol = 5e-2f;         // Relative tolerance for comparison
    float abs_tol = 1e-4f;         // Absolute tolerance for near-zero gradients
    float fd_eps = 1e-3f;          // Epsilon for finite differences
    float max_precent_bad = 0.003f;// Maximum allowed percentage of failures
    float min_value = -1.0f;       // Minimum random value
    float max_value = 1.0f;        // Maximum random value
};
LUISA_STRUCT(AdCheckOptions, repeats, rel_tol, abs_tol, fd_eps, max_precent_bad, min_value, max_value) {};

using B = Buffer<float>;

// Helper template for testing N-argument functions with autodiff
// Compares autodiff gradients against central finite differences
template<int N, typename F>
void test_ad_helper(luisa::string_view name, Device &device, F &&f_, AdCheckOptions options = AdCheckOptions{}) {
    if (auto filter = std::getenv("LUISA_AD_TEST_FILTER")) {
        if (luisa::string{name}.find(filter) == luisa::string::npos) {
            return;
        }
    }
    if (auto repeats = std::getenv("LUISA_AD_TEST_REPEATS")) {
        char *end = nullptr;
        auto value = std::strtoul(repeats, &end, 10);
        if (end != repeats && value > 0u) {
            options.repeats = static_cast<uint32_t>(value);
        }
    }
    auto stream = device.create_stream(StreamTag::GRAPHICS);
    auto rng = std::mt19937{0x5eed1234u};

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
            if (!std::isfinite(fd) || !std::isfinite(ad)) {
                if (bad_count <= 20) {
                    error_msg.append(luisa::format("x[{}] = {}, fd = {}, ad = {}\n", j, input_data[j][i], fd, ad));
                }
                bad_count++;
                continue;
            }
            const auto diff = std::abs(fd - ad);
            const auto rel_diff = diff / std::max(std::abs(fd), options.abs_tol);
            if (diff > options.abs_tol && rel_diff > options.rel_tol) {
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

template<typename A, typename B>
[[nodiscard]] auto test_outer_product(A &&a, B &&b) noexcept {
    return def<float2x2>(
        luisa::compute::detail::FunctionBuilder::current()->call(
            Type::of<float2x2>(), CallOp::OUTER_PRODUCT,
            {luisa::compute::detail::extract_expression(std::forward<A>(a)),
             luisa::compute::detail::extract_expression(std::forward<B>(b))}));
}

template<typename A, typename B>
[[nodiscard]] auto test_binary_fmod(A &&a, B &&b) noexcept {
    return def<float>(
        luisa::compute::detail::FunctionBuilder::current()->binary(
            Type::of<float>(), BinaryOp::MOD,
            luisa::compute::detail::extract_expression(std::forward<A>(a)),
            luisa::compute::detail::extract_expression(std::forward<B>(b))));
}

template<typename A, typename B>
[[nodiscard]] auto test_binary_fmod2(A &&a, B &&b) noexcept {
    return def<float2>(
        luisa::compute::detail::FunctionBuilder::current()->binary(
            Type::of<float2>(), BinaryOp::MOD,
            luisa::compute::detail::extract_expression(std::forward<A>(a)),
            luisa::compute::detail::extract_expression(std::forward<B>(b))));
}

void test_autodiff_full(Device &device) {

    // luisa::log_level_info();

    // Test trigonometric functions
    TEST_AD_1(sin, -1.0, 1.0);
    TEST_AD_1(cos, -1.0, 1.0);
    TEST_AD_1(tan, -1.0, 1.0);
    TEST_AD_1(asin, -0.8, 0.8);
    TEST_AD_1(acos, -0.8, 0.8);
    TEST_AD_1(atan, -1.0, 1.0);

    // Test hyperbolic functions
    TEST_AD_1(sinh, -1.0, 1.0);
    TEST_AD_1(cosh, -1.0, 1.0);
    TEST_AD_1(tanh, -1.0, 1.0);
    TEST_AD_1(asinh, -1.0, 1.0);
    TEST_AD_1(acosh, 1.2, 3.0);
    TEST_AD_1(atanh, -0.8, 0.8);

    // Test exponential and logarithmic functions
    TEST_AD_1(exp, -1.0, 1.0);
    TEST_AD_1(exp2, -1.0, 1.0);
    [&] {
        auto options = AdCheckOptions{};
        options.min_value = 0.01f;
        options.max_value = 10.0f;
        test_ad_helper<1>(
            "log", device, [&](auto x) { return luisa::compute::log(x); }, options);
    }();
    [&] {
        auto options = AdCheckOptions{};
        options.min_value = 0.2f;
        options.max_value = 0.8f;
        test_ad_helper<1>(
            "smoothstep_x", device, [](auto x) { return smoothstep(0.0f, 1.0f, x); }, options);
    }();

    // Test float2 vector operations
    {
        test_ad_helper<2>("float2_length", device, [](auto x, auto y) { return length(make_float2(x, y)); });
        test_ad_helper<2>("float2_dot2", device, [](auto x, auto y) { return dot(make_float2(x, y), make_float2(x, y)); });
        test_ad_helper<4>("float2_dot", device, [](auto x, auto y, auto z, auto w) { return dot(make_float2(x, y), make_float2(z, w)); });
        test_ad_helper<3>("float2_overwrite_member_insert", device, [](auto x, auto y, auto z) {
            auto v = def(make_float2(x, y));
            v.x = z;
            return v.x + v.y;
        });
    }

    // Test matrix autodiff operations
    {
        auto options = AdCheckOptions{};
        options.min_value = -0.8f;
        options.max_value = 0.8f;
        options.repeats = 256u * 1024u;
        test_ad_helper<6>("matrix2_linalg_mul", device, [](auto a, auto b, auto c, auto d, auto x, auto y) {
            auto m = make_float2x2(
                make_float2(a + 1.5f, b),
                make_float2(c, d + 1.25f));
            auto v = make_float2(x, y);
            auto r = m * v;
            return dot(r, make_float2(0.7f, -0.35f)) + 0.1f * dot(v, v);
        }, options);
        test_ad_helper<8>("matrix2_matrix_mul", device, [](auto a, auto b, auto c, auto d,
                                                            auto e, auto f, auto g, auto h) {
            auto lhs = make_float2x2(
                make_float2(a + 1.3f, b),
                make_float2(c, d + 1.1f));
            auto rhs = make_float2x2(
                make_float2(e + 0.9f, f * 0.5f),
                make_float2(g * 0.5f, h + 1.2f));
            auto r = lhs * rhs;
            return dot(r * make_float2(0.25f, -0.4f), make_float2(0.6f, -0.2f));
        }, options);
        test_ad_helper<4>("float2_outer_product", device, [](auto ax, auto ay, auto bx, auto by) {
            auto a = make_float2(ax, ay);
            auto b = make_float2(bx, by);
            auto m = test_outer_product(a, b);
            return dot(m * make_float2(0.8f, -0.45f), make_float2(-0.2f, 0.7f));
        }, options);
        test_ad_helper<5>("matrix2_component_scalar", device, [](auto a, auto b, auto c, auto d, auto s) {
            auto m = make_float2x2(
                make_float2(a + 1.4f, b * 0.3f + 0.2f),
                make_float2(c * 0.25f - 0.1f, d + 1.3f));
            auto scale = s + 1.6f;
            auto shifted = m + 2.3f;
            auto r = (m * scale) + (0.7f + s) / shifted - shifted / (scale + 1.1f);
            return dot(r * make_float2(0.35f, -0.55f), make_float2(0.8f, -0.25f));
        }, options);
    }
    {
        auto options = AdCheckOptions{};
        options.min_value = -0.5f;
        options.max_value = 0.5f;
        options.repeats = 256u * 1024u;
        options.fd_eps = 1e-2f;
        test_ad_helper<4>("matrix2_det_inverse", device, [](auto a, auto b, auto c, auto d) {
            auto m = make_float2x2(
                make_float2(a + 1.8f, b * 0.25f),
                make_float2(c * 0.25f, d + 1.6f));
            auto inv = inverse(m);
            return determinant(m) + dot(inv * make_float2(0.3f, -0.7f), make_float2(0.5f, 0.25f));
        }, options);
    }
    {
        auto options = AdCheckOptions{};
        options.min_value = -0.9f;
        options.max_value = 0.9f;
        options.repeats = 256u * 1024u;
        test_ad_helper<2>("binary_fmod", device, [](auto x, auto y) {
            auto denom = y + 1.6f;
            return test_binary_fmod(1.7f * x, denom) + 0.2f * x * y;
        }, options);
        test_ad_helper<4>("binary_fmod2", device, [](auto x, auto y, auto z, auto w) {
            auto a = make_float2(1.3f * x, -1.1f * y);
            auto b = make_float2(z + 1.5f, w + 1.7f);
            auto r = test_binary_fmod2(a, b);
            return dot(r, make_float2(0.6f, -0.4f)) + 0.1f * (x * z + y * w);
        }, options);
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
        test_ad_helper<6>("float3_reflect", device, [](auto ix, auto iy, auto iz, auto nx, auto ny, auto nz) {
            auto i = make_float3(ix, iy, iz);
            auto n = normalize(make_float3(nx + 0.35f, ny - 0.2f, nz + 1.1f));
            auto r = reflect(i, n);
            return dot(r, make_float3(0.7f, -0.25f, 0.45f)) + 0.15f * length(r);
        });
        test_ad_helper<3>("float3_faceforward", device, [](auto nx, auto ny, auto nz) {
            auto n = make_float3(nx, ny, nz);
            auto ff = faceforward(n, make_float3(0.15f, -1.0f, 0.2f), make_float3(0.0f, 1.0f, 0.0f));
            return dot(ff, make_float3(0.25f, -0.75f, 0.5f));
        });
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

    // Test callable inlining before XIR autodiff with nested CFG in callable body
    {
        Callable nested = [](Float a, Float b, Float c) noexcept {
            auto y = def(sin(a * b) + c * c);
            $if (dispatch_x() % 2u == 0u) {
                y = y + cos(a + c) * b;
            }
            $else {
                y = y + exp(0.2f * b) * a;
            };
            auto tag = def((dispatch_x() % 3u).template cast<int>());
            $switch (tag) {
                $case (0) {
                    y = y + tanh(a - b);
                };
                $case (1) {
                    y = y + sqrt(c * c + 1.0f);
                };
                $default {
                    y = y + 0.25f * a * c;
                };
            };
            return y;
        };
        auto options = AdCheckOptions{};
        options.min_value = 0.15f;
        options.max_value = 0.85f;
        options.repeats = 256u * 1024u;
        test_ad_helper<3>("callable_nested_cfg", device, [&](auto a, auto b, auto c) {
            return nested(a, b, c);
        }, options);
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

    // Test switch control flow with nested differentiable branches
    {
        auto options = AdCheckOptions{};
        options.min_value = 0.15f;
        options.max_value = 0.85f;
        options.repeats = 256u * 1024u;
        test_ad_helper<4>("switch_nested_cfg", device, [](auto a, auto b, auto c, auto d) {
            auto y = def(a * b + 0.25f * c);
            auto tag = def((dispatch_x() % 3u).template cast<int>());
            $switch (tag) {
                $case (0) {
                    y = sin(y + c) + d * d;
                };
                $case (1) {
                    y = cos(y * b) + a * d;
                };
                $default {
                    y = sqrt(y * y + 1.0f) + c * d;
                };
            };
            $if (b > 0.5f) {
                y = y + tanh(c + d);
            }
            $else {
                y = y + exp(a * 0.2f);
            };
            return y;
        }, options);
    }

    // Test fixed-trip loop AD with nested CFG in a simulation-like update
    {
        auto options = AdCheckOptions{};
        options.min_value = 0.2f;
        options.max_value = 0.8f;
        options.repeats = 256u * 1024u;
        test_ad_helper<4>("fixed_loop_simulation_cfg", device, [](auto px, auto py, auto vx, auto mass) {
            auto p = def(make_float2(px, py));
            auto v = def(make_float2(vx, 0.3f * py));
            auto inv_m = 1.0f / (mass + 1.5f);
            $for (step, 0, 4) {
                auto step_f = step.template cast<float>();
                auto f = def(make_float2(sin(p.y + step_f * 0.1f),
                                         cos(p.x - step_f * 0.2f)));
                $switch (step) {
                    $case (1) {
                        f.x += p.x * 0.25f;
                    };
                    $case (2) {
                        f.y -= p.y * 0.15f;
                    };
                    $default {
                        f += make_float2(0.1f * vx, -0.05f * py);
                    };
                };
                v += f * inv_m * 0.125f;
                p += v * 0.125f;
                $if (dispatch_x() % 2u == 0u) {
                    v *= 0.85f;
                }
                $else {
                    v += make_float2(0.02f, -0.015f);
                };
            };
            auto n = normalize(make_float3(p.x, p.y, 1.0f));
            auto l = normalize(make_float3(0.3f + vx, 0.4f + py, 1.2f));
            auto shade = max(dot(n, l), 0.0f);
            return shade + 0.1f * length(v);
        }, options);
    }

    // Test runtime-bound loop AD with bounded dynamic unrolling
    {
        auto options = AdCheckOptions{};
        options.min_value = 0.2f;
        options.max_value = 0.8f;
        options.repeats = 256u * 1024u;
        test_ad_helper<4>("dynamic_bound_loop_simulation_cfg", device, [](auto px, auto py, auto vx, auto mass) {
            auto p = def(make_float2(px, py));
            auto v = def(make_float2(vx, 0.2f + 0.3f * py));
            auto inv_m = 1.0f / (mass + 1.4f);
            auto steps = def((dispatch_x() % 5u).template cast<int>());
            for (auto step : dynamic_range(steps)) {
                auto sf = step.template cast<float>();
                auto force = def(make_float2(sin(p.y + 0.08f * sf),
                                             cos(p.x - 0.06f * sf)));
                $if (step % 2 == 0) {
                    force += make_float2(0.05f * px, -0.03f * mass);
                }
                $else {
                    force += make_float2(0.02f * vx, 0.04f * py);
                };
                v += force * inv_m * 0.09f;
                p += v * 0.1f;
            }
            auto n = normalize(make_float3(p.x, p.y, 1.0f));
            auto l = normalize(make_float3(0.25f + vx, 0.35f + py, 1.25f));
            return max(dot(n, l), 0.0f) + 0.04f * dot(v, v);
        }, options);
    }

    // Test explicit positive-step loop AD with simulation-like state updates
    {
        auto options = AdCheckOptions{};
        options.min_value = 0.2f;
        options.max_value = 0.8f;
        options.repeats = 256u * 1024u;
        test_ad_helper<4>("positive_step_loop_simulation_cfg", device, [](auto px, auto py, auto vx, auto mass) {
            auto p = def(make_float2(px, py));
            auto v = def(make_float2(vx, 0.2f + 0.25f * py));
            auto inv_m = 1.0f / (mass + 1.75f);
            $for (step, 0, 6, 2) {
                auto step_f = step.template cast<float>();
                auto force = def(make_float2(sin(p.y + 0.07f * step_f),
                                             cos(p.x - 0.11f * step_f)));
                $if (step == 2) {
                    force += make_float2(p.x * p.y, -0.2f * vx);
                }
                $else {
                    force += make_float2(0.05f * mass, 0.03f * py);
                };
                v += force * inv_m * 0.1f;
                p += v * (0.08f + 0.01f * step_f);
            };
            auto n = normalize(make_float3(p.x, p.y, 1.0f));
            auto l = normalize(make_float3(0.25f + vx, 0.45f + py, 1.35f));
            return max(dot(n, l), 0.0f) + 0.05f * dot(v, v);
        }, options);
    }

    // Test descending explicit-step loop AD with render-like light accumulation
    {
        auto options = AdCheckOptions{};
        options.min_value = -0.75f;
        options.max_value = 0.75f;
        options.repeats = 256u * 1024u;
        test_ad_helper<4>("descending_step_diff_render_cfg", device, [](auto nx, auto ny, auto rough, auto albedo) {
            auto n = normalize(make_float3(nx, ny, 1.2f));
            auto v = normalize(make_float3(0.1f, -0.35f, 1.0f));
            auto color = def(0.04f * albedo);
            $for (light, 3, 0, -1) {
                auto lf = light.template cast<float>();
                auto l = normalize(make_float3(0.22f + 0.15f * lf,
                                               -0.45f + 0.08f * lf,
                                               1.05f + 0.04f * lf));
                auto ndotl = max(dot(n, l), 0.0f);
                auto h = normalize(l + v);
                auto spec = pow(max(dot(n, h), 0.0f), 1.5f + lf);
                $switch (light) {
                    $case (1) {
                        color += albedo * ndotl + (0.15f + rough * rough) * spec;
                    };
                    $case (2) {
                        auto refl = reflect(-v, n);
                        color += 0.55f * albedo * ndotl + rough * max(dot(refl, l), 0.0f);
                    };
                    $default {
                        color += 0.35f * albedo * ndotl + 0.25f * spec;
                    };
                };
            };
            return color;
        }, options);
    }

    // Test fixed-trip loop AD when a structured continue skips one simulation step
    {
        auto options = AdCheckOptions{};
        options.min_value = -0.65f;
        options.max_value = 0.65f;
        options.repeats = 256u * 1024u;
        test_ad_helper<4>("continue_loop_simulation_cfg", device, [](auto px, auto py, auto vx, auto mass) {
            auto p = def(make_float2(px, py));
            auto v = def(make_float2(vx, 0.15f + 0.2f * mass));
            auto inv_m = 1.0f / (mass * mass + 1.4f);
            $for (step, 0, 5) {
                $if (step == 2) {
                    v += make_float2(0.04f * px, -0.03f * py);
                    $continue;
                };
                auto sf = step.template cast<float>();
                auto force = make_float2(sin(p.y + 0.09f * sf),
                                          cos(p.x - 0.07f * sf));
                v += force * inv_m * (0.06f + 0.01f * sf);
                p += v * 0.11f;
            };
            auto n = normalize(make_float3(p.x, p.y, 1.1f));
            auto l = normalize(make_float3(0.3f + vx, 0.25f + py, 1.2f));
            return max(dot(n, l), 0.0f) + 0.04f * dot(v, v);
        }, options);
    }

    // Test fixed-trip loop AD when a render-style light loop exits early
    {
        auto options = AdCheckOptions{};
        options.min_value = -0.6f;
        options.max_value = 0.6f;
        options.repeats = 256u * 1024u;
        test_ad_helper<4>("break_loop_diff_render_cfg", device, [](auto nx, auto ny, auto rough, auto albedo) {
            auto n = normalize(make_float3(nx, ny, 1.15f));
            auto v = normalize(make_float3(-0.15f, 0.25f, 1.0f));
            auto color = def(0.03f * albedo);
            $for (light, 0, 6) {
                auto lf = light.template cast<float>();
                auto l = normalize(make_float3(0.18f + 0.12f * lf,
                                               -0.25f + 0.07f * lf,
                                               1.0f + 0.05f * lf));
                auto ndotl = max(dot(n, l), 0.0f);
                auto h = normalize(l + v);
                color += albedo * ndotl + (0.08f + rough * rough) * pow(max(dot(n, h), 0.0f), 1.2f + 0.15f * lf);
                $if (light == 3) {
                    $break;
                };
                color += 0.02f * rough * lf;
            };
            return color;
        }, options);
    }

    // Test switch-case break/continue exits in a fixed-trip AD loop
    {
        auto options = AdCheckOptions{};
        options.min_value = -0.55f;
        options.max_value = 0.55f;
        options.repeats = 256u * 1024u;
        test_ad_helper<4>("switch_early_exit_loop_cfg", device, [](auto px, auto py, auto vx, auto mass) {
            auto p = def(make_float2(px, py));
            auto v = def(make_float2(vx, 0.1f + 0.2f * mass));
            auto inv_m = 1.0f / (1.6f + mass * mass);
            $for (step, 0, 6) {
                $switch (step) {
                    $case (1) {
                        v += make_float2(0.03f * px, -0.02f * py);
                        $continue;
                    };
                    $case (4) {
                        p += 0.05f * v;
                        $break;
                    };
                    $default {
                        auto sf = step.template cast<float>();
                        auto force = make_float2(sin(p.y + 0.08f * sf),
                                                  cos(p.x - 0.05f * sf));
                        v += force * inv_m * 0.07f;
                    };
                };
                p += v * 0.12f;
            };
            auto n = normalize(make_float3(p.x, p.y, 1.0f));
            auto l = normalize(make_float3(0.35f + vx, -0.2f + py, 1.15f));
            return max(dot(n, l), 0.0f) + 0.03f * dot(v, v);
        }, options);
    }

    // Test render-like differentiable shading with fixed looped light accumulation
    {
        auto options = AdCheckOptions{};
        options.min_value = -0.75f;
        options.max_value = 0.75f;
        options.repeats = 256u * 1024u;
        test_ad_helper<4>("diff_render_shading_cfg", device, [](auto nx, auto ny, auto rough, auto albedo) {
            auto n = normalize(make_float3(nx, ny, 1.25f));
            auto v = normalize(make_float3(0.2f, -0.3f, 1.0f));
            auto color = def(0.05f * albedo);
            $for (light, 0, 3) {
                auto lf = light.template cast<float>();
                auto l = normalize(make_float3(0.35f + 0.17f * lf,
                                               -0.4f + 0.13f * lf,
                                               1.1f + 0.05f * lf));
                auto ndotl = max(dot(n, l), 0.0f);
                auto h = normalize(l + v);
                auto spec = pow(max(dot(n, h), 0.0f), 2.0f + lf);
                $if ((dispatch_x() + light) % 2u == 0u) {
                    color += albedo * ndotl + (0.2f + rough * rough) * spec;
                }
                $else {
                    auto refl = reflect(-v, n);
                    color += 0.6f * albedo * ndotl + rough * max(dot(refl, l), 0.0f);
                };
            };
            return color;
        }, options);
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
