/**
 * @file test/feat/common/test_autodiff_full.cpp
 * @author sailing-innocent
 * @date 2023/08/03
 * @brief the full feature test for autodiff
*/

#include "common/config.h"

#include <cmath>
#include <luisa/runtime/context.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/buffer.h>
#include <luisa/runtime/stream.h>
#include <luisa/dsl/sugar.h>

#include <random>

using namespace luisa;
using namespace luisa::compute;

namespace luisa::test {

struct AdCheckOptions {
    uint32_t repeats = 1024 * 1024;
    float rel_tol = 5e-2f;
    float fd_eps = 1e-3f;
    float max_precent_bad = 0.003f;
    float min_value = -1.0f;
    float max_value = 1.0f;
};

}// namespace luisa::test

LUISA_STRUCT(luisa::test::AdCheckOptions, repeats, rel_tol, fd_eps, max_precent_bad, min_value, max_value) {};

namespace luisa::test {

using B = Buffer<float>;

template<int N, typename F>
int test_ad_helper(luisa::string_view name, Device &device, F &&f_, AdCheckOptions options = AdCheckOptions{}) {
    auto stream = device.create_stream(StreamTag::GRAPHICS);
    auto rng = std::mt19937{std::random_device{}()};

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
    const auto inputs = [&] {
        auto inputs = luisa::vector<B>();
        for (auto i = 0; i < N; i++) {
            auto tmp = device.create_buffer<float>(options.repeats);
            stream << tmp.copy_from(input_data[i].data()) << synchronize();
            inputs.emplace_back(std::move(tmp));
        }
        return inputs;
    }();

    const auto dinputs_fd = [&] {
        auto dinputs_fd = luisa::vector<B>();
        for (auto i = 0; i < N; i++) {
            dinputs_fd.emplace_back(device.create_buffer<float>(options.repeats));
        }
        return dinputs_fd;
    }();
    const auto dinputs_ad = [&] {
        auto dinputs_ad = luisa::vector<B>();
        for (auto i = 0; i < N; i++) {
            dinputs_ad.emplace_back(device.create_buffer<float>(options.repeats));
        }
        return dinputs_ad;
    }();
    auto f = [&](luisa::span<Var<float>> x) {
        auto impl = [&]<size_t... i>(std::index_sequence<i...>) noexcept {
            return f_(x[i]...);
        };
        return impl(std::make_index_sequence<N>{});
    };
    Kernel1D fd_kernel = [&](Var<AdCheckOptions> options) {
        const auto i = dispatch_x();
        auto x = luisa::vector<Var<float>>();
        for (auto j = 0; j < N; j++) {
            x.emplace_back(def(inputs[j]->read(i)));
        }
        auto eval_f = [&](int comp, Expr<float> dx) {
            auto x_copy = x;
            x_copy[comp] += dx;
            auto y = f(x_copy);
            return y;
        };
        auto dx = luisa::vector<Var<float>>();
        for (auto j = 0; j < N; j++) {
            auto f_plus_xi = eval_f(j, options.fd_eps);
            auto f_minus_xi = eval_f(j, -options.fd_eps);
            dx.emplace_back(def((f_plus_xi - f_minus_xi) / (2 * options.fd_eps)));
        }
        for (auto j = 0; j < N; j++) {
            dinputs_fd[j]->write(i, dx[j]);
        }
    };
    Kernel1D ad_kernel = [&](Var<AdCheckOptions> options) {
        const auto i = dispatch_x();
        auto x = luisa::vector<Var<float>>();
        for (auto j = 0; j < N; j++) {
            x.emplace_back(def(inputs[j]->read(i)));
        }
        $autodiff {
            for (auto j = 0; j < N; j++) {
                requires_grad(x[j]);
            }
            auto y = f(x);
            backward(y);
            for (auto j = 0; j < N; j++) {
                dinputs_ad[j]->write(i, grad(x[j]));
            }
        };
    };
    auto o = luisa::compute::ShaderOption{.enable_fast_math = false};
    stream
        << device.compile(fd_kernel, o)(options).dispatch(options.repeats)
        << device.compile(ad_kernel, o)(options).dispatch(options.repeats)
        << synchronize();
    const auto fd_data = [&] {
        auto fd_data = luisa::vector<luisa::vector<float>>();
        for (auto i = 0; i < N; i++) {
            luisa::vector<float> tmp;
            tmp.resize(options.repeats);
            stream << dinputs_fd[i].copy_to(tmp.data()) << synchronize();
            fd_data.emplace_back(std::move(tmp));
        }
        return fd_data;
    }();
    const auto ad_data = [&] {
        auto ad_data = luisa::vector<luisa::vector<float>>();
        for (auto i = 0; i < N; i++) {
            luisa::vector<float> tmp;
            tmp.resize(options.repeats);
            stream << dinputs_ad[i].copy_to(tmp.data()) << synchronize();
            ad_data.emplace_back(std::move(tmp));
        }
        return ad_data;
    }();
    size_t bad_count = 0;
    luisa::string error_msg;
    for (size_t i = 0; i < options.repeats; i++) {
        for (size_t j = 0; j < N; j++) {
            const auto fd = fd_data[j][i];
            const auto ad = ad_data[j][i];
            const auto diff = std::abs(fd - ad);
            const auto rel_diff = diff / std::abs(fd);
            if (rel_diff > options.rel_tol) {
                bad_count++;
            }
        }
    }
    const auto bad_percent = static_cast<float>(bad_count) / (options.repeats * N);
    CHECK_MESSAGE(bad_percent < options.max_precent_bad, "Test `{}` First 20 errors:\n{}\nTest `{}`: Bad percent {}% is greater than max percent {}%.\n", name, error_msg, name, bad_percent * 100, options.max_precent_bad * 100);
    return 0;
}

}// namespace luisa::test

TEST_SUITE("ir") {
    LUISA_TEST_CASE_WITH_DEVICE("autodiff::sin",
                                luisa::test::test_ad_helper<1>(
                                    "sin", device, [&](auto x) { return sin(x); },
                                    luisa::test::AdCheckOptions{
                                        .min_value = -1.0,
                                        .max_value = 1.0}) == 0);
    LUISA_TEST_CASE_WITH_DEVICE("autodiff::cos",
                                luisa::test::test_ad_helper<1>(
                                    "cos", device, [&](auto x) { return cos(x); },
                                    luisa::test::AdCheckOptions{
                                        .min_value = -1.0,
                                        .max_value = 1.0}) == 0);
    LUISA_TEST_CASE_WITH_DEVICE("autodiff::tan",
                                luisa::test::test_ad_helper<1>(
                                    "tan", device, [&](auto x) { return tan(x); },
                                    luisa::test::AdCheckOptions{
                                        .min_value = -1.0,
                                        .max_value = 1.0}) == 0);
    LUISA_TEST_CASE_WITH_DEVICE("autodiff::asin",
                                luisa::test::test_ad_helper<1>(
                                    "asin", device, [&](auto x) { return asin(x); },
                                    luisa::test::AdCheckOptions{
                                        .min_value = -1.0,
                                        .max_value = 1.0}) == 0);
    LUISA_TEST_CASE_WITH_DEVICE("autodiff::acos",
                                luisa::test::test_ad_helper<1>(
                                    "acos", device, [&](auto x) { return acos(x); },
                                    luisa::test::AdCheckOptions{
                                        .min_value = -1.0,
                                        .max_value = 1.0}) == 0);
    LUISA_TEST_CASE_WITH_DEVICE("autodiff::atan",
                                luisa::test::test_ad_helper<1>(
                                    "atan", device, [&](auto x) { return atan(x); },
                                    luisa::test::AdCheckOptions{
                                        .min_value = -1.0,
                                        .max_value = 1.0}) == 0);
    LUISA_TEST_CASE_WITH_DEVICE("autodiff::sinh",
                                luisa::test::test_ad_helper<1>(
                                    "sinh", device, [&](auto x) { return sinh(x); },
                                    luisa::test::AdCheckOptions{
                                        .min_value = -1.0,
                                        .max_value = 1.0}) == 0);
    LUISA_TEST_CASE_WITH_DEVICE("autodiff::cosh",
                                luisa::test::test_ad_helper<1>(
                                    "cosh", device, [&](auto x) { return cosh(x); },
                                    luisa::test::AdCheckOptions{
                                        .min_value = -1.0,
                                        .max_value = 1.0}) == 0);
    LUISA_TEST_CASE_WITH_DEVICE("autodiff::tanh",
                                luisa::test::test_ad_helper<1>(
                                    "tanh", device, [&](auto x) { return tanh(x); },
                                    luisa::test::AdCheckOptions{
                                        .min_value = -1.0,
                                        .max_value = 1.0}) == 0);
    LUISA_TEST_CASE_WITH_DEVICE("autodiff::asinh",
                                luisa::test::test_ad_helper<1>(
                                    "asinh", device, [&](auto x) { return asinh(x); },
                                    luisa::test::AdCheckOptions{
                                        .min_value = -1.0,
                                        .max_value = 1.0}) == 0);
    LUISA_TEST_CASE_WITH_DEVICE("autodiff::acosh",
                                luisa::test::test_ad_helper<1>(
                                    "acosh", device, [&](auto x) { return acosh(x); },
                                    luisa::test::AdCheckOptions{
                                        .min_value = -1.0,
                                        .max_value = 1.0}) == 0);
    LUISA_TEST_CASE_WITH_DEVICE("autodiff::atanh",
                                luisa::test::test_ad_helper<1>(
                                    "atanh", device, [&](auto x) { return atanh(x); },
                                    luisa::test::AdCheckOptions{
                                        .min_value = -1.0,
                                        .max_value = 1.0}) == 0);

    LUISA_TEST_CASE_WITH_DEVICE("autodiff::exp",
                                luisa::test::test_ad_helper<1>(
                                    "exp", device, [&](auto x) { return exp(x); },
                                    luisa::test::AdCheckOptions{
                                        .min_value = -1.0,
                                        .max_value = 1.0}) == 0);
    LUISA_TEST_CASE_WITH_DEVICE("autodiff::exp2",
                                luisa::test::test_ad_helper<1>(
                                    "exp2", device, [&](auto x) { return exp2(x); },
                                    luisa::test::AdCheckOptions{
                                        .min_value = -1.0,
                                        .max_value = 1.0}) == 0);
    LUISA_TEST_CASE_WITH_DEVICE("autodiff::log",
                                luisa::test::test_ad_helper<1>(
                                    "log", device, [&](auto x) { return log(x); },
                                    luisa::test::AdCheckOptions{
                                        .min_value = 0.001,
                                        .max_value = 1.0}) == 0);

    LUISA_TEST_CASE_WITH_DEVICE("autodiff::float2_length",
                                luisa::test::test_ad_helper<2>("float2_length", device, [](auto x, auto y) {
                                    return length(make_float2(x, y));
                                }) == 0);
    LUISA_TEST_CASE_WITH_DEVICE("autodiff::float2_dot2",
                                luisa::test::test_ad_helper<2>("float2_dot2", device, [](auto x, auto y) {
                                    return dot(make_float2(x, y), make_float2(x, y));
                                }) == 0);
    LUISA_TEST_CASE_WITH_DEVICE("autodiff::float2_dot",
                                luisa::test::test_ad_helper<4>("float2_dot", device, [](auto x, auto y, auto u, auto v) {
                                    return dot(make_float2(x, y), make_float2(u, v));
                                }) == 0);

    LUISA_TEST_CASE_WITH_DEVICE("autodiff::float3_length",
                                luisa::test::test_ad_helper<3>("float3_length", device, [](auto x, auto y, auto z) {
                                    return length(make_float3(x, y, z));
                                }) == 0);
    LUISA_TEST_CASE_WITH_DEVICE("autodiff::float3_dot",
                                luisa::test::test_ad_helper<6>("float3_dot", device, [](auto x, auto y, auto z, auto u, auto v, auto w) {
                                    return dot(make_float3(x, y, z), make_float3(u, v, w));
                                }) == 0);
    LUISA_TEST_CASE_WITH_DEVICE("autodiff::float3_dot2",
                                luisa::test::test_ad_helper<3>("float3_dot2", device, [](auto x, auto y, auto z) {
                                    return dot(make_float3(x, y, z), make_float3(x, y, z));
                                }) == 0);
    LUISA_TEST_CASE_WITH_DEVICE("autodiff::float3_cross_x",
                                luisa::test::test_ad_helper<6>("float3_cross_x", device, [](auto x, auto y, auto z, auto u, auto v, auto w) {
                                    return cross(make_float3(x, y, z), make_float3(u, v, w)).x;
                                }) == 0);

    // TODO: cross is crushed until 2023-08-03
    // LUISA_TEST_CASE_WITH_DEVICE("autodiff::float3_cross_y",
    //                             luisa::test::test_ad_helper<6>("float3_cross_y", device, [](auto x, auto y, auto z, auto u, auto v, auto w) {
    //                                 return cross(make_float3(x, y, z), make_float3(u, v, w)).y;
    //                             }) == 0);
    // LUISA_TEST_CASE_WITH_DEVICE("autodiff::float3_cross_z",
    //                             luisa::test::test_ad_helper<6>("float3_cross_z", device, [](auto x, auto y, auto z, auto u, auto v, auto w) {
    //                                 return cross(make_float3(x, y, z), make_float3(u, v, w)).z;
    //                             }) == 0);
}