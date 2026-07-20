// Comprehensive test for nested callables, $lambda, $outline, and Lambda.
//
// This test verifies that DSL variable captures through outlined/lambda
// function boundaries propagate correctly — both reads and mutations.
// It exercises the exact patterns used in path_tracing_nested_callable.cpp
// which currently renders a black image, suggesting a capture-mutation bug.
//
// Corner cases tested:
//  1.  $lambda: simple capture read
//  2.  $lambda: simple capture mutation (write-back)
//  3.  $lambda: multiple captured variables mutated
//  4.  $lambda: nested (2 levels deep) with mutation
//  5.  $lambda: with parameters
//  6.  $lambda: with return value
//  7.  $outline: capture mutation
//  8.  Callable calling Callable (deep composition)
//  9.  luisa::optional<Var<T>> emplace inside $lambda
// 10.  Buffer read/write inside $lambda
// 11.  Control flow ($if / $for) inside $lambda
// 12.  RNG (LCG) state mutation through Lambda boundary
// 13.  Accumulation (radiance += ...) through nested $lambda
// 14.  $lambda inside $for loop
// 15.  Mixed: Callable invoked from inside $lambda

#include <cmath>
#include <numeric>
#include <iostream>
#include <vector>

#include <luisa/core/clock.h>
#include <luisa/core/logging.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/context.h>
#include <luisa/runtime/stream.h>
#include <luisa/runtime/buffer.h>
#include <luisa/dsl/syntax.h>
#include <luisa/dsl/sugar.h>
#include "ut/ut.hpp"
#include "test_device.h"

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;
using namespace boost::ut::literals;

static constexpr uint N = 1024u;

// ── helpers ──────────────────────────────────────────────────────────

static void check(const char *name,
                  const std::vector<float> &result,
                  const std::vector<float> &expected) {
    uint mismatch_count = 0u;
    for (uint i = 0u; i < result.size(); i++) {
        float diff = std::abs(result[i] - expected[i]);
        if (!std::isfinite(result[i]) || !std::isfinite(expected[i]) || diff > 1e-4f) {
            if (mismatch_count < 5u) {
                LUISA_WARNING("{}: mismatch at [{}]: got {} expected {} (diff {})",
                              name, i, result[i], expected[i], diff);
            }
            mismatch_count++;
        }
    }
    expect(mismatch_count == 0u) << name << ": " << mismatch_count << "/" << result.size() << " mismatches";
}

// ── LCG matching the path tracer ─────────────────────────────────────
// state = state * 1664525u + 1013904223u;
// return cast<float>(state & 0x00ffffffu) * (1.0f / 0x01000000u);

static uint host_lcg_state_update(uint s) {
    return s * 1664525u + 1013904223u;
}
static float host_lcg_value(uint s) {
    return static_cast<float>(s & 0x00ffffffu) * (1.0f / static_cast<float>(0x01000000u));
}

int test_nested_callable(Device &device) {
    log_level_verbose();
    Stream stream = device.create_stream();

    // ── buffers ──────────────────────────────────────────────────────
    Buffer<float> buf_in = device.create_buffer<float>(N);
    Buffer<float> buf_out = device.create_buffer<float>(N);

    std::vector<float> host_in(N);
    std::vector<float> host_out(N);
    std::vector<float> host_expected(N);

    // Fill host_in with [1, 2, 3, …, N]
    std::iota(host_in.begin(), host_in.end(), 1.0f);

    // ================================================================
    // Test 1: $lambda — simple capture read
    // ================================================================
    {
        Kernel1D kernel = [&](BufferVar<float> in, BufferVar<float> out) noexcept {
            UInt idx = dispatch_id().x;
            Float val = in.read(idx);
            Float result = def(0.0f);
            $lambda({
                result = val;// read captured 'val', write captured 'result'
            })();
            out.write(idx, result);
        };
        auto shader = device.compile(kernel);
        stream << buf_in.copy_from(luisa::span{host_in})
               << shader(buf_in, buf_out).dispatch(N)
               << buf_out.copy_to(luisa::span{host_out})
               << synchronize();
        // expected: identity
        check("test1_lambda_capture_read", host_out, host_in);
    }

    // ================================================================
    // Test 2: $lambda — simple capture mutation
    // ================================================================
    {
        Kernel1D kernel = [&](BufferVar<float> in, BufferVar<float> out) noexcept {
            UInt idx = dispatch_id().x;
            Float val = in.read(idx);
            $lambda({
                val = val + 10.0f;// mutate captured 'val'
            })();
            out.write(idx, val);// should see mutation
        };
        auto shader = device.compile(kernel);
        stream << buf_in.copy_from(luisa::span{host_in})
               << shader(buf_in, buf_out).dispatch(N)
               << buf_out.copy_to(luisa::span{host_out})
               << synchronize();
        for (uint i = 0; i < N; i++) host_expected[i] = host_in[i] + 10.0f;
        check("test2_lambda_capture_mutation", host_out, host_expected);
    }

    // ================================================================
    // Test 3: $lambda — multiple captured variables mutated
    // ================================================================
    {
        Kernel1D kernel = [&](BufferVar<float> in, BufferVar<float> out) noexcept {
            UInt idx = dispatch_id().x;
            Float a = in.read(idx);
            Float b = def(0.0f);
            Float c = def(0.0f);
            $lambda({
                b = a * 2.0f;
                c = a * 3.0f;
                a = a + 1.0f;
            })();
            out.write(idx, a + b + c);// (x+1) + 2x + 3x = 6x + 1
        };
        auto shader = device.compile(kernel);
        stream << buf_in.copy_from(luisa::span{host_in})
               << shader(buf_in, buf_out).dispatch(N)
               << buf_out.copy_to(luisa::span{host_out})
               << synchronize();
        for (uint i = 0; i < N; i++) host_expected[i] = 6.0f * host_in[i] + 1.0f;
        check("test3_lambda_multi_capture_mutation", host_out, host_expected);
    }

    // ================================================================
    // Test 4: nested $lambda — 2 levels deep with mutation
    //         This is THE pattern from path_tracing_nested_callable.
    // ================================================================
    {
        Kernel1D kernel = [&](BufferVar<float> in, BufferVar<float> out) noexcept {
            UInt idx = dispatch_id().x;
            Float val = in.read(idx);
            Float accum = def(0.0f);
            $lambda({
                accum += val;// first level mutates accum
                Float inner_add = val * 2.0f;
                $lambda({
                    accum += inner_add;// second level mutates accum through 2 boundaries
                })();
            })();
            out.write(idx, accum);// expected: val + val*2 = val*3
        };
        auto shader = device.compile(kernel);
        stream << buf_in.copy_from(luisa::span{host_in})
               << shader(buf_in, buf_out).dispatch(N)
               << buf_out.copy_to(luisa::span{host_out})
               << synchronize();
        for (uint i = 0; i < N; i++) host_expected[i] = host_in[i] * 3.0f;
        check("test4_nested_lambda_2_levels", host_out, host_expected);
    }

    // ================================================================
    // Test 5: $lambda with parameters
    // ================================================================
    {
        Kernel1D kernel = [&](BufferVar<float> in, BufferVar<float> out) noexcept {
            UInt idx = dispatch_id().x;
            Float val = in.read(idx);
            auto add_to = $lambda((Float amount) {
                val += amount;// captures 'val', takes 'amount' as parameter
            });
            add_to(5.0f);
            add_to(7.0f);
            out.write(idx, val);// expected: val + 5 + 7 = val + 12
        };
        auto shader = device.compile(kernel);
        stream << buf_in.copy_from(luisa::span{host_in})
               << shader(buf_in, buf_out).dispatch(N)
               << buf_out.copy_to(luisa::span{host_out})
               << synchronize();
        for (uint i = 0; i < N; i++) host_expected[i] = host_in[i] + 12.0f;
        check("test5_lambda_with_params", host_out, host_expected);
    }

    // ================================================================
    // Test 6: $lambda with return value
    // ================================================================
    {
        Kernel1D kernel = [&](BufferVar<float> in, BufferVar<float> out) noexcept {
            UInt idx = dispatch_id().x;
            Float val = in.read(idx);
            auto square = $lambda((Float x) {
                return x * x;
            });
            Float result = square(val);
            out.write(idx, result);
        };
        auto shader = device.compile(kernel);
        stream << buf_in.copy_from(luisa::span{host_in})
               << shader(buf_in, buf_out).dispatch(N)
               << buf_out.copy_to(luisa::span{host_out})
               << synchronize();
        for (uint i = 0; i < N; i++) host_expected[i] = host_in[i] * host_in[i];
        check("test6_lambda_return_value", host_out, host_expected);
    }

    // ================================================================
    // Test 7: $outline — capture mutation
    // ================================================================
    {
        Kernel1D kernel = [&](BufferVar<float> in, BufferVar<float> out) noexcept {
            UInt idx = dispatch_id().x;
            Float val = in.read(idx);
            $outline {
                val = val + 100.0f;
            };
            out.write(idx, val);
        };
        auto shader = device.compile(kernel);
        stream << buf_in.copy_from(luisa::span{host_in})
               << shader(buf_in, buf_out).dispatch(N)
               << buf_out.copy_to(luisa::span{host_out})
               << synchronize();
        for (uint i = 0; i < N; i++) host_expected[i] = host_in[i] + 100.0f;
        check("test7_outline_capture_mutation", host_out, host_expected);
    }

    // ================================================================
    // Test 8: Callable calling Callable (deep composition)
    // ================================================================
    {
        Callable add5 = [](Float x) noexcept {
            return x + 5.0f;
        };
        Callable add10 = [&add5](Float x) noexcept {
            return add5(add5(x));
        };
        Kernel1D kernel = [&](BufferVar<float> in, BufferVar<float> out) noexcept {
            UInt idx = dispatch_id().x;
            Float val = in.read(idx);
            out.write(idx, add10(val));
        };
        auto shader = device.compile(kernel);
        stream << buf_in.copy_from(luisa::span{host_in})
               << shader(buf_in, buf_out).dispatch(N)
               << buf_out.copy_to(luisa::span{host_out})
               << synchronize();
        for (uint i = 0; i < N; i++) host_expected[i] = host_in[i] + 10.0f;
        check("test8_callable_composition", host_out, host_expected);
    }

    // ================================================================
    // Test 9: $lambda with return value used via local variable
    //         (replaces the old optional<Var<T>> emplace pattern
    //         which leaks expressions across FunctionBuilder scopes)
    // ================================================================
    {
        Kernel1D kernel = [&](BufferVar<float> in, BufferVar<float> out) noexcept {
            UInt idx = dispatch_id().x;
            Float val = in.read(idx);
            auto compute = $lambda((Float v) {
                return v * 4.0f;
            });
            Float result = compute(val);
            out.write(idx, result);
        };
        auto shader = device.compile(kernel);
        stream << buf_in.copy_from(luisa::span{host_in})
               << shader(buf_in, buf_out).dispatch(N)
               << buf_out.copy_to(luisa::span{host_out})
               << synchronize();
        for (uint i = 0; i < N; i++) host_expected[i] = host_in[i] * 4.0f;
        check("test9_optional_emplace_in_lambda", host_out, host_expected);
    }

    // ================================================================
    // Test 9b: luisa::optional<Var<T>> emplace in one $lambda,
    //          deref in a SIBLING $lambda. This is the exact pattern
    //          historically used by test_path_tracing_nested_callable
    //          (emplace in outer lambda, deref in sibling lambda) which
    //          rendered all-black on cuda. Expected: out[i] = in[i]*2+1
    // ================================================================
    {
        Kernel1D kernel = [&](BufferVar<float> in, BufferVar<float> out) noexcept {
            UInt idx = dispatch_id().x;
            Float v = in.read(idx);
            luisa::optional<Float> doubled;
            $lambda({
                doubled.emplace(v * 2.0f);
            })();
            Float result = def(0.0f);
            $lambda({
                result = *doubled + 1.0f;
            })();
            out.write(idx, result);
        };
        auto shader = device.compile(kernel);
        stream << buf_in.copy_from(luisa::span{host_in})
               << shader(buf_in, buf_out).dispatch(N)
               << buf_out.copy_to(luisa::span{host_out})
               << synchronize();
        for (uint i = 0; i < N; i++) host_expected[i] = host_in[i] * 2.0f + 1.0f;
        check("test9b_optional_sibling_lambda_deref", host_out, host_expected);
    }

    // ================================================================
    // Test 10: Buffer read/write inside $lambda
    // ================================================================
    {
        Buffer<float> buf_temp = device.create_buffer<float>(N);
        Kernel1D kernel = [&](BufferVar<float> in, BufferVar<float> out) noexcept {
            UInt idx = dispatch_id().x;
            Float val = in.read(idx);
            $lambda({
                buf_temp->write(idx, val * 2.0f);
            })();
            // read back from temp buffer
            Float tmp = buf_temp->read(idx);
            out.write(idx, tmp);
        };
        auto shader = device.compile(kernel);
        stream << buf_in.copy_from(luisa::span{host_in})
               << shader(buf_in, buf_out).dispatch(N)
               << buf_out.copy_to(luisa::span{host_out})
               << synchronize();
        for (uint i = 0; i < N; i++) host_expected[i] = host_in[i] * 2.0f;
        check("test10_buffer_ops_in_lambda", host_out, host_expected);
    }

    // ================================================================
    // Test 11: Control flow ($if / $for) inside $lambda
    // ================================================================
    {
        Kernel1D kernel = [&](BufferVar<float> in, BufferVar<float> out) noexcept {
            UInt idx = dispatch_id().x;
            Float val = in.read(idx);
            Float sum = def(0.0f);
            $lambda({
                // accumulate val 3 times using $for
                $for (i, 3u) {
                    $if (val > 0.0f) {
                        sum += val;
                    };
                };
            })();
            out.write(idx, sum);// expected: val * 3 (all inputs > 0)
        };
        auto shader = device.compile(kernel);
        stream << buf_in.copy_from(luisa::span{host_in})
               << shader(buf_in, buf_out).dispatch(N)
               << buf_out.copy_to(luisa::span{host_out})
               << synchronize();
        for (uint i = 0; i < N; i++) host_expected[i] = host_in[i] * 3.0f;
        check("test11_control_flow_in_lambda", host_out, host_expected);
    }

    // ================================================================
    // Test 12: RNG (LCG) state mutation through Lambda boundary
    //          This is the EXACT broken pattern from the path tracer:
    //          Callable lcg captures &state via Lambda, mutates state.
    // ================================================================
    {
        Buffer<uint> seed_buf = device.create_buffer<uint>(N);
        std::vector<uint> host_seeds(N);
        for (uint i = 0; i < N; i++) host_seeds[i] = i + 1u;

        Kernel1D kernel = [&](BufferVar<uint> seeds, BufferVar<float> out) noexcept {
            UInt idx = dispatch_id().x;
            UInt state = seeds.read(idx);

            // lcg callable captures &state — exactly as in the path tracer
            Callable lcg = [&state]() noexcept {
                constexpr auto lcg_a = 1664525u;
                constexpr auto lcg_c = 1013904223u;
                state = state * lcg_a + lcg_c;
                return cast<float>(state & 0x00ffffffu) *
                       (1.0f / static_cast<float>(0x01000000u));
            };

            Float r1 = def(0.0f);
            Float r2 = def(0.0f);

            // call lcg through $lambda (the path tracer pattern)
            $lambda({
                r1 = lcg();
                $lambda({
                    r2 = lcg();// nested: state mutated through 2 lambda levels
                })();
            })();

            // write r1 + r2 (both should be valid random numbers, not 0)
            out.write(idx, r1 + r2);
            // also write back state so we can verify it advanced
            seeds.write(idx, state);
        };
        auto shader = device.compile(kernel);
        std::vector<float> rng_out(N);
        std::vector<uint> seed_out(N);
        stream << seed_buf.copy_from(luisa::span{host_seeds})
               << shader(seed_buf, buf_out).dispatch(N)
               << buf_out.copy_to(luisa::span{rng_out})
               << seed_buf.copy_to(luisa::span{seed_out})
               << synchronize();

        // Compute expected on host
        std::vector<float> rng_expected(N);
        for (uint i = 0; i < N; i++) {
            uint s = host_seeds[i];
            s = host_lcg_state_update(s);
            float v1 = host_lcg_value(s);
            s = host_lcg_state_update(s);
            float v2 = host_lcg_value(s);
            rng_expected[i] = v1 + v2;
        }
        check("test12_lcg_through_nested_lambda", rng_out, rng_expected);

        // Also verify final state advanced correctly
        bool state_ok = true;
        for (uint i = 0; i < N; i++) {
            uint expected_state = host_seeds[i];
            expected_state = host_lcg_state_update(expected_state);
            expected_state = host_lcg_state_update(expected_state);
            if (seed_out[i] != expected_state) {
                LUISA_WARNING("test12_state: mismatch at [{}]: got {} expected {}",
                              i, seed_out[i], expected_state);
                state_ok = false;
            }
        }
        expect(state_ok) << "test12_lcg_state_writeback";
    }

    // ================================================================
    // Test 13: Accumulation through nested $lambda
    //          radiance += beta * value — the core path tracing pattern
    // ================================================================
    {
        Kernel1D kernel = [&](BufferVar<float> in, BufferVar<float> out) noexcept {
            UInt idx = dispatch_id().x;
            Float beta = in.read(idx);// treat input as 'beta'
            Float radiance = def(0.0f);
            // outer lambda: adds beta to radiance, then calls inner
            $lambda({
                radiance += beta * 1.0f;// direct light contribution
                Float inner_contrib = beta * 0.5f;
                $lambda({
                    radiance += inner_contrib;// indirect contribution through 2 levels
                })();
            })();
            // one more single-level addition
            $lambda({
                radiance += beta * 0.25f;
            })();
            out.write(idx, radiance);// expected: beta*(1.0 + 0.5 + 0.25) = beta*1.75
        };
        auto shader = device.compile(kernel);
        stream << buf_in.copy_from(luisa::span{host_in})
               << shader(buf_in, buf_out).dispatch(N)
               << buf_out.copy_to(luisa::span{host_out})
               << synchronize();
        for (uint i = 0; i < N; i++) host_expected[i] = host_in[i] * 1.75f;
        check("test13_accumulation_nested_lambda", host_out, host_expected);
    }

    // ================================================================
    // Test 14: $lambda inside $for loop
    //          Each iteration should see the updated state from prior.
    // ================================================================
    {
        Kernel1D kernel = [&](BufferVar<float> in, BufferVar<float> out) noexcept {
            UInt idx = dispatch_id().x;
            Float val = in.read(idx);
            Float accum = def(0.0f);
            $for (i, 4u) {
                $lambda({
                    accum += val;// each iteration adds val
                })();
            };
            out.write(idx, accum);// expected: val * 4
        };
        auto shader = device.compile(kernel);
        stream << buf_in.copy_from(luisa::span{host_in})
               << shader(buf_in, buf_out).dispatch(N)
               << buf_out.copy_to(luisa::span{host_out})
               << synchronize();
        for (uint i = 0; i < N; i++) host_expected[i] = host_in[i] * 4.0f;
        check("test14_lambda_in_for_loop", host_out, host_expected);
    }

    // ================================================================
    // Test 15: Mixed — Callable invoked from inside $lambda
    // ================================================================
    {
        Callable double_it = [](Float x) noexcept {
            return x * 2.0f;
        };
        Kernel1D kernel = [&](BufferVar<float> in, BufferVar<float> out) noexcept {
            UInt idx = dispatch_id().x;
            Float val = in.read(idx);
            Float result = def(0.0f);
            $lambda({
                result = double_it(val);// call Callable from inside lambda
            })();
            out.write(idx, result);
        };
        auto shader = device.compile(kernel);
        stream << buf_in.copy_from(luisa::span{host_in})
               << shader(buf_in, buf_out).dispatch(N)
               << buf_out.copy_to(luisa::span{host_out})
               << synchronize();
        for (uint i = 0; i < N; i++) host_expected[i] = host_in[i] * 2.0f;
        check("test15_callable_inside_lambda", host_out, host_expected);
    }

    // ================================================================
    // Test 16: $lambda — captured Callable (Lambda wrapping a Lambda)
    //          The `sample_f = $lambda(...)` then `$lambda({ sample_f(...); })()`
    //          pattern from the path tracer.
    // ================================================================
    {
        Kernel1D kernel = [&](BufferVar<float> in, BufferVar<float> out) noexcept {
            UInt idx = dispatch_id().x;
            Float val = in.read(idx);
            Float result = def(0.0f);
            auto inner_fn = $lambda((Float x) {
                result = x * 3.0f;// captures result, takes x as param
            });
            $lambda({
                Float tmp = val + 1.0f;
                inner_fn(tmp);// invoke Lambda from inside another $lambda
            })();
            out.write(idx, result);// expected: (val + 1) * 3
        };
        auto shader = device.compile(kernel);
        stream << buf_in.copy_from(luisa::span{host_in})
               << shader(buf_in, buf_out).dispatch(N)
               << buf_out.copy_to(luisa::span{host_out})
               << synchronize();
        for (uint i = 0; i < N; i++) host_expected[i] = (host_in[i] + 1.0f) * 3.0f;
        check("test16_lambda_calling_lambda", host_out, host_expected);
    }

    // ================================================================
    // Test 17: $outline inside $lambda
    // ================================================================
    {
        Kernel1D kernel = [&](BufferVar<float> in, BufferVar<float> out) noexcept {
            UInt idx = dispatch_id().x;
            Float val = in.read(idx);
            $lambda({
                $outline {
                    val = val + 50.0f;
                };
            })();
            out.write(idx, val);
        };
        auto shader = device.compile(kernel);
        stream << buf_in.copy_from(luisa::span{host_in})
               << shader(buf_in, buf_out).dispatch(N)
               << buf_out.copy_to(luisa::span{host_out})
               << synchronize();
        for (uint i = 0; i < N; i++) host_expected[i] = host_in[i] + 50.0f;
        check("test17_outline_inside_lambda", host_out, host_expected);
    }

    // ================================================================
    // Test 18: Triple-nested $lambda with mutation propagation
    // ================================================================
    {
        Kernel1D kernel = [&](BufferVar<float> in, BufferVar<float> out) noexcept {
            UInt idx = dispatch_id().x;
            Float val = in.read(idx);
            Float result = def(0.0f);
            $lambda({
                result += 1.0f;
                $lambda({
                    result += 2.0f;
                    $lambda({
                        result += val;// 3 levels deep, captures from outermost scope
                    })();
                })();
            })();
            out.write(idx, result);// expected: 1 + 2 + val = val + 3
        };
        auto shader = device.compile(kernel);
        stream << buf_in.copy_from(luisa::span{host_in})
               << shader(buf_in, buf_out).dispatch(N)
               << buf_out.copy_to(luisa::span{host_out})
               << synchronize();
        for (uint i = 0; i < N; i++) host_expected[i] = host_in[i] + 3.0f;
        check("test18_triple_nested_lambda", host_out, host_expected);
    }

    // ================================================================
    // Test 19: $lambda with luisa::optional and nested read
    //          Full path tracer pattern:
    //          optional<Float> pp;
    //          $lambda({ pp.emplace(...);
    //              $lambda({ use *pp; })();
    //          })();
    // ================================================================
    {
        Kernel1D kernel = [&](BufferVar<float> in, BufferVar<float> out) noexcept {
            UInt idx = dispatch_id().x;
            Float val = in.read(idx);
            Float result = def(0.0f);
            luisa::optional<Float> pp;
            $lambda({
                pp.emplace(val * 2.0f);
                $lambda({
                    result = *pp + 1.0f;// read optional through 2 lambda levels
                })();
            })();
            out.write(idx, result);// expected: val * 2 + 1
        };
        auto shader = device.compile(kernel);
        stream << buf_in.copy_from(luisa::span{host_in})
               << shader(buf_in, buf_out).dispatch(N)
               << buf_out.copy_to(luisa::span{host_out})
               << synchronize();
        for (uint i = 0; i < N; i++) host_expected[i] = host_in[i] * 2.0f + 1.0f;
        check("test19_optional_nested_lambda", host_out, host_expected);
    }

    // ================================================================
    // Test 20: Full path tracer mini-pattern
    //          Combines: LCG state, optional, nested $lambda, accumulation
    // ================================================================
    {
        Buffer<uint> seed_buf = device.create_buffer<uint>(N);
        std::vector<uint> host_seeds(N);
        for (uint i = 0; i < N; i++) host_seeds[i] = i + 42u;

        Kernel1D kernel = [&](BufferVar<uint> seeds, BufferVar<float> out) noexcept {
            UInt idx = dispatch_id().x;
            UInt state = seeds.read(idx);

            Callable lcg = [&state]() noexcept {
                constexpr auto lcg_a = 1664525u;
                constexpr auto lcg_c = 1013904223u;
                state = state * lcg_a + lcg_c;
                return cast<float>(state & 0x00ffffffu) *
                       (1.0f / static_cast<float>(0x01000000u));
            };

            Float radiance = def(0.0f);
            Float beta = def(1.0f);

            // Simulate one path tracing bounce
            luisa::optional<Float> pp;
            luisa::optional<Float> albedo;

            $lambda({
                // "sample light" — consume 2 random numbers
                Float ux = lcg();
                Float uy = lcg();
                pp.emplace(ux);
                albedo.emplace(uy);

                // "shadow test" — nested lambda reads optional, accumulates
                $lambda({
                    Float contribution = *pp * *albedo;
                    radiance += beta * contribution;
                })();
            })();

            // "sample BSDF" — another lambda consuming RNG
            auto sample_bsdf = $lambda((Float u1, Float u2) {
                beta *= (u1 + u2) * 0.5f;// attenuate beta
            });

            $lambda({
                Float bx = lcg();
                Float by = lcg();
                sample_bsdf(bx, by);
            })();

            out.write(idx, radiance + beta);
            seeds.write(idx, state);
        };

        auto shader = device.compile(kernel);
        std::vector<float> pt_out(N);
        std::vector<uint> seed_out(N);
        stream << seed_buf.copy_from(luisa::span{host_seeds})
               << shader(seed_buf, buf_out).dispatch(N)
               << buf_out.copy_to(luisa::span{pt_out})
               << seed_buf.copy_to(luisa::span{seed_out})
               << synchronize();

        // Host-side reference computation
        std::vector<float> pt_expected(N);
        for (uint i = 0; i < N; i++) {
            uint s = host_seeds[i];
            float beta_h = 1.0f;
            float radiance_h = 0.0f;

            // "sample light"
            s = host_lcg_state_update(s);
            float ux = host_lcg_value(s);
            s = host_lcg_state_update(s);
            float uy = host_lcg_value(s);
            float pp_h = ux;
            float albedo_h = uy;

            // "shadow test"
            float contribution = pp_h * albedo_h;
            radiance_h += beta_h * contribution;

            // "sample BSDF"
            s = host_lcg_state_update(s);
            float bx = host_lcg_value(s);
            s = host_lcg_state_update(s);
            float by = host_lcg_value(s);
            beta_h *= (bx + by) * 0.5f;

            pt_expected[i] = radiance_h + beta_h;
        }
        check("test20_full_path_tracer_pattern", pt_out, pt_expected);

        // Verify state
        bool state_ok = true;
        for (uint i = 0; i < N; i++) {
            uint expected_state = host_seeds[i];
            for (int j = 0; j < 4; j++) expected_state = host_lcg_state_update(expected_state);
            if (seed_out[i] != expected_state) {
                LUISA_WARNING("test20_state: mismatch at [{}]: got {} expected {}", i, seed_out[i], expected_state);
                state_ok = false;
            }
        }
        expect(state_ok) << "test20_lcg_state_final";
    }

    LUISA_INFO("All nested callable tests completed.");
    return 0;
}

int main(int argc, char *argv[]) {
    auto dc = luisa::test::create_device_from_ut(argc, argv);
    if (!dc) {
        return 0;
    }
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));

    auto &device = dc->device;
    test_nested_callable(device);
}
