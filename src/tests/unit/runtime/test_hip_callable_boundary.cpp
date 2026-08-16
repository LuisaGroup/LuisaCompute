#include "ut/ut.hpp"

#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <utility>

#include <luisa/core/binary_io.h>
#include <luisa/core/logging.h>
#include <luisa/core/stl/vector.h>
#include <luisa/dsl/sugar.h>
#include <luisa/runtime/buffer.h>
#include <luisa/runtime/context.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>

using namespace boost::ut;
using namespace luisa;
using namespace luisa::compute;

namespace {

class CapturingBinaryIO final : public BinaryIO {

private:
    mutable size_t _last_shader_cache_size{};

public:
    void clear_shader_cache() const noexcept override {
        _last_shader_cache_size = 0u;
    }

    [[nodiscard]] luisa::unique_ptr<BinaryStream>
    read_shader_bytecode(luisa::string_view) const noexcept override {
        return nullptr;
    }

    [[nodiscard]] luisa::unique_ptr<BinaryStream>
    read_shader_cache(luisa::string_view) const noexcept override {
        return nullptr;
    }

    [[nodiscard]] luisa::unique_ptr<BinaryStream>
    read_internal_shader(luisa::string_view) const noexcept override {
        return nullptr;
    }

    luisa::filesystem::path write_shader_bytecode(
        luisa::string_view,
        luisa::span<const std::byte>) const noexcept override {
        return {};
    }

    luisa::filesystem::path write_shader_cache(
        luisa::string_view,
        luisa::span<const std::byte> data) const noexcept override {
        _last_shader_cache_size = data.size_bytes();
        return {};
    }

    luisa::filesystem::path write_internal_shader(
        luisa::string_view,
        luisa::span<const std::byte>) const noexcept override {
        return {};
    }

    void reset_capture() const noexcept {
        _last_shader_cache_size = 0u;
    }

    [[nodiscard]] size_t last_shader_cache_size() const noexcept {
        return _last_shader_cache_size;
    }
};

static constexpr auto callable_round_count = 192u;

[[nodiscard]] uint32_t scramble_reference(uint32_t value) noexcept {
    for (auto round = 0u; round < callable_round_count; round++) {
        auto right_shift = round % 15u + 1u;
        auto left_shift = round % 13u + 1u;
        auto multiplier = 0x9e3779b1u + round * 0x85ebca6bu;
        auto additive = 0xc2b2ae35u + round * 0x27d4eb2du;
        value = (value ^ (value >> right_shift)) * multiplier;
        value ^= value << left_shift;
        value += additive;
    }
    return value;
}

struct CompileResult {
    size_t artifact_size{};
    luisa::vector<uint32_t> values;
};

[[nodiscard]] luisa::vector<float>
evaluate_normalized_vectors_through_callable(
    Device &device,
    luisa::span<const float3> inputs,
    uint32_t seed,
    bool enable_fast_math,
    uint64_t *structure_hash = nullptr) noexcept {
    Callable project = [](Float3 direction, UInt value) noexcept {
        for (auto round = 0u;
             round < callable_round_count; round++) {
            auto right_shift = round % 15u + 1u;
            auto left_shift = round % 13u + 1u;
            auto multiplier =
                0x9e3779b1u + round * 0x85ebca6bu;
            auto additive =
                0xc2b2ae35u + round * 0x27d4eb2du;
            value = (value ^ (value >> right_shift)) * multiplier;
            value ^= value << left_shift;
            value += additive;
        }
        return direction.x + 2.0f * direction.y +
               4.0f * direction.z +
               cast<float>(value & 255u);
    };
    Kernel1D kernel = [&project](BufferFloat output,
                                 BufferFloat3 input,
                                 UInt initial_value) noexcept {
        const auto index = dispatch_x();
        const auto direction = normalize(input.read(index));
        output.write(
            index,
            project(direction, initial_value + index));
    };
    if (structure_hash != nullptr) {
        *structure_hash = kernel.function()->function().hash();
    }

    auto shader = device.compile(
        kernel,
        ShaderOption{
            .enable_cache = false,
            .enable_fast_math = enable_fast_math});
    auto input = device.create_buffer<float3>(inputs.size());
    auto output = device.create_buffer<float>(inputs.size());
    luisa::vector<float> values(inputs.size());
    auto stream = device.create_stream();
    stream << input.copy_from(inputs)
           << shader(output, input, seed).dispatch(inputs.size())
           << output.copy_to(values.data())
           << synchronize();
    return values;
}

[[nodiscard]] std::string read_text_file(
    const std::filesystem::path &path) {
    std::ifstream stream{path};
    return {
        std::istreambuf_iterator<char>{stream},
        std::istreambuf_iterator<char>{}};
}

[[nodiscard]] std::string_view amdgpu_kernel_body(
    const std::string &module) noexcept {
    constexpr auto kernel_prefix =
        std::string_view{"define amdgpu_kernel "};
    auto begin = module.find(kernel_prefix);
    if (begin == std::string::npos) { return {}; }
    auto end = module.find("\n}", begin);
    if (end == std::string::npos) { return {}; }
    return std::string_view{module}.substr(
        begin, end + 2u - begin);
}

[[nodiscard]] std::string_view llvm_function_body(
    const std::string &module,
    std::string_view function_name) noexcept {
    for (auto name = module.find(function_name);
         name != std::string::npos;
         name = module.find(function_name, name + 1u)) {
        auto begin = module.rfind('\n', name);
        begin = begin == std::string::npos ? 0u : begin + 1u;
        if (!std::string_view{module}.substr(begin).starts_with(
                "define ")) {
            continue;
        }
        auto end = module.find("\n}", name);
        if (end == std::string::npos) { return {}; }
        return std::string_view{module}.substr(
            begin, end + 2u - begin);
    }
    return {};
}

[[nodiscard]] bool contains_dynamic_fp_operation(
    std::string_view body) noexcept {
    constexpr std::array operations{
        " fadd ", " fsub ", " fmul ", " fdiv ",
        " frem ", " fneg ", " fcmp "};
    for (auto operation : operations) {
        if (body.find(operation) != std::string_view::npos) {
            return true;
        }
    }
    return false;
}

[[nodiscard]] size_t count_occurrences(
    std::string_view text, std::string_view pattern) noexcept {
    auto count = size_t{0u};
    for (auto position = text.find(pattern);
         position != std::string_view::npos;
         position = text.find(pattern, position + pattern.size())) {
        ++count;
    }
    return count;
}

[[nodiscard]] CompileResult compile_reused_callable(
    Device &device, CapturingBinaryIO &binary_io,
    uint32_t reuse_count, uint32_t seed) noexcept {
    Callable scramble = [](UInt value) noexcept {
        for (auto round = 0u;
             round < callable_round_count; round++) {
            auto right_shift = round % 15u + 1u;
            auto left_shift = round % 13u + 1u;
            auto multiplier =
                0x9e3779b1u + round * 0x85ebca6bu;
            auto additive =
                0xc2b2ae35u + round * 0x27d4eb2du;
            value = (value ^ (value >> right_shift)) * multiplier;
            value ^= value << left_shift;
            value += additive;
        }
        return value;
    };
    Kernel1D kernel = [&scramble, reuse_count](
                          BufferUInt output,
                          UInt initial_value) noexcept {
        for (auto use = 0u; use < reuse_count; use++) {
            output.write(
                use,
                scramble(initial_value +
                         use * 0x165667b1u));
        }
    };

    binary_io.reset_capture();
    auto shader = device.compile(
        kernel, ShaderOption{.enable_cache = true});
    auto artifact_size = binary_io.last_shader_cache_size();
    auto output = device.create_buffer<uint32_t>(reuse_count);
    luisa::vector<uint32_t> values(reuse_count);
    auto stream = device.create_stream();
    stream << shader(output, seed).dispatch(1u)
           << output.copy_to(values.data())
           << synchronize();
    return CompileResult{
        .artifact_size = artifact_size,
        .values = std::move(values)};
}

void compile_minimal_ray_query(Device &device) noexcept {
    Kernel1D kernel = [](BufferUInt output, AccelVar accel) noexcept {
        auto ray = make_ray(
            make_float3(0.0f, 0.0f, -1.0f),
            make_float3(0.0f, 0.0f, 1.0f));
        auto all_hit = accel.traverse(ray, {})
                           .on_surface_candidate(
                               [](auto &candidate) noexcept {
                                   candidate.commit();
                               })
                           .trace();
        auto any_hit = accel.traverse_any(ray, {})
                           .on_surface_candidate(
                               [](auto &candidate) noexcept {
                                   candidate.commit();
                               })
                           .trace();
        output.write(dispatch_x(), all_hit->prim ^ any_hit->prim);
    };
    static_cast<void>(device.compile(
        kernel, ShaderOption{.enable_cache = false}));
}

void compile_observing_ray_query(Device &device) noexcept {
    Kernel1D kernel = [](BufferUInt output, AccelVar accel) noexcept {
        auto ray = make_ray(
            make_float3(0.0f, 0.0f, -1.0f),
            make_float3(0.0f, 0.0f, 1.0f));
        Float observed_t_max = -1.0f;
        auto hit = accel.traverse(ray, {})
                       .on_surface_candidate(
                           [&](SurfaceCandidate &candidate) noexcept {
                               // Put the world-ray observation in a nested
                               // generated Callable. The compact-transaction
                               // eligibility proof must follow this edge, not
                               // merely scan the immediate handler body.
                               Callable observe = [&candidate]() noexcept {
                                   return candidate.ray()->t_max();
                               };
                               observed_t_max = observe();
                               candidate.commit();
                           })
                       .trace();
        output.write(
            dispatch_x(),
            hit->prim ^ cast<uint>(observed_t_max > 0.0f));
    };
    static_cast<void>(device.compile(
        kernel, ShaderOption{.enable_cache = false}));
}

void compile_large_ray_query_environment(
    Device &device, bool observe_post_state,
    bool observe_world_ray = false) noexcept {
    Kernel1D kernel = [observe_post_state, observe_world_ray](
                          BufferUInt output,
                          BufferUInt input_0,
                          BufferUInt input_1,
                          BufferUInt input_2,
                          BufferUInt input_3,
                          BufferUInt input_4,
                          BufferUInt input_5,
                          BufferUInt input_6,
                          BufferUInt input_7,
                          BufferUInt input_8,
                          AccelVar accel) noexcept {
        auto ray = make_ray(
            make_float3(0.0f, 0.0f, -1.0f),
            make_float3(0.0f, 0.0f, 1.0f));
        auto hit = accel.traverse(ray, {})
                       .on_surface_candidate(
                           [&](SurfaceCandidate &candidate) noexcept {
                               // Every buffer is consumed by the handler, so
                               // argument-demand projection cannot shrink this
                               // environment below the ordinary 64-byte native
                               // budget even after each buffer descriptor is
                               // projected to its demanded device-pointer leaf.
                               // The output write is the semantic result when
                               // the query post-state is discarded.
                               auto checksum =
                                   input_0.read(0u) ^ input_1.read(0u) ^
                                   input_2.read(0u) ^ input_3.read(0u) ^
                                   input_4.read(0u) ^ input_5.read(0u) ^
                                   input_6.read(0u) ^ input_7.read(0u) ^
                                   input_8.read(0u);
                               if (observe_world_ray) {
                                   checksum ^= cast<uint>(
                                       candidate.ray()->t_max() > 0.0f);
                               }
                               output.write(dispatch_x(), checksum);
                               candidate.commit();
                           })
                       .trace();
        if (observe_post_state) {
            output.write(dispatch_x(), hit->prim);
        }
    };
    static_cast<void>(device.compile(
        kernel, ShaderOption{.enable_cache = false}));
}

void compile_large_pure_closest_reduction(
    Device &device, bool explicitly_terminate) noexcept {
    Kernel1D kernel = [explicitly_terminate](
                          BufferUInt output,
                          BufferUInt input_0,
                          BufferUInt input_1,
                          BufferUInt input_2,
                          BufferUInt input_3,
                          BufferUInt input_4,
                          BufferUInt input_5,
                          BufferUInt input_6,
                          BufferUInt input_7,
                          BufferUInt input_8,
                          AccelVar accel) noexcept {
        auto ray = make_ray(
            make_float3(0.0f, 0.0f, -1.0f),
            make_float3(0.0f, 0.0f, 1.0f));
        auto hit = accel.traverse(ray, {})
                       .on_surface_candidate(
                           [&](SurfaceCandidate &candidate) noexcept {
                               // These nine live resource handles keep the
                               // projected callback environment above 64
                               // bytes even when unused descriptor-size leaves
                               // are removed. Resource reads are pure and the only
                               // state transition is a conditional closest-hit
                               // commit, so enumeration order is unobservable
                               // unless terminate() is present.
                               auto checksum =
                                   input_0.read(0u) ^ input_1.read(0u) ^
                                   input_2.read(0u) ^ input_3.read(0u) ^
                                   input_4.read(0u) ^ input_5.read(0u) ^
                                   input_6.read(0u) ^ input_7.read(0u) ^
                                   input_8.read(0u);
                               Callable commit_if_even =
                                   [&candidate](UInt value) noexcept {
                                       $if ((value & 1u) == 0u) {
                                           candidate.commit();
                                       };
                                   };
                               commit_if_even(checksum);
                               if (explicitly_terminate) {
                                   candidate.terminate();
                               }
                           })
                       .on_procedural_candidate(
                           [](ProceduralCandidate &candidate) noexcept {
                               // A procedural leaf has one candidate, so the
                               // native closest and resumable HIPRT state
                               // machines expose the same active max here.
                               const auto distance =
                                   candidate.ray()->t_max() * 0.5f;
                               $if (distance > 0.0f) {
                                   candidate.commit(distance);
                               };
                           })
                       .trace();
        output.write(dispatch_x(), hit->prim);
    };
    static_cast<void>(device.compile(
        kernel, ShaderOption{.enable_cache = false}));
}

void compile_mixed_ray_query_pipelines(Device &device) noexcept {
    Callable compact_trace = [](
                                 AccelVar accel,
                                 Var<Ray> ray) noexcept {
        auto compact_hit =
            accel.traverse_any(ray, {})
                .on_surface_candidate(
                    [](SurfaceCandidate &candidate) noexcept {
                        candidate.commit();
                    })
                .trace();
        return compact_hit->prim;
    };
    compact_trace.function_builder()->set_name(
        "hip_mixed_exact_query_state_domain");

    Callable observed_trace = [](
                                  AccelVar accel,
                                  Var<Ray> ray,
                                  BufferUInt input_0,
                                  BufferUInt input_1,
                                  BufferUInt input_2,
                                  BufferUInt input_3,
                                  BufferUInt input_4,
                                  BufferUInt input_5,
                                  BufferUInt input_6,
                                  BufferUInt input_7,
                                  BufferUInt input_8) noexcept {
        auto observed_hit =
            accel.traverse(ray, {})
                .on_surface_candidate(
                    [&](SurfaceCandidate &candidate) noexcept {
                        auto checksum =
                            input_0.read(0u) ^ input_1.read(0u) ^
                            input_2.read(0u) ^ input_3.read(0u) ^
                            input_4.read(0u) ^ input_5.read(0u) ^
                            input_6.read(0u) ^ input_7.read(0u) ^
                            input_8.read(0u);
                        checksum ^= cast<uint>(
                            candidate.ray()->t_max() > 0.0f);
                        $if ((checksum & 1u) == 0u) {
                            candidate.commit();
                        };
                    })
                .trace();
        return observed_hit->prim;
    };
    observed_trace.function_builder()->set_name(
        "hip_mixed_resumable_query_state_domain");

    Kernel1D kernel = [&compact_trace, &observed_trace](
                          BufferUInt output,
                          BufferUInt input_0,
                          BufferUInt input_1,
                          BufferUInt input_2,
                          BufferUInt input_3,
                          BufferUInt input_4,
                          BufferUInt input_5,
                          BufferUInt input_6,
                          BufferUInt input_7,
                          BufferUInt input_8,
                          AccelVar accel) noexcept {
        auto ray = make_ray(
            make_float3(0.0f, 0.0f, -1.0f),
            make_float3(0.0f, 0.0f, 1.0f));
        output.write(0u, compact_trace(accel, ray));
        // This post-state observation and the nine independently demanded
        // pointer leaves make only this Callable's function-owned state domain
        // budget-constrained. The compact Callable owns a distinct state and
        // must remain synchronous in the same generated module.
        output.write(
            1u,
            observed_trace(
                accel, ray, input_0, input_1, input_2, input_3,
                input_4, input_5, input_6, input_7, input_8));
    };
    static_cast<void>(device.compile(
        kernel, ShaderOption{.enable_cache = false}));
}

void compile_object_ray_reduction(
    Device &device, bool observe_world_ray) noexcept {
    Kernel1D kernel = [observe_world_ray](
                          BufferUInt output,
                          AccelVar accel) noexcept {
        auto ray = make_ray(
            make_float3(0.0f, 0.0f, -1.0f),
            make_float3(0.0f, 0.0f, 1.0f));
        auto hit = accel.traverse(ray, {})
                       .on_surface_candidate(
                           [](SurfaceCandidate &candidate) noexcept {
                               candidate.commit();
                           })
                       .on_procedural_candidate(
                           [&](ProceduralCandidate &candidate) noexcept {
                               const auto object_ray = candidate.object_ray();
                               auto distance = object_ray->t_max() * 0.5f;
                               if (observe_world_ray) {
                                   // Both values are demanded by this one
                                   // handler invocation. They cannot alias one
                                   // compact ray field even though surface and
                                   // procedural handler domains may otherwise
                                   // use distinct representations independently.
                                   const auto world_ray = candidate.ray();
                                   distance = min(
                                       distance,
                                       world_ray->t_max() * 0.5f);
                               }
                               $if (distance > 0.0f) {
                                   candidate.commit(distance);
                               };
                           })
                       .trace();
        output.write(dispatch_x(), hit->prim);
    };
    static_cast<void>(device.compile(
        kernel, ShaderOption{.enable_cache = false}));
}

enum class EffectOnlyRayQueryCase {
    proven,
    nested_commit,
    opacity_write,
};

void compile_effect_only_ray_query(
    Device &device, EffectOnlyRayQueryCase test_case) noexcept {
    Kernel1D kernel = [test_case](
                          BufferUInt output,
                          AccelVar accel) noexcept {
        auto ray = make_ray(
            make_float3(0.0f, 0.0f, -1.0f),
            make_float3(0.0f, 0.0f, 1.0f));
        UInt callback_count = 0u;
        const auto ignored =
            accel.traverse(ray, {})
                .on_surface_candidate(
                    [&](SurfaceCandidate &candidate) noexcept {
                        const auto hit = candidate.hit();
                        output.write(
                            dispatch_x(),
                            hit->prim ^ callback_count);
                        if (test_case ==
                            EffectOnlyRayQueryCase::proven) {
                            // Active-query provenance must cross this direct
                            // Callable edge. TERMINATE preserves the effect-
                            // only quotient and arbitrary buffer writes remain
                            // observable in their original candidate order.
                            Callable terminate_after_two =
                                [&candidate](UInt count) noexcept {
                                    $if (count == 2u) {
                                        candidate.terminate();
                                    };
                                };
                            callback_count += 1u;
                            terminate_after_two(callback_count);
                        } else if (
                            test_case ==
                            EffectOnlyRayQueryCase::nested_commit) {
                            // A commit hidden behind the same call boundary
                            // must invalidate the proof.
                            Callable commit =
                                [&candidate]() noexcept {
                                    candidate.commit();
                                };
                            commit();
                        } else {
                            // Even without a query commit, mutation of the
                            // accel opacity domain can invalidate the monotone
                            // non-opaque certificate during traversal.
                            accel.set_instance_opaque(hit->inst, false);
                        }
                    })
                .trace();
        static_cast<void>(ignored);
    };
    static_cast<void>(device.compile(
        kernel, ShaderOption{.enable_cache = false}));
}

void compile_terminal_predicate_ray_query(
    Device &device, bool observe_full_hit,
    bool write_opacity = false) noexcept {
    Kernel1D kernel = [observe_full_hit, write_opacity](
                          BufferUInt output,
                          AccelVar accel) noexcept {
        auto ray = make_ray(
            make_float3(0.0f, 0.0f, -1.0f),
            make_float3(0.0f, 0.0f, 1.0f));
        auto hit =
            accel.traverse_any(ray, {})
                .on_surface_candidate(
                    [&](SurfaceCandidate &candidate) noexcept {
                        const auto candidate_hit = candidate.hit();
                        if (write_opacity) {
                            // Opacity is part of the traversal state, not the
                            // parent-visible hit-kind quotient. A write from
                            // this callback must therefore keep the native
                            // terminal route while forcing live opacity reads
                            // for all later candidates.
                            accel.set_instance_opaque(
                                candidate_hit->inst, true);
                        }
                        $if ((candidate_hit->prim & 1u) == 0u) {
                            candidate.commit();
                        };
                    })
                .trace();
        if (observe_full_hit) {
            // One identity field invalidates the terminal predicate quotient.
            output.write(dispatch_x(), hit->prim);
        } else {
            output.write(dispatch_x(), cast<uint>(hit->miss()));
        }
    };
    static_cast<void>(device.compile(
        kernel, ShaderOption{.enable_cache = false}));
}

}// namespace

int main(int argc, char *argv[]) {
    auto program_path =
        argc > 0 && argv != nullptr ? argv[0] : "";
    CapturingBinaryIO binary_io;
    Context context{program_path};
    DeviceConfig config{.binary_io = &binary_io};
    auto device = context.create_device("hip", &config);

    std::error_code filesystem_error;
    const auto original_directory =
        std::filesystem::current_path(filesystem_error);
    const auto dump_directory =
        std::filesystem::temp_directory_path(filesystem_error) /
        ("luisa_hip_callable_boundary_" +
         std::to_string(
             std::chrono::steady_clock::now()
                 .time_since_epoch()
                 .count()));
    std::filesystem::create_directories(
        dump_directory, filesystem_error);
    std::filesystem::current_path(
        dump_directory, filesystem_error);
    expect(!filesystem_error)
        << "failed to prepare isolated HIP LLVM dump directory";
#if defined(_WIN32)
    _putenv_s("LUISA_DUMP_LLVM_IR", "1");
#else
    setenv("LUISA_DUMP_LLVM_IR", "1", 1);
#endif

    "HIP preserves shared DSL callable boundaries after optimization"_test =
        [&] {
            constexpr auto seed = 0x12345678u;
            constexpr auto shared_use_count = 16u;
            auto single = compile_reused_callable(
                device, binary_io, 1u, seed);
            auto shared = compile_reused_callable(
                device, binary_io, shared_use_count, seed);

            expect(single.artifact_size != 0u);
            expect(shared.artifact_size != 0u);
            for (auto use = 0u; use < shared_use_count; use++) {
                auto input = seed + use * 0x165667b1u;
                expect(shared.values[use] ==
                       scramble_reference(input));
            }
            expect(single.values.front() == shared.values.front());

            LUISA_INFO(
                "HIP callable-boundary regression: one use = {} bytes, "
                "{} uses = {} bytes.",
                single.artifact_size, shared_use_count,
                shared.artifact_size);

            // The callable body is intentionally much larger than its call
            // sites. Reusing it must grow the code object by calls and stores,
            // not by sixteen copies of the body. The fixed allowance covers
            // target-dependent code-object metadata without hiding renewed
            // megakernel expansion.
            constexpr auto maximum_reuse_overhead = 96u * 1024u;
            expect(shared.artifact_size <=
                   single.artifact_size + maximum_reuse_overhead)
                << "shared Luisa Callable was repeatedly expanded into the "
                   "HIP kernel";
        };

    "HIP preserves fixed-vector reductions before callable boundaries"_test =
        [&] {
            constexpr auto seed = 0x89abcdefu;
            constexpr std::array inputs{
                float3{0.0f, 0.0f, 0.75f},
                float3{0.25f, -0.5f, 2.0f},
                float3{-1.0f, 3.0f, 0.125f},
                float3{4.0f, -2.0f, 1.0f},
                float3{0.5f, 0.25f, -0.75f},
                float3{-2.0f, -1.0f, 3.0f},
                float3{7.0f, 0.125f, -0.25f},
                float3{0.0f, 0.0f, 1.5f},
                float3{1.0f, 1.0f, 1.0f},
                float3{-1.0f, 1.0f, -1.0f},
                float3{0.125f, 8.0f, 0.5f},
                float3{3.0f, 2.0f, -4.0f},
                float3{-0.5f, 0.75f, 0.25f},
                float3{9.0f, -3.0f, 2.0f},
                float3{0.0625f, 0.125f, 0.25f},
                float3{-6.0f, 5.0f, 4.0f}};
            uint64_t fast_structure_hash{};
            uint64_t strict_structure_hash{};
            auto fast = evaluate_normalized_vectors_through_callable(
                device, inputs, seed, true, &fast_structure_hash);
            auto strict = evaluate_normalized_vectors_through_callable(
                device, inputs, seed, false, &strict_structure_hash);
            expect(fast_structure_hash == strict_structure_hash)
                << "fast-math policy unexpectedly changed the DSL AST";
            for (auto i = 0u; i < inputs.size(); i++) {
                const auto &v = inputs[i];
                const auto squared_length =
                    v.x * v.x + v.y * v.y + v.z * v.z;
                const auto inverse_length =
                    1.0f / std::sqrt(squared_length);
                const auto direction = v * inverse_length;
                const auto expected =
                    direction.x + 2.0f * direction.y +
                    4.0f * direction.z +
                    static_cast<float>(
                        scramble_reference(seed + i) & 255u);
                expect(std::abs(fast[i] - expected) <= 2.0e-5f)
                    << "fixed-vector reduction changed across an "
                       "out-of-line HIP Callable in fast mode";
                expect(std::abs(strict[i] - expected) <= 2.0e-5f)
                    << "fixed-vector reduction changed across an "
                       "out-of-line HIP Callable in strict mode";
            }
        };

    "HIP RayQuery projects candidate-only callback transactions"_test =
        [&] {
            compile_minimal_ray_query(device);
            compile_observing_ray_query(device);
            compile_large_ray_query_environment(device, false);
            compile_large_ray_query_environment(device, true);
            compile_large_ray_query_environment(device, false, true);
            compile_large_pure_closest_reduction(device, false);
            compile_large_pure_closest_reduction(device, true);
            compile_mixed_ray_query_pipelines(device);
            compile_object_ray_reduction(device, false);
            compile_object_ray_reduction(device, true);
            compile_effect_only_ray_query(
                device, EffectOnlyRayQueryCase::proven);
            compile_effect_only_ray_query(
                device, EffectOnlyRayQueryCase::nested_commit);
            compile_effect_only_ray_query(
                device, EffectOnlyRayQueryCase::opacity_write);
            compile_terminal_predicate_ray_query(device, false);
            compile_terminal_predicate_ray_query(device, true);
            compile_terminal_predicate_ray_query(
                device, false, true);
            const auto before_module = read_text_file(
                dump_directory / "hip_kernel_before_opt_4.ll");
            const auto observing_before_module = read_text_file(
                dump_directory / "hip_kernel_before_opt_5.ll");
            const auto module = read_text_file(
                dump_directory / "hip_kernel_final_4.ll");
            const auto observing_module = read_text_file(
                dump_directory / "hip_kernel_final_5.ll");
            const auto handler_only_before_module = read_text_file(
                dump_directory / "hip_kernel_before_opt_6.ll");
            const auto observed_large_before_module = read_text_file(
                dump_directory / "hip_kernel_before_opt_7.ll");
            const auto before_root =
                amdgpu_kernel_body(before_module);
            const auto observing_before_root =
                amdgpu_kernel_body(observing_before_module);
            const auto root = amdgpu_kernel_body(module);
            const auto observing_dispatcher = llvm_function_body(
                observing_module,
                "@luisa_ray_query_pipeline_dispatch");
            const auto handler_only_before_root =
                amdgpu_kernel_body(handler_only_before_module);
            const auto observed_large_before_root =
                amdgpu_kernel_body(observed_large_before_module);
            const auto full_candidate_large_before_module = read_text_file(
                dump_directory / "hip_kernel_before_opt_8.ll");
            const auto full_candidate_large_before_root =
                amdgpu_kernel_body(full_candidate_large_before_module);
            const auto pure_reduction_before_module = read_text_file(
                dump_directory / "hip_kernel_before_opt_9.ll");
            const auto pure_reduction_before_root =
                amdgpu_kernel_body(pure_reduction_before_module);
            const auto terminating_reduction_before_module = read_text_file(
                dump_directory / "hip_kernel_before_opt_10.ll");
            const auto terminating_reduction_before_root =
                amdgpu_kernel_body(terminating_reduction_before_module);
            const auto mixed_before_module = read_text_file(
                dump_directory / "hip_kernel_before_opt_11.ll");
            const auto object_ray_before_module = read_text_file(
                dump_directory / "hip_kernel_before_opt_12.ll");
            const auto joint_rays_before_module = read_text_file(
                dump_directory / "hip_kernel_before_opt_13.ll");
            const auto effect_only_before_module = read_text_file(
                dump_directory / "hip_kernel_before_opt_14.ll");
            const auto effect_commit_before_module = read_text_file(
                dump_directory / "hip_kernel_before_opt_15.ll");
            const auto effect_opacity_before_module = read_text_file(
                dump_directory / "hip_kernel_before_opt_16.ll");
            const auto terminal_predicate_before_module = read_text_file(
                dump_directory / "hip_kernel_before_opt_17.ll");
            const auto terminal_full_hit_before_module = read_text_file(
                dump_directory / "hip_kernel_before_opt_18.ll");
            const auto terminal_mutable_opacity_before_module = read_text_file(
                dump_directory / "hip_kernel_before_opt_19.ll");
            const auto mixed_before_root =
                amdgpu_kernel_body(mixed_before_module);
            const auto object_ray_before_root =
                amdgpu_kernel_body(object_ray_before_module);
            const auto joint_rays_before_root =
                amdgpu_kernel_body(joint_rays_before_module);
            const auto effect_only_before_root =
                amdgpu_kernel_body(effect_only_before_module);
            const auto effect_commit_before_root =
                amdgpu_kernel_body(effect_commit_before_module);
            const auto effect_opacity_before_root =
                amdgpu_kernel_body(effect_opacity_before_module);
            const auto terminal_predicate_before_root =
                amdgpu_kernel_body(terminal_predicate_before_module);
            const auto terminal_full_hit_before_root =
                amdgpu_kernel_body(terminal_full_hit_before_module);
            const auto terminal_mutable_opacity_before_root =
                amdgpu_kernel_body(
                    terminal_mutable_opacity_before_module);
            const auto mixed_exact_state_domain =
                llvm_function_body(
                    mixed_before_module,
                    "hip_mixed_exact_query_state_domain");
            const auto mixed_resumable_state_domain =
                llvm_function_body(
                    mixed_before_module,
                    "hip_mixed_resumable_query_state_domain");
            expect(!before_root.empty() &&
                   !observing_before_root.empty() && !root.empty() &&
                   !observing_dispatcher.empty() &&
                   !handler_only_before_root.empty() &&
                   !observed_large_before_root.empty() &&
                   !full_candidate_large_before_root.empty() &&
                   !pure_reduction_before_root.empty() &&
                   !terminating_reduction_before_root.empty() &&
                   !mixed_before_root.empty() &&
                   !object_ray_before_root.empty() &&
                   !joint_rays_before_root.empty() &&
                   !effect_only_before_root.empty() &&
                   !effect_commit_before_root.empty() &&
                   !effect_opacity_before_root.empty() &&
                   !terminal_predicate_before_root.empty() &&
                   !terminal_full_hit_before_root.empty() &&
                   !terminal_mutable_opacity_before_root.empty() &&
                   !mixed_exact_state_domain.empty() &&
                   !mixed_resumable_state_domain.empty())
                << "failed to locate the generated RayQuery functions";
            const auto uses_gfx12_hardware_stack =
                module.find(
                    "@llvm.amdgcn.ds.bvh.stack.push8.pop1.rtn") !=
                std::string::npos;
            const auto uses_static_native_closest =
                before_root.find(
                    "@luisa_pipeline_ray_query_trace_all_native_closest_"
                    "global_stack_stable_opacity(") !=
                std::string_view::npos;
            // Selection is a codegen property, while outlining is an LLVM
            // profitability decision. Inspect the generated root before the
            // ordinary inliner instead of requiring trace wrappers to survive
            // in final IR.
            if (uses_gfx12_hardware_stack) {
                expect(uses_static_native_closest &&
                       before_root.find(
                           "@luisa_pipeline_ray_query_trace_all_stable_opacity(") ==
                           std::string_view::npos &&
                       before_root.find(
                           "@luisa_pipeline_ray_query_trace_all_native_closest_stable_opacity(") ==
                           std::string_view::npos)
                    << "gfx12 closest reduction did not select one static-"
                       "global HIPRT closest transaction";
            } else {
                expect(before_root.find(
                           "@luisa_pipeline_ray_query_trace_all_native_closest_stable_opacity(") !=
                       std::string_view::npos)
                    << "software closest reduction did not select one native "
                       "HIPRT closest traversal";
            }
            expect(before_root.find(
                       "@luisa_pipeline_ray_query_trace_any_stable_opacity(") !=
                   std::string_view::npos)
                << "RayQueryAny without device opacity writes did not select "
                   "its stable-opacity native traversal";
            expect(before_root.find(
                       "@luisa_pipeline_ray_query_trace_all(") ==
                       std::string_view::npos &&
                   before_root.find(
                       "@luisa_pipeline_ray_query_trace_any(") ==
                       std::string_view::npos)
                << "stable-opacity RayQuery retained a mutable-opacity "
                   "traversal entry point";
            expect(before_root.find(
                       "@luisa_pipeline_ray_query_trace(") ==
                   std::string_view::npos)
                << "synchronous RayQuery regressed to a runtime-kind "
                   "traversal entry point";
            expect(root.find(
                       "alloca { i64, i64, i64, i64, i64 }") ==
                   std::string_view::npos)
                << "HIP RayQuery regressed to the 40-byte source-layout "
                   "surrogate";
            expect(root.find("alloca i32") == std::string_view::npos)
                << "HIP RayQuery identity escaped the traversal state into "
                   "a separate private token allocation";
            expect(module.find("ray.query.context.projected") ==
                   std::string::npos)
                << "empty RayQuery callback environment was "
                   "materialized in private memory";
            if (uses_gfx12_hardware_stack) {
                // Neither compact RayQueryAll nor RayQueryAny observes
                // query-object identity. The hardware traversal still requires
                // an addressable 112-byte transaction state, but no extra
                // pointer-to-integer identity may escape it.
                expect(root.find("ray.query.state.address") ==
                           std::string_view::npos &&
                       root.find("ray.query.identity.address") ==
                           std::string_view::npos)
                    << "candidate-only RayQuery materialized an unobservable "
                       "private-state identity";
            }

            // The second kernel hides a world-ray observation in a nested
            // Callable. The fixed-point proof must follow that call edge and
            // retain the full query transaction, while constant specialization
            // still removes pipeline identity from its three-argument
            // {query, context, candidate-kind} ABI.
            expect(observing_dispatcher.find("load i32, ptr %0") !=
                       std::string_view::npos &&
                   observing_dispatcher.find("i32 44") !=
                       std::string_view::npos)
                << "nested world-ray observation was not decoded from the "
                   "dedicated query identity";
            expect(observing_before_root.find(
                       "ray.query.identity.address") !=
                       std::string_view::npos &&
                   observing_before_root.find(
                       "ray.query.identity.field") !=
                       std::string_view::npos)
                << "full-state RayQuery failed to materialize identity at "
                   "its observable callback boundary";
            expect(observing_module.find(
                       "@luisa_pipeline_ray_query_dispatch_compact(") ==
                   std::string_view::npos)
                << "nested world-ray observation unsafely selected the "
                   "candidate-only transaction";
            expect(observing_before_root.find(
                       "@luisa_pipeline_ray_query_trace_all_native_closest") ==
                   std::string_view::npos)
                << "mutable world-ray t_max observation unsafely selected "
                   "the order-independent closest reduction";
            expect(module.find("luisa-specialize-constant-argument") ==
                       std::string_view::npos &&
                   observing_module.find(
                       "luisa-specialize-constant-argument") ==
                       std::string_view::npos)
                << "internal constant-specialization marker escaped HIP "
                   "codegen";
            if (uses_gfx12_hardware_stack) {
                // RayQueryAny retains the exact single-region hardware
                // frontier, while the proven closest reduction uses HIPRT's
                // 24-entry-per-lane static global spill stack. The two routes
                // are non-reentrant and overlay one arena; for a 256-thread
                // block the latter dominates at 24 * 256 dwords.
                expect(module.find(
                           "@luisa_hiprt_shared_stack_cache = internal "
                           "addrspace(3) global [6144 x i32]") !=
                       std::string::npos)
                    << "gfx12 mixed native/hardware query did not reserve "
                       "the maximum route-local LDS footprint";
                expect(module.find(", i32 16)") !=
                       std::string::npos)
                    << "gfx12 BVH-stack intrinsic did not receive the "
                       "native 16-entry frontier capacity";
            }

            // The two large-environment kernels differ only in whether the
            // parent observes the committed-hit post-state. A local query
            // whose only result is handler side effects may retain the native
            // synchronous traversal even when its projected capture product
            // exceeds the ordinary budget. Adding one post-state read must
            // fail the closed-use proof and select the resumable ABI.
            expect(handler_only_before_root.find(
                       "@luisa_pipeline_ray_query_trace_all_stable_opacity(") !=
                   std::string_view::npos)
                << "large handler-only RayQuery did not retain native "
                   "synchronous traversal";
            expect(count_occurrences(
                       handler_only_before_root,
                       "ray.query.context.projected.field") >= 9u)
                << "large handler-only RayQuery regression did not exercise "
                   "an environment above the ordinary native budget";
            expect(observed_large_before_module.find(
                       "@luisa_ray_query_proceed(") !=
                       std::string_view::npos &&
                   observed_large_before_root.find(
                       "@luisa_pipeline_ray_query_trace_all_stable_opacity(") ==
                       std::string_view::npos)
                << "observable large RayQuery post-state did not fail closed "
                   "to the resumable traversal ABI";
            expect(full_candidate_large_before_module.find(
                       "@luisa_ray_query_proceed(") !=
                       std::string_view::npos &&
                   full_candidate_large_before_root.find(
                       "@luisa_pipeline_ray_query_trace_all_stable_opacity(") ==
                       std::string_view::npos)
                << "large full-candidate RayQuery did not fail closed to the "
                   "resumable traversal ABI";

            // A large environment is not itself a semantic reason to expose
            // HIPRT's continuation frontier. Prove the candidate callbacks
            // are a pure closest-hit reduction and lower the complete query
            // to one target-native closest transaction. The surface commit is
            // intentionally in a nested Callable to exercise active-query
            // provenance across a call edge; the procedural handler exercises
            // the formally equivalent active-ray frontier used by ribbons.
            // Explicit termination changes the observable candidate prefix and
            // must fail the same proof.
            const auto expected_pure_reduction_trace =
                uses_gfx12_hardware_stack ?
                    "@luisa_pipeline_ray_query_trace_all_native_closest_"
                    "global_stack_stable_opacity(" :
                    "@luisa_pipeline_ray_query_trace_all_native_closest_stable_opacity(";
            expect(pure_reduction_before_root.find(
                       expected_pure_reduction_trace) !=
                       std::string_view::npos &&
                   pure_reduction_before_module.find(
                       "@luisa_ray_query_proceed(") ==
                       std::string::npos)
                << "large pure closest reduction did not lower to one "
                   "target-native transaction";
            expect(count_occurrences(
                       pure_reduction_before_root,
                       "ray.query.context.projected.field") >= 9u)
                << "pure closest regression did not retain the intended "
                   "large callback environment";
            expect(terminating_reduction_before_module.find(
                       "@luisa_ray_query_proceed(") !=
                       std::string::npos &&
                   terminating_reduction_before_root.find(
                       "@luisa_pipeline_ray_query_trace_all_native_closest") ==
                       std::string_view::npos &&
                   (!uses_gfx12_hardware_stack ||
                    terminating_reduction_before_root.find(
                        "@luisa_pipeline_ray_query_trace_all_stable_opacity(") ==
                        std::string_view::npos))
                << "explicitly terminating RayQuery did not fail closed to "
                   "the resumable traversal ABI";

            // Handler observations are a tagged product
            //   SurfaceObservation x ProceduralObservation,
            // not one union. The object-only procedural handler therefore
            // retains native closest traversal with encoded mask 4 << 8,
            // while its surface handler remains candidate-only. Demanding
            // world and object rays in the same procedural invocation needs
            // two simultaneous representations and selects exact state.
            expect(object_ray_before_root.find(
                       expected_pure_reduction_trace) !=
                       std::string_view::npos &&
                   object_ray_before_root.find("i32 1024") !=
                       std::string_view::npos &&
                   object_ray_before_module.find(
                       "@luisa_pipeline_ray_query_dispatch_compact_object_ray(") !=
                       std::string_view::npos &&
                   object_ray_before_root.find(
                       "ray.query.identity.address") ==
                       std::string_view::npos &&
                   object_ray_before_module.find(
                       "@luisa_ray_query_proceed(") ==
                       std::string_view::npos)
                << "procedural object-ray reduction did not retain the "
                   "split-mask native closest route and object-ray quotient";
            expect(joint_rays_before_module.find(
                       "@luisa_ray_query_proceed(") !=
                       std::string_view::npos &&
                   joint_rays_before_root.find(
                       "@luisa_pipeline_ray_query_trace_all_native_closest") ==
                       std::string_view::npos)
                << "one handler observing both ray spaces did not select "
                   "the exact resumable state domain";

            // Effect-only RayQueryAll is a separate formal quotient from a
            // closest-hit reduction: query post-state is dead, handlers may
            // perform arbitrary ordered side effects, and the active query is
            // proven to admit only reads/terminate. A nested commit must fail
            // that interprocedural proof. A device opacity write must also
            // keep the exact route because it can invalidate the accel proof
            // certificate during the same traversal.
            const auto expected_effect_only_trace =
                uses_gfx12_hardware_stack ?
                    "@luisa_pipeline_ray_query_trace_all_hardware_effect(" :
                    "@luisa_pipeline_ray_query_trace_all_native_effect(";
            expect(effect_only_before_root.find(
                       expected_effect_only_trace) !=
                       std::string_view::npos &&
                   effect_only_before_module.find(
                       "@luisa_ray_query_proceed(") ==
                       std::string::npos)
                << "proven effect-only RayQueryAll did not select one "
                   "specialized candidate-effect traversal";
            expect(effect_commit_before_root.find(
                       "native_effect") == std::string_view::npos &&
                   effect_commit_before_root.find(
                       "hardware_effect") == std::string_view::npos &&
                   effect_commit_before_root.find(
                       "@luisa_pipeline_ray_query_trace_all_stable_opacity(") !=
                       std::string_view::npos)
                << "nested active-query commit did not reject native "
                   "effect-only enumeration";
            expect(effect_opacity_before_root.find(
                       "native_effect") == std::string_view::npos &&
                   effect_opacity_before_root.find(
                       "hardware_effect") == std::string_view::npos &&
                   effect_opacity_before_root.find(
                       "@luisa_pipeline_ray_query_trace_all(") !=
                       std::string_view::npos)
                << "kernel-reachable opacity mutation did not retain the "
                   "exact mutable-opacity traversal";

            const auto expected_terminal_trace =
                uses_gfx12_hardware_stack ?
                    "@luisa_pipeline_ray_query_trace_any_native_terminal_"
                    "global_stack(" :
                    "@luisa_pipeline_ray_query_trace_any_native_terminal(";
            expect(terminal_predicate_before_root.find(
                       expected_terminal_trace) != std::string_view::npos &&
                   terminal_predicate_before_root.find(
                       "@luisa_pipeline_ray_query_trace_any_stable_opacity(") ==
                       std::string_view::npos)
                << "hit-kind-only RayQueryAny did not select native terminal "
                   "AnyHit traversal";
            expect(terminal_full_hit_before_root.find(
                       "@luisa_pipeline_ray_query_trace_any_stable_opacity(") !=
                       std::string_view::npos &&
                   terminal_full_hit_before_root.find(
                       "native_terminal") == std::string_view::npos)
                << "RayQueryAny identity observation did not fail closed to "
                   "the complete committed-hit transaction";
            expect(terminal_mutable_opacity_before_root.find(
                       expected_terminal_trace) != std::string_view::npos)
                << "kernel-reachable opacity write rejected the live-opacity "
                   "native terminal traversal";

            if (uses_gfx12_hardware_stack) {
                // Traversal control is outlined from the owning Callables,
                // while their state allocations remain in the named function
                // bodies. Check route coexistence module-wide and state
                // representation ownership in the individual domains.
                expect(mixed_before_module.find(
                           "@luisa_pipeline_ray_query_trace_any_stable_opacity(") !=
                           std::string::npos &&
                       mixed_before_module.find(
                           "@luisa_ray_query_proceed(") !=
                           std::string::npos)
                    << "function-domain retry did not retain synchronous and "
                       "resumable gfx12 RayQuery routes in one module";
                expect(mixed_exact_state_domain.find(
                           "alloca [112 x i8]") !=
                           std::string_view::npos &&
                       mixed_exact_state_domain.find(
                           "alloca [224 x i8]") ==
                           std::string_view::npos &&
                       mixed_resumable_state_domain.find(
                           "alloca [224 x i8]") !=
                           std::string_view::npos)
                    << "mixed gfx12 RayQuery pipelines did not allocate one "
                       "ABI-consistent state representation per function";
                expect(mixed_before_module.find(
                           "@luisa_hiprt_shared_stack_cache = internal "
                           "addrspace(3) global [4608 x i32]") !=
                       std::string::npos)
                    << "mixed gfx12 RayQuery did not overlay one-region "
                       "16-entry and two-region 9-entry frontiers at the "
                       "common 576-dword stride for eight waves";
            }
        };

    "HIP lowers fixed-vector dot products and preserves FP mode"_test =
        [&] {
            auto dumped_module_count = 0u;
            auto retained_vector_reduction = false;
            for (const auto &entry :
                 std::filesystem::directory_iterator(dump_directory)) {
                const auto filename =
                    entry.path().filename().string();
                if (!filename.starts_with("hip_kernel_final_") ||
                    entry.path().extension() != ".ll") {
                    continue;
                }
                ++dumped_module_count;
                const auto module = read_text_file(entry.path());
                retained_vector_reduction |=
                    module.find("llvm.vector.reduce.fadd") !=
                    std::string::npos;
            }
            constexpr auto callable_and_fp_shader_count = 4u;
            constexpr auto ray_query_shader_count = 16u;
            expect(dumped_module_count ==
                   callable_and_fp_shader_count + ray_query_shader_count)
                << "expected one final HIP LLVM module per compiled shader";
            expect(!retained_vector_reduction)
                << "fixed-vector dot product retained the target-unstable "
                   "LLVM reduction intrinsic";

            // The two normalize kernels have an identical DSL AST and differ
            // only in ShaderOption::enable_fast_math. The dump counter follows
            // compile order: module 2 is fast and module 3 is strict. Inspect
            // the generated root rather than linked OCML helpers so this is a
            // direct test of option propagation into user instructions.
            const auto fast_module = read_text_file(
                dump_directory / "hip_kernel_final_2.ll");
            const auto strict_module = read_text_file(
                dump_directory / "hip_kernel_final_3.ll");
            const auto fast_root = amdgpu_kernel_body(fast_module);
            const auto strict_root = amdgpu_kernel_body(strict_module);
            expect(!fast_root.empty() && !strict_root.empty())
                << "failed to locate generated AMDGPU root kernels";
            expect(contains_dynamic_fp_operation(fast_root) &&
                   contains_dynamic_fp_operation(strict_root))
                << "fast/strict regression kernel lost its dynamic FP work";
            expect(fast_root.find(" fast ") != std::string_view::npos)
                << "ShaderOption::enable_fast_math did not reach HIP LLVM IR";
            expect(strict_root.find(" fast ") == std::string_view::npos)
                << "strict HIP LLVM IR unexpectedly retained fast-math flags";
        };

    std::filesystem::current_path(
        original_directory, filesystem_error);
    std::filesystem::remove_all(
        dump_directory, filesystem_error);
}
