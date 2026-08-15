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
                               // budget. The output write is the semantic result
                               // when the query post-state is discarded.
                               auto checksum =
                                   input_0.read(0u) ^ input_1.read(0u) ^
                                   input_2.read(0u) ^ input_3.read(0u) ^
                                   input_4.read(0u) ^ input_5.read(0u) ^
                                   input_6.read(0u) ^ input_7.read(0u);
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
            expect(!before_root.empty() &&
                   !observing_before_root.empty() && !root.empty() &&
                   !observing_dispatcher.empty() &&
                   !handler_only_before_root.empty() &&
                   !observed_large_before_root.empty() &&
                   !full_candidate_large_before_root.empty())
                << "failed to locate the generated RayQuery functions";
            // Selection is a codegen property, while outlining is an LLVM
            // profitability decision. Inspect the generated root before the
            // ordinary inliner instead of requiring trace wrappers to survive
            // in final IR.
            expect(before_root.find(
                       "@luisa_pipeline_ray_query_trace_all_stable_opacity(") !=
                   std::string_view::npos)
                << "RayQueryAll without device opacity writes did not select "
                   "its stable-opacity native traversal";
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
            expect(module.find(
                       "@luisa_ray_query_pipeline_dispatch(") ==
                   std::string_view::npos)
                << "candidate-only RayQuery retained the full-state "
                   "dispatcher";
            expect(module.find(
                       "@luisa_pipeline_ray_query_dispatch_compact(") ==
                   std::string_view::npos)
                << "compact RayQuery transaction remained as an outlined "
                   "state-pointer boundary";

            const auto uses_native_gfx12_stack =
                module.find("@llvm.amdgcn.ds.bvh.stack.push8.pop1.rtn") !=
                std::string::npos;
            if (uses_native_gfx12_stack) {
                // A candidate-only query has no cross-function identity in
                // the semantic state. Local provenance must therefore let
                // SROA remove both the 112-byte aggregate and its pointer-
                // integer escape after native traversal is inlined.
                expect(root.find("alloca [112 x i8]") ==
                       std::string_view::npos)
                    << "candidate-only RayQuery state escaped scalar "
                       "replacement";
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
            expect(module.find("luisa-specialize-constant-argument") ==
                       std::string_view::npos &&
                   observing_module.find(
                       "luisa-specialize-constant-argument") ==
                       std::string_view::npos)
                << "internal constant-specialization marker escaped HIP "
                   "codegen";
            if (uses_native_gfx12_stack) {
                // A 256-thread block contains eight wave32 waves. Native
                // synchronous traversal uses one 16-entry instance-aware
                // stack, hence 16 * 32 * 8 = 4096 dwords. Its intrinsic
                // immediate and LDS allocation must change together.
                expect(module.find("[4096 x i32]") !=
                       std::string::npos)
                    << "synchronous gfx12 RayQuery did not use the native "
                       "single-region LDS frontier";
                expect(module.find(", i32 16)") !=
                       std::string::npos)
                    << "gfx12 BVH-stack intrinsic did not receive the "
                       "native 16-entry frontier capacity";
                expect(module.find("[8192 x i32]") ==
                       std::string::npos)
                    << "synchronous gfx12 RayQuery retained an unreachable "
                       "second stack region";
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
                       "ray.query.context.projected.field") >= 8u)
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
            expect(dumped_module_count == 9u)
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
