#include "ut/ut.hpp"
#include <utility>
#include <luisa/luisa-compute.h>
#include <luisa/xir/translators/ast2xir.h>
#include <luisa/xir/translators/xir2ast.h>
#include <luisa/xir/translators/xir2text.h>
#include <luisa/xir/verifier.h>

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::compute::xir;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

[[nodiscard]] auto first_kernel_definition(Module *module) noexcept {
    for (auto *f : module->function_list()) {
        if (f->derived_function_tag() == DerivedFunctionTag::KERNEL) { return static_cast<FunctionDefinition *>(f); }
    }
    return static_cast<FunctionDefinition *>(nullptr);
}

struct RoundtripResult {
    luisa::unique_ptr<Module> module;
    luisa::string text;
};

[[nodiscard]] RoundtripResult roundtrip(compute::Function function) noexcept {
    auto module = ast_to_xir_translate(function, {});
    expect(module != nullptr);
    if (module == nullptr) { return {}; }
    xir_to_ast_normalize_module(module.get());
    expect(xir_verify_module(module.get(), {.require_no_phi = true}).succeeded());
    auto *def = first_kernel_definition(module.get());
    expect(def != nullptr);
    if (def == nullptr) { return {}; }
    auto ast = xir_to_ast_translate(*def, {});
    expect(ast != nullptr);
    if (ast == nullptr) { return {}; }
    auto rebuilt = ast_to_xir_translate(ast->function(), {});
    expect(rebuilt != nullptr);
    if (rebuilt == nullptr) { return {}; }
    expect(xir_verify_module(
               rebuilt.get(),
               {.require_canonical_break_continue_targets = true})
               .succeeded());
    auto text = xir_to_text_translate(rebuilt.get(), false);
    return {.module = std::move(rebuilt), .text = std::move(text)};
}

[[nodiscard]] size_t count_occurrences(luisa::string_view text, luisa::string_view needle) noexcept {
    size_t count = 0u;
    for (auto offset = text.find(needle); offset != luisa::string_view::npos;
         offset = text.find(needle, offset + needle.size())) {
        count++;
    }
    return count;
}

}// namespace

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));

    "xir_to_ast_roundtrip_callable_chain"_test = [] {
        Callable add_one = [](Float x) noexcept { return x + 1.0f; };
        Callable add_two = [&add_one](Float x) noexcept { return add_one(add_one(x)); };
        Kernel1D kernel = [&add_two](BufferFloat buffer) noexcept {
            auto idx = dispatch_id().x;
            buffer->write(idx, add_two(buffer->read(idx)));
        };
        auto result = roundtrip(kernel.function()->function());
        auto &text = result.text;
        expect(count_occurrences(text, "arithmetic binary_add") >= 2u);
        expect(text.find("resource_read buffer_read") != string::npos);
        expect(text.find("resource_write buffer_write") != string::npos);
    };

    "xir_to_ast_roundtrip_void_callable"_test = [] {
        Callable write_one = [](BufferFloat buffer, UInt index) noexcept {
            buffer->write(index, 1.0f);
        };
        Kernel1D kernel = [&write_one](BufferFloat buffer) noexcept {
            write_one(buffer, dispatch_id().x);
        };
        auto result = roundtrip(kernel.function()->function());
        auto &text = result.text;
        expect(text.find("resource_write buffer_write") != string::npos);
    };

    "xir_to_ast_roundtrip_structured_control_flow"_test = [] {
        Kernel1D kernel = [](BufferFloat buffer) noexcept {
            auto idx = dispatch_id().x;
            auto x = buffer->read(idx);
            Var<float> y = 0.0f;
            $if (x > 0.0f) {
                y = x * 2.0f;
            } $else {
                y = -x;
            };
            buffer->write(idx, y);
        };
        auto result = roundtrip(kernel.function()->function());
        auto &text = result.text;
        auto *definition = first_kernel_definition(result.module.get());
        expect(definition != nullptr);
        auto if_count = 0u;
        if (definition != nullptr) {
            definition->traverse_instructions(
                [&](Instruction *inst) noexcept {
                    if_count += inst->isa<IfInst>();
                });
        }
        // Normalization crosses the explicit plain-CFG boundary before
        // if-conversion. This side-effect-free diamond is therefore expected
        // to become a select rather than to be reconstructed as an IfInst.
        expect(that % if_count == 0u);
        expect(text.find("arithmetic select") != string::npos);
        expect(text.find("arithmetic binary_mul") != string::npos);
        expect(text.find("arithmetic unary_minus") != string::npos);
        expect(text.find("resource_write buffer_write") != string::npos);
    };

    "xir_to_ast_roundtrip_for_loop"_test = [] {
        Kernel1D kernel = [](BufferFloat buffer) noexcept {
            auto idx = dispatch_id().x;
            Float sum = 0.0f;
            $for (i, 4u) {
                sum += cast<float>(i);
            };
            buffer->write(idx, sum);
        };
        auto result = roundtrip(kernel.function()->function());
        auto &text = result.text;
        expect(text.find("simple_loop") != string::npos);
        expect(count_occurrences(text, "arithmetic binary_add") >= 2u);
        expect(text.find("resource_write buffer_write") != string::npos);
    };

    "xir_to_ast_roundtrip_nested_continue_runs_update"_test = [] {
        Kernel1D kernel = [](BufferUInt buffer) noexcept {
            UInt sum = 0u;
            $for (i, 0u, 4u) {
                $if (i == 1u) {
                    $continue;
                };
                sum += i;
            };
            buffer->write(dispatch_id().x, sum);
        };
        auto result = roundtrip(kernel.function()->function());
        expect(result.module != nullptr);
        if (result.module == nullptr) { return; }
        auto *kernel_definition = first_kernel_definition(result.module.get());
        expect(kernel_definition != nullptr);
        if (kernel_definition == nullptr) { return; }
        auto equality_if_count = 0u;
        auto skipped_body_action_is_guarded = false;
        auto induction_update_is_common = false;
        for (auto *block : kernel_definition->basic_blocks()) {
            for (auto *inst : block->instructions()) {
                if (inst->isa<IfInst>()) {
                    auto *if_inst = static_cast<IfInst *>(inst);
                    auto *condition = if_inst->condition();
                    if (condition->isa<ArithmeticInst>() && static_cast<ArithmeticInst *>(condition)->op() == ArithmeticOp::BINARY_EQUAL) {
                        equality_if_count++;
                        auto count_adds_and_stores = [](BasicBlock *branch) noexcept {
                            auto add_count = 0u;
                            auto store_count = 0u;
                            if (branch != nullptr) {
                                for (auto *branch_inst : branch->instructions()) {
                                    add_count += branch_inst->isa<ArithmeticInst>() &&
                                                 static_cast<ArithmeticInst *>(branch_inst)->op() ==
                                                     ArithmeticOp::BINARY_ADD;
                                    store_count += branch_inst->isa<StoreInst>();
                                }
                            }
                            return std::pair{add_count, store_count};
                        };
                        auto true_adds =
                            count_adds_and_stores(
                                if_inst->true_block())
                                .first;
                        auto [false_adds, false_stores] =
                            count_adds_and_stores(if_inst->false_block());
                        auto [merge_adds, merge_stores] =
                            count_adds_and_stores(if_inst->merge_block());
                        skipped_body_action_is_guarded |=
                            true_adds == 0u &&
                            false_adds == 1u && false_stores >= 1u;
                        induction_update_is_common |=
                            merge_adds == 1u && merge_stores >= 1u;
                    }
                }
            }
        }
        expect(equality_if_count == 1u);
        expect(skipped_body_action_is_guarded);
        expect(induction_update_is_common);
        expect(result.text.find("resource_write buffer_write") != string::npos);
    };

    "xir_to_ast_roundtrip_path_tracing_kernel"_test = [] {
        Callable intersect_sphere = [](Float3 origin, Float3 direction, Float3 center, Float radius) noexcept {
            auto oc = origin - center;
            auto b = dot(oc, direction);
            auto c = dot(oc, oc) - radius * radius;
            auto h = b * b - c;
            return select(-b - sqrt(max(h, 0.0f)), 1e20f, h > 0.0f);
        };
        Callable shade = [](Float3 normal, Float3 throughput) noexcept {
            auto light = normalize(make_float3(0.3f, 0.7f, -0.2f));
            auto n_dot_l = max(dot(normal, light), 0.0f);
            return throughput * (0.1f + 0.9f * n_dot_l);
        };
        Kernel2D kernel = [&intersect_sphere, &shade](ImageFloat output, UInt frame_index) noexcept {
            auto coord = make_uint2(dispatch_id().x, dispatch_id().y);
            auto resolution = make_float2(cast<float>(dispatch_size().x), cast<float>(dispatch_size().y));
            auto uv = (make_float2(coord) + 0.5f) / resolution * 2.0f - 1.0f;
            Float3 origin = make_float3(0.0f, 0.0f, 3.0f);
            Float3 direction = normalize(make_float3(uv, -1.5f));
            Float3 throughput = make_float3(1.0f);
            Float3 radiance = make_float3(0.0f);
            Bool active = true;
            $for (depth, 4u) {
                auto t = intersect_sphere(origin, direction, make_float3(0.0f), 1.0f);
                auto missed = t > 1e10f;
                radiance += ite(active & missed, throughput * make_float3(0.02f, 0.04f, 0.08f), make_float3(0.0f));
                auto hit = origin + t * direction;
                auto normal = normalize(hit);
                radiance += ite(active & !missed, shade(normal, throughput), make_float3(0.0f));
                throughput = ite(active & !missed, throughput * make_float3(0.55f, 0.50f, 0.45f), throughput);
                origin = ite(active & !missed, hit + normal * 1e-3f, origin);
                direction = ite(active & !missed, reflect(direction, normal), direction);
                active = active & !missed;
            };
            auto color = radiance / cast<float>(frame_index + 1u);
            output.write(coord, make_float4(color, 1.0f));
        };
        auto result = roundtrip(kernel.function()->function());
        auto &text = result.text;
        expect(text.find("arithmetic sqrt") != string::npos);
        expect(text.find("simple_loop") != string::npos);
        expect(text.find("resource_write texture2d_write") != string::npos);
    };

    "xir_to_ast_roundtrip_sdf_rendering_kernel"_test = [] {
        Callable sdf = [](Float3 p) noexcept {
            auto sphere = length(p - make_float3(0.0f, 0.0f, -1.0f)) - 0.5f;
            auto plane = p.y + 0.4f;
            auto box_p = abs(p - make_float3(0.7f, 0.0f, -1.2f)) - make_float3(0.25f);
            auto box = length(max(box_p, 0.0f)) + min(max(max(box_p.x, box_p.y), box_p.z), 0.0f);
            return min(min(sphere, plane), box);
        };
        Callable ray_march = [&sdf](Float3 origin, Float3 direction) noexcept {
            Float t = 0.0f;
            Bool active = true;
            $for (step, 48u) {
                auto d = sdf(origin + t * direction);
                active = active & d >= 1e-3f & t <= 20.0f;
                t += ite(active, d, 0.0f);
            };
            return t;
        };
        Callable normal_at = [&sdf](Float3 p) noexcept {
            auto e = 1e-3f;
            auto dx = sdf(p + make_float3(e, 0.0f, 0.0f)) - sdf(p - make_float3(e, 0.0f, 0.0f));
            auto dy = sdf(p + make_float3(0.0f, e, 0.0f)) - sdf(p - make_float3(0.0f, e, 0.0f));
            auto dz = sdf(p + make_float3(0.0f, 0.0f, e)) - sdf(p - make_float3(0.0f, 0.0f, e));
            return normalize(make_float3(dx, dy, dz));
        };
        Kernel2D kernel = [&ray_march, &normal_at](ImageFloat output) noexcept {
            auto coord = make_uint2(dispatch_id().x, dispatch_id().y);
            auto resolution = make_float2(cast<float>(dispatch_size().x), cast<float>(dispatch_size().y));
            auto uv = (make_float2(coord) + 0.5f) / resolution * 2.0f - 1.0f;
            auto origin = make_float3(0.0f, 0.0f, 2.5f);
            auto direction = normalize(make_float3(uv, -1.8f));
            auto t = ray_march(origin, direction);
            Float3 color = make_float3(0.0f);
            $if (t < 20.0f) {
                auto hit = origin + t * direction;
                auto n = normal_at(hit);
                color = make_float3(max(dot(n, normalize(make_float3(0.4f, 0.8f, 0.2f))), 0.0f));
            } $else {
                color = make_float3(0.02f, 0.03f, 0.05f);
            };
            output.write(coord, make_float4(color, 1.0f));
        };
        auto result = roundtrip(kernel.function()->function());
        auto &text = result.text;
        expect(text.find("call") != string::npos);
        expect(text.find("loop") != string::npos);
        expect(text.find("if") != string::npos);
        expect(text.find("resource_write texture2d_write") != string::npos);
    };

    "xir_to_ast_roundtrip_resource_io"_test = [] {
        Kernel1D kernel = [](BufferFloat input, BufferFloat output) noexcept {
            auto idx = dispatch_id().x;
            output->write(idx, input->read(idx) + 1.0f);
        };
        auto result = roundtrip(kernel.function()->function());
        auto &text = result.text;
        expect(text.find("resource_read buffer_read") != string::npos);
        expect(text.find("resource_write buffer_write") != string::npos);
    };

    return 0;
}
