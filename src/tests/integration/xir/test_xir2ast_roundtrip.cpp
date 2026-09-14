#include "ut/ut.hpp"
#include <utility>
#include <optional>
#include <unordered_map>
#include <vector>
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

using UIntBufferWrites = std::vector<std::pair<uint32_t, uint32_t>>;

// Execute only the scalar subset used by the continue regression. Undefined
// stores are permitted until overwritten; reading one, encountering an unknown
// operation, or failing to return within the bound rejects the execution.
[[nodiscard]] std::optional<UIntBufferWrites> execute_uint_continue_kernel(
    FunctionDefinition *definition, uint32_t limit, uint32_t skip) {
    std::unordered_map<const Value *, uint32_t> values;
    std::unordered_map<const Value *, std::optional<uint32_t>> memory;
    Value *buffer = nullptr;
    auto argument_index = 0u;
    for (auto *argument : definition->arguments()) {
        if (argument->derived_argument_tag() == DerivedArgumentTag::RESOURCE) {
            if (buffer != nullptr || !argument->type()->is_buffer()) { return std::nullopt; }
            buffer = argument;
        } else {
            if (argument->derived_argument_tag() != DerivedArgumentTag::VALUE ||
                argument->type() != Type::of<uint32_t>() || argument_index >= 2u) { return std::nullopt; }
            values.emplace(argument, argument_index++ == 0u ? limit : skip);
        }
    }
    if (buffer == nullptr || argument_index != 2u) { return std::nullopt; }
    auto read = [&](const Value *value) -> std::optional<uint32_t> {
        if (value == nullptr ||
            (value->type() != Type::of<uint32_t>() && value->type() != Type::of<bool>())) { return std::nullopt; }
        if (value->isa<xir::Constant>()) {
            auto *constant = static_cast<const xir::Constant *>(value);
            return value->type()->is_bool() ? uint32_t(constant->as<bool>()) : constant->as<uint32_t>();
        }
        auto iter = values.find(value);
        return iter == values.end() ? std::nullopt : std::optional{iter->second};
    };
    UIntBufferWrites writes;
    auto *block = definition->body_block();
    for (auto steps = 0u; steps < 10000u; ++steps) {
        if (block == nullptr || !block->is_terminated()) { return std::nullopt; }
        // A stale definition from a previous loop iteration cannot satisfy a
        // missing SSA operand in this iteration.
        for (auto *inst : block->instructions()) { values.erase(inst); }
        BasicBlock *next = nullptr;
        for (auto *inst : block->instructions()) {
            if (inst->isa<AllocaInst>()) {
                if (inst->type() != Type::of<uint32_t>() && inst->type() != Type::of<bool>()) { return std::nullopt; }
                memory[inst] = std::nullopt;
            } else if (inst->isa<LoadInst>()) {
                auto iter = memory.find(static_cast<LoadInst *>(inst)->variable());
                if (iter == memory.end() || !iter->second.has_value()) { return std::nullopt; }
                values[inst] = *iter->second;
            } else if (inst->isa<StoreInst>()) {
                auto *store = static_cast<StoreInst *>(inst);
                auto iter = memory.find(store->variable());
                if (iter == memory.end()) { return std::nullopt; }
                if (store->value()->isa<Undefined>()) {
                    iter->second = std::nullopt;
                } else {
                    auto value = read(store->value());
                    if (!value.has_value()) { return std::nullopt; }
                    iter->second = *value;
                }
            } else if (inst->isa<ArithmeticInst>()) {
                auto *arithmetic = static_cast<ArithmeticInst *>(inst);
                if (arithmetic->operand_count() != 1u && arithmetic->operand_count() != 2u) { return std::nullopt; }
                auto x = read(arithmetic->operand(0u));
                if (!x.has_value()) { return std::nullopt; }
                if (arithmetic->operand_count() == 1u) {
                    if (arithmetic->op() != ArithmeticOp::UNARY_BIT_NOT || arithmetic->type() != Type::of<bool>()) { return std::nullopt; }
                    values[inst] = !*x;
                } else {
                    auto y = read(arithmetic->operand(1u));
                    if (!y.has_value()) { return std::nullopt; }
                    switch (arithmetic->op()) {
                        case ArithmeticOp::BINARY_ADD: values[inst] = *x + *y; break;
                        case ArithmeticOp::BINARY_LESS: values[inst] = *x < *y; break;
                        case ArithmeticOp::BINARY_EQUAL: values[inst] = *x == *y; break;
                        default: return std::nullopt;
                    }
                }
            } else if (inst->isa<ResourceWriteInst>()) {
                auto *write = static_cast<ResourceWriteInst *>(inst);
                if (write->op() != ResourceWriteOp::BUFFER_WRITE || write->operand_count() != 3u || write->operand(0u) != buffer) { return std::nullopt; }
                auto index = read(write->operand(1u));
                auto value = read(write->operand(2u));
                if (!index.has_value() || !value.has_value()) { return std::nullopt; }
                writes.emplace_back(*index, *value);
            } else if (inst->isa<IfInst>() || inst->isa<ConditionalBranchInst>()) {
                auto *branch = static_cast<ConditionalBranchTerminatorInstruction *>(inst);
                auto condition = read(branch->condition());
                if (!condition.has_value()) { return std::nullopt; }
                next = *condition ? branch->true_block() : branch->false_block();
            } else if (inst->isa<BranchInst>() || inst->isa<BreakInst>() || inst->isa<ContinueInst>()) {
                next = static_cast<BranchTerminatorInstruction *>(inst)->target_block();
            } else if (inst->isa<SimpleLoopInst>()) {
                next = static_cast<SimpleLoopInst *>(inst)->body_block();
            } else if (inst->isa<LoopInst>()) {
                next = static_cast<LoopInst *>(inst)->prepare_block();
            } else if (inst->isa<ReturnInst>()) {
                if (static_cast<ReturnInst *>(inst)->return_value() != nullptr) { return std::nullopt; }
                return writes;
            } else {
                return std::nullopt;
            }
        }
        block = next;
    }
    return std::nullopt;
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
        Kernel1D kernel = [](BufferUInt buffer, UInt limit, UInt skip) noexcept {
            UInt sum = 0u;
            $for (i, 0u, limit) {
                buffer->write(i + 1u, i);
                $if (i == skip) {
                    $continue;
                };
                sum += i;
            };
            buffer->write(0u, sum);
        };
        auto result = roundtrip(kernel.function()->function());
        expect(result.module != nullptr);
        if (result.module == nullptr) { return; }
        auto *kernel_definition = first_kernel_definition(result.module.get());
        expect(kernel_definition != nullptr);
        if (kernel_definition == nullptr) { return; }
        // The update can be emitted once at a shared merge or in both mutually
        // exclusive arms before Continue. Its physical location is incidental;
        // every iteration must execute it exactly once, including a skipped
        // body. The complete write trace proves both iteration order/count and
        // the final accumulated value after the XIR -> AST -> XIR round trip.
        for (auto limit : {0u, 1u, 2u, 4u, 9u, 16u}) {
            for (auto skip : {0u, 1u, limit == 0u ? 0u : limit - 1u, limit, UINT32_MAX}) {
                UIntBufferWrites expected;
                auto sum = 0u;
                for (auto i = 0u; i < limit; ++i) {
                    expected.emplace_back(i + 1u, i);
                    if (i != skip) { sum += i; }
                }
                expected.emplace_back(0u, sum);
                auto actual = execute_uint_continue_kernel(kernel_definition, limit, skip);
                expect(actual.has_value()) << "limit=" << limit << " skip=" << skip;
                if (actual.has_value()) {
                    auto equal = static_cast<bool>(*actual == expected);
                    expect(equal) << "iteration/write trace limit=" << limit << " skip=" << skip;
                    expect(actual->size() == size_t{limit} + 1u);
                }
            }
        }
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
