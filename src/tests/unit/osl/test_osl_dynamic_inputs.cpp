// Test for the OSL dynamic-input feature (osl::ShaderGroup::mark_dynamic_input()).
//
// A dynamic input is a shader parameter whose value is delivered by the host at
// translate time through OSLTranslator::emit_dynamic_input() instead of through a
// parameter override or an upstream connection:
//   ShaderGroup::mark_dynamic_input() -> ShaderLayer::dynamic_inputs
//   -> detail::link_shader_group() (mark_dyn + skip_init_params + validation)
//   -> detail::analyze_group_strings_and_constants() (top lattice + isconnected)
//   -> detail::analyze_group_derivatives() (has_deriv seeds D_src)
//   -> Impl::allocate_group_storage() -> OSLTranslator::emit_dynamic_input()
//   -> OSLTranslator::Impl::try_emit_instruction() ("isconnected")
// so this test exercises that path end to end (translate + XIR verify + XIR->AST
// + execute on the auto-selected test device).
//
// This test covers:
// - a scalar dynamic input reaching the shader (+ isconnected() == 1 for it)
// - has_deriv=true actually delivering dx/dy (deriv-analysis D_src seeding)
// - a struct dynamic input fanning out to its leaf fields
// - the layer index handed to the hook being the *group* index even when a
//   preceding dead layer was pruned
// - fail-closed behaviour (default hook, unknown param, non-param, closure array)
// - a dynamic input on an array whose length cannot be resolved is rejected
//   instead of being silently dropped
// - isconnected() answering exactly 1 for the dynamic fields, not for every
//   sibling of the struct that contains one
// - a parameter that is both connected and marked as a dynamic input is rejected
//   instead of silently discarding the host value
// - a dynamic string input: raw token copies work, string ops over it fail closed

#include "ut/ut.hpp"
#include "test_osl_common.h"

#include <cmath>
#include <cstdlib>

using namespace boost::ut;
using namespace luisa;
using namespace luisa::compute;
using namespace luisa::test::osl_test;

namespace {

[[nodiscard]] bool near_f(float a, float b, float tol = 1e-4f) noexcept {
    return std::abs(a - b) <= tol;
}

[[nodiscard]] bool near_v3(luisa::float3 a, luisa::float3 b, float tol = 1e-4f) noexcept {
    return near_f(a.x, b.x, tol) && near_f(a.y, b.y, tol) && near_f(a.z, b.z, tol);
}

[[nodiscard]] bool is_prim(const compute::osl::Type *t,
                           compute::osl::SimpleType::Primitive p) noexcept {
    return t != nullptr && t->tag() == compute::osl::Type::Tag::SIMPLE &&
           static_cast<const compute::osl::SimpleType *>(t)->primitive() == p;
}

struct DynEntry {
    float f{0.0f};
    float dx{0.0f};
    float dy{0.0f};
    int32_t i{0};
    luisa::float3 v3{0.0f};
    luisa::string str;
};

struct DynCall {
    uint32_t layer{0u};
    luisa::string name;
    int array_length{0};
    bool want_deriv{false};
    bool closure{false};
};

// A TestOSLTranslator that actually supplies "dynamic" (host-provided) inputs.
class DynamicInputTranslator final : public TestOSLTranslator {

private:
    luisa::unordered_map<luisa::string, DynEntry> _values;

    [[nodiscard]] static luisa::string _key(uint32_t layer, luisa::string_view name) {
        return luisa::format("{}:{}", layer, name);
    }

public:
    luisa::vector<DynCall> calls;

    void set(uint32_t layer, luisa::string_view name, DynEntry e) {
        _values.insert_or_assign(_key(layer, name), e);
    }

protected:
    [[nodiscard]] compute::osl::DualValue emit_dynamic_input(
        uint32_t layer_index, luisa::string_view param_name,
        const compute::osl::Type *param_type, int array_length,
        bool want_deriv) noexcept override {
        calls.emplace_back(DynCall{.layer = layer_index,
                                   .name = luisa::string{param_name},
                                   .array_length = array_length,
                                   .want_deriv = want_deriv,
                                   .closure = false});
        auto it = _values.find(_key(layer_index, param_name));
        if (it == _values.end()) {
            report_error(luisa::format("dynamic input test: no dynamic value for layer {} '{}'.",
                                       layer_index, param_name));
            return {};
        }
        const auto &e = it->second;
        compute::osl::DualValue dv;
        if (array_length > 0) {
            report_error(luisa::format("dynamic input test: dynamic arrays are not exercised ('{}').", param_name));
            return {};
        }
        if (is_prim(param_type, compute::osl::SimpleType::Primitive::FLOAT)) {
            dv.val = const_float(e.f);
            if (want_deriv) {
                dv.dx = const_float(e.dx);
                dv.dy = const_float(e.dy);
            }
        } else if (is_prim(param_type, compute::osl::SimpleType::Primitive::INT)) {
            dv.val = const_int(e.i);
        } else if (is_prim(param_type, compute::osl::SimpleType::Primitive::POINT) ||
                   is_prim(param_type, compute::osl::SimpleType::Primitive::VECTOR) ||
                   is_prim(param_type, compute::osl::SimpleType::Primitive::NORMAL) ||
                   is_prim(param_type, compute::osl::SimpleType::Primitive::COLOR)) {
            dv.val = const_float3(e.v3);
        } else if (is_prim(param_type, compute::osl::SimpleType::Primitive::STRING)) {
            dv.val = const_uint(intern_string(e.str));
        } else {
            report_error(luisa::format("dynamic input test: unsupported dynamic input type '{}' for '{}'.",
                                       param_type ? param_type->identifier() : luisa::string_view{"null"}, param_name));
            return {};
        }
        return dv;
    }

    [[nodiscard]] compute::osl::ClosureValue emit_dynamic_closure_input(
        uint32_t layer_index, luisa::string_view param_name) noexcept override {
        calls.emplace_back(DynCall{.layer = layer_index,
                                   .name = luisa::string{param_name},
                                   .array_length = 0,
                                   .want_deriv = false,
                                   .closure = true});
        // Same TestClosure layout TestOSLTranslator uses: tag 4 == "emission".
        auto *ct = compute::Type::of<TestClosure>();
        return {.value = xir_builder().call(
                    ct, compute::xir::ArithmeticOp::AGGREGATE,
                    {const_float3(luisa::make_float3(1.0f)), const_float3(luisa::make_float3(0.0f)),
                     const_uint(4u), const_float(0.0f), const_float(0.0f), const_float(0.0f)})};
    }
};

// Translate only (no device); used for the fail-closed checks.
[[nodiscard]] compute::osl::TranslationResult translate_only(
    compute::osl::OSLTranslator &translator,
    const compute::osl::ShaderGroup &group) {
    compute::xir::Module module;
    auto *kernel = module.create_kernel();
    return translator.translate(group, kernel);
}

void append_calls(luisa::string &out, const luisa::vector<DynCall> &calls) {
    for (const auto &c : calls) {
        out.append(luisa::format("[layer={} name='{}' len={} deriv={} closure={}] ",
                                 c.layer, c.name, c.array_length, c.want_deriv, c.closure));
    }
}

}// namespace

int main(int argc, char *argv[]) {
    set_osl_test_args(argc, argv);

    // A single-layer shader with one scalar input parameter that we feed from the
    // host through the dynamic-input hook instead of through set_parameter().
    static constexpr auto basic_oso = R"oso(
OpenShadingLanguage 1.00
shader dyn_basic
param	float	gain	1.0
oparam	float	f0	0
oparam	int	i0	0
const	float	$two	2.0
code ___main___
	mul		f0 gain $two
	isconnected	i0 gain
end
)oso";

    "dynamic_scalar_input_and_isconnected"_test = [] {
        auto shader = parse_oso_source(basic_oso);
        expect(shader != nullptr);
        if (!shader) { return; }

        // 1. Dynamic input marked -> the hook value must reach the shader.
        {
            osl::ShaderGroup group;
            auto layer = group.add_layer("main", shader.get());
            group.mark_output(layer, "f0");
            group.mark_output(layer, "i0");
            group.mark_dynamic_input(layer, "gain");

            DynamicInputTranslator translator;
            translator.set(layer, "gain", DynEntry{.f = 3.0f});
            auto res = compile_and_run_with_translator(translator, group, {TestShaderGlobals{}});
            expect(res.translation.succeeded()) << "dynamic input translation failed";
            expect(res.ast_ok);
            expect(res.shader_compile_ok);
            expect(translator.calls.size() == 1u) << "emit_dynamic_input was not called exactly once";
            if (!translator.calls.empty()) {
                expect(translator.calls[0].name == "gain");
                expect(translator.calls[0].layer == 0u);
                expect(translator.calls[0].array_length == 0);
                expect(translator.calls[0].want_deriv == false);
            }
            if (res.outputs.size() == 1u) {
                expect(near_f(res.outputs[0].f0, 6.0f))// gain(3.0) * 2.0
                    << luisa::format("f0 = {} (expected 6.0)", res.outputs[0].f0);
                expect(res.outputs[0].i0 == 1)
                    << luisa::format("isconnected(dynamic param) = {} (expected 1)", res.outputs[0].i0);
            }
        }

        // 2. Control: identical shader without mark_dynamic_input -> initial value 1.0,
        //    isconnected() == 0.
        {
            osl::ShaderGroup group;
            auto layer = group.add_layer("main", shader.get());
            group.mark_output(layer, "f0");
            group.mark_output(layer, "i0");

            DynamicInputTranslator translator;
            translator.set(layer, "gain", DynEntry{.f = 3.0f});
            auto res = compile_and_run_with_translator(translator, group, {TestShaderGlobals{}});
            expect(res.translation.succeeded());
            expect(res.shader_compile_ok);
            expect(translator.calls.empty()) << "hook must not be called without mark_dynamic_input";
            if (res.outputs.size() == 1u) {
                expect(near_f(res.outputs[0].f0, 2.0f))// default 1.0 * 2.0
                    << luisa::format("f0 = {} (expected 2.0)", res.outputs[0].f0);
                expect(res.outputs[0].i0 == 0);
            }
        }
    };

    // Derivative plumbing: has_deriv=true must seed D_src so that a Dx() consumer
    // gets the host-supplied derivative instead of static zero.
    static constexpr auto deriv_oso = R"oso(
OpenShadingLanguage 1.00
shader dyn_deriv
param	float	gain	1.0
oparam	float	f0	0
oparam	float	f1	0
oparam	float	f2	0
local	float	lv
local	float	dl
local	float	dly
code ___main___
	assign		lv gain
	Dx		dl lv
	Dy		dly lv
	assign		f0 lv
	assign		f1 dl
	assign		f2 dly
end
)oso";

    "dynamic_input_derivatives"_test = [] {
        auto shader = parse_oso_source(deriv_oso);
        expect(shader != nullptr);
        if (!shader) { return; }

        struct Case {
            bool mark{false};
            bool has_deriv{false};
            float expect_f0{1.0f};
            float expect_f1{0.0f};
            float expect_f2{0.0f};
            bool expect_want_deriv{false};
        };
        const Case cases[] = {
            {.mark = false, .has_deriv = false, .expect_f0 = 1.0f, .expect_f1 = 0.0f, .expect_f2 = 0.0f, .expect_want_deriv = false},
            {.mark = true, .has_deriv = false, .expect_f0 = 3.0f, .expect_f1 = 0.0f, .expect_f2 = 0.0f, .expect_want_deriv = false},
            {.mark = true, .has_deriv = true, .expect_f0 = 3.0f, .expect_f1 = 0.5f, .expect_f2 = 0.25f, .expect_want_deriv = true},
        };
        for (const auto &c : cases) {
            osl::ShaderGroup group;
            auto layer = group.add_layer("main", shader.get());
            group.mark_output(layer, "f0");
            group.mark_output(layer, "f1");
            group.mark_output(layer, "f2");
            if (c.mark) { group.mark_dynamic_input(layer, "gain", c.has_deriv); }

            DynamicInputTranslator translator;
            translator.set(layer, "gain", DynEntry{.f = 3.0f, .dx = 0.5f, .dy = 0.25f});
            auto res = compile_and_run_with_translator(translator, group, {TestShaderGlobals{}});
            expect(res.translation.succeeded());
            expect(res.ast_ok);
            expect(res.shader_compile_ok);
            if (c.mark) {
                expect(translator.calls.size() == 1u);
                if (!translator.calls.empty()) {
                    expect(translator.calls[0].want_deriv == c.expect_want_deriv)
                        << luisa::format("want_deriv = {} (expected {}) for has_deriv={}",
                                         translator.calls[0].want_deriv, c.expect_want_deriv, c.has_deriv);
                }
            }
            if (res.outputs.size() == 1u) {
                expect(near_f(res.outputs[0].f0, c.expect_f0))
                    << luisa::format("f0 = {} (expected {})", res.outputs[0].f0, c.expect_f0);
                expect(near_f(res.outputs[0].f1, c.expect_f1))
                    << luisa::format("Dx = {} (expected {}) for has_deriv={}",
                                     res.outputs[0].f1, c.expect_f1, c.has_deriv);
                expect(near_f(res.outputs[0].f2, c.expect_f2))
                    << luisa::format("Dy = {} (expected {}) for has_deriv={}",
                                     res.outputs[0].f2, c.expect_f2, c.has_deriv);
            }
        }
    };

    // A struct parameter marked as a dynamic input must fan out to its leaf fields
    // (link_shader_group::mark_dyn) so that the field storages get host values.
    static constexpr auto struct_oso = R"oso(
OpenShadingLanguage 1.00
shader dyn_struct
param	struct Pair	sp		%structfields{a,c}
param	float	sp.a	1.0	%mystruct{sp} %mystructfield{0}
param	color	sp.c	1 1 1	%mystruct{sp} %mystructfield{1}
oparam	float	f0	0
oparam	color	v0	0 0 0
code ___main___
	assign		f0 sp.a
	assign		v0 sp.c
end
)oso";

    "dynamic_struct_input_fans_out_to_fields"_test = [] {
        auto shader = parse_oso_source(struct_oso);
        expect(shader != nullptr);
        if (!shader) { return; }

        osl::ShaderGroup group;
        auto layer = group.add_layer("main", shader.get());
        group.mark_output(layer, "f0");
        group.mark_output(layer, "v0");
        group.mark_dynamic_input(layer, "sp");

        DynamicInputTranslator translator;
        translator.set(layer, "sp.a", DynEntry{.f = 4.0f});
        translator.set(layer, "sp.c", DynEntry{.v3 = luisa::make_float3(1.0f, 2.0f, 3.0f)});
        auto res = compile_and_run_with_translator(translator, group, {TestShaderGlobals{}});
        auto call_dump = luisa::string{};
        append_calls(call_dump, translator.calls);
        expect(res.translation.succeeded()) << "struct dynamic input translation failed";
        expect(res.ast_ok);
        expect(res.shader_compile_ok);
        expect(translator.calls.size() == 2u) << call_dump;
        if (res.outputs.size() == 1u) {
            expect(near_f(res.outputs[0].f0, 4.0f))
                << luisa::format("f0 = {} (expected 4.0)", res.outputs[0].f0);
            expect(near_v3(res.outputs[0].v0, luisa::make_float3(1.0f, 2.0f, 3.0f)))
                << luisa::format("v0 = ({}, {}, {}) (expected 1,2,3)",
                                 res.outputs[0].v0.x, res.outputs[0].v0.y, res.outputs[0].v0.z);
        }
    };

    // The layer index handed to the hook is documented as the ShaderGroup layer
    // index. Force dead-layer pruning in front of the dynamic layer so that the
    // group index and the internal active index differ.
    static constexpr auto dead_layer_oso = R"oso(
OpenShadingLanguage 1.00
shader review_dead
oparam	float	dead_out	0
const	float	$c9	999.0
code ___main___
	assign		dead_out $c9
end
)oso";
    static constexpr auto src_layer_oso = R"oso(
OpenShadingLanguage 1.00
shader review_src
oparam	float	out_s	0
const	float	$two	2.0
code ___main___
	assign		out_s $two
end
)oso";
    static constexpr auto dst_layer_oso = R"oso(
OpenShadingLanguage 1.00
shader review_dst
param	float	in_s	0
param	float	extra	1.0
oparam	float	f0	0
code ___main___
	add		f0 in_s extra
end
)oso";

    "dynamic_input_layer_index_is_group_index"_test = [] {
        auto s_dead = parse_oso_source(dead_layer_oso);
        auto s_src = parse_oso_source(src_layer_oso);
        auto s_dst = parse_oso_source(dst_layer_oso);
        expect(s_dead != nullptr && s_src != nullptr && s_dst != nullptr);
        if (!s_dead || !s_src || !s_dst) { return; }

        osl::ShaderGroup group;
        static_cast<void>(group.add_layer("dead", s_dead.get()));// index 0, pruned
        auto idx_src = group.add_layer("src", s_src.get());       // index 1
        auto idx_dst = group.add_layer("dst", s_dst.get());       // index 2
        group.connect_shaders(idx_src, "out_s", idx_dst, "in_s");
        group.mark_output(idx_dst, "f0");
        group.mark_dynamic_input(idx_dst, "extra");

        DynamicInputTranslator translator;
        translator.set(idx_dst, "extra", DynEntry{.f = 5.0f});
        auto res = compile_and_run_with_translator(translator, group, {TestShaderGlobals{}});
        auto call_dump = luisa::string{};
        append_calls(call_dump, translator.calls);
        expect(res.translation.succeeded()) << "multi-layer dynamic input translation failed";
        expect(res.translation.eliminated_layer_count == 1u) << "the dead layer was not pruned";
        expect(res.ast_ok);
        expect(res.shader_compile_ok);
        expect(translator.calls.size() == 1u) << call_dump;
        if (!translator.calls.empty()) {
            expect(translator.calls[0].layer == idx_dst)
                << luisa::format("hook layer index = {} (expected group index {}), active index is {}",
                                 translator.calls[0].layer, idx_dst, res.translation.active_layer_count - 1u);
        }
        if (res.outputs.size() == 1u) {
            expect(near_f(res.outputs[0].f0, 7.0f))// in_s(2.0) + extra(5.0)
                << luisa::format("f0 = {} (expected 7.0)", res.outputs[0].f0);
        }
    };

    // Fail-closed behaviour of the new API surface.
    "dynamic_input_fail_closed"_test = [] {
        auto shader = parse_oso_source(basic_oso);
        expect(shader != nullptr);
        if (!shader) { return; }

        // 1. Default OSLTranslator hook -> "not implemented" error, no silent zero.
        {
            osl::ShaderGroup group;
            auto layer = group.add_layer("main", shader.get());
            group.mark_output(layer, "f0");
            group.mark_dynamic_input(layer, "gain");
            TestOSLTranslator translator;
            auto res = translate_only(translator, group);
            expect(!res.succeeded()) << "default emit_dynamic_input must fail closed";
        }

        // 2. Unknown parameter name.
        {
            osl::ShaderGroup group;
            auto layer = group.add_layer("main", shader.get());
            group.mark_output(layer, "f0");
            group.mark_dynamic_input(layer, "does_not_exist");
            DynamicInputTranslator translator;
            auto res = translate_only(translator, group);
            expect(!res.succeeded()) << "unknown dynamic input must fail closed";
        }

        // 3. Non-parameter symbol (a local) is rejected.
        {
            static constexpr auto local_oso = R"oso(
OpenShadingLanguage 1.00
shader dyn_bad
oparam	float	f0	0
local	float	lv
const	float	$two	2.0
code ___main___
	assign		lv $two
	assign		f0 lv
end
)oso";
            auto bad = parse_oso_source(local_oso);
            expect(bad != nullptr);
            if (bad) {
                osl::ShaderGroup group;
                auto layer = group.add_layer("main", bad.get());
                group.mark_output(layer, "f0");
                group.mark_dynamic_input(layer, "lv");
                DynamicInputTranslator translator;
                auto res = translate_only(translator, group);
                expect(!res.succeeded()) << "dynamic input on a local must fail closed";
            }
        }

        // 4. Dynamic closure arrays are explicitly unsupported.
        {
            static constexpr auto closure_arr_oso = R"oso(
OpenShadingLanguage 1.00
shader dyn_carr
param	closure color [2]	carr
oparam	float	f0	0
const	float	$two	2.0
code ___main___
	assign		f0 $two
end
)oso";
            auto carr = parse_oso_source(closure_arr_oso);
            expect(carr != nullptr);
            if (carr) {
                osl::ShaderGroup group;
                auto layer = group.add_layer("main", carr.get());
                group.mark_output(layer, "f0");
                group.mark_dynamic_input(layer, "carr");
                DynamicInputTranslator translator;
                auto res = translate_only(translator, group);
                expect(!res.succeeded()) << "dynamic closure array must fail closed";
            }
        }
    };

    // Dynamic closure input: the scalar closure path of allocate_group_storage.
    static constexpr auto closure_oso = R"oso(
OpenShadingLanguage 1.00
shader dyn_closure
param	closure color	cin
global	closure color	Ci
const	float	$czero	0
local	closure color	c_zero
code ___main___
	assign		c_zero $czero
	add		Ci cin c_zero
end
)oso";

    "dynamic_closure_input"_test = [] {
        auto shader = parse_oso_source(closure_oso);
        expect(shader != nullptr);
        if (!shader) { return; }

        osl::ShaderGroup group;
        auto layer = group.add_layer("main", shader.get());
        DynamicInputTranslator translator;
        group.mark_dynamic_input(layer, "cin");
        auto res = compile_and_run_with_translator(translator, group, {TestShaderGlobals{}});
        auto call_dump = luisa::string{};
        append_calls(call_dump, translator.calls);
        expect(res.translation.succeeded()) << "dynamic closure input translation failed";
        expect(res.ast_ok);
        expect(res.shader_compile_ok);
        expect(translator.calls.size() == 1u) << call_dump;
        if (!translator.calls.empty()) {
            expect(translator.calls[0].closure) << "emit_dynamic_closure_input was not used";
        }
        if (res.outputs.size() == 1u) {
            // TestOSLTranslator closure tags: emission == 4
            expect(res.outputs[0].ci.kind_tag == 4u)
                << luisa::format("Ci tag = {} (expected 4 = emission)", res.outputs[0].ci.kind_tag);
            expect(near_v3(res.outputs[0].ci.weight, luisa::make_float3(1.0f)))
                << luisa::format("Ci weight = ({}, {}, {})",
                                 res.outputs[0].ci.weight.x,
                                 res.outputs[0].ci.weight.y,
                                 res.outputs[0].ci.weight.z);
        }
    };

    // A dynamic *string* input: the raw token copy works, but any op that needs a
    // compile-time-bounded string set cannot be served by a runtime-only value.
    static constexpr auto str_assign_oso = R"oso(
OpenShadingLanguage 1.00
shader dyn_str
param	string	src	"default"
oparam	string	out_s	""
code ___main___
	assign		out_s src
end
)oso";
    static constexpr auto str_len_oso = R"oso(
OpenShadingLanguage 1.00
shader dyn_strlen
param	string	src	"default"
oparam	int	i0	0
code ___main___
	strlen		i0 src
end
)oso";

    "dynamic_string_input"_test = [] {
        // 1. Token plumbing works for a plain copy.
        {
            auto shader = parse_oso_source(str_assign_oso);
            expect(shader != nullptr);
            if (shader) {
                osl::ShaderGroup group;
                auto layer = group.add_layer("main", shader.get());
                group.mark_output(layer, "out_s");
                group.mark_dynamic_input(layer, "src");
                DynamicInputTranslator translator;
                translator.set(layer, "src", DynEntry{.str = "hello"});
                auto res = compile_and_run_with_translator(translator, group, {TestShaderGlobals{}});
                expect(res.translation.succeeded()) << "dynamic string input translation failed";
                expect(res.shader_compile_ok);
                expect(translator.calls.size() == 1u);
                if (res.outputs.size() == 1u) {
                    auto tok = static_cast<osl::StringToken>(res.outputs[0].i0);
                    auto sv = res.translation.string_table.lookup(tok);
                    expect(sv == "hello")
                        << luisa::format("output string token {} -> '{}' (expected 'hello')", tok, sv);
                }
            }
        }
        // 2. A string op over an unbounded (top) lattice must fail closed, not
        //    silently read an unbound runtime token.
        {
            auto shader = parse_oso_source(str_len_oso);
            expect(shader != nullptr);
            if (shader) {
                osl::ShaderGroup group;
                auto layer = group.add_layer("main", shader.get());
                group.mark_output(layer, "i0");
                group.mark_dynamic_input(layer, "src");
                DynamicInputTranslator translator;
                translator.set(layer, "src", DynEntry{.str = "hello"});
                auto res = translate_only(translator, group);
                expect(!res.succeeded()) << "strlen() over a dynamic string input must fail closed";
            }
        }
    };

    // An unbounded array parameter whose length cannot be resolved has no storage;
    // marking it dynamic cannot deliver a value, so it must be reported instead of
    // being silently dropped (fail closed).
    static constexpr auto unbounded_array_oso = R"oso(
OpenShadingLanguage 1.00
shader dyn_arr
param	float []	vals
oparam	int	i0	0
code ___main___
	arraylength	i0 vals
end
)oso";

    "dynamic_unbounded_array_input_fails_closed"_test = [] {
        auto shader = parse_oso_source(unbounded_array_oso);
        expect(shader != nullptr);
        if (!shader) { return; }
        osl::ShaderGroup group;
        auto layer = group.add_layer("main", shader.get());
        group.mark_output(layer, "i0");
        group.mark_dynamic_input(layer, "vals");
        DynamicInputTranslator translator;
        auto res = translate_only(translator, group);
        auto call_dump = luisa::string{};
        append_calls(call_dump, translator.calls);
        expect(!res.succeeded())
            << luisa::format("dynamic input on an array with unresolved length was silently ignored "
                             "(hook calls: {})",
                             call_dump);
    };

    // isconnected() must report exactly the dynamic input fields, not every sibling
    // of the struct that contains one.
    static constexpr auto isconnected_field_oso = R"oso(
OpenShadingLanguage 1.00
shader dyn_sib
param	struct Pair	sp		%structfields{a,c}
param	float	sp.a	1.0	%mystruct{sp} %mystructfield{0}
param	color	sp.c	1 1 1	%mystruct{sp} %mystructfield{1}
oparam	int	i0	0
oparam	int	i1	0
oparam	int	i2	0
oparam	color	v0	0 0 0
code ___main___
	isconnected	i0 sp.a
	isconnected	i1 sp.c
	isconnected	i2 sp
	assign		v0 sp.c
end
)oso";

    "dynamic_field_isconnected_is_exact"_test = [] {
        auto shader = parse_oso_source(isconnected_field_oso);
        expect(shader != nullptr);
        if (!shader) { return; }

        // Only sp.a is a dynamic input; sp.c keeps its .oso default.
        {
            osl::ShaderGroup group;
            auto layer = group.add_layer("main", shader.get());
            group.mark_output(layer, "i0");
            group.mark_output(layer, "i1");
            group.mark_output(layer, "i2");
            group.mark_output(layer, "v0");
            group.mark_dynamic_input(layer, "sp.a");
            DynamicInputTranslator translator;
            translator.set(layer, "sp.a", DynEntry{.f = 4.0f});
            auto res = compile_and_run_with_translator(translator, group, {TestShaderGlobals{}});
            expect(res.translation.succeeded());
            expect(res.shader_compile_ok);
            if (res.outputs.size() == 1u) {
                expect(res.outputs[0].i0 == 1) << "isconnected(sp.a) must be 1 (sp.a is dynamic)";
                expect(res.outputs[0].i1 == 0)
                    << luisa::format("isconnected(sp.c) = {} (expected 0: sp.c is neither connected nor dynamic)",
                                     res.outputs[0].i1);
                expect(near_v3(res.outputs[0].v0, luisa::make_float3(1.0f)))
                    << luisa::format("sp.c = ({}, {}, {}) (expected the .oso default 1,1,1)",
                                     res.outputs[0].v0.x, res.outputs[0].v0.y, res.outputs[0].v0.z);
            }
        }
        // The whole struct marked dynamic: every field is a dynamic input.
        {
            osl::ShaderGroup group;
            auto layer = group.add_layer("main", shader.get());
            group.mark_output(layer, "i0");
            group.mark_output(layer, "i1");
            group.mark_output(layer, "i2");
            group.mark_output(layer, "v0");
            group.mark_dynamic_input(layer, "sp");
            DynamicInputTranslator translator;
            translator.set(layer, "sp.a", DynEntry{.f = 4.0f});
            translator.set(layer, "sp.c", DynEntry{.v3 = luisa::make_float3(1.0f, 2.0f, 3.0f)});
            auto res = compile_and_run_with_translator(translator, group, {TestShaderGlobals{}});
            expect(res.translation.succeeded());
            expect(res.shader_compile_ok);
            if (res.outputs.size() == 1u) {
                expect(res.outputs[0].i0 == 1) << "isconnected(sp.a) must be 1";
                expect(res.outputs[0].i1 == 1) << "isconnected(sp.c) must be 1";
                expect(res.outputs[0].i2 == 1) << "isconnected(sp) must be 1 when all fields are dynamic";
            }
        }
    };


    // A parameter that is both fed by an upstream connection and marked as a
    // dynamic input is contradictory: the connection copy is emitted after the
    // dynamic-input store, so the host value is silently discarded (observed before
    // the fix: the hook was called with 100.0 and f0 = 2 + 1 = 3). Reject it, also
    // when a struct root and one of its fields overlap.
    static constexpr auto pair_src_oso = R"oso(
OpenShadingLanguage 1.00
shader review_pair_src
oparam	struct Pair	out_p		%structfields{a,c}
oparam	float	out_p.a	1	%mystruct{out_p} %mystructfield{0}
oparam	color	out_p.c	1 0 0	%mystruct{out_p} %mystructfield{1}
code ___main___
end
)oso";
    static constexpr auto pair_dst_oso = R"oso(
OpenShadingLanguage 1.00
shader review_pair_dst
param	struct Pair	sp		%structfields{a,c}
param	float	sp.a	0	%mystruct{sp} %mystructfield{0}
param	color	sp.c	0 0 0	%mystruct{sp} %mystructfield{1}
oparam	float	f0	0
code ___main___
	assign		f0 sp.a
end
)oso";

    "dynamic_input_conflicts_with_connection"_test = [] {
        auto expect_rejected = [](const char *src_oso, const char *dst_oso,
                                  luisa::string_view src_param, luisa::string_view dst_param,
                                  luisa::string_view dynamic_param) {
            auto s_src = parse_oso_source(src_oso);
            auto s_dst = parse_oso_source(dst_oso);
            expect(s_src != nullptr && s_dst != nullptr);
            if (!s_src || !s_dst) { return; }
            osl::ShaderGroup group;
            auto idx_src = group.add_layer("src", s_src.get());
            auto idx_dst = group.add_layer("dst", s_dst.get());
            group.connect_shaders(idx_src, src_param, idx_dst, dst_param);
            group.mark_output(idx_dst, "f0");
            group.mark_dynamic_input(idx_dst, dynamic_param);
            DynamicInputTranslator translator;
            // A struct dynamic input is fanned out to its fields, so supply the
            // field values as well as the requested name.
            translator.set(idx_dst, dynamic_param, DynEntry{.f = 100.0f});
            translator.set(idx_dst, luisa::format("{}.a", dynamic_param), DynEntry{.f = 100.0f});
            translator.set(idx_dst, luisa::format("{}.c", dynamic_param), DynEntry{.v3 = luisa::make_float3(100.0f)});
            auto res = translate_only(translator, group);
            auto joined = luisa::string{};
            for (const auto &e : res.errors) {
                joined.append(luisa::format("[{}] ", e.message));
            }
            LUISA_INFO("connected '{}' + dynamic '{}': {}", dst_param, dynamic_param, joined);
            expect(!res.succeeded())
                << luisa::format("connected '{}' + dynamic '{}' must fail closed (errors: {})",
                                 dst_param, dynamic_param, joined);
        };
        // 1. Scalar parameter connected and marked dynamic.
        expect_rejected(src_layer_oso, dst_layer_oso, "out_s", "in_s", "in_s");
        // 2. Whole struct parameter connected, one of its fields marked dynamic.
        expect_rejected(pair_src_oso, pair_dst_oso, "out_p", "sp", "sp.a");
        // 3. Whole struct parameter marked dynamic, one of its fields connected.
        expect_rejected(pair_src_oso, pair_dst_oso, "out_p.a", "sp.a", "sp");
    };

    return 0;
}
