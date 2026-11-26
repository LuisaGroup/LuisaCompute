#include "ut/ut.hpp"
#include "test_osl_common.h"

#include <cmath>
#include <cstdlib>
#include <filesystem>

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
}// namespace

int main(int argc, char *argv[]) {
    set_osl_test_args(argc, argv);

    "multi_layer_group_dead_layer_pruning_messages_and_closures"_test = [] {
        // Layer 0 (generator): outputs scalar factor & color, sets intra-group message "shared_gain"
        static constexpr auto gen_oso = R"oso(
OpenShadingLanguage 1.00
shader layer_gen
param	float	base_scale	1.0
oparam	float	out_s	0
oparam	color	out_c	0 0 0
global	float	u
local	float	msg_val
const	float	$c2	2.0
const	string	$msg_k	"shared_gain"
code ___main___
	mul		out_s u base_scale
	mul		msg_val base_scale $c2
	setmessage	$msg_k msg_val
	assign		out_c out_s
end
)oso";
        // Layer 1 (dead_unused): connected to nothing requested
        static constexpr auto dead_oso = R"oso(
OpenShadingLanguage 1.00
shader layer_dead
oparam	float	dead_out	0
const	float	$c9	999.0
code ___main___
	assign		dead_out $c9
end
)oso";
        // Layer 2 (surface): receives generator outputs, reads "shared_gain" message,
        // checks isconnected, branches on closures, and writes Ci + diagnostics outputs!
        static constexpr auto surf_oso = R"oso(
OpenShadingLanguage 1.00
surface layer_surf
param	color	in_col	1 1 1
param	float	unconnected_p	7.0
oparam	float	f0	0
oparam	float	f1	0
oparam	float	f2	0
oparam	int	i0	0
oparam	int	i1	0
oparam	color	v0	0 0 0
global	normal	N
global	closure color	Ci
local	closure color	c_diff
local	closure color	c_emit
local	closure color	c_zero
local	float	dx_in
local	float	got_gain
local	int	cond_b
const	string	$msg_k	"shared_gain"
const	string	$s_diff	"diffuse"
const	string	$s_emit	"emission"
const	color	$em_w	0.2 0.3 0.4
const	float	$thresh	1.0
const	float	$fzero	0.0
const	int	$czero	0
code ___main___
	isconnected	i0 in_col
	isconnected	i1 unconnected_p
	getmessage	cond_b $msg_k got_gain
	compref		f0 in_col $czero
	Dx		dx_in f0
	assign		f1 dx_in
	assign		f2 got_gain
	assign		v0 in_col
	closure		c_diff $s_diff N
	closure		c_emit $s_emit
	assign		c_zero $czero
	gt		cond_b f0 $thresh
	if		cond_b 16 19
	mul		c_diff c_diff in_col
	add		Ci c_diff c_zero
	jump		19
	mul		c_emit c_emit $em_w
	mul		c_diff c_diff in_col
	add		Ci c_diff c_emit
end
)oso";

        auto s_gen = parse_oso_source(gen_oso);
        auto s_dead = parse_oso_source(dead_oso);
        auto s_surf = parse_oso_source(surf_oso);
        expect(s_gen != nullptr && s_dead != nullptr && s_surf != nullptr);

        osl::ShaderGroup group;
        uint32_t idx_gen = group.add_layer("gen", s_gen.get());
        static_cast<void>(group.add_layer("dead", s_dead.get()));
        uint32_t idx_surf = group.add_layer("surf", s_surf.get());

        // Override base_scale = 3.0 on gen
        group.set_parameter(idx_gen, "base_scale", 3.0f);
        // Connect gen.out_c -> surf.in_col
        group.connect_shaders(idx_gen, "out_c", idx_surf, "in_col");

        // Request surf's oparams (note `dead` has no requested outputs and no downstream edges!)
        for (auto name : {"f0", "f1", "f2", "i0", "i1", "v0"}) {
            group.mark_output(idx_surf, name, false);
        }

        TestShaderGlobals g0;
        g0.u = 0.25f;
        g0.dudx = 0.2f;
        g0.N = luisa::make_float3(0.0f, 1.0f, 0.0f);// f0 = 0.25*3 = 0.75 <= 1 -> diff*col + emit*$em_w

        TestShaderGlobals g1;
        g1.u = 0.5f;
        g1.dudx = 0.2f;
        g1.N = luisa::make_float3(0.0f, 0.0f, 1.0f);// f0 = 0.5*3 = 1.5 > 1 -> diff*col + zero

        auto res = compile_and_run_osl_group(group, {g0, g1});
        expect(res.translation.succeeded());
        expect(res.translation.active_layer_count == 2u);
        expect(res.translation.eliminated_layer_count == 1u);
        expect(res.ast_ok);
        expect(res.shader_compile_ok);
        if (res.outputs.size() < 2u) { return; }

        const auto &o0 = res.outputs[0];
        const auto &o1 = res.outputs[1];

        // Connected/unconnected
        expect(o0.i0 == 1);
        expect(o0.i1 == 0);
        // Intra-group setmessage("shared_gain", 3.0 * 2.0 = 6.0)
        expect(near_f(o0.f2, 6.0f));
        // Cross-layer out_s = 3.0*u -> dx = 3.0*dudx = 0.6!
        expect(near_f(o0.f0, 0.75f));
        expect(near_f(o0.f1, 0.6f));
        expect(near_v3(o0.v0, luisa::float3{0.75f, 0.75f, 0.75f}));

        // Closure branch 0: diffuse(tag=1) * (0.75) + emission(tag=4) * (0.2, 0.3, 0.4)
        expect(o0.ci.kind_tag == (1u | (4u << 4u)));
        expect(near_v3(o0.ci.weight, luisa::float3{0.95f, 1.05f, 1.15f}));
        expect(near_v3(o0.ci.arg0, g0.N));

        // Closure branch 1: diffuse(tag=1) * 1.5 + zero(tag=0) -> tag stays 1!
        expect(near_f(o1.f0, 1.5f));
        expect(o1.ci.kind_tag == 1u);
        expect(near_v3(o1.ci.weight, luisa::float3{1.5f, 1.5f, 1.5f}));
        expect(near_v3(o1.ci.arg0, g1.N));
    };

    "output_binding_arrays_structs_generic_shader_and_closure_pos_args"_test = [] {
        static constexpr auto generic_oso = R"oso(
OpenShadingLanguage 1.00
shader generic_p_writer
global	point	P
const	vector	$off	1 2 3
code ___main___
	add		P P $off
end
)oso";
        static constexpr auto out_oso = R"oso(
OpenShadingLanguage 1.00
surface struct_arr_closure
param	string	dist	"ggx"
oparam	float[3]	arr_out	0 0 0
oparam	struct Pair	s_out		%structfields{a,c}
oparam	float	s_out.a	0	%mystruct{s_out} %mystructfield{0}
oparam	color	s_out.c	0 0 0	%mystruct{s_out} %mystructfield{1}
oparam	int	i0	0
global	normal	N
global	closure color	Ci
local	closure color	c0
local	closure color	c1
const	int	$i0	0
const	int	$i1	1
const	int	$i2	2
const	float	$f3	3.0
const	float	$f7	7.0
const	float	$f11	11.0
const	float	$r	0.35
const	color	$col	0.2 0.4 0.6
const	string	$micro	"microfacet"
const	string	$custom	"custom_pos_bsdf"
const	string	$ggx	"ggx"
const	int	$c9	9
code ___main___
	aassign		arr_out $i0 $f3
	aassign		arr_out $i1 $f7
	aassign		arr_out $i2 $f11
	assign		s_out.a $f7
	assign		s_out.c $col
	assign		i0 $c9
	closure		c0 $micro $ggx N
	closure		c1 $custom N dist $r
	add		Ci c0 c1
end
)oso";

        class InspectorTranslator final : public TestOSLTranslator {
        public:
            bool saw_microfacet_ok{false};
            bool saw_custom_ok{false};
            bool saw_arr_binding_ok{false};
            bool saw_struct_binding_ok{false};

        protected:
            [[nodiscard]] size_t closure_positional_arg_count(
                luisa::string_view name,
                luisa::span<const osl::Type *const> arg_types) const noexcept override {
                if (name == "custom_pos_bsdf") {
                    return arg_types.size();// all positional (`N`, `dist` string, `roughness`)
                }
                return TestOSLTranslator::closure_positional_arg_count(name, arg_types);
            }

            [[nodiscard]] osl::ClosureValue emit_closure(
                luisa::string_view name,
                luisa::span<const osl::HookArg> args,
                luisa::span<const osl::NamedOption> opts) noexcept override {
                if (name == "microfacet") {
                    saw_microfacet_ok = (args.size() == 2u && opts.empty() &&
                                         args[0].constant.has_value() &&
                                         luisa::holds_alternative<luisa::string_view>(*args[0].constant) &&
                                         luisa::get<luisa::string_view>(*args[0].constant) == "ggx");
                    return TestOSLTranslator::emit_closure(name, args, opts);
                }
                if (name == "custom_pos_bsdf") {
                    saw_custom_ok = (args.size() == 3u && opts.empty() &&
                                     args[1].constant.has_value() &&
                                     luisa::holds_alternative<luisa::string_view>(*args[1].constant) &&
                                     luisa::get<luisa::string_view>(*args[1].constant) == "ggx");
                    return TestOSLTranslator::emit_closure("diffuse", args.subspan(0u, 1u), {});
                }
                return TestOSLTranslator::emit_closure(name, args, opts);
            }

            void on_end_group(const osl::ShaderGroup &group,
                              luisa::span<const osl::OutputBinding> outputs) noexcept override {
                for (const auto &out : outputs) {
                    if (out.field_name == "arr_out") {
                        saw_arr_binding_ok = (out.array_length == 3 && out.elements.size() == 3u);
                    } else if (out.field_name == "s_out") {
                        saw_struct_binding_ok = (out.fields.size() == 2u &&
                                                 out.fields[0].field_name == "s_out.a" &&
                                                 out.fields[1].field_name == "s_out.c");
                    }
                }
                TestOSLTranslator::on_end_group(group, outputs);
            }
        };

        auto sg = parse_oso_source(generic_oso);
        auto so = parse_oso_source(out_oso);
        expect(sg != nullptr && so != nullptr);
        if (!sg || !so) { return; }

        osl::ShaderGroup g;
        static_cast<void>(g.add_layer("gen_writer", sg.get()));
        uint32_t main_li = g.add_layer("surf", so.get());
        g.mark_output(main_li, "arr_out");
        g.mark_output(main_li, "s_out");
        g.mark_output(main_li, "i0");

        InspectorTranslator tr;
        TestShaderGlobals in;
        in.N = luisa::make_float3(0.0f, 0.0f, 1.0f);
        auto res = compile_and_run_with_translator(tr, g, {in});
        expect(res.translation.succeeded());
        expect(res.translation.active_layer_count == 2u);
        expect(res.translation.eliminated_layer_count == 0u);
        expect(tr.saw_microfacet_ok);
        expect(tr.saw_custom_ok);
        expect(tr.saw_arr_binding_ok);
        expect(tr.saw_struct_binding_ok);
        expect(res.ast_ok);
        expect(res.shader_compile_ok);
        if (res.outputs.empty()) { return; }
        const auto &o = res.outputs[0];
        // TestOSLTranslator stores generic writer global P into v1 (field 3) and array/struct into f0..f3, v0
        expect(near_v3(o.v1, luisa::float3{1.0f, 2.0f, 3.0f}));
        expect(near_f(o.f0, 3.0f));
        expect(near_f(o.f1, 7.0f));
        expect(near_f(o.f2, 11.0f));
        expect(near_f(o.f3, 7.0f));
        expect(near_v3(o.v0, luisa::float3{0.2f, 0.4f, 0.6f}));
        expect(o.i0 == 9);
    };

    "real_blender_cycles_25_plus_oso_suite"_test = [] {
        const char *env_dir = std::getenv("LUISA_OSL_CYCLES_SHADER_DIR");
        if (env_dir == nullptr || *env_dir == '\0') {
            return;
        }
        const std::filesystem::path cycles_dir{env_dir};
        if (!std::filesystem::exists(cycles_dir)) {
            return;
        }
        const char *cycles_shaders[] = {
            "node_absorption_volume.oso",
            "node_add_closure.oso",
            "node_background.oso",
            "node_blackbody.oso",
            "node_boolean_math.oso",
            "node_brightness.oso",
            "node_bump.oso",
            "node_camera.oso",
            "node_checker_texture.oso",
            "node_clamp.oso",
            "node_combine_color.oso",
            "node_combine_xyz.oso",
            "node_convert_from_color.oso",
            "node_convert_from_float.oso",
            "node_convert_from_int.oso",
            "node_convert_from_normal.oso",
            "node_convert_from_point.oso",
            "node_convert_from_vector.oso",
            "node_diffuse_bsdf.oso",
            "node_displacement.oso",
            "node_emission.oso",
            "node_fresnel.oso",
            "node_gamma.oso",
            "node_geometry.oso",
            "node_glass_bsdf.oso",
            "node_glossy_bsdf.oso",
            "node_gradient_texture.oso",
            "node_holdout.oso",
            "node_hsv.oso",
            "node_invert.oso",
            "node_layer_weight.oso",
            "node_light_path.oso",
            "node_map_range.oso",
            "node_mapping.oso",
            "node_math.oso",
            "node_metallic_bsdf.oso",
            "node_mix_closure.oso",
            "node_mix_color.oso",
            "node_mix_float.oso",
            "node_noise_texture.oso",
            "node_normal.oso",
            "node_normal_map.oso",
            "node_refraction_bsdf.oso",
            "node_rgb_ramp.oso",
            "node_rgb_to_bw.oso",
            "node_scatter_volume.oso",
            "node_separate_color.oso",
            "node_separate_xyz.oso",
            "node_sheen_bsdf.oso",
            "node_toon_bsdf.oso",
            "node_translucent_bsdf.oso",
            "node_transparent_bsdf.oso",
            "node_value.oso",
            "node_vector_math.oso",
            "node_vector_rotate.oso",
            "node_vector_transform.oso",
            "node_wavelength.oso",
        };
        size_t tested_count = 0u;
        size_t ok_count = 0u;
        for (const char *name : cycles_shaders) {
            auto p = cycles_dir / name;
            if (!std::filesystem::exists(p)) { continue; }
            ++tested_count;
            auto p_str = luisa::string{p.generic_string()};
            auto s = osl::OSOParser::parse_file(p_str);
            expect(s != nullptr) << name;
            if (!s) { continue; }
            osl::ShaderGroup g;
            uint32_t li = g.add_layer("n", s.get());
            for (auto &&sym : s->symbols()) {
                if (sym->tag() == osl::Symbol::Tag::SYM_OUTPUT_PARAM && sym->is_root()) {
                    g.mark_output(li, sym->identifier(), false);
                }
            }
            xir::Module mod;
            auto *fn = mod.create_kernel();
            CyclesSmokeTranslator tr;
            auto res = tr.translate(g, fn, {.verify_xir = true, .run_xir_passes = true, .restructure_for_ast = true});
            bool verified = res.succeeded();
            if (verified) {
                auto post = xir::xir_verify_module(&mod, {.require_no_phi = true, .require_canonical_break_continue_targets = true});
                verified = post.succeeded() && (xir::xir_to_ast_translate(*fn, {.strict = true}) != nullptr);
            }
            if (verified) { ++ok_count; }
            expect(verified) << name << ": "
                             << (res.errors.empty() ? luisa::string{"XIR verify/AST failure"} : res.errors[0].message);
        }
        expect(tested_count >= 25u);
        expect(ok_count == tested_count);
    };

    return 0;
}
