#include "ut/ut.hpp"
#include "test_osl_common.h"

#include <cmath>
#include <cstdlib>
#include <filesystem>

using namespace boost::ut;
using namespace luisa;
using namespace luisa::compute;
using namespace luisa::test::osl_test;

int main(int argc, char *argv[]) {
    set_osl_test_args(argc, argv);

    "for_while_dowhile_break_continue_and_nested_if"_test = [] {
        static constexpr auto oso = R"oso(
OpenShadingLanguage 1.00
shader loops_cfg_test
param	int	n	8
oparam	int	i0	0
oparam	int	i1	0
oparam	int	i2	0
local	int	k
local	int	cond
local	int	acc
const	int	$c0	0
const	int	$c1	1
const	int	$c2	2
const	int	$c6	6
const	int	$c7	7
const	int	$c100	100
code ___main___
	assign		acc $c0
	for		cond 3 4 11 12
	assign		k $c0
	lt		cond k n
	eq		cond k $c2
	if		cond 7 7
	continue
	eq		cond k $c6
	if		cond 10 10
	break
	add		acc acc k
	add		k k $c1
	assign		i0 acc
	assign		acc $c1
	while		cond 15 16 17 17
	lt		cond acc $c100
	mul		acc acc $c2
	assign		i1 acc
	assign		acc $c0
	dowhile		cond 20 21 22 22
	lt		cond acc $c0
	add		acc acc $c7
	assign		i2 acc
end
)oso";
        auto res = compile_and_run_single_oso(oso, {{}});
        expect(res.translation.succeeded());
        expect(res.ast_ok);
        expect(res.shader_compile_ok);
        if (res.outputs.empty()) { return; }
        const auto &o = res.outputs[0];
        // for: 0 + 1 + 3 + 4 + 5 = 13 (k=2 skipped via continue, k=6 broke out)
        expect(o.i0 == 13);
        // while: 1 * 2^7 = 128
        expect(o.i1 == 128);
        // dowhile: executes body once despite cond < 0 initially false -> 7
        expect(o.i2 == 7);
    };

    "functioncall_early_return_exit_and_param_init_ranges"_test = [] {
        static constexpr auto oso = R"oso(
OpenShadingLanguage 1.00
shader call_exit_param_test
param	int	dyn_p	5	%meta{int,init,1}
param	int	x	0
oparam	int	i0	0
oparam	int	i1	0
oparam	int	i2	0
global	float	u
local	int	cond
const	int	$c0	0
const	int	$c5	5
const	int	$c10	10
const	int	$c11	11
const	int	$c77	77
const	int	$c99	99
const	float	$f05	0.5
const	string	$fn_early	"helper_early"
const	string	$fn_exit	"helper_exit"
code dyn_p
	add		dyn_p x $c11
code ___main___
	assign		i2 dyn_p
	functioncall	$fn_early 8
	eq		cond x $c0
	if		cond 7 7
	assign		i0 $c99
	return
	assign		i0 $c10
	functioncall	$fn_exit 12
	gt		cond x $c5
	if		cond 12 12
	exit
	assign		i1 $c77
end
)oso";
        auto shader = parse_oso_source(oso);
        expect(shader != nullptr);

        // Case A: default params (dyn_p init code range runs: dyn_p = x + 11)
        {
            osl::ShaderGroup g;
            uint32_t li = g.add_layer("main", shader.get());
            g.set_parameter(li, "x", int32_t{0});
            g.mark_output(li, "i0");
            g.mark_output(li, "i1");
            g.mark_output(li, "i2");
            auto res = compile_and_run_osl_group(g, {{}});
            expect(res.translation.succeeded());
            expect(res.ast_ok);
            expect(res.shader_compile_ok);
            if (!res.outputs.empty()) {
                expect(res.outputs[0].i0 == 99);// early return inside functioncall
                expect(res.outputs[0].i1 == 77);// did not exit early
                expect(res.outputs[0].i2 == 11);// code dyn_p executed: 0 + 11
            }
        }

        // Case B: x = 10 -> triggers `exit` before i1 = 77; dyn_p overridden skips init code range!
        {
            osl::ShaderGroup g;
            uint32_t li = g.add_layer("main", shader.get());
            g.set_parameter(li, "x", int32_t{10});
            g.set_parameter(li, "dyn_p", int32_t{123});
            g.mark_output(li, "i0");
            g.mark_output(li, "i1");
            g.mark_output(li, "i2");
            auto res = compile_and_run_osl_group(g, {{}});
            expect(res.translation.succeeded());
            expect(res.ast_ok);
            expect(res.shader_compile_ok);
            if (!res.outputs.empty()) {
                expect(res.outputs[0].i0 == 10);// helper_early normal path
                expect(res.outputs[0].i1 == 0);// exit skipped assign i1 77
                expect(res.outputs[0].i2 == 123);// param override skipped `code dyn_p`
            }
        }
    };

    "real_blender_cycles_oso_smoke"_test = [] {
        const char *env_dir = std::getenv("LUISA_OSL_CYCLES_SHADER_DIR");
        if (env_dir == nullptr || *env_dir == '\0') {
            return;
        }
        const std::filesystem::path cycles_dir{env_dir};
        if (!std::filesystem::exists(cycles_dir)) {
            return;
        }
        const char *sample_shaders[] = {
            "node_combine_xyz.oso",
            "node_separate_xyz.oso",
            "node_gamma.oso",
            "node_invert.oso",
            "node_value.oso",
            "node_diffuse_bsdf.oso",
            "node_emission.oso",
        };
        for (const char *name : sample_shaders) {
            auto p = cycles_dir / name;
            if (!std::filesystem::exists(p)) { continue; }
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
            TestOSLTranslator tr;
            auto res = tr.translate(g, fn, {.verify_xir = true, .run_xir_passes = true, .restructure_for_ast = true});
            expect(res.succeeded()) << name << ": "
                                    << (res.errors.empty() ? luisa::string{} : res.errors[0].message);
        }
    };

    "exit_inside_param_init_and_non_argrw_sincos_kill"_test = [] {
        // Param init `code dyn_p` conditionally exits whole layer when u < 0.5;
        // also non-%argrw `sincos` must kill both output symbols s_v and c_v in constant analysis.
        static constexpr auto cond_oso = R"oso(
OpenShadingLanguage 1.00
shader init_exit_and_sincos
param	int	dyn_p	5		%meta{int,init,1}
oparam	int	i0	0
oparam	int	i1	0
oparam	float	f0	0
oparam	float	f1	0
global	float	u
local	int	flag
local	float	s_v
local	float	c_v
const	float	$fzero	0.0
const	float	$fhalf	0.5
const	float	$fone	1.0
const	float	$fang	1.57079637
const	int	$c10	10
const	int	$c20	20
code dyn_p
	lt		flag u $fhalf
	if		flag 3 3
	exit
	assign		dyn_p $c10
code ___main___
	assign		i0 $c20
	assign		i1 dyn_p
	assign		s_v $fone
	assign		c_v $fone
	sincos		$fang s_v c_v
	gt		flag c_v $fhalf
	if		flag 13 15
	assign		f0 $fone
	jump		15
	assign		f0 $fzero
	nop
	assign		f1 s_v
end
)oso";
        TestShaderGlobals g_exit;
        g_exit.u = 0.25f;// triggers `exit` inside `code dyn_p`
        TestShaderGlobals g_run;
        g_run.u = 0.75f;// falls through into `___main___`

        auto res = compile_and_run_single_oso(cond_oso, {g_exit, g_run});
        expect(res.translation.succeeded());
        expect(res.ast_ok);
        expect(res.shader_compile_ok);
        if (res.outputs.size() >= 2u) {
            expect(res.outputs[0].i0 == 0);
            expect(res.outputs[0].i1 == 0);
            expect(res.outputs[1].i0 == 20);
            expect(res.outputs[1].i1 == 10);
            // cos(pi/2) ~= 0 < 0.5 -> false branch f0 == 0.0 (not stale 1.0), sin(pi/2) ~= 1.0
            expect(std::abs(res.outputs[1].f0 - 0.0f) <= 1e-4f);
            expect(std::abs(res.outputs[1].f1 - 1.0f) <= 1e-4f);
        }

        // Unconditional `exit` in param init completely skips `___main___`
        static constexpr auto uncond_oso = R"oso(
OpenShadingLanguage 1.00
shader uncond_init_exit
param	int	dyn_p	5		%meta{int,init,1}
oparam	int	i0	0
const	int	$c99	99
code dyn_p
	exit
code ___main___
	assign		i0 $c99
end
)oso";
        auto res2 = compile_and_run_single_oso(uncond_oso, {{}});
        expect(res2.translation.succeeded());
        expect(res2.ast_ok);
        expect(res2.shader_compile_ok);
        if (!res2.outputs.empty()) {
            expect(res2.outputs[0].i0 == 0);
        }
    };

    return 0;
}
