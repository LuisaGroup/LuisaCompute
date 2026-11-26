#include "ut/ut.hpp"
#include "test_osl_common.h"

#include <cmath>

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

    "constant_string_folding_and_builtins"_test = [] {
        static constexpr auto oso = R"oso(
OpenShadingLanguage 1.00
shader str_fold_test
oparam	int	i0	0
oparam	int	i1	0
oparam	int	i2	0
oparam	int	i3	0
oparam	float	f0	0
local	string	s1
local	string	s2
local	int	t_sw
local	int	t_eq
local	int	t_i
const	string	$p1	"foo_"
const	string	$p2	"bar_%02d"
const	string	$num_s	"42"
const	string	$flt_s	"3.5"
const	string	$pref	"foo"
const	string	$suf	"07"
const	string	$bar	"bar"
const	int	$c7	7
const	int	$c4	4
const	int	$c3	3
code ___main___
	format		s1 $p2 $c7
	concat		s2 $p1 s1
	strlen		i0 s2
	startswith	t_sw s2 $pref
	endswith	t_i s2 $suf
	substr		s1 s2 $c4 $c3
	eq		t_eq s1 $bar
	add		i2 t_sw t_i
	add		i2 i2 t_eq
	stoi		i3 $num_s
	stof		f0 $flt_s
	getchar		i1 s2 $c4
end
)oso";
        auto res = compile_and_run_single_oso(oso, {{}});
        expect(res.translation.succeeded());
        expect(res.translation.string_switch_count == 0u);// All statically folded!
        expect(res.ast_ok);
        expect(res.shader_compile_ok);
        if (res.outputs.empty()) { return; }
        const auto &o = res.outputs[0];
        // "foo_bar_07" -> len 10; starts_with("foo")=1, ends_with("07")=1, substr(4,3)=="bar" -> i2=3
        expect(o.i0 == 10);
        expect(o.i1 == static_cast<int32_t>('b'));
        expect(o.i2 == 3);
        expect(o.i3 == 42);
        expect(near_f(o.f0, 3.5f));
    };

    "dynamic_string_branch_switch_expansion_positional_and_named_opts"_test = [] {
        // Test runtime branches creating bounded string sets fed into:
        // 1) transform(space, "common", P) -> positional dynamic string
        // 2) texture(fname, u, v, "wrap", wrap_opt, "alpha", alpha_out) -> positional + named option dynamic string!
        // 3) getattribute(attr, ...) -> dynamic attribute name
        static constexpr auto oso = R"oso(
OpenShadingLanguage 1.00
shader dyn_str_hooks
oparam	point	v0	0 0 0
oparam	color	v1	0 0 0
oparam	float	f0	0
oparam	float	f1	0
oparam	int	i0	0
global	point	P
global	float	u
global	float	v
local	string	space_s
local	string	tex_s
local	string	wrap_s
local	string	attr_s
local	int	cond_u
local	int	cond_v
const	float	$half	0.5
const	string	$obj	"object"
const	string	$shd	"shader"
const	string	$com	"common"
const	string	$ramp	"ramp.tx"
const	string	$chk	"checker.tx"
const	string	$clamp	"clamp"
const	string	$black	"black"
const	string	$kwrap	"wrap"
const	string	$kalpha	"alpha"
const	string	$a_obj	"custom:scale"
const	string	$a_idx	"object:index"
code ___main___
	lt		cond_u u $half
	if		cond_u 4 5
	assign		space_s $obj
	jump		5
	assign		space_s $shd
	transform	v0 space_s $com P
	lt		cond_v v $half
	if		cond_v 11 13
	assign		tex_s $ramp
	assign		wrap_s $clamp
	jump		13
	assign		tex_s $ramp
	assign		wrap_s $black
	texture		v1 tex_s u v $kwrap wrap_s $kalpha f0 	%argrw{"wrrrrrrw"}
	if		cond_u 18 20
	assign		attr_s $a_obj
	getattribute	i0 attr_s f1
	jump		20
	assign		attr_s $a_idx
	getattribute	i0 attr_s f1
end
)oso";
        TestShaderGlobals g0;
        g0.P = luisa::make_float3(1.0f, 2.0f, 3.0f);
        g0.u = 0.2f;
        g0.v = 0.25f;// object space, wrap="clamp", in bounds, custom:scale

        TestShaderGlobals g1;
        g1.P = luisa::make_float3(1.0f, 2.0f, 3.0f);
        g1.u = 1.5f;// outside [0, 1]!
        g1.v = 0.75f;// shader space, wrap="black" -> out of bounds black (0,0,0), alpha=0!

        auto res = compile_and_run_single_oso(oso, {g0, g1});
        expect(res.translation.succeeded());
        expect(res.translation.string_switch_count >= 2u);
        expect(res.ast_ok);
        expect(res.shader_compile_ok);
        if (res.outputs.size() < 2u) { return; }

        // Lane 0: object->common on (1,2,3) = (12, 1.5, 14)
        expect(near_v3(res.outputs[0].v0, luisa::float3{12.0f, 1.5f, 14.0f}));
        // ramp.tx at (0.2, 0.25): r=0.1625, g=0.2375, b=0.045, alpha=0.05
        expect(near_v3(res.outputs[0].v1, luisa::float3{0.1625f, 0.2375f, 0.045f}));
        expect(near_f(res.outputs[0].f0, 0.05f));
        expect(res.outputs[0].i0 == 1);
        expect(near_f(res.outputs[0].f1, 2.25f));

        // Lane 1: shader->common on (1,2,3): 0.5*P + (0,2,0) = (0.5, 3.0, 1.5)
        expect(near_v3(res.outputs[1].v0, luisa::float3{0.5f, 3.0f, 1.5f}));
        // wrap="black" with u=1.5 -> (0,0,0), alpha=0
        expect(near_v3(res.outputs[1].v1, luisa::float3{0.0f, 0.0f, 0.0f}));
        expect(near_f(res.outputs[1].f0, 0.0f));
        expect(res.outputs[1].i0 == 1);
        // tid=1 -> object:index = 101.0f
        expect(near_f(res.outputs[1].f1, 101.0f));
    };

    "trace_gettextureinfo_raytype_and_backfacing"_test = [] {
        static constexpr auto oso = R"oso(
OpenShadingLanguage 1.00
shader scene_queries_test
oparam	int	i0	0
oparam	int	i1	0
oparam	int	i2	0
oparam	int	i3	0
oparam	float	f0	0
oparam	float	f1	0
oparam	point	v0	0 0 0
oparam	normal	v1	0 0 0
global	point	P
global	vector	I
local	float	keep_f
local	int	res_ch
const	string	$ramp	"ramp.tx"
const	string	$exists	"exists"
const	string	$chans	"channels"
const	string	$reso	"resolution"
const	string	$missing	"does_not_exist.tx"
const	string	$tr_src	"trace"
const	string	$tr_dist	"hitdist"
const	string	$tr_p	"P"
const	string	$tr_n	"N"
const	string	$rt_cam	"camera"
const	string	$rt_shd	"shadow"
const	string	$kmax	"maxdist"
const	float	$fmax	20.0
const	float	$fseed	77.0
code ___main___
	trace		i0 P I $kmax $fmax
	getmessage	i1 $tr_src $tr_dist f0
	getmessage	i2 $tr_src $tr_p v0
	getmessage	i3 $tr_src $tr_n v1
	assign		f1 $fseed
	gettextureinfo	i2 $missing $exists f1
	gettextureinfo	i2 $ramp $reso f1
	gettextureinfo	i2 $ramp $chans res_ch
	raytype		i3 $rt_cam
	backfacing	i1
	add		i1 i1 res_ch
end
)oso";
        TestShaderGlobals g0;
        g0.P = luisa::make_float3(1.0f, 2.0f, 5.0f);
        g0.I = luisa::make_float3(0.0f, 0.0f, -1.0f);// hits z=0 at t=5, p=(1,2,0)
        g0.ray_type_mask = 1u;// camera=1
        g0.backfacing_flag = 1;

        TestShaderGlobals g1;
        g1.P = luisa::make_float3(1.0f, 2.0f, 5.0f);
        g1.I = luisa::make_float3(0.0f, 0.0f, 1.0f);// shoots away (t=-5, miss)
        g1.ray_type_mask = 2u;// shadow -> camera=0
        g1.backfacing_flag = 0;

        auto res = compile_and_run_single_oso(oso, {g0, g1});
        expect(res.translation.succeeded());
        expect(res.ast_ok);
        expect(res.shader_compile_ok);
        if (res.outputs.size() < 2u) { return; }

        expect(res.outputs[0].i0 == 1);
        expect(near_f(res.outputs[0].f0, 5.0f));
        expect(near_v3(res.outputs[0].v0, luisa::float3{1.0f, 2.0f, 0.0f}));
        expect(near_v3(res.outputs[0].v1, luisa::float3{0.0f, 0.0f, 1.0f}));
        expect(near_f(res.outputs[0].f1, 512.0f));
        expect(res.outputs[0].i1 == 5);// backfacing(1) + channels(4)
        expect(res.outputs[0].i2 == 1);// gettextureinfo ok
        expect(res.outputs[0].i3 == 1);// raytype("camera") == 1

        expect(res.outputs[1].i0 == 0);// missed
        expect(near_f(res.outputs[1].f0, 0.0f));
        expect(res.outputs[1].i1 == 4);// backfacing(0) + channels(4)
        expect(res.outputs[1].i3 == 0);// raytype("camera") == 0
    };

    "extended_string_ops_format_and_3arg_transform"_test = [] {
        static constexpr auto oso = R"oso(
OpenShadingLanguage 1.00
shader ext_str_ops
oparam	int	i0	0
oparam	int	i1	0
oparam	int	i2	0
oparam	int	i3	0
oparam	point	v0	0 0 0
global	float	u
global	point	P
local	string	s_sub1
local	string	s_sub2
local	string	s_dyn
local	string	s_fmt
local	string[4]	s_arr
local	point	pt_p
local	int	flag
local	int	idx
local	int	n_sp1
local	int	n_sp2
local	int	n_sp3
local	int	ch_short
local	int	ch_long
local	int	eq_tmp
local	string	w0
local	string	w2
local	string	c1
const	string	$abcdef	"abcdef"
const	string	$exp_sub1	"bcde"
const	string	$exp_sub2	"cde"
const	string	$alpha	"alpha"
const	string	$zeta	"zeta"
const	string	$mu	"mu"
const	string	$short_s	"mesh0"
const	string	$long_s	"long_material_name_0123"
const	string	$ws_in	"  foo\tbar  baz "
const	string	$csv_in	"a,b,c"
const	string	$colon_in	"1:2:3:4:5"
const	string	$comma	","
const	string	$colon	":"
const	string	$exp_foo	"foo"
const	string	$exp_baz	"baz"
const	string	$exp_bc	"b,c"
const	string	$lit_ok	"static_ok"
const	string	$obj	"object"
const	int	$c0	0
const	int	$c1	1
const	int	$c2	2
const	int	$cn1	-1
const	int	$cn4	-4
const	int	$c5	5
const	int	$c22	22
const	int	$c23	23
const	float	$fhalf	0.5
const	float	$f5	5.0
code ___main___
	substr		s_sub1 $abcdef $c1 $cn1
	substr		s_sub2 $abcdef $cn4 $cn1
	eq		i0 s_sub1 $exp_sub1
	eq		eq_tmp s_sub2 $exp_sub2
	add		i0 i0 eq_tmp
	lt		flag u $fhalf
	if		flag 8 9
	assign		s_dyn $alpha
	assign		s_dyn $zeta
	lt		i1 s_dyn $mu
	mul		pt_p P $f5
	lt		flag u $fhalf
	if		flag 14 15
	assign		idx $c0
	assign		idx $c5
	getchar		ch_short $short_s idx
	lt		flag u $fhalf
	if		flag 19 20
	assign		idx $c22
	assign		idx $c23
	getchar		ch_long $long_s idx
	split		n_sp1 $ws_in s_arr
	aref		w0 s_arr $c0
	aref		w2 s_arr $c2
	eq		eq_tmp w0 $exp_foo
	add		i2 n_sp1 eq_tmp
	eq		eq_tmp w2 $exp_baz
	add		i2 i2 eq_tmp
	split		n_sp2 $csv_in s_arr $comma $c1
	aref		c1 s_arr $c1
	eq		eq_tmp c1 $exp_bc
	add		i2 i2 n_sp2
	add		i2 i2 eq_tmp
	split		n_sp3 $colon_in s_arr $colon
	add		i2 i2 n_sp3
	format		s_fmt $lit_ok u
	eq		i3 s_fmt $lit_ok
	add		ch_short ch_short ch_long
	add		i3 i3 ch_short
	transform	v0 $obj pt_p
end
)oso";
        TestShaderGlobals g0;
        g0.u = 0.25f;// true branches: s_dyn="alpha" (<"mu"=>1), idx=0 ('m'=109), idx=22 ('3'=51)
        g0.P = luisa::make_float3(12.0f, 1.5f, 14.0f) / 5.0f;// becomes (12, 1.5, 14) -> inverse object_to_common gives (1, 2, 3)

        TestShaderGlobals g1;
        g1.u = 0.75f;// false branches: s_dyn="zeta" (<"mu"=>0), idx=5 (0), idx=23 (0)
        g1.P = g0.P;

        auto res = compile_and_run_single_oso(oso, {g0, g1});
        expect(res.translation.succeeded());
        expect(res.ast_ok);
        expect(res.shader_compile_ok);
        if (res.outputs.size() < 2u) { return; }

        // g0 checks
        expect(res.outputs[0].i0 == 2); // negative-len substr x2
        expect(res.outputs[0].i1 == 1); // "alpha" < "mu"
        expect(res.outputs[0].i2 == 12);// split checks: (3+1+1) + (2+1) + 4 = 12
        expect(res.outputs[0].i3 == 1 + 109 + 51);// format + short/long dynamic getchar
        expect(near_v3(res.outputs[0].v0, luisa::float3{1.0f, 2.0f, 3.0f}, 1e-4f));

        // g1 checks
        expect(res.outputs[1].i0 == 2);
        expect(res.outputs[1].i1 == 0);// "zeta" < "mu" == 0
        expect(res.outputs[1].i2 == 12);
        expect(res.outputs[1].i3 == 1);// OOB getchar -> 0
        expect(near_v3(res.outputs[1].v0, luisa::float3{1.0f, 2.0f, 3.0f}, 1e-4f));
    };

    "custom_intern_string_and_message_string_propagation"_test = [] {
        static constexpr auto l0_oso = R"oso(
OpenShadingLanguage 1.00
shader msg_sender
const	string	$mk	"cross_space"
const	string	$mv	"camera"
code ___main___
	setmessage	$mk $mv
end
)oso";
        static constexpr auto l1_oso = R"oso(
OpenShadingLanguage 1.00
shader msg_consumer
oparam	point	v0	0 0 0
oparam	point	v1	0 0 0
oparam	string	i0	""
global	float	u
global	point	P
local	string	intra_sp
local	string	cross_sp
local	string	dyn_sp
local	int	flag
local	int	ok
const	string	$intra_k	"intra_space"
const	string	$cross_k	"cross_space"
const	string	$obj	"object"
const	string	$shd	"shader"
const	string	$com	"common"
const	float	$half	0.5
code ___main___
	lt		flag u $half
	if		flag 3 4
	assign		dyn_sp $obj
	assign		dyn_sp $shd
	setmessage	$intra_k dyn_sp
	getmessage	ok $intra_k intra_sp
	transform	v0 intra_sp $com P
	getmessage	ok $cross_k cross_sp
	transform	v1 cross_sp $com P
	assign		i0 intra_sp
end
)oso";
        class CustomInternTranslator final : public TestOSLTranslator {
        public:
            [[nodiscard]] uint32_t intern_string(luisa::string_view sv) noexcept override {
                uint32_t h = 2166136261u;
                for (char c : sv) {
                    h ^= static_cast<uint8_t>(c);
                    h *= 16777619u;
                }
                return 0x81000000u | (h & 0x00ffffffu);
            }
        };

        auto s0 = parse_oso_source(l0_oso);
        auto s1 = parse_oso_source(l1_oso);
        expect(s0 != nullptr && s1 != nullptr);

        osl::ShaderGroup g;
        static_cast<void>(g.add_layer("sender", s0.get()));
        uint32_t c_idx = g.add_layer("consumer", s1.get());
        for (auto name : {"v0", "v1", "i0"}) {
            g.mark_output(c_idx, name, false);
        }

        CustomInternTranslator tr;
        TestShaderGlobals g0;
        g0.u = 0.25f;// object -> common
        g0.P = luisa::make_float3(1.0f, 2.0f, 3.0f);
        TestShaderGlobals g1;
        g1.u = 0.75f;// shader -> common
        g1.P = luisa::make_float3(2.0f, 4.0f, 6.0f);

        auto res = compile_and_run_with_translator(tr, g, {g0, g1});
        expect(res.translation.succeeded());
        expect(res.ast_ok);
        expect(res.shader_compile_ok);
        if (res.outputs.size() < 2u) { return; }

        // v0: object -> common on g0, shader -> common on g1
        expect(near_v3(res.outputs[0].v0, luisa::float3{12.0f, 1.5f, 14.0f}, 1e-4f));
        expect(near_v3(res.outputs[1].v0, luisa::float3{1.0f, 4.0f, 3.0f}, 1e-4f));
        // v1: camera -> common (x, y, -z + 5)
        expect(near_v3(res.outputs[0].v1, luisa::float3{1.0f, 2.0f, 2.0f}, 1e-4f));
        expect(near_v3(res.outputs[1].v1, luisa::float3{2.0f, 4.0f, -1.0f}, 1e-4f));
        // i0 holds custom intern_string token cast to int32
        uint32_t exp_tok0 = tr.intern_string("object");
        uint32_t exp_tok1 = tr.intern_string("shader");
        expect(static_cast<uint32_t>(res.outputs[0].i0) == exp_tok0);
        expect(static_cast<uint32_t>(res.outputs[1].i0) == exp_tok1);
    };

    return 0;
}
