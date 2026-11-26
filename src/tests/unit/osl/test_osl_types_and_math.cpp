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

    "types_structs_arrays_and_bitwise"_test = [] {
        static constexpr auto clean_oso = R"oso(
OpenShadingLanguage 1.00
shader types_test
param	int[4]	iarr	10 20 30 40
oparam	int	i0	0
oparam	int	i1	0
oparam	int	i2	0
oparam	int	i3	0
oparam	float	f0	0
oparam	float	f1	0
oparam	vector	v0	0 0 0
local	struct Pair	p	%structfields{a,c}
local	float	p.a	%mystruct{p} %mystructfield{0}
local	color	p.c	%mystruct{p} %mystructfield{1}
local	float[3]	farr
local	int	t1
local	int	t2
const	int	$c0	0
const	int	$c1	1
const	int	$c2	2
const	int	$c3	3
const	int	$c99	99
const	int	$cn5	-5
const	int	$c12	12
const	int	$c10	10
const	int	$c7	7
const	float	$cf25	2.5
const	color	$ccol	0.25 0.5 0.75
const	float	$cf9	9.0
const	float	$cfn3	-3.0
code ___main___
	assign		p.a $cf25
	assign		p.c $ccol
	compassign	p.c $c1 $cf25
	aassign		farr $c0 p.a
	aassign		farr $c1 $cfn3
	aassign		farr $c2 $cf9
	compref		f0 p.c $c1
	arraylength	i0 iarr
	aref		i1 iarr $c99
	aref		t1 iarr $cn5
	add		i2 i1 t1
	aassign		farr $c99 $cf25
	aref		f1 farr $c2
	bitand		t1 $c12 $c10
	bitor		t2 $c12 $c10
	xor		i3 t1 t2
	shr		t1 $c12 $c1
	sub		i3 i3 t1
	add		i3 i3 $c7
	assign		v0 p.c
end
)oso";
        luisa::vector<TestShaderGlobals> inputs(1);
        auto res = compile_and_run_single_oso(clean_oso, inputs);
        expect(res.translation.succeeded());
        expect(res.ast_ok);
        expect(res.shader_compile_ok);
        if (res.outputs.empty()) { return; }
        const auto &o = res.outputs[0];
        expect(o.i0 == 4);
        expect(o.i1 == 40);
        expect(o.i2 == 50);// clamped 99->40, -5->10
        // (12 & 10)=8, (12 | 10)=14, 8^14=6, (12 >> 1)=6, 6 - 6 + 7 = 7
        expect(o.i3 == 7);
        expect(near_f(o.f0, 2.5f));
        expect(near_f(o.f1, 2.5f));// clamped index 99 wrote to farr[2]
        expect(near_v3(o.v0, luisa::float3{0.25f, 2.5f, 0.75f}));
    };

    "math_transcendentals_and_portable_ops"_test = [] {
        static constexpr auto oso = R"oso(
OpenShadingLanguage 1.00
shader math_test
oparam	float	f0	0
oparam	float	f1	0
oparam	float	f2	0
oparam	float	f3	0
oparam	float	f4	0
oparam	float	f5	0
oparam	float	f6	0
oparam	float	f7	0
oparam	vector	v0	0 0 0
oparam	vector	v1	0 0 0
oparam	int	i0	0
oparam	int	i1	0
global	float	u
global	float	v
local	float	s
local	float	c
const	int	$ci0	0
const	int	$ci_n7	-7
const	float	$cf_neg27	-27.0
const	float	$cf_small	1.0e-5
const	float	$cf_one	1.0
const	float	$cf_zero	0.0
const	float	$cf_n23	-2.3
const	vector	$cv1	1.0 2.0 3.0
const	vector	$cv2	4.0 5.0 6.0
const	vector	$cv_cond	0.0 1.0 0.0
code ___main___
	erf		f0 u
	erfc		f1 u
	expm1		f2 $cf_small
	cbrt		f3 $cf_neg27
	logb		f4 v
	sincos		u s c
	add		f5 s c
	mod		f6 $cf_n23 $cf_one
	fmod		f7 $cf_n23 $cf_one
	mod		i0 $ci_n7 $ci0
	div		i1 $ci_n7 $ci0
	select		v0 $cv1 $cv2 $cv_cond
	smoothstep	v1 $cf_zero $cf_one u
end
)oso";
        TestShaderGlobals g;
        g.u = 0.6f;
        g.v = 16.0f;
        auto res = compile_and_run_single_oso(oso, {g});
        expect(res.translation.succeeded());
        expect(res.ast_ok);
        expect(res.shader_compile_ok);
        if (res.outputs.empty()) { return; }
        const auto &o = res.outputs[0];
        expect(near_f(o.f0, std::erf(0.6f), 1e-5f));
        expect(near_f(o.f1, std::erfc(0.6f), 1e-5f));
        expect(near_f(o.f2, std::expm1(1.0e-5f), 1e-8f));
        expect(near_f(o.f3, -3.0f, 1e-4f));
        expect(near_f(o.f4, 4.0f, 1e-4f));
        expect(near_f(o.f5, std::sin(0.6f) + std::cos(0.6f), 1e-4f));
        // mod(-2.3, 1.0) = -2.3 - 1.0*floor(-2.3) = 0.7; fmod(-2.3, 1.0) = -0.3
        expect(near_f(o.f6, 0.7f, 1e-4f));
        expect(near_f(o.f7, -0.3f, 1e-4f));
        expect(o.i0 == 0);
        expect(o.i1 == 0);
        // componentwise select: cond=(0,1,0) -> (1, 5, 3)
        expect(near_v3(o.v0, luisa::float3{1.0f, 5.0f, 3.0f}));
        float sm = 0.6f * 0.6f * (3.0f - 2.0f * 0.6f);
        expect(near_v3(o.v1, luisa::float3{sm, sm, sm}));
    };

    "matrices_and_space_transform_semantics"_test = [] {
        static constexpr auto oso = R"oso(
OpenShadingLanguage 1.00
shader matrix_space_test
oparam	matrix	m0	1
oparam	point	v0	0 0 0
oparam	vector	v1	0 0 0
oparam	normal	v2	0 0 0
oparam	point	v3	0 0 0
oparam	float	f0	0
oparam	float	f1	0
oparam	float	f2	0
global	point	P
local	matrix	m_a
local	matrix	m_b
local	int	ok
const	string	$cs_obj	"object"
const	string	$cs_com	"common"
const	int	$c1	1
const	int	$c2	2
const	int	$c3	3
const	float	$f0	0.0
const	float	$f1	1.0
const	float	$f2	2.0
const	float	$f3	3.0
const	float	$f4	4.0
const	float	$f9	9.0
const	normal	$n_y	0 1 0
code ___main___
	transform	v0 $cs_obj $cs_com P
	transformv	v1 $cs_obj $cs_com P
	transformn	v2 $cs_obj $cs_com $n_y
	getmatrix	ok $cs_obj $cs_com m_a
	transform	v3 m_a P
	matrix		m_b $f2 $f0 $f0 $f0  $f0 $f3 $f0 $f0  $f0 $f0 $f4 $f0  $f1 $f2 $f3 $f1
	mxcompref	f0 m_b $c3 $c1
	mxcompassign	m_b $c3 $c2 $f9
	mxcompref	f1 m_b $c3 $c2
	transpose	m0 m_b
	determinant	f2 m_b
end
)oso";
        TestShaderGlobals g;
        g.P = luisa::make_float3(1.0f, 2.0f, 3.0f);
        auto res = compile_and_run_single_oso(oso, {g});
        expect(res.translation.succeeded());
        expect(res.ast_ok);
        expect(res.shader_compile_ok);
        if (res.outputs.empty()) { return; }
        const auto &o = res.outputs[0];
        // M_obj->com: [2 0.5 0 0; 0 3 0 0; 0 0 4 0; 10 -5 2 1]
        // point (1,2,3) -> (1*2 + 10, 1*0.5 + 2*3 - 5, 3*4 + 2) = (12, 1.5, 14)
        expect(near_v3(o.v0, luisa::float3{12.0f, 1.5f, 14.0f}));
        // vector (1,2,3) -> (2, 6.5, 12) (no translation)
        expect(near_v3(o.v1, luisa::float3{2.0f, 6.5f, 12.0f}));
        // normal (0,1,0) -> M3^{-1}*(0,1,0)^T = (-0.5/6, 1/3, 0) = (-1/12, 1/3, 0)
        expect(near_v3(o.v2, luisa::float3{-1.0f / 12.0f, 1.0f / 3.0f, 0.0f}, 1e-4f));
        expect(near_v3(o.v3, o.v0));
        expect(near_f(o.f0, 2.0f));// m_b[3][1]
        expect(near_f(o.f1, 9.0f));// mutated m_b[3][2]
        expect(near_f(o.f2, 24.0f));// det(diag(2,3,4,1))
        // m0 = transpose(m_b), where L_orig had row 3=(1,2,9,1) as col 3;
        // L_{M^T} = (M^T)^T = M, so col 3 of o.m0 contains col 3 of OSL M = (0,0,0,1), and row 3 = (1,2,9,1) -> m0[2][3] == 9
        expect(near_f(o.m0[2][3], 9.0f));
    };

    "color_transforms_and_default_noise"_test = [] {
        static constexpr auto oso = R"oso(
OpenShadingLanguage 1.00
shader color_noise_test
oparam	color	v0	0 0 0
oparam	color	v1	0 0 0
oparam	color	v2	0 0 0
oparam	float	f0	0
oparam	float	f1	0
oparam	float	f2	0
oparam	float	f3	0
oparam	float	f4	0
global	point	P
local	color	tmp_c
const	string	$rgb	"rgb"
const	string	$hsv	"hsv"
const	string	$xyz	"XYZ"
const	string	$perlin	"perlin"
const	string	$simplex	"simplex"
const	color	$col	0.8 0.3 0.2
const	point	$pperiod	4.0 4.0 4.0
const	float	$bbtemp	6500.0
code ___main___
	transformc	tmp_c $rgb $hsv $col
	transformc	v0 $hsv $rgb tmp_c
	transformc	tmp_c $rgb $xyz $col
	transformc	v1 $xyz $rgb tmp_c
	luminance	f0 $col
	blackbody	v2 $bbtemp
	noise		f1 $perlin P
	snoise		f2 P
	pnoise		f3 $simplex P $pperiod
	cellnoise	f4 P
end
)oso";
        TestShaderGlobals g;
        g.P = luisa::make_float3(1.37f, -0.42f, 2.81f);
        TestShaderGlobals g_per = g;
        g_per.P = g.P + luisa::make_float3(4.0f, 8.0f, -4.0f);
        auto res = compile_and_run_single_oso(oso, {g, g_per});
        expect(res.translation.succeeded());
        expect(res.ast_ok);
        expect(res.shader_compile_ok);
        if (res.outputs.size() < 2u) { return; }
        const auto &o0 = res.outputs[0];
        const auto &o1 = res.outputs[1];
        expect(near_v3(o0.v0, luisa::float3{0.8f, 0.3f, 0.2f}, 1e-3f));
        expect(near_v3(o0.v1, luisa::float3{0.8f, 0.3f, 0.2f}, 1e-3f));
        expect(near_f(o0.f0, 0.2126f * 0.8f + 0.7152f * 0.3f + 0.0722f * 0.2f, 1e-4f));
        // 6500K blackbody normalized luminance near white
        expect(o0.v2.x > 0.5f && o0.v2.y > 0.5f && o0.v2.z > 0.5f);
        // Noise in bounded range and periodic pnoise matches modulo period
        expect(o0.f1 >= -1.0f && o0.f1 <= 1.0f);
        expect(o0.f2 >= -1.5f && o0.f2 <= 1.5f);
        expect(o0.f4 >= 0.0f && o0.f4 <= 1.0f);
        expect(near_f(o0.f3, o1.f3, 1e-4f));
    };

    "int_limits_asin_acos_safe_pow_and_struct_assign"_test = [] {
        static constexpr auto oso = R"oso(
OpenShadingLanguage 1.00
shader edge_math_struct_test
param	int	in_min	-2147483648
param	int	in_neg1	-1
oparam	int	i0	0
oparam	int	i1	0
oparam	int	i2	0
oparam	float	f0	0
oparam	float	f1	0
oparam	float	f2	0
oparam	float	f3	0
oparam	float	f4	0
oparam	float	f5	0
oparam	float	f6	0
oparam	float	f7	0
oparam	vector	v0	0 0 0
local	struct Pair	src_s		%structfields{a,c}
local	float	src_s.a	0	%mystruct{src_s} %mystructfield{0}
local	color	src_s.c	0 0 0	%mystruct{src_s} %mystructfield{1}
local	struct Pair	dst_s		%structfields{a,c}
local	float	dst_s.a	0	%mystruct{dst_s} %mystructfield{0}
local	color	dst_s.c	0 0 0	%mystruct{dst_s} %mystructfield{1}
const	int	$c_min	-2147483648
const	int	$c_neg1	-1
const	int	$c_big	100000001
const	float	$f_over	1.001
const	float	$f_under	-1.001
const	float	$fn2	-2.0
const	float	$f2	2.0
const	float	$f3	3.0
const	float	$fhalf	0.5
const	float	$f0	0.0
const	float	$f95	9.5
const	color	$col	0.1 0.2 0.3
local	int	t_q
local	int	t_m
local	int	dyn_q
local	int	dyn_m
global	float	u
local	int	neg1_rt
code ___main___
	div		t_q $c_min $c_neg1
	mod		t_m $c_min $c_neg1
	assign		neg1_rt in_neg1
	gt		i2 u $f2
	if		i2 5 6
	assign		neg1_rt $c_neg1
	div		dyn_q in_min neg1_rt
	mod		dyn_m in_min neg1_rt
	eq		i0 t_q $c_min
	eq		dyn_q dyn_q $c_min
	add		i0 i0 dyn_q
	add		i1 t_m dyn_m
	assign		i2 $c_big
	asin		f0 $f_over
	acos		f1 $f_under
	pow		f2 $fn2 $f2
	pow		f3 $fn2 $f3
	pow		f4 $fn2 $fhalf
	pow		f5 $f0 $f0
	assign		src_s.a $f95
	assign		src_s.c $col
	assign		dst_s src_s
	assign		f6 dst_s.a
	assign		v0 dst_s.c
	acos		f7 $f_over
end
)oso";
        TestShaderGlobals g;
        g.u = 0.0f;
        auto res = compile_and_run_single_oso(oso, {g});
        expect(res.translation.succeeded());
        expect(res.ast_ok);
        expect(res.shader_compile_ok);
        if (res.outputs.empty()) { return; }
        const auto &o = res.outputs[0];
        expect(o.i0 == 2);
        expect(o.i1 == 0);
        expect(o.i2 == 100000001);
        expect(near_f(o.f0, 1.5707963f, 1e-3f));
        expect(near_f(o.f1, 3.1415926f, 1e-3f));
        expect(near_f(o.f2, 4.0f, 1e-4f));
        expect(near_f(o.f3, -8.0f, 1e-4f));
        expect(near_f(o.f4, 0.0f, 1e-4f));
        expect(near_f(o.f5, 1.0f, 1e-4f));
        expect(near_f(o.f6, 9.5f, 1e-4f));
        expect(near_f(o.f7, 0.0f, 1e-4f));
        expect(near_v3(o.v0, luisa::float3{0.1f, 0.2f, 0.3f}));
    };

    "splines_all_bases_and_splineinverse"_test = [] {
        static constexpr auto oso = R"oso(
OpenShadingLanguage 1.00
shader spline_test
param	float[6]	k_lin	99.0 0.0 10.0 20.0 30.0 99.0
param	float[7]	k_bez	0.0 1.0 2.0 3.0 4.0 5.0 6.0
param	float[5]	k_desc	50.0 40.0 20.0 0.0 -10.0
oparam	float	f0	0
oparam	float	f1	0
oparam	float	f2	0
oparam	float	f3	0
oparam	float	f4	0
oparam	float	f5	0
oparam	float	f6	0
oparam	float	f7	0
oparam	color	v0	0 0 0
const	string	$b_const	"constant"
const	string	$b_lin	"linear"
const	string	$b_cr	"catmull-rom"
const	string	$b_bsp	"bspline"
const	string	$b_bez	"bezier"
const	string	$b_herm	"hermite"
const	float	$x025	0.25
const	float	$x05	0.5
const	float	$x075	0.75
const	float	$f0	0.0
const	float	$f1	1.0
const	float	$f2	2.0
const	float	$f3	3.0
const	float	$f4	4.0
const	float	$f10	10.0
const	float	$f25	25.0
const	float	$v99	99.0
const	color	$c0	0 0 0
const	color	$c1	0.2 0.4 0.6
const	color	$c2	0.8 0.6 0.4
const	color	$c3	1 1 1
const	int	$n4	4
code ___main___
	spline		f0 $b_const $x075 $v99 $f2 $f3 $v99
	spline		f1 $b_lin $x05 k_lin
	spline		f2 $b_cr $x05 $f0 $f1 $f2 $f3
	spline		f3 $b_bsp $x05 $f0 $f1 $f2 $f3
	spline		f4 $b_bez $x025 k_bez
	spline		f5 $b_herm $x05 $f0 $f4 $f1 $f4
	splineinverse	f6 $b_lin $f25 k_lin
	splineinverse	f7 $b_lin $f10 $n4 k_desc
	spline		v0 $b_lin $x05 $c0 $c1 $c2 $c3
end
)oso";
        auto res = compile_and_run_single_oso(oso, {{}});
        expect(res.translation.succeeded());
        expect(res.ast_ok);
        expect(res.shader_compile_ok);
        if (res.outputs.empty()) { return; }
        const auto &o = res.outputs[0];

        // "constant" ignores 1st and last 2 -> constant $f2 = 2.0
        expect(near_f(o.f0, 2.0f));
        // "linear" on 6 knots -> 3 segments [0..10, 10..20, 20..30], x=0.5 -> mid seg 1 = 15.0
        expect(near_f(o.f1, 15.0f));
        // "catmull-rom" at u=0.5 for (0,1,2,3) -> 1.5
        expect(near_f(o.f2, 1.5f));
        // "bspline" at u=0.5 for (0,1,2,3) -> (0.125*0 + 2.875*1 + 2.875*2 + 0.125*3)/6 = 9/6 = 1.5
        expect(near_f(o.f3, 1.5f));
        // "bezier" on 7 knots (2 segs), x=0.25 -> seg 0 u=0.5 on (0,1,2,3) -> 1.5
        expect(near_f(o.f4, 1.5f));
        // "hermite" on (0, 4, 1, 4) at u=0.5 -> h00*0 + h10*4 + h01*1 + h11*4 = 0 + 0.125*4 + 0.5*1 - 0.125*4 = 0.5
        expect(near_f(o.f5, 0.5f));
        // splineinverse("linear", 25, k_lin) -> between k_lin[3]=20 and k_lin[4]=30 at 2.5/3 = 5/6
        expect(near_f(o.f6, 5.0f / 6.0f, 1e-4f));
        // 5-arg descending splineinverse("linear", 10, 4, k_desc) uses first 4 knots [50,40,20,0] -> segment [40..20] -> v=10 clamps? wait linear uses y1=40..y2=20, v=10 <= 20 -> clamps to 1.0!
        expect(near_f(o.f7, 1.0f, 1e-4f));
        // color linear midpoint between $c1 and $c2
        expect(near_v3(o.v0, luisa::float3{0.5f, 0.5f, 0.5f}, 1e-4f));
    };

    "official_mandelbrot_oso_execution"_test = [] {
        static constexpr auto mandelbrot_oso = R"oso(
OpenShadingLanguage 1.00
surface mandelbrot
param	point	center	0 0 0
param	float	scale	2
param	int	iters	100
param	int	julia	0
param	point	P_julia	0 0 0
param	color[]	colormap	0 0 0.00999999978 0 0 0.00999999978 0 0 0.5 0.75 0.25 0 0.949999988 0.949999988 0 1 1 1 1 1 1
oparam	float	fout	0
oparam	color	Cout	0 0 0
global	float	u
global	float	v
local	point	cent
local	point	c
local	point	z
local	point	iota
local	int	i
local	float	___345_x
local	float	___345_y
local	float	___348_f
temp	point	$tmp1
const	float	$const2	0.5
temp	float	$tmp2
temp	float	$tmp3
const	float	$const3	2
const	int	$const4	1
temp	float	$tmp4
const	float	$const5	1
temp	float	$tmp5
temp	float	$tmp6
const	int	$const6	0
const	float	$const7	0
temp	point	$tmp7
const	point	$const8	1 2 0
temp	int	$tmp8
temp	int	$tmp9
temp	float	$tmp10
const	float	$const9	4
temp	int	$tmp11
temp	int	$tmp12
temp	int	$tmp13
temp	point	$tmp14
temp	float	$tmp15
temp	float	$tmp16
temp	float	$tmp17
temp	float	$tmp18
temp	float	$tmp19
temp	point	$tmp20
temp	float	$tmp21
temp	float	$tmp22
temp	float	$tmp23
temp	float	$tmp24
temp	float	$tmp25
temp	int	$tmp26
temp	float	$tmp27
temp	float	$tmp28
temp	float	$tmp29
temp	float	$tmp30
temp	float	$tmp31
temp	float	$tmp32
temp	float	$tmp33
temp	float	$tmp34
const	string	$const10	"linear"
code ___main___
	assign		cent center
	sub		$tmp2 u $const2
	mul		$tmp3 $const3 $tmp2
	sub		$tmp4 $const5 v
	sub		$tmp5 $tmp4 $const2
	mul		$tmp6 $const3 $tmp5
	point		$tmp1 $tmp3 $tmp6 $const7
	mul		$tmp7 scale $tmp1
	add		c $tmp7 cent
	assign		z c
	assign		iota $const8
	for		$tmp13 13 21 38 39
	assign		i $const4
	lt		$tmp8 i iters
	neq		$tmp9 $tmp8 $const6
	if		$tmp9 20 20
	dot		$tmp10 z z
	lt		$tmp11 $tmp10 $const9
	neq		$tmp12 $tmp11 $const6
	assign		$tmp9 $tmp12
	neq		$tmp13 $tmp9 $const6
	compref		___345_x z $const6
	compref		___345_y z $const4
	if		julia 31 38
	mul		$tmp15 ___345_x ___345_x
	mul		$tmp16 ___345_y ___345_y
	sub		$tmp17 $tmp15 $tmp16
	mul		$tmp18 $const3 ___345_x
	mul		$tmp19 $tmp18 ___345_y
	point		$tmp14 $tmp17 $tmp19 $const7
	add		z $tmp14 P_julia
	mul		$tmp21 ___345_x ___345_x
	mul		$tmp22 ___345_y ___345_y
	sub		$tmp23 $tmp21 $tmp22
	mul		$tmp24 $const3 ___345_x
	mul		$tmp25 $tmp24 ___345_y
	point		$tmp20 $tmp23 $tmp25 $const7
	add		z $tmp20 c
	add		i i $const4
	lt		$tmp26 i iters
	if		$tmp26 52 53
	assign		fout i
	assign		$tmp28 i
	assign		$tmp27 $tmp28
	assign		$tmp30 iters
	div		$tmp29 $tmp27 $tmp30
	assign		$tmp33 iters
	assign		$tmp32 $tmp33
	log10		$tmp31 $tmp32
	div		$tmp34 $const5 $tmp31
	pow		___348_f $tmp29 $tmp34
	spline		Cout $const10 ___348_f colormap
	assign		Cout $const6
end
)oso";
        TestShaderGlobals g_in;
        g_in.u = 0.5f;
        g_in.v = 0.5f;// c = (0,0,0), stays inside Mandelbrot set -> Cout = (0,0,0)
        TestShaderGlobals g_out;
        g_out.u = 0.0f;
        g_out.v = 0.5f;// c = (-2,0,0), |z|^2 = 4 on iter 1 -> fout = 1, f = 0.1 -> Cout = (0,0,0.206)

        auto res = compile_and_run_single_oso(mandelbrot_oso, {g_in, g_out});
        expect(res.translation.succeeded());
        expect(res.ast_ok);
        expect(res.shader_compile_ok);
        if (res.outputs.size() < 2u) { return; }
        expect(near_f(res.outputs[0].f0, 0.0f));
        expect(near_v3(res.outputs[0].v0, luisa::float3{0.0f, 0.0f, 0.0f}));
        expect(near_f(res.outputs[1].f0, 1.0f));
        expect(near_v3(res.outputs[1].v0, luisa::float3{0.0f, 0.0f, 0.206f}, 1e-3f));
    };

    "bounds_check_mode_guard"_test = [] {
        static constexpr auto oso = R"oso(
OpenShadingLanguage 1.00
shader guard_test
param	float[3]	farr	10.0 20.0 30.0
param	float[]	empty_arr
oparam	float	f0	0
oparam	float	f1	0
oparam	float	f2	0
oparam	float	f3	0
global	float	u
local	int	len_e
local	int	cond
const	int	$cn1	-1
const	int	$c0	0
const	int	$c2	2
const	int	$c3	3
const	float	$fhalf	0.5
const	float	$f999	999.0
code ___main___
	aref		f0 farr $cn1
	aref		f1 farr $c3
	aassign		farr $c3 $f999
	aref		f2 farr $c2
	gt		cond u $fhalf
	if		cond 7 7
	aref		f3 empty_arr $c0
end
)oso";
        osl::TranslatorOptions opts;
        opts.bounds_check = osl::BoundsCheckMode::GUARD;
        auto res = compile_and_run_single_oso(oso, {{}}, {}, opts);
        expect(res.translation.succeeded());
        expect(res.ast_ok);
        expect(res.shader_compile_ok);
        if (res.outputs.empty()) { return; }
        const auto &o = res.outputs[0];
        expect(near_f(o.f0, 0.0f)); // -1 guarded -> 0 (not farr[0]=10)
        expect(near_f(o.f1, 0.0f)); // 3 guarded -> 0 (not farr[2]=30)
        expect(near_f(o.f2, 30.0f));// OOB aassign at 3 did NOT overwrite farr[2]
        expect(near_f(o.f3, 0.0f)); // empty unbounded array behind runtime guard succeeded cleanly
    };

    return 0;
}
