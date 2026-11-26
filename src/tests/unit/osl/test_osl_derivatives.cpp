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

    "zero_derivative_overhead_when_unused"_test = [] {
        static constexpr auto oso = R"oso(
OpenShadingLanguage 1.00
shader no_derivs_test
oparam	float	f0	0
oparam	point	v0	0 0 0
global	point	P
global	float	u
global	float	v
local	float	x
local	point	q
const	float	$c2	2.0
code ___main___
	mul		x u $c2
	sin		x x
	add		f0 x v
	add		q P P
	normalize	v0 q
end
)oso";
        TestShaderGlobals g;
        g.P = luisa::make_float3(1.0f, 2.0f, 2.0f);
        g.u = 0.4f;
        g.v = 0.7f;
        auto res = compile_and_run_single_oso(oso, {g});
        expect(res.translation.succeeded());
        expect(res.translation.dual_symbol_count == 0u);// No Dx/Dy/area/filterwidth/tex -> zero dual registers!
        expect(res.ast_ok);
        expect(res.shader_compile_ok);
        if (!res.outputs.empty()) {
            expect(near_f(res.outputs[0].f0, std::sin(0.8f) + 0.7f));
            expect(near_v3(res.outputs[0].v0, luisa::normalize(luisa::make_float3(2.0f, 4.0f, 4.0f))));
        }
    };

    "chain_rules_area_filterwidth_calculatenormal_and_texture_derivs"_test = [] {
        // Verify exact forward derivatives against analytic calculus:
        // f(u,v) = u*u + sin(v) + sqrt(u)
        //   df/du = 2*u + 0.5/sqrt(u), df/dv = cos(v)
        //   Dx(f) = df/du * dudx + df/dv * dvdx
        //   Dy(f) = df/du * dudy + df/dv * dvdy
        // Vec chain rule: np = normalize(P), dp = dot(P, N), tp = transform("object", "common", P)
        // Differential geometry: area(P), calculatenormal(P), filterwidth(u)
        // Texture derivatives: implicit & explicit
        static constexpr auto oso = R"oso(
OpenShadingLanguage 1.00
shader derivs_test
oparam	float	f0	0
oparam	float	f1	0
oparam	float	f2	0
oparam	float	f3	0
oparam	float	f4	0
oparam	float	f5	0
oparam	vector	v0	0 0 0
oparam	vector	v1	0 0 0
oparam	vector	v2	0 0 0
oparam	vector	v3	0 0 0
oparam	color	v4	0 0 0
oparam	color	v5	0 0 0
global	point	P
global	normal	N
global	float	u
global	float	v
local	float	u2
local	float	sv
local	float	r_u
local	float	fval
local	point	np
local	point	tp
local	float	unused_tmp
local	color	tex_imp
local	color	tex_exp
const	float	$c2	2.0
const	float	$c05	0.5
const	string	$obj	"object"
const	string	$com	"common"
const	string	$ramp	"ramp.tx"
const	float	$ex_dsdx	0.4
const	float	$ex_dtdx	-0.2
const	float	$ex_dsdy	0.1
const	float	$ex_dtdy	0.8
code ___main___
	mul		u2 u u
	sin		sv v
	sqrt		r_u u
	add		fval u2 sv
	add		fval fval r_u
	assign		f0 fval
	Dx		f1 fval
	Dy		f2 fval
	area		f3 P
	filterwidth	f4 u
	calculatenormal	v0 P
	normalize	np P
	Dx		v1 np
	dot		unused_tmp np N
	Dx		f5 unused_tmp
	transform	tp $obj $com P
	Dx		v2 tp
	Dy		v3 tp
	texture		tex_imp $ramp u v 	%argderivs{2,3}
	Dx		v4 tex_imp
	texture		tex_exp $ramp u v $ex_dsdx $ex_dtdx $ex_dsdy $ex_dtdy 	%argderivs{2,3}
	Dy		v5 tex_exp
end
)oso";
        TestShaderGlobals g;
        g.u = 0.64f;
        g.dudx = 0.15f;
        g.dudy = -0.08f;
        g.v = 0.5f;
        g.dvdx = -0.10f;
        g.dvdy = 0.25f;
        g.P = luisa::make_float3(1.0f, 2.0f, 2.0f);// |P| = 3.0
        g.dPdx = luisa::make_float3(0.3f, 0.1f, -0.2f);
        g.dPdy = luisa::make_float3(-0.1f, 0.4f, 0.2f);
        g.N = luisa::make_float3(0.2f, -0.5f, 0.8f);

        auto res = compile_and_run_single_oso(oso, {g});
        expect(res.translation.succeeded());
        expect(res.translation.dual_symbol_count > 0u);
        expect(res.ast_ok);
        expect(res.shader_compile_ok);
        if (res.outputs.empty()) { return; }
        const auto &o = res.outputs[0];

        // Analytic scalar chain rule
        float exp_val = g.u * g.u + std::sin(g.v) + std::sqrt(g.u);
        float dfdu = 2.0f * g.u + 0.5f / std::sqrt(g.u);
        float dfdv = std::cos(g.v);
        float exp_dx = dfdu * g.dudx + dfdv * g.dvdx;
        float exp_dy = dfdu * g.dudy + dfdv * g.dvdy;
        expect(near_f(o.f0, exp_val));
        expect(near_f(o.f1, exp_dx));
        expect(near_f(o.f2, exp_dy));

        // area(P) = |dPdx x dPdy|, calculatenormal(P) = dPdx x dPdy
        luisa::float3 calc_n = luisa::cross(g.dPdx, g.dPdy);
        expect(near_f(o.f3, luisa::length(calc_n)));
        expect(near_f(o.f4, std::hypot(g.dudx, g.dudy)));
        expect(near_v3(o.v0, calc_n));

        // d(normalize(P))/dx = (dPdx - n * dot(n, dPdx)) / |P|
        float lenP = luisa::length(g.P);
        luisa::float3 n = g.P / lenP;
        luisa::float3 dndx = (g.dPdx - n * luisa::dot(n, g.dPdx)) / lenP;
        expect(near_v3(o.v1, dndx));
        expect(near_f(o.f5, luisa::dot(dndx, g.N)));

        // Affine transform derivative = linear 3x3 * dPdx / dPdy
        auto vec_mul_m3 = [](luisa::float3 d) {
            return luisa::make_float3(2.0f * d.x, 0.5f * d.x + 3.0f * d.y, 4.0f * d.z);
        };
        expect(near_v3(o.v2, vec_mul_m3(g.dPdx)));
        expect(near_v3(o.v3, vec_mul_m3(g.dPdy)));

        // Ramp texture derivatives
        luisa::float3 exp_tex_dx = luisa::make_float3(
            0.5f * g.dudx + 0.25f * g.dvdx,
            0.25f * g.dudx + 0.75f * g.dvdx,
            0.1f * g.dudx + 0.1f * g.dvdx);
        luisa::float3 exp_tex_dy_explicit = luisa::make_float3(
            0.5f * 0.1f + 0.25f * 0.8f,
            0.25f * 0.1f + 0.75f * 0.8f,
            0.1f * 0.1f + 0.1f * 0.8f);
        expect(near_v3(o.v4, exp_tex_dx));
        expect(near_v3(o.v5, exp_tex_dy_explicit));
    };

    "dz_perspective_reflect_refract_spline_and_local_msg_derivs"_test = [] {
        static constexpr auto oso = R"oso(
OpenShadingLanguage 1.00
shader deriv_extra_test
oparam	float	f0	0
oparam	float	f1	0
oparam	float	f2	0
oparam	float	f3	0
oparam	float	f4	0
oparam	vector	v0	0 0 0
oparam	vector	v1	0 0 0
oparam	vector	v2	0 0 0
oparam	vector	v3	0 0 0
oparam	vector	v4	0 0 0
global	point	P
global	float	u
global	float	v
local	matrix	mpersp
local	point	pt_p
local	vector	inc_v
local	vector	nrm_v
local	vector	refl_v
local	vector	refr_v
local	float	u2
local	float	sp_val
local	float	sp_inv
local	float	sp_x
local	float	k2
local	float	eta_v
local	float	msg_in
local	float	msg_out
local	int	msg_ok
local	string	msg_key
const	string	$pfx	"deriv_"
const	string	$sfx	"msg"
const	string	$cr	"catmull-rom"
const	float	$f0	0.0
const	float	$fhalf	0.5
const	float	$f1	1.0
const	float	$f2	2.0
const	float	$f01	0.1
const	float	$f04	0.4
const	float	$f03	0.3
const	float	$f075	0.75
const	float	$f09	0.9
const	vector	$vn_y	0 1 0
code ___main___
	mul		u2 u u
	Dz		f0 u2
	matrix		mpersp $f1 $f0 $f0 $f0  $f0 $f1 $f0 $f0  $f0 $f0 $f1 $fhalf  $f0 $f0 $f0 $f1
	transform	pt_p mpersp P
	Dx		v0 pt_p
	Dy		v1 pt_p
	mul		nrm_v P $f01
	add		nrm_v nrm_v $vn_y
	reflect		refl_v P nrm_v
	Dx		v2 refl_v
	mul		eta_v v $f01
	add		eta_v eta_v $f075
	mul		inc_v $vn_y $f01
	neg		inc_v inc_v
	add		inc_v inc_v P
	refract		refr_v inc_v $vn_y eta_v
	Dx		v3 refr_v
	Dy		v4 refr_v
	mul		k2 $fhalf v
	add		k2 k2 $f2
	spline		sp_val $cr u $f0 $f1 k2 $f2
	Dx		f1 sp_val
	Dy		f2 sp_val
	mul		sp_x u $f04
	add		sp_x sp_x $f03
	splineinverse	sp_inv $cr sp_x $f0 $f01 $f09 $f1
	Dx		f3 sp_inv
	concat		msg_key $pfx $sfx
	mul		msg_in u $f2
	setmessage	msg_key msg_in
	getmessage	msg_ok msg_key msg_out
	Dx		f4 msg_out
end
)oso";
        TestShaderGlobals g;
        g.P = luisa::make_float3(1.2f, -1.5f, 1.0f);
        g.dPdx = luisa::make_float3(0.4f, -0.2f, 0.3f);
        g.dPdy = luisa::make_float3(-0.1f, 0.5f, 0.25f);
        g.u = 0.42f;
        g.dudx = 0.6f;
        g.dudy = -0.2f;
        g.v = 0.35f;
        g.dvdx = 0.15f;
        g.dvdy = 0.8f;

        auto res = compile_and_run_single_oso(oso, {g});
        expect(res.translation.succeeded());
        expect(res.ast_ok);
        expect(res.shader_compile_ok);
        if (res.outputs.empty()) { return; }
        const auto &o = res.outputs[0];

        // 1. Dz == 0
        expect(near_f(o.f0, 0.0f));

        // 2. Perspective point transform derivative via double-precision central finite difference
        const double deps = 1e-6;
        struct D3 { double x, y, z; };
        auto persp_fn = [](double px, double py, double pz) -> D3 {
            double w = 1.0 + 0.5 * pz;
            return {px / w, py / w, pz / w};
        };
        auto fd_d3 = [&](auto fn, double hx, double hy, double hz) -> luisa::float3 {
            auto a = fn(double{g.P.x} + deps * hx, double{g.P.y} + deps * hy, double{g.P.z} + deps * hz);
            auto b = fn(double{g.P.x} - deps * hx, double{g.P.y} - deps * hy, double{g.P.z} - deps * hz);
            return luisa::make_float3(
                static_cast<float>((a.x - b.x) / (2.0 * deps)),
                static_cast<float>((a.y - b.y) / (2.0 * deps)),
                static_cast<float>((a.z - b.z) / (2.0 * deps)));
        };
        expect(near_v3(o.v0, fd_d3(persp_fn, g.dPdx.x, g.dPdx.y, g.dPdx.z), 1e-4f));
        expect(near_v3(o.v1, fd_d3(persp_fn, g.dPdy.x, g.dPdy.y, g.dPdy.z), 1e-4f));

        // 3. reflect(P, (0,1,0) + 0.1*P) derivative via double-precision central finite difference
        auto r_d = [](double px, double py, double pz) -> D3 {
            double nx = 0.1 * px, ny = 1.0 + 0.1 * py, nz = 0.1 * pz;
            double d = px * nx + py * ny + pz * nz;
            return {px - 2.0 * d * nx, py - 2.0 * d * ny, pz - 2.0 * d * nz};
        };
        expect(near_v3(o.v2, fd_d3(r_d, g.dPdx.x, g.dPdx.y, g.dPdx.z), 1e-4f));

        // 4. refract(I, N, eta) derivative via double-precision central finite difference
        auto refr_fn = [](double px, double py, double pz, double v_val) -> D3 {
            double ix = px, iy = py - 0.1, iz = pz;
            double eta = 0.75 + 0.1 * v_val;
            double c = iy;
            double k = 1.0 - eta * eta * (1.0 - c * c);
            double s = eta * c + std::sqrt(k);
            return {eta * ix, eta * iy - s, eta * iz};
        };
        auto fd_refr = [&](luisa::float3 dp, float dv) -> luisa::float3 {
            auto a = refr_fn(double{g.P.x} + deps * dp.x, double{g.P.y} + deps * dp.y,
                             double{g.P.z} + deps * dp.z, double{g.v} + deps * dv);
            auto b = refr_fn(double{g.P.x} - deps * dp.x, double{g.P.y} - deps * dp.y,
                             double{g.P.z} - deps * dp.z, double{g.v} - deps * dv);
            return luisa::make_float3(
                static_cast<float>((a.x - b.x) / (2.0 * deps)),
                static_cast<float>((a.y - b.y) / (2.0 * deps)),
                static_cast<float>((a.z - b.z) / (2.0 * deps)));
        };
        expect(near_v3(o.v3, fd_refr(g.dPdx, g.dvdx), 1e-4f));
        expect(near_v3(o.v4, fd_refr(g.dPdy, g.dvdy), 1e-4f));

        // 5. spline & splineinverse derivatives via double-precision central finite difference
        auto cr_eval = [](double x, double k0, double k1, double k2v, double k3) {
            double x2 = x * x, x3 = x2 * x;
            double a = -0.5 * k0 + 1.5 * k1 - 1.5 * k2v + 0.5 * k3;
            double b = k0 - 2.5 * k1 + 2.0 * k2v - 0.5 * k3;
            double c = -0.5 * k0 + 0.5 * k2v;
            double d = k1;
            return a * x3 + b * x2 + c * x + d;
        };
        float fd_sp_dx = static_cast<float>(
            (cr_eval(double{g.u} + deps * g.dudx, 0.0, 1.0, 2.0 + 0.5 * (double{g.v} + deps * g.dvdx), 2.0) -
             cr_eval(double{g.u} - deps * g.dudx, 0.0, 1.0, 2.0 + 0.5 * (double{g.v} - deps * g.dvdx), 2.0)) /
            (2.0 * deps));
        float fd_sp_dy = static_cast<float>(
            (cr_eval(double{g.u} + deps * g.dudy, 0.0, 1.0, 2.0 + 0.5 * (double{g.v} + deps * g.dvdy), 2.0) -
             cr_eval(double{g.u} - deps * g.dudy, 0.0, 1.0, 2.0 + 0.5 * (double{g.v} - deps * g.dvdy), 2.0)) /
            (2.0 * deps));
        expect(near_f(o.f1, fd_sp_dx, 1e-4f));
        expect(near_f(o.f2, fd_sp_dy, 1e-4f));

        // Inverse deriv: dx/dt * 0.4 * dudx where cr_eval(x, 0, 0.1, 0.9, 1) == 0.4*u + 0.3
        auto cr_inv = [&](double target) {
            double lo = 0.0, hi = 1.0;
            for (int it = 0; it < 52; ++it) {
                double mid = 0.5 * (lo + hi);
                if (cr_eval(mid, 0.0, 0.1, 0.9, 1.0) < target) { lo = mid; }
                else { hi = mid; }
            }
            return 0.5 * (lo + hi);
        };
        float fd_inv_dx = static_cast<float>(
            (cr_inv(0.4 * (double{g.u} + deps * g.dudx) + 0.3) -
             cr_inv(0.4 * (double{g.u} - deps * g.dudx) + 0.3)) /
            (2.0 * deps));
        expect(near_f(o.f3, fd_inv_dx, 1e-4f));

        // 6. Folded local-string message derivative: Dx(2*u) = 2*dudx
        expect(near_f(o.f4, 2.0f * g.dudx, 1e-4f));
    };

    return 0;
}
