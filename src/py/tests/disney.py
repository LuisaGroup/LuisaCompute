# ported to python, from https://github.com/Twinklebear/ChameleonRT/blob/master/backends/optix/disney_bsdf.h

import luisa
from luisa.mathtypes import *
import math
import glass
M_PI = math.pi
M_1_PI = 1/math.pi


@luisa.func
def pow2(x: float):
	return x*x

@luisa.func
def luminance(c: float3):
    return 0.2126 * c.x + 0.7152 * c.y + 0.0722 * c.z

@luisa.func
def all_zero(v: float3):
	return v.x == 0. and v.y == 0. and v.z == 0.

@luisa.func
def reflect(i: float3, n: float3):
	return i - 2. * n * dot(i, n)

@luisa.func
def refract_ray(i: float3, n: float3, eta: float):
	n_dot_i = dot(n, i);
	k = 1. - eta * eta * (1. - n_dot_i * n_dot_i)
	if k < 0.:
		return make_float3(0.)
	return eta * i - (eta * n_dot_i + sqrt(k)) * n

@luisa.func
def saturate(x: float):
	return clamp(x, 0., 1.)


# Disney BSDF functions, for additional details and examples see:
# - https://blog.selfshadow.com/publications/s2012-shading-course/burley/s2012_pbs_disney_brdf_notes_v3.pdf
# - https://www.shadertoy.com/view/XdyyDd
# - https://github.com/wdas/brdf/blob/master/src/brdfs/disney.brdf
# - https://schuttejoe.github.io/post/disneybsdf/
#
# Variable naming conventions with the Burley course notes:
# V -> w_o
# L -> w_i
# H -> w_h


# signature: (mat, n, w_o, v_x, v_y, [rng])
# pdf, w_i, brdf
# 'disney_brdf', 'disney_pdf', 'sample_disney_brdf'


DisneyMaterial = luisa.StructType(
	base_color= float3,
	metallic= float,

	specular= float,
	roughness= float,
	specular_tint= float,
	anisotropy= float,

	sheen= float,
	sheen_tint= float,
	clearcoat= float,
	clearcoat_gloss= float,

	ior= float,
	specular_transmission= float
)

DisneyMaterial.default = DisneyMaterial(
    base_color      = float3(0.7),
    metallic        = float(0),
    specular        = float(0),
    roughness       = float(0.5),
    specular_tint   = float(0),
    anisotropy      = float(0),
    sheen           = float(0),
    sheen_tint      = float(0),
    clearcoat       = float(0),
    clearcoat_gloss = float(0),
    ior             = float(1),
    specular_transmission = float(0)
)


@luisa.func
def same_hemisphere(w_o: float3, w_i: float3, n: float3):
	return dot(w_o, n) * dot(w_i, n) > 0.


# Sample the hemisphere using a cosine weighted distribution,
# returns a vector in a hemisphere oriented about (0, 0, 1)
@luisa.func
def cos_sample_hemisphere(u: float2):
	s = 2. * u - make_float2(1.)
	d = float2()
	radius = 0.
	theta = 0.
	if s.x == 0. and s.y == 0.:
		d = s
	else:
		if abs(s.x) > abs(s.y):
			radius = s.x
			theta  = M_PI / 4. * (s.y / s.x)
		else:
			radius = s.y
			theta  = M_PI / 2. - M_PI / 4 * (s.x / s.y)

	d = radius * make_float2(cos(theta), sin(theta))
	return make_float3(d.x, d.y, sqrt(max(0., 1. - d.x * d.x - d.y * d.y)))


@luisa.func
def spherical_dir(sin_theta: float, cos_theta: float, phi: float):
	return make_float3(sin_theta * cos(phi), sin_theta * sin(phi), cos_theta)

@luisa.func
def power_heuristic(n_f: float, pdf_f: float, n_g: float, pdf_g: float):
	f = n_f * pdf_f
	g = n_g * pdf_g
	return (f * f) / (f * f + g * g)

@luisa.func
def schlick_weight(cos_theta: float):
	return pow(saturate(1 - cos_theta), 5)


# Complete Fresnel Dielectric computation, for transmission at ior near 1
# they mention having issues with the Schlick approximation.
# eta_i: material on incident side's ior
# eta_t: material on transmitted side's ior
@luisa.func
def fresnel_dielectric(cos_theta_i: float, eta_i: float, eta_t: float):
	g = pow2(eta_t) / pow2(eta_i) - 1. + pow2(cos_theta_i)
	if g < 0.:
		return 1.
	return 0.5 * pow2(g - cos_theta_i) / pow2(g + cos_theta_i) \
		* (1. + pow2(cos_theta_i * (g + cos_theta_i) - 1.) / pow2(cos_theta_i * (g - cos_theta_i) + 1.))


# D_GTR1: Generalized Trowbridge-Reitz with gamma=1
# Burley notes eq. 4
@luisa.func
def gtr_1(cos_theta_h: float, alpha: float):
	if alpha >= 1.:
		return M_1_PI
	alpha_sqr = alpha * alpha
	return M_1_PI * (alpha_sqr - 1.) / (log(alpha_sqr) * (1. + (alpha_sqr - 1.) * cos_theta_h * cos_theta_h))


# D_GTR2: Generalized Trowbridge-Reitz with gamma=2
# Burley notes eq. 8
@luisa.func
def gtr_2(cos_theta_h: float, alpha: float):
	alpha_sqr = alpha * alpha
	return M_1_PI * alpha_sqr / pow2(1. + (alpha_sqr - 1.) * cos_theta_h * cos_theta_h)


# D_GTR2 Anisotropic: Anisotropic generalized Trowbridge-Reitz with gamma=2
# Burley notes eq. 13
@luisa.func
def gtr_2_aniso(h_dot_n: float, h_dot_x: float, h_dot_y: float, alpha: float2):
	return M_1_PI / (alpha.x * alpha.y
			* pow2(pow2(h_dot_x / alpha.x) + pow2(h_dot_y / alpha.y) + h_dot_n * h_dot_n))


@luisa.func
def smith_shadowing_ggx(n_dot_o: float, alpha_g: float):
	a = alpha_g * alpha_g
	b = n_dot_o * n_dot_o
	return 1. / (n_dot_o + sqrt(a + b - a * b))


@luisa.func
def smith_shadowing_ggx_aniso(n_dot_o: float, o_dot_x: float, o_dot_y: float, alpha: float2):
	return 1. / (n_dot_o + sqrt(pow2(o_dot_x * alpha.x) + pow2(o_dot_y * alpha.y) + pow2(n_dot_o)))


# Sample a reflection direction the hemisphere oriented along n and spanned by v_x, v_y using the random samples in s
@luisa.func
def sample_lambertian_dir(n: float3, v_x: float3, v_y: float3, s: float2):
	hemi_dir = normalize(cos_sample_hemisphere(s))
	return hemi_dir.x * v_x + hemi_dir.y * v_y + hemi_dir.z * n


# Sample the microfacet normal vectors for the various microfacet distributions
@luisa.func
def sample_gtr_1_h(n: float3, v_x: float3, v_y: float3, alpha: float, s: float2):
	phi_h = 2. * M_PI * s.x
	alpha_sqr = alpha * alpha
	cos_theta_h_sqr = (1. - pow(alpha_sqr, 1. - s.y)) / (1. - alpha_sqr)
	cos_theta_h = sqrt(cos_theta_h_sqr)
	sin_theta_h = sqrt(max(0.0, 1. - cos_theta_h_sqr))
	hemi_dir = normalize(spherical_dir(sin_theta_h, cos_theta_h, phi_h))
	return hemi_dir.x * v_x + hemi_dir.y * v_y + hemi_dir.z * n


@luisa.func
def sample_gtr_2_h(n: float3, v_x: float3, v_y: float3, alpha: float, s: float2):
	phi_h = 2. * M_PI * s.x
	cos_theta_h_sqr = (1. - s.y) / (1. + (alpha * alpha - 1.) * s.y)
	cos_theta_h = sqrt(cos_theta_h_sqr)
	sin_theta_h = sqrt(max(0.0, 1. - cos_theta_h_sqr))
	hemi_dir = normalize(spherical_dir(sin_theta_h, cos_theta_h, phi_h))
	return hemi_dir.x * v_x + hemi_dir.y * v_y + hemi_dir.z * n


@luisa.func
def sample_gtr_2_aniso_h(n: float3, v_x: float3, v_y: float3, alpha: float2, s: float2):
	x = 2. * M_PI * s.x
	w_h = sqrt(s.y / (1. - s.y)) * (alpha.x * cos(x) * v_x + alpha.y * sin(x) * v_y) + n
	return normalize(w_h)


@luisa.func
def lambertian_pdf(w_i: float3, n: float3):
	d = dot(w_i, n)
	if d > 0.:
		return d * M_1_PI
	return 0.


@luisa.func
def gtr_1_pdf(w_o: float3, w_i: float3, n: float3, alpha: float):
	if not same_hemisphere(w_o, w_i, n):
		return 0.
	w_h = normalize(w_i + w_o)
	cos_theta_h = dot(n, w_h)
	d = gtr_1(cos_theta_h, alpha)
	return d * cos_theta_h / (4. * dot(w_o, w_h))


@luisa.func
def gtr_2_pdf(w_o: float3, w_i: float3, n: float3, alpha: float):
	if not same_hemisphere(w_o, w_i, n):
		return 0.
	w_h = normalize(w_i + w_o)
	cos_theta_h = dot(n, w_h)
	d = gtr_2(cos_theta_h, alpha)
	return d * cos_theta_h / (4. * dot(w_o, w_h))


@luisa.func
def gtr_2_transmission_pdf(w_o: float3, w_i: float3, n: float3,
	alpha: float, ior: float):

	if same_hemisphere(w_o, w_i, n):
		return 0.
	entering = dot(w_o, n) > 0.
	eta_o = 1.  if entering else ior
	eta_i = ior if entering else 1.
	w_h = -normalize(w_o + w_i * eta_i / eta_o)
	cos_theta_h = abs(dot(n, w_h))
	i_dot_h = dot(w_i, w_h)
	o_dot_h = dot(w_o, w_h)
	d = gtr_2(cos_theta_h, alpha)
	# dwh_dwi = o_dot_h * pow2(eta_o) / pow2(eta_o * o_dot_h + eta_i * i_dot_h)
	return d * cos_theta_h # * abs(dwh_dwi)


@luisa.func
def gtr_2_aniso_pdf(w_o: float3, w_i: float3, n: float3,
	v_x: float3, v_y: float3, alpha: float2):

	if not same_hemisphere(w_o, w_i, n):
		return 0.
	w_h = normalize(w_i + w_o)
	cos_theta_h = dot(n, w_h)
	d = gtr_2_aniso(cos_theta_h, abs(dot(w_h, v_x)), abs(dot(w_h, v_y)), alpha)
	return d * cos_theta_h / (4. * dot(w_o, w_h))


@luisa.func
def disney_diffuse(mat: DisneyMaterial, n: float3,
	w_o: float3, w_i: float3):

	w_h = normalize(w_i + w_o)
	n_dot_o = abs(dot(w_o, n))
	n_dot_i = abs(dot(w_i, n))
	i_dot_h = dot(w_i, w_h)
	fd90 = 0.5 + 2. * mat.roughness * i_dot_h * i_dot_h
	fi = schlick_weight(n_dot_i)
	fo = schlick_weight(n_dot_o)
	return mat.base_color * M_1_PI * lerp(1., fd90, fi) * lerp(1., fd90, fo)


@luisa.func
def disney_microfacet_isotropic(mat: DisneyMaterial, n: float3,
	w_o: float3, w_i: float3):

	w_h = normalize(w_i + w_o)
	lum = luminance(mat.base_color)
	tint = mat.base_color / lum if lum > 0. else make_float3(1.)
	spec = lerp(mat.specular * 0.08 * lerp(make_float3(1.), tint, mat.specular_tint), mat.base_color, mat.metallic)

	alpha = max(0.001, mat.roughness * mat.roughness)
	d = gtr_2(dot(n, w_h), alpha)
	f = lerp(spec, make_float3(1.), schlick_weight(dot(w_i, w_h)))
	g = smith_shadowing_ggx(dot(n, w_i), alpha) * smith_shadowing_ggx(dot(n, w_o), alpha)
	return d * f * g


@luisa.func
def disney_microfacet_transmission_isotropic(mat: DisneyMaterial, n: float3,
	w_o: float3, w_i: float3):

	o_dot_n = dot(w_o, n)
	i_dot_n = dot(w_i, n)
	if o_dot_n == 0. or i_dot_n == 0.:
		return make_float3(0.)

	entering = o_dot_n > 0.
	eta_o = 1.      if entering else mat.ior
	eta_i = mat.ior if entering else 1.
	w_h = normalize(w_o + w_i * eta_i / eta_o)

	alpha = max(0.001, mat.roughness * mat.roughness)
	d = gtr_2(abs(dot(n, w_h)), alpha)

	f = fresnel_dielectric(abs(dot(w_i, n)), eta_o, eta_i)
	g = smith_shadowing_ggx(abs(dot(n, w_i)), alpha) * smith_shadowing_ggx(abs(dot(n, w_o)), alpha)
	# f=0.0
	# g=1.0

	i_dot_h = dot(w_i, w_h)
	o_dot_h = dot(w_o, w_h)

	c = abs(o_dot_h) / abs(dot(w_o, n)) * abs(i_dot_h) / abs(dot(w_i, n)) \
		* pow2(eta_o) / pow2(eta_o * o_dot_h + eta_i * i_dot_h)

	return mat.base_color * c * (1. - f) * g * d


@luisa.func
def disney_microfacet_anisotropic(mat: DisneyMaterial, n: float3,
	w_o: float3, w_i: float3, v_x: float3, v_y: float3):

	w_h = normalize(w_i + w_o)
	lum = luminance(mat.base_color)
	tint = mat.base_color / lum if lum > 0. else make_float3(1.)
	spec = lerp(mat.specular * 0.08 * lerp(make_float3(1.), tint, mat.specular_tint), mat.base_color, mat.metallic)

	aspect = sqrt(1. - mat.anisotropy * 0.9)
	a = mat.roughness * mat.roughness
	alpha = make_float2(max(0.001, a / aspect), max(0.001, a * aspect))
	d = gtr_2_aniso(dot(n, w_h), abs(dot(w_h, v_x)), abs(dot(w_h, v_y)), alpha)
	f = lerp(spec, make_float3(1.), schlick_weight(dot(w_i, w_h)))
	g = smith_shadowing_ggx_aniso(dot(n, w_i), abs(dot(w_i, v_x)), abs(dot(w_i, v_y)), alpha) \
		* smith_shadowing_ggx_aniso(dot(n, w_o), abs(dot(w_o, v_x)), abs(dot(w_o, v_y)), alpha)
	return d * f * g


@luisa.func
def disney_clear_coat(mat: DisneyMaterial, n: float3,
	w_o: float3, w_i: float3):

	w_h = normalize(w_i + w_o)
	alpha = lerp(0.1, 0.001, mat.clearcoat_gloss)
	d = gtr_1(dot(n, w_h), alpha)
	f = lerp(0.04, 1., schlick_weight(dot(w_i, n)))
	g = smith_shadowing_ggx(dot(n, w_i), 0.25) * smith_shadowing_ggx(dot(n, w_o), 0.25)
	return mat.clearcoat * d * f * g
	# return 0.25 * mat.clearcoat * d * f * g


@luisa.func
def disney_sheen(mat: DisneyMaterial, n: float3,
	w_o: float3, w_i: float3):

	w_h = normalize(w_i + w_o)
	lum = luminance(mat.base_color)
	tint =  mat.base_color / lum if lum > 0. else make_float3(1.)
	sheen_color = lerp(make_float3(1.), tint, mat.sheen_tint)
	f = schlick_weight(dot(w_i, n))
	return f * mat.sheen * sheen_color


@luisa.func
def disney_brdf(mat: DisneyMaterial, n: float3,
	w_o: float3, w_i: float3, v_x: float3, v_y: float3):

	if mat.specular_transmission > 0.:
		return float3(0.)
		
	if not same_hemisphere(w_o, w_i, n):
		if mat.specular_transmission > 0.:
			spec_trans = disney_microfacet_transmission_isotropic(mat, n, w_o, w_i)
			return spec_trans * (1. - mat.metallic) * mat.specular_transmission
		return make_float3(0.)

	coat = disney_clear_coat(mat, n, w_o, w_i)
	sheen = disney_sheen(mat, n, w_o, w_i)
	diffuse = disney_diffuse(mat, n, w_o, w_i)
	gloss = float3()
	if mat.anisotropy == 0.:
		gloss = disney_microfacet_isotropic(mat, n, w_o, w_i)
	else:
		gloss = disney_microfacet_anisotropic(mat, n, w_o, w_i, v_x, v_y)

	return (diffuse + sheen) * (1. - mat.metallic) * (1. - mat.specular_transmission) + gloss + coat


@luisa.func
def disney_pdf(mat: DisneyMaterial, n: float3,
	w_o: float3, w_i: float3, v_x: float3, v_y: float3):

	alpha = max(0.001, mat.roughness * mat.roughness)
	aspect = sqrt(1. - mat.anisotropy * 0.9)
	alpha_aniso = make_float2(max(0.001, alpha / aspect), max(0.001, alpha * aspect))

	clearcoat_alpha = lerp(0.1, 0.001, mat.clearcoat_gloss)

	diffuse = lambertian_pdf(w_i, n)
	clear_coat = gtr_1_pdf(w_o, w_i, n, clearcoat_alpha)

	n_comp = 3.
	microfacet = 0.
	microfacet_transmission = 0.
	if mat.anisotropy == 0.:
		microfacet = gtr_2_pdf(w_o, w_i, n, alpha)
	else:
		microfacet = gtr_2_aniso_pdf(w_o, w_i, n, v_x, v_y, alpha_aniso)

	if mat.specular_transmission > 0.:
		return 0.
		n_comp = 4. if dot(w_o, n) > 0. else 1.
		microfacet_transmission = gtr_2_transmission_pdf(w_o, w_i, n, alpha, mat.ior)


	if mat.metallic == 1.:
		n_comp -= 1 # don't sample diffuse
		diffuse = 0.
	if mat.clearcoat == 0.:
		n_comp -= 1 # don't sample coat
		clear_coat = 0.

	return (diffuse + microfacet + microfacet_transmission + clear_coat) / n_comp



# Sample a component of the Disney BRDF, returns the sampled BRDF color,
# ray reflection direction (w_i) and sample PDF.
# return struct(pdf: float, w_i: float3, brdf: float3)
@luisa.func
def sample_disney_brdf(mat: DisneyMaterial, n: float3,
	w_o: float3, v_x: float3, v_y: float3, rng):

	if mat.specular_transmission != 0.:
		return glass.sample_brdf(n, w_o, v_x, v_y, rng, mat.ior)

	n_component = 4
	if mat.specular_transmission == 0.:
		n_component -= 1
	if mat.metallic == 1.:
		n_component -= 1 # don't sample diffuse
	if mat.clearcoat == 0.:
		n_component -= 1 # don't sample coat

	component = int(rng.next() * n_component)
	component = clamp(component, 0, n_component-1)

	if mat.metallic == 1.:
		component += 1 # skip diffuse
	if mat.clearcoat == 0.:
		if component >= 2:
			component += 1 # skip coat

	samples = make_float2(rng.next(), rng.next())
	if component == 0:
		# Sample diffuse component
		w_i = sample_lambertian_dir(n, v_x, v_y, samples)
	elif component == 1:
		alpha = max(0.001, mat.roughness * mat.roughness)
		if mat.anisotropy == 0.:
			w_h = sample_gtr_2_h(n, v_x, v_y, alpha, samples)
			# print(f"alpha = {alpha},  w_h = {w_h}")
		else:
			aspect = sqrt(1. - mat.anisotropy * 0.9)
			alpha_aniso = make_float2(max(0.001, alpha / aspect), max(0.001, alpha * aspect))
			w_h = sample_gtr_2_aniso_h(n, v_x, v_y, alpha_aniso, samples)
		w_i = reflect(-w_o, w_h)

		# Invalid reflection, terminate ray
		if not same_hemisphere(w_o, w_i, n):
			return struct(pdf=0., w_i=make_float3(0.), brdf=make_float3(0.))

	elif component == 2:
		# Sample clear coat component
		alpha = lerp(0.1, 0.001, mat.clearcoat_gloss)
		w_h = sample_gtr_1_h(n, v_x, v_y, alpha, samples)
		w_i = reflect(-w_o, w_h)

		# Invalid reflection, terminate ray
		if not same_hemisphere(w_o, w_i, n):
			return struct(pdf=0., w_i=make_float3(0.), brdf=make_float3(0.))

	else:
		# Sample microfacet transmission component
		alpha = max(0.001, mat.roughness * mat.roughness)
		w_h = sample_gtr_2_h(n, v_x, v_y, alpha, samples)
		# print(f"alpha = {alpha},  w_h = {w_h}")
		if dot(w_o, w_h) < 0.:
			w_h = -w_h
		entering = dot(w_o, n) > 0.
		w_i = refract_ray(-w_o, w_h, 1. / mat.ior if entering else mat.ior)

		# Total internal reflection
		if all_zero(w_i):
			return struct(pdf=1., w_i=reflect(-w_o, w_h if entering else -w_h), brdf=make_float3(1.))
		# if all_zero(w_i):
		# 	return struct(pdf=0., w_i=make_float3(0.), brdf=make_float3(0.))

	return struct(
		pdf = disney_pdf(mat, n, w_o, w_i, v_x, v_y),
		w_i = w_i,
		brdf = disney_brdf(mat, n, w_o, w_i, v_x, v_y)
	)

__all__ = ['DisneyMaterial', 'disney_brdf', 'disney_pdf', 'sample_disney_brdf']
