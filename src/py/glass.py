import luisa
from luisa.mathtypes import *


@luisa.func
def brdf():
	return 0

@luisa.func
def pdf():
	return 0

@luisa.func
def sample_brdf_local(wo: float3, sampler, IOR: float):
	ior = IOR if wo.z > 0.0 else 1/IOR

	cosi = abs(wo.z)
	sini = sqrt(max(0.0, 1.0 - cosi * cosi))
	sint = sini / ior;
	if sint >= 1:
		r = 1.0 # total reflect
	else:
		cost = sqrt(1-sint*sint);
		r1 = (ior*cosi - cost) / (ior*cosi + cost)
		r2 = (cosi - ior*cost) / (cosi + ior*cost)
		# reflectivity
		r = 0.5 * (r2*r2 + r1*r1);

	if r == 1.0 or sampler.next() < r: # reflected
		wi = float3(-wo.x, -wo.y, wo.z)
		val = r/abs(wi.z)
		return struct(pdf=r*1e4, w_i=wi, brdf=float3(val*1e4))
	else: # transmitted
		wi = float3(-wo.x/ior, -wo.y/ior, -wo.z * cost/abs(wo.z));
		val = (1-r) / (abs(wi.z) * ior * ior);
		return struct(pdf=(1-r)*1e4, w_i=wi, brdf=float3(val*1e4))
		# the 1/ior^2 coefficient is used to convert energy to radiance

@luisa.func
def sample_brdf(n: float3, w_o: float3, binormal: float3, tangent: float3, sampler, IOR: float):
	wo_local = float3(dot(w_o, binormal), dot(w_o, tangent), dot(w_o, n))
	s = sample_brdf_local(wo_local, sampler, IOR)
	s.w_i = s.w_i.x * binormal + s.w_i.y * tangent + s.w_i.z * n
	return s
