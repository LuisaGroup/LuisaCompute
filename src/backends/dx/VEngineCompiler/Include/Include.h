#include "ShaderConfig.cginc"
#include "Matrix.cginc"
SamplerState pointWrapSampler  : register(s0);
SamplerState pointClampSampler  : register(s1);
SamplerState bilinearWrapSampler  : register(s2);
SamplerState bilinearClampSampler  : register(s3);
SamplerState trilinearWrapSampler  : register(s4);
SamplerState trilinearClampSampler  : register(s5);
SamplerState anisotropicWrapSampler  : register(s6);
SamplerState anisotropicClampSampler  : register(s7);
SamplerState pointClampSampler_linearMip : register(s10);
SamplerState pointWrapSampler_linearMip : register(s11);

SamplerComparisonState linearShadowSampler : register(s8);      //tex.SampleCmpLevelZero(linearShadowSampler, uv, testZ)
SamplerComparisonState cubemapShadowSampler : register(s9);
SamplerComparisonState pointShadowSampler : register(s12);

float4x4 _inverse(float4x4 m) {
    float n11 = m[0][0], n12 = m[1][0], n13 = m[2][0], n14 = m[3][0];
    float n21 = m[0][1], n22 = m[1][1], n23 = m[2][1], n24 = m[3][1];
    float n31 = m[0][2], n32 = m[1][2], n33 = m[2][2], n34 = m[3][2];
    float n41 = m[0][3], n42 = m[1][3], n43 = m[2][3], n44 = m[3][3];

    float t11 = n23 * n34 * n42 - n24 * n33 * n42 + n24 * n32 * n43 - n22 * n34 * n43 - n23 * n32 * n44 + n22 * n33 * n44;
    float t12 = n14 * n33 * n42 - n13 * n34 * n42 - n14 * n32 * n43 + n12 * n34 * n43 + n13 * n32 * n44 - n12 * n33 * n44;
    float t13 = n13 * n24 * n42 - n14 * n23 * n42 + n14 * n22 * n43 - n12 * n24 * n43 - n13 * n22 * n44 + n12 * n23 * n44;
    float t14 = n14 * n23 * n32 - n13 * n24 * n32 - n14 * n22 * n33 + n12 * n24 * n33 + n13 * n22 * n34 - n12 * n23 * n34;

    float det = n11 * t11 + n21 * t12 + n31 * t13 + n41 * t14;
    float idet = 1.0f / det;

    float4x4 ret;

    ret[0][0] = t11 * idet;
    ret[0][1] = (n24 * n33 * n41 - n23 * n34 * n41 - n24 * n31 * n43 + n21 * n34 * n43 + n23 * n31 * n44 - n21 * n33 * n44) * idet;
    ret[0][2] = (n22 * n34 * n41 - n24 * n32 * n41 + n24 * n31 * n42 - n21 * n34 * n42 - n22 * n31 * n44 + n21 * n32 * n44) * idet;
    ret[0][3] = (n23 * n32 * n41 - n22 * n33 * n41 - n23 * n31 * n42 + n21 * n33 * n42 + n22 * n31 * n43 - n21 * n32 * n43) * idet;

    ret[1][0] = t12 * idet;
    ret[1][1] = (n13 * n34 * n41 - n14 * n33 * n41 + n14 * n31 * n43 - n11 * n34 * n43 - n13 * n31 * n44 + n11 * n33 * n44) * idet;
    ret[1][2] = (n14 * n32 * n41 - n12 * n34 * n41 - n14 * n31 * n42 + n11 * n34 * n42 + n12 * n31 * n44 - n11 * n32 * n44) * idet;
    ret[1][3] = (n12 * n33 * n41 - n13 * n32 * n41 + n13 * n31 * n42 - n11 * n33 * n42 - n12 * n31 * n43 + n11 * n32 * n43) * idet;

    ret[2][0] = t13 * idet;
    ret[2][1] = (n14 * n23 * n41 - n13 * n24 * n41 - n14 * n21 * n43 + n11 * n24 * n43 + n13 * n21 * n44 - n11 * n23 * n44) * idet;
    ret[2][2] = (n12 * n24 * n41 - n14 * n22 * n41 + n14 * n21 * n42 - n11 * n24 * n42 - n12 * n21 * n44 + n11 * n22 * n44) * idet;
    ret[2][3] = (n13 * n22 * n41 - n12 * n23 * n41 - n13 * n21 * n42 + n11 * n23 * n42 + n12 * n21 * n43 - n11 * n22 * n43) * idet;

    ret[3][0] = t14 * idet;
    ret[3][1] = (n13 * n24 * n31 - n14 * n23 * n31 + n14 * n21 * n33 - n11 * n24 * n33 - n13 * n21 * n34 + n11 * n23 * n34) * idet;
    ret[3][2] = (n14 * n22 * n31 - n12 * n24 * n31 - n14 * n21 * n32 + n11 * n24 * n32 + n12 * n21 * n34 - n11 * n22 * n34) * idet;
    ret[3][3] = (n12 * n23 * n31 - n13 * n22 * n31 + n13 * n21 * n32 - n11 * n23 * n32 - n12 * n21 * n33 + n11 * n22 * n33) * idet;

    return ret;
}

float3x3 _inverse(float3x3 m)
{
    float3 c = m.c0;
    float3 c2 = m.c1;
    float3 c3 = m.c2;
    float3 lhs = float3(c2.x, c3.x, c.x);
    float3 flt = float3(c2.y, c3.y, c.y);
    float3 rhs = float3(c2.z, c3.z, c.z);
    float3 flt2 = flt * rhs.yzx - flt.yzx * rhs;
    float3 c4 = lhs.yzx * rhs - lhs * rhs.yzx;
    float3 c5 = lhs * flt.yzx - lhs.yzx * flt;
    float rhs2 = 1f / dot(lhs.zxy * flt2, 1);
    return float3x3(flt2, c4, c5) * rhs2;
}

////vec,scalar,float
Z _acosh(Z v){return log(v + sqrt(v * v - 1));}
////vec,scalar,float
Z _asinh(Z v){return log(v + sqrt(v * v + 1));}
////vec,scalar,float
Z _atanh(Z v){return 0.5 * log((1 + v) / (1 - v));}
////vec,scalar,float
Z _exp10(Z v){return pow(10, v)};
////vec,scalar,float,uint,int
Z _copysign(Z x, Z y) { (abs(sign(x) - sign(y)) > 1) ? -x : x; }
////vec,scalar,float
Z _length_sqr(Z x){ return dot(x,x);}
////vec,scalar,float
Z _distance_sqr(Z x, Z y){ return _length_sqr(x - y);}

#define _min3(X,Y,Z) (min((X),min((Y),(Z))))
