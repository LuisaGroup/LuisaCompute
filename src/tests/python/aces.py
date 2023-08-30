from luisa import *
from luisa.builtin import *


@func
def linear_to_srgb(x: float3):
    return clamp(select(1.055 * x ** (1.0 / 2.4) - 0.055,
                        12.92 * x,
                        x <= 0.00031308),
                 0.0, 1.0)


@func
def srgb_to_acescg(col):
    mat = transpose(float3x3(0.6124941985, 0.3387372519, 0.0488555261,
                             0.0705942516, 0.9176714837, 0.0117043061,
                             0.0207273350, 0.1068822318, 0.8723380622))
    return mat * col


@func
def acescg_to_srgb(col):
    mat = transpose(float3x3(1.7310585561, -0.6039691407, -0.0801447831,
                             -0.1314365771, 1.1347744211, -0.0086903805,
                             -0.0245283648, -0.1257564506, 1.0656754216))
    return mat * col
