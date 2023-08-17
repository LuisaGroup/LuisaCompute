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
    mat = transpose(float3x3(0.6131178129, 0.3411819959, 0.0457873443,
                           0.0699340823, 0.9181030375, 0.0119327755,
                           0.0204629926, 0.1067686634, 0.8727159106))
    return mat * col


@func
def acescg_to_srgb(col):
    mat = transpose(float3x3(1.7048873310, -0.6241572745, -0.0808867739,
                           -0.1295209353, 1.1383993260, -0.0087792418,
                           -0.0241270599, -0.1246206123, 1.1488221099))
    return mat * col
