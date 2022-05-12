# Julia set demo ported from the Taichi tutorial

import luisa
from luisa.mathtypes import *

luisa.init()

n = 320
pixels = luisa.Texture2D.empty(n * 2, n, 4, dtype=float)

@luisa.func
def complex_sqr(z):
    return float2(z[0]**2 - z[1]**2, z[1] * z[0] * 2)

@luisa.func
def paint(t):
    coord = dispatch_id().xy
    c = float2(-0.8, cos(t) * 0.2)
    z = 2 * coord / n - float2(2, 1)
    iterations = 0
    while length(z) < 20 and iterations < 50:
        z = complex_sqr(z) + c
        iterations += 1
    color = float3(1 - iterations * 0.02)
    pixels.write(coord, float4(color, 1))

gui = luisa.GUI("Julia Set", resolution=(n * 2, n))

i = 0
while gui.running():
    paint(i * 0.03, dispatch_size=(n * 2, n, 1))
    gui.set_image(pixels)
    gui.show()
    i = i + 1