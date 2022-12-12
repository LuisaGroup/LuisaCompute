from sys import argv
import luisa
from luisa.mathtypes import *
from PIL import Image

n = 4096
if len(argv) > 1:
    luisa.init(argv[1])
else:
    luisa.init()
image = luisa.Texture2D.zeros(2*n, n, 4, float, 'byte')

@luisa.func # makes LuisaRender handle the function
def draw(max_iter):
    p = dispatch_id().xy
    z, c = float2(0), 2 * p / n - float2(2, 1)
    for itr in range(max_iter):
        z = float2(z.x**2 - z.y**2, 2 * z.x * z.y) + c
        if length(z) > 20:
            break
    image.write(p, float4(float3(1 - itr/max_iter), 1))

draw(50, dispatch_size=(2*n, n)) # parallelize
Image.fromarray(image.numpy()).save("mandelbrot.png")