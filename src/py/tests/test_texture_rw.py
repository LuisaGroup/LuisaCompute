import luisa
from sys import argv

import luisa
from luisa.mathtypes import *
from luisa.texture2d import Texture2D
if len(argv) > 1:
    luisa.init(argv[1])
else:
    luisa.init()
# load texture as hdr
tex = Texture2D.from_hdr_image("camera.png")
res = (tex.width, tex.height)
# take half resolution
outTex = Texture2D(res[0] // 2, res[1] // 2, 4, float, storage="byte")
@luisa.func
def Linear2SRGB():
    set_block_size(16,16,1)
    upCoord = dispatch_id().xy * 2
    x = tex.read(upCoord) + tex.read(upCoord + int2(1,0)) + tex.read(upCoord + int2(0,1)) + tex.read(upCoord + int2(1,1))
    x /= float4(4)
    outTex.write(dispatch_id().xy, clamp(select(1.055 * x ** (1.0 / 2.4) - 0.055, 12.92 * x, x <= 0.00031308), 0.0, 1.0))
# downsample and transform to ldr srgb format
Linear2SRGB(dispatch_size=(outTex.width, outTex.height, 1))
outTex.to_image("out_camera.png")