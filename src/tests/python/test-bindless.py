from luisa import *
from luisa.builtin import *
from luisa.types import *
import numpy as np
init()


@func
def write_texture(tex):
    set_block_size(16, 16, 1)
    index = dispatch_id().xy
    size = dispatch_size().xy
    uv = (float2(index) + 0.5) / float2(size)
    tex.write(index, float4(uv, 0.5, 1))
# Tonemapping


@func
def aces_tonemapping(x: float3):
    a = 2.51
    b = 0.03
    c = 2.43
    d = 0.59
    e = 0.14
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0)


@func
def read_texture(bindless, display_tex):
    set_block_size(16, 16, 1)
    index = dispatch_id().xy
    size = dispatch_size().xy
    uv = (float2(index) + 0.5) / float2(size)
    uv = (1.0 - uv) * 2.0
    bindless_index = 55
    color = bindless.texture2d_sample_mip(bindless_index, uv, 0).xyz
    color = aces_tonemapping(color)
    display_tex.write(index, float4(color, 1))


bindless_array = BindlessArray()
res = 1024, 1024
input_tex = Texture2D(*res, 4, float, storage="BYTE")
display_tex = Texture2D(*res, 4, float, storage="BYTE")
address = lcapi.Address.MIRROR
bindless_array.emplace(
    55, input_tex, filter=lcapi.Filter.LINEAR_POINT, address=address)
useless_buffer = Buffer(1, float4)
bindless_array.emplace(55, useless_buffer)

gui = GUI("Test Bindless", res)
while gui.running():
    # 更新bindless到GPU
    bindless_array.update()
    write_texture(input_tex, dispatch_size=(*res, 1))
    read_texture(bindless_array, display_tex, dispatch_size=(*res, 1))
    gui.set_image(display_tex)
    gui.show()
synchronize()
