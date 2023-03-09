from luisa import *
from luisa.types import *
from luisa.builtin import *
from luisa.util import *
import math
import numpy as np
import random
init()


class ImagePair:
    def __init__(self, width, height, channel, storage):
        self.prev = Texture2D(width, height, channel, uint, 1, storage)
        self.curr = Texture2D(width, height, channel, uint, 1, storage)

    def swap(self):
        tmp = self.prev
        self.prev = self.curr
        self.curr = tmp


@func
def read_state(prev, uv):
    return ite(prev.read(uv).x == 255, 1, 0)


@func
def kernel(prev, curr):
    set_block_size(16, 16, 1)
    count = uint()
    uv = dispatch_id().xy
    size = dispatch_size().xy
    state = read_state(prev, uv)
    p = int2(uv)
    for dy in range(-1, 2):
        for dx in range(-1, 2):
            if (dx != 0 or dy != 0):
                q = p + int2(dx, dy) + int2(size)
                neighbor = read_state(prev, uint2(q) % size)
                count += neighbor
    c0 = ite(count == 2, 1, 0)
    c1 = ite(count == 3, 1, 0)
    curr.write(uv, uint4(uint3(ite(((state & c0) | c1) != 0, 255, 0)), 255))


res = (128, 128)
super_sampling = 8

@func
def display_kernel(in_tex, out_tex):
    set_block_size(16, 16, 1)
    uv = dispatch_id().xy
    coord = uv // super_sampling
    value = in_tex.read(coord)
    out_tex.write(uv, float4(value) / 255.)


image_pair = ImagePair(*res, 4, "BYTE")
# reset
host_image = np.empty(res[0] * res[1], dtype=np.uint32)
for i in range(len(host_image)):
    x = 0
    if random.random() < 0.25:
        x = 255
    host_image[i] = x * 0x00010101 | 0xff000000
image_pair.prev.copy_from(host_image)
synchronize()
window_size = (res[0] * super_sampling, res[1] * super_sampling)
display = Texture2D(*window_size, 4, float, 1, "BYTE")
gui = GUI("Test game-of-life", window_size)
while gui.running():
    kernel(image_pair.prev, image_pair.curr, dispatch_size=(*res, 1))
    display_kernel(image_pair.prev, display, dispatch_size=(*window_size, 1))
    image_pair.swap()
    gui.set_image(display)
    gui.show()
synchronize()
