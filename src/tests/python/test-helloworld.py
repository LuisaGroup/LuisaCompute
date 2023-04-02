from luisa import *
from luisa.types import *
init()
res = 1024, 1024
img = Image2D(*res, 4, float, storage="BYTE")


@func
def shader():
    coord = dispatch_id().xy
    size = dispatch_size().xy
    uv = (float2(coord) + float(0.5)) / float2(size)
    img.write(coord, float4(uv, 0.5, float(1.)))


shader(dispatch_size=(*res, 1))
img.to_image("helloworld.png")
print("finished")