from luisa import *
from luisa.types import *
init()

res = 1024, 1024
img = Image2D(*res, 4, float, storage="BYTE")


@func
def shader():
    coord = dispatch_id().xy
    size = dispatch_size().xy
    uv = (float2(coord) + 0.5) / float2(size)
    img.write(coord, float4(uv, 0.5, 1.))

####### 16-bit float version
# @func
# def shader():
#     coord = dispatch_id().xy
#     size = dispatch_size().xy
#     uv = (half2(coord) + half(0.5)) / half2(size)
#     img.write(coord, half4(uv, 0.5, half(1.)))
####### 16-bit float version

shader(dispatch_size=(*res, 1))
img.to_image("helloworld.png")
print("finished")
