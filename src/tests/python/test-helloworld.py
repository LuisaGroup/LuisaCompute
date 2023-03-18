from luisa import *
init()
res = 1024, 1024
img = Texture2D(*res, 4, float, storage="BYTE")


@func
def shader():
    coord = dispatch_id().xy
    size = dispatch_size().xy
    uv = (float2(coord) + 0.5) / float2(size)
    img.write(coord, float4(uv, 0.5, 1.))


shader(dispatch_size=(*res, 1))
img.to_image("helloworld.png")
