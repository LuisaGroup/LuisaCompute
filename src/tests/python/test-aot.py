from luisa import *
# use headless mode, no runtime available
init_headless("dx")


@func
def out_kernel(tex):
    set_block_size(16, 16, 1)
    index = dispatch_id().xy
    size = dispatch_size().xy
    uv = (float2(index) + 0.5) / float2(size)
    tex.write(index, float4(uv, 0.5, 1.))


out_kernel.save(
    (Texture2DType(float, 4), ),
    "out_kernel.bytes",
    False
)
print("compile finished.")
