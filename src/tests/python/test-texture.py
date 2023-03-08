from luisa import *
from luisa.builtin import *
from luisa.types import *
init()

texture = Texture2D.from_ldr_image("logo.png")
res = texture.width, texture.height
display_tex = Texture2D(*res, 4, float, storage="BYTE")
# texture = Texture2D(*res, 4, float, storage="FLOAT")
block_size = 16


@func
def write_texture():
    set_block_size(block_size, block_size, 1)
    index = dispatch_id().xy
    color = texture.read(index)
    index = dispatch_size().xy - index.xy - 1
    display_tex.write(index, color)


gui = GUI("Test Texture", res)
while gui.running():
    write_texture(dispatch_size=(*res, 1))
    gui.set_image(display_tex)
    frame_time = gui.show()
synchronize()
