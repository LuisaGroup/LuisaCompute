import numpy as np
import LuisaCompute as lc
from astpretty import pprint
import inspect
import ast


class Nothing:
    def __getitem__(self, item):
        print(item)


def aces_tonemapping(x: float) -> float:
    a = 2.51
    b = 0.03
    c = 2.43
    d = 0.59
    e = 0.14
    x = (x * (a * x + b)) / (x * (c * x + d) + e)
    return max(min(x, 1.0), 0.0)


if __name__ == "__main__":
    lc.set_log_level_verbose()
    device = lc.Device("cuda", 0)
    buffer = device.create_buffer(lc.float4, 4096)
    image = device.create_image(lc.PIXEL_FORMAT_RGBA32F, 256)
    exit()
    image_slice = image[:64, 64:64 + 64]
    print(f"size = {image_slice.size}, total_size = {image_slice.total_size}, offset = {image_slice.offset}")
    view = buffer[1024:2048]
    in_data = np.array(np.random.rand(1024, 4), dtype=np.float32)
    out_data = np.empty([1024 * 4], dtype=np.float32)
    in_pixels = np.array(np.random.rand(256, 256, 4), dtype=np.float32)
    out_pixels = np.empty_like(in_pixels)
    with device.default_stream:
        view.copy_from(in_data)
        image.copy_from(in_pixels)
        view.copy_to(out_data)
        image.copy_to(out_pixels)
    with device.default_stream as s:
        s.synchronize()
    print(in_data)
    print(out_data)
    print(in_pixels)
    print(out_pixels)
    assert (in_data.reshape([-1]) == out_data).all()
    assert (in_pixels == out_pixels).all()

    src = inspect.getsource(aces_tonemapping)
    pprint(ast.parse(src, type_comments=True))

    c = lc.Constant([[1, 1], [2, 2], [3, 3]])
    print(c.element)
    print(c.array)
