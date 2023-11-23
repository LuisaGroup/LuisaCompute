# Quick Start

```python
import luisa
from luisa import float2, float3, float4

n = 2048
luisa.init()
image = luisa.Image2D(2*n, n, 4, dtype=float)

@luisa.func
def draw(max_iter):
    p = dispatch_id().xy
    z,c = float2(0), 2 * p / n - float2(2, 1)
    for itr in range(max_iter):
        z = float2(z.x**2 - z.y**2, 2 * z.x * z.y) + c
        if length (z) > 20:
            break
    image.write(p , float4(float3(1 - itr/max_iter), 1))
    
# parallelized over (2*n, n) threads
draw(50, dispatch_size=(2*n, n)) 
image.to_image("mandelbrot.png")
```