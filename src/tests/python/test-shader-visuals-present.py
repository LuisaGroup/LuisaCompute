from luisa import *
from luisa.types import *
from luisa.builtin import *
from luisa.util import *
import math
import numpy as np
init()


@func
def comp(p):
    p = asin(sin(p) * .9)
    return length(p) - 1.


@func
def erot(p, ax, ro):
    return lerp(dot(p, ax) * ax, p, cos(ro)) + sin(ro) * cross(ax, p)


@func
def smin(a, b, k):
    h = max(0., k - abs(b - a)) / k
    return min(a, b) + h * h * h * k / 6.


@func
def wrot(p):
    return float4(dot(p, float4(1.)), p.yzw + p.zwy - p.wyz - p.xxx) * .5


Data = StructType(
    lazors=float,
    doodad=float,
    p2=float3,
    d1=float,
    d2=float,
    d3=float,
)


@func
def scene(p, t, data):
    bpm = 125.
    data.p2 = erot(p, float3(0., 1., 0.), t)
    data.p2 = erot(data.p2, float3(0., 0., 1.), t / 3.)
    data.p2 = erot(data.p2, float3(1., 0., 0.), t / 5.)
    bpt = time / 60. * bpm
    p4 = float4(data.p2, 0.)
    p4 = lerp(p4, wrot(p4), smoothstep(-.5, .5, sin(bpt / 4.)))
    p4 = abs(p4)
    p4 = lerp(p4, wrot(p4), smoothstep(-.5, .5, sin(bpt)))
    fctr = smoothstep(-.5, .5, sin(bpt / 2.))
    fctr2 = smoothstep(.9, 1., sin(bpt / 16.))
    data.doodad = length(max(abs(p4) - lerp(0.05, 0.07, fctr), 0.) +
                         lerp(-0.1, .2, fctr)) - lerp(.15, .55, fctr * fctr) + fctr2
    p.x += asin(sin(t / 80.) * .99) * 80.
    data.lazors = length(
        asin(sin(erot(p, float3(1., 0., 0.), t * .2).yz * .5 + 1.)) / .5) - .1
    data.d1 = comp(p)
    data.d2 = comp(erot(p + 5., normalize(float3(1., 3., 4.)), .4))
    data.d3 = comp(erot(p + 10., normalize(float3(3., 2., 1.)), 1.))
    return min(data.doodad, min(data.lazors, .3 - smin(smin(data.d1, data.d2, .05), data.d3, .05)))


@func
def norm(p, t, data):
    precis = ite(length(p) < 1., .005, .01)
    k = float3x3(p, p, p) - float3x3(precis, 0.,
                                     0., 0., precis, 0., 0., 0., precis)
    return normalize(scene(p, t, data) - float3(scene(k[0], t, data), scene(k[1], t, data), scene(k[2], t, data)))


@func
def render_kernel(image, time):
    bpm = 125.
    fragCoord = float2(dispatch_id().xy)
    iResolution = float2(dispatch_size().xy)
    uv = (fragCoord - .5 * iResolution) / iResolution.y

    bpt = time / 60. * bpm
    bp = lerp(pow(sin(fract(bpt) * math.pi / 2.), 20.) + floor(bpt), bpt, .4)
    t = bp
    cam = normalize(float3(.8 + sin(bp * 3.14 / 4.) * .3, uv))
    init = float3(-1.5 + sin(bp * 3.14) * .2, 0., 0.) + cam * .2
    init = erot(init, float3(0., 1., 0.), sin(bp * .2) * .4)
    init = erot(init, float3(0., 0., 1.), cos(bp * .2) * .4)
    cam = erot(cam, float3(0., 1., 0.), sin(bp * .2) * .4)
    cam = erot(cam, float3(0., 0., 1.), cos(bp * .2) * .4)
    p = init
    atten = (1.)
    tlen = (0.)
    glo = (0.)
    fog = (0.)
    dlglo = (0.)
    trg = (False)
    dist = (0.)
    data = Data()
    for i in range(80):
        dist = scene(p, t, data)
        hit = dist * dist < 1e-6
        glo += .2 / (1. + data.lazors * data.lazors * 20.) * atten
        dlglo += .2 / (1. + data.doodad * data.doodad * 20.) * atten
        if (hit and ((sin(data.d3 * 45.) < -0.4 and (dist != data.doodad)) or (dist == data.doodad and sin(pow(length(data.p2 * data.p2 * data.p2), .3) * 120.) > .4)) and dist != data.lazors):
            trg = trg or dist == data.doodad
            hit = False
            n = norm(p, t, data)
            atten *= 1. - abs(dot(cam, n)) * .98
            cam = reflect(cam, n)
            dist = .1
        p += cam * dist
        tlen += dist
        fog += dist * atten / 30.
        if hit:
            break
    fog = smoothstep(0., 1., fog)
    lz = data.lazors == dist
    dl = data.doodad == dist
    fogcol = lerp(float3(.5, .8, 1.2), float3(.4, .6, .9), length(uv))
    n = norm(p, t, data)
    r = reflect(cam, n)
    ss = smoothstep(-.3, .3, scene(p + float3(.3), t, data)) + .5
    fact = length(sin(r * (ite(dl, 4., 3.)))
                  * .5 + .5) / sqrt(3.) * .7 + .3
    matcol = lerp(float3(.9, .4, .3), float3(.3, .4, .8),
                  smoothstep(-1., 1., sin(data.d1 * 5. + time * 2.)))
    matcol = lerp(matcol, float3(.5, .4, 1.), smoothstep(
        0., 1., sin(data.d2 * 5. + time * 2.)))
    matcol = ite(dl, lerp(1., matcol, .1) * .2 + .1, matcol)
    col = matcol * fact * ss + pow(fact, 10.)
    col = ite(lz, float3(4.), col)
    fragColor = col * atten + glo * glo + fogcol * glo
    fragColor = lerp(fragColor, fogcol, fog)
    fragColor = ite(dl, fragColor, abs(
        erot(fragColor, normalize(sin(p * 2.)), .2 * (1. - fog))))
    fragColor = ite(trg or dl, fragColor, fragColor +
                    dlglo * dlglo * .1 * float3(.4, .6, .9))
    fragColor = sqrt(fragColor)
    color = smoothstep(0., 1.2, fragColor)
    image.write(dispatch_id().xy, float4(pow(color, 2.2), 1.))


@func
def clear_kernel(image):
    coord = dispatch_id().xy
    image.write(coord, float4(0.3, 0.4, 0.5, 1.))


res = 1280, 720
image = Texture2D(*res, 4, float, storage="BYTE")
gui = GUI("Test shadertoy", res)
clear_kernel(image, dispatch_size=(*res, 1))
time = 0.0
while gui.running():
    gui.set_image(image)
    render_kernel(image, time, dispatch_size=(*res, 1))
    # use seconds
    time += gui.show() / 1000.0
synchronize()
