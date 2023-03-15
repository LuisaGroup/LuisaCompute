from luisa import *
from luisa.builtin import *
from luisa.types import *
from luisa.util import *
init()


@func
def dot2(v):
    return dot(v, v)


@func
def ndot(a, b):
    return a.x*b.x - a.y*b.y


@func
def sdPlane(p):
    return p.y


@func
def sdSphere(p, s):
    return length(p)-s


@func
def sdBox(p, b):
    d = abs(p) - b
    return min(max(d.x, max(d.y, d.z)), 0.0) + length(max(d, 0.0))


@func
def sdBoxFrame(p, b, e):
    p = abs(p)-b
    q = abs(p+e)-e
    return min(min(
        length(max(float3(p.x, q.y, q.z), 0.0)) +
        min(max(p.x, max(q.y, q.z)), 0.0),
        length(max(float3(q.x, p.y, q.z), 0.0))+min(max(q.x, max(p.y, q.z)), 0.0)),
        length(max(float3(q.x, q.y, p.z), 0.0))+min(max(q.x, max(q.y, p.z)), 0.0))


@func
def sdEllipsoid(p, r):
    k0 = length(p/r)
    k1 = length(p/(r*r))
    return k0*(k0-1.0)/k1


@func
def sdTorus(p, t):
    return length(float2(length(p.xz)-t.x, p.y))-t.y


@func
def sdCappedTorus(p, sc, ra, rb):
    p.x = abs(p.x)
    k = ite((sc.y*p.x > sc.x*p.y), dot(p.xy, sc), length(p.xy))
    return sqrt(dot(p, p) + ra*ra - 2.0*ra*k) - rb


@func
def sdHexPrism(p, h):
    q = abs(p)
    k = float3(-0.8660254, 0.5, 0.57735)
    p = abs(p)
    p -= float3(2.0*min(dot(k.xy, p.xy), 0.0)*k.xy, 0)
    d = float2(
        length(p.xy - float2(clamp(p.x, -k.z*h.x, k.z*h.x), h.x)) *
        sign(p.y - h.x),
        p.z-h.y)
    return min(max(d.x, d.y), 0.0) + length(max(d, 0.0))


@func
def sdOctogonPrism(p, r, h):
    k = float3(-0.9238795325, 0.3826834323, 0.4142135623)
    p = abs(p)
    p -= float3(2.0*min(dot(float2(k.x, k.y), p.xy), 0.0)*float2(k.x, k.y), 0)
    p -= float3(2.0*min(dot(float2(-k.x, k.y), p.xy), 0.0)
                * float2(-k.x, k.y), 0)
    p -= float3(float2(clamp(p.x, -k.z*r, k.z*r), r), 0)
    d = float2(length(p.xy)*sign(p.y), p.z-h)
    return min(max(d.x, d.y), 0.0) + length(max(d, 0.0))


@func
def sdCapsule(p, a, b, r):
    pa = p-a
    ba = b-a
    h = clamp(dot(pa, ba)/dot(ba, ba), 0.0, 1.0)
    return length(pa - ba*h) - r


@func
def sdRoundCone0(p, r1, r2, h):
    q = float2(length(p.xz), p.y)
    b = (r1-r2)/h
    a = sqrt(1.0-b*b)
    k = dot(q, float2(-b, a))
    if k < 0.0:
        return length(q) - r1
    if k > a*h:
        return length(q-float2(0.0, h)) - r2
    return dot(q, float2(a, b)) - r1


@func
def sdRoundCone1(p, a, b, r1, r2):
    ba = b - a
    l2 = dot(ba, ba)
    rr = r1 - r2
    a2 = l2 - rr*rr
    il2 = 1.0/l2

    pa = p - a
    y = dot(pa, ba)
    z = y - l2
    x2 = dot2(pa*l2 - ba*y)
    y2 = y*y*l2
    z2 = z*z*l2

    k = sign(rr)*rr*rr*x2
    if (sign(z)*a2*z2 > k):
        return sqrt(x2 + z2) * il2 - r2
    if (sign(y)*a2*y2 < k):
        return sqrt(x2 + y2) * il2 - r1
    return (sqrt(x2*a2*il2)+y*rr)*il2 - r1


@func
def sdTriPrism(p, h):
    k = sqrt(3.)
    h.x *= 0.5*k
    p /= float3(float2(h.x), 1)
    p.x = abs(p.x) - 1.0
    p.y = p.y + 1.0/k
    if (p.x+k*p.y > 0.0):
        p = float3(float2(p.x-k*p.y, -k*p.x-p.y)/2.0, p.z)
    p.x -= clamp(p.x, -2.0, 0.0)
    d1 = length(p.xy)*sign(-p.y)*h.x
    d2 = abs(p.z)-h.y
    return length(max(float2(d1, d2), 0.0)) + min(max(d1, d2), 0.)


@func
def sdCylinder0(p, h):
    d = abs(float2(length(p.xz), p.y)) - h
    return min(max(d.x, d.y), 0.0) + length(max(d, 0.0))


@func
def sdCylinder1(p, a, b, r):
    pa = p - a
    ba = b - a
    baba = dot(ba, ba)
    paba = dot(pa, ba)

    x = length(pa*baba-ba*paba) - r*baba
    y = abs(paba-baba*0.5)-baba*0.5
    x2 = x*x
    y2 = y*y*baba
    d = ite((max(x, y) < 0.0), -min(x2, y2),
            (ite((x > 0.0), x2, 0.0)+ite((y > 0.0), y2, 0.0)))
    return sign(d)*sqrt(abs(d))/baba


@func
def sdCone(p, c, h):
    q = h*float2(c.x, -c.y)/c.y
    w = float2(length(p.xz), p.y)

    a = w - q*clamp(dot(w, q)/dot(q, q), 0.0, 1.0)
    b = w - q*float2(clamp(w.x/q.x, 0.0, 1.0), 1.0)
    k = sign(q.y)
    d = min(dot(a, a), dot(b, b))
    s = max(k*(w.x*q.y-w.y*q.x), k*(w.y-q.y))
    return sqrt(d)*sign(s)


@func
def sdCappedCone0(p, h, r1, r2):
    q = float2(length(p.xz), p.y)

    k1 = float2(r2, h)
    k2 = float2(r2-r1, 2.0*h)
    ca = float2(q.x-min(q.x, ite((q.y < 0.0), r1, r2)), abs(q.y)-h)
    cb = q - k1 + k2*clamp(dot(k1-q, k2)/dot2(k2), 0.0, 1.0)
    s = ite((cb.x < 0.0 and ca.y < 0.0), -1.0, 1.0)
    return s*sqrt(min(dot2(ca), dot2(cb)))


@func
def sdCappedCone1(p, a, b, ra, rb):
    rba = rb-ra
    baba = dot(b-a, b-a)
    papa = dot(p-a, p-a)
    paba = dot(p-a, b-a)/baba

    x = sqrt(papa - paba*paba*baba)

    cax = max(0.0, x-ite((paba < 0.5), ra, rb))
    cay = abs(paba-0.5)-0.5

    k = rba*rba + baba
    f = clamp((rba*(x-ra)+paba*baba)/k, 0.0, 1.0)

    cbx = x-ra - f*rba
    cby = paba - f

    s = ite((cbx < 0.0 and cay < 0.0), -1.0, 1.0)
    return s*sqrt(min(cax*cax + cay*cay*baba,
                      cbx*cbx + cby*cby*baba))


@func
def sdSolidAngle(pos, c, ra):
    p = float2(length(pos.xz), pos.y)
    l = length(p) - ra
    m = length(p - c*clamp(dot(p, c), 0.0, ra))
    return max(l, m*sign(c.y*p.x-c.x*p.y))


@func
def sdOctahedron(p, s):
    p = abs(p)
    m = p.x + p.y + p.z - s
    if (3.0*p.x < m):
        q = p.xyz
    elif (3.0*p.y < m):
        q = p.yzx
    elif (3.0*p.z < m):
        q = p.zxy
    else:
        return m*0.57735027
    k = clamp(0.5*(q.z-q.y+s), 0.0, s)
    return length(float3(q.x, q.y-s+k, q.z-k))


@func
def sdPyramid(p, h):
    m2 = h*h + 0.25

    # symmetry
    p_xz = p.xz
    p_xz = abs(p_xz)
    p_xz = ite((p_xz.y > p_xz.x), p_xz.yx, p_xz)
    p_xz -= 0.5
    p = float3(p_xz.x, p.y, p_xz.y)

    # project into face plane (2D)
    q = float3(p.z, h*p.y - 0.5*p.x, h*p.x + 0.5*p.y)

    s = max(-q.x, 0.0)
    t = clamp((q.y-0.5*p.z)/(m2+0.25), 0.0, 1.0)

    a = m2*(q.x+s)*(q.x+s) + q.y*q.y
    b = m2*(q.x+0.5*t)*(q.x+0.5*t) + (q.y-m2*t)*(q.y-m2*t)

    d2 = ite(min(q.y, -q.x*m2-q.y*0.5) > 0.0, 0.0, min(a, b))

    # recover 3D and scale, and add sign
    return sqrt((d2+q.z*q.z)/m2) * sign(max(q.z, -p.y))


@func
def sdRhombus(p, la, lb, h, ra):
    p = abs(p)
    b = float2(la, lb)
    f = clamp((ndot(b, b-2.0*p.xz))/dot(b, b), -1.0, 1.0)
    q = float2(length(p.xz-0.5*b*float2(1.0-f, 1.0+f))
               * sign(p.x*b.y+p.z*b.x-b.x*b.y)-ra, p.y-h)
    return min(max(q.x, q.y), 0.0) + length(max(q, 0.0))


@func
def sdHorseshoe(p, c, r, le, w):
    p.x = abs(p.x)
    l = length(p.xy)
    p = float3(float2x2(-c.x, c.y, c.y, c.x) * p.xy, p.z)
    p = float3(float2(ite((p.y > 0.0 or p.x > 0.0), p.x, l *
                          sign(-c.x)), ite((p.x > 0.0), p.y, l)), p.z)
    p = float3(float2(p.x, abs(p.y-r))-float2(le, 0.0), p.z)

    q = float2(length(max(p.xy, 0.0)) + min(0.0, max(p.x, p.y)), p.z)
    d = abs(q) - w
    return min(max(d.x, d.y), 0.0) + length(max(d, 0.0))


@func
def sdU(p, r, le, w):
    p.x = ite((p.y > 0.0), abs(p.x), length(p.xy))
    p.x = abs(p.x-r)
    p.y = p.y - le
    k = max(p.x, p.y)
    q = float2(ite((k < 0.0), -k, length(max(p.xy, 0.0))), abs(p.z)) - w
    return length(max(q, 0.0)) + min(max(q.x, q.y), 0.0)


@func
def opU(d1, d2):
    return ite(d1.x < d2.x, d1, d2)


@func
def map(pos):
    res = float2(pos.y, 0.0)

    # bounding box
    if (sdBox(pos-float3(-2.0, 0.3, 0.25), float3(0.3, 0.3, 1.0)) < res.x):
        res = opU(res, float2(sdSphere(pos-float3(-2.0, 0.25, 0.0), 0.25), 26.9))
        res = opU(res, float2(
            sdRhombus((pos-float3(-2.0, 0.25, 1.0)).xzy, 0.15, 0.25, 0.04, 0.08), 17.0))

    # bounding box
    if (sdBox(pos-float3(0.0, 0.3, -1.0), float3(0.35, 0.3, 2.5)) < res.x):
        res = opU(res, float2(sdCappedTorus((pos-float3(0.0, 0.30, 1.0))
                  * float3(1, -1, 1), float2(0.866025, -0.5), 0.25, 0.05), 25.0))
    res = opU(res, float2(sdBoxFrame(pos-float3(0.0, 0.25, 0.0),
              float3(0.3, 0.25, 0.2), 0.025), 16.9))
    res = opU(res, float2(
        sdCone(pos-float3(0.0, 0.45, -1.0), float2(0.6, 0.8), 0.45), 55.0))
    res = opU(res, float2(sdCappedCone0(
        pos-float3(0.0, 0.25, -2.0), 0.25, 0.25, 0.1), 13.67))
    res = opU(res, float2(sdSolidAngle(
        pos-float3(0.0, 0.00, -3.0), float2(3, 4)/5.0, 0.4), 49.13))

    # bounding box
    if (sdBox(pos-float3(1.0, 0.3, -1.0), float3(0.35, 0.3, 2.5)) < res.x):
        res = opU(res, float2(
            sdTorus((pos-float3(1.0, 0.30, 1.0)).xzy, float2(0.25, 0.05)), 7.1))
    res = opU(res, float2(
        sdBox(pos-float3(1.0, 0.25, 0.0), float3(0.3, 0.25, 0.1)), 3.0))
    res = opU(res, float2(sdCapsule(pos-float3(1.0, 0.00, -1.0),
              float3(-0.1, 0.1, -0.1), float3(0.2, 0.4, 0.2), 0.1), 31.9))
    res = opU(res, float2(sdCylinder0(
        pos-float3(1.0, 0.25, -2.0), float2(0.15, 0.25)), 8.0))
    res = opU(res, float2(sdHexPrism(
        pos-float3(1.0, 0.2, -3.0), float2(0.2, 0.05)), 18.4))

    # bounding box
    if (sdBox(pos-float3(-1.0, 0.35, -1.0), float3(0.35, 0.35, 2.5)) < res.x):
        res = opU(res, float2(
            sdPyramid(pos-float3(-1.0, -0.6, -3.0), 1.0), 13.56))
        res = opU(res, float2(sdOctahedron(
            pos-float3(-1.0, 0.15, -2.0), 0.35), 23.56))
    res = opU(res, float2(sdTriPrism(
        pos-float3(-1.0, 0.15, -1.0), float2(0.3, 0.05)), 43.5))
    res = opU(res, float2(sdEllipsoid(
        pos-float3(-1.0, 0.25, 0.0), float3(0.2, 0.25, 0.05)), 43.17))
    res = opU(res, float2(sdHorseshoe(pos-float3(-1.0, 0.25, 1.0),
              float2(cos(1.3), sin(1.3)), 0.2, 0.3, float2(0.03, 0.08)), 11.5))

    # bounding box
    if (sdBox(pos-float3(2.0, 0.3, -1.0), float3(0.35, 0.3, 2.5)) < res.x):
        res = opU(res, float2(sdOctogonPrism(
            pos-float3(2.0, 0.2, -3.0), 0.2, 0.05), 51.8))
    res = opU(res, float2(sdCylinder1(pos-float3(2.0, 0.14, -2.0),
              float3(0.1, -0.1, 0.0), float3(-0.2, 0.35, 0.1), 0.08), 31.2))
    res = opU(res, float2(sdCappedCone1(pos-float3(2.0, 0.09, -1.0),
              float3(0.1, 0.0, 0.0), float3(-0.2, 0.40, 0.1), 0.15, 0.05), 46.1))
    res = opU(res, float2(sdRoundCone1(pos-float3(2.0, 0.15, 0.0),
              float3(0.1, 0.0, 0.0), float3(-0.1, 0.35, 0.1), 0.15, 0.05), 51.7))
    res = opU(res, float2(sdRoundCone0(
        pos-float3(2.0, 0.20, 1.0), 0.2, 0.1, 0.3), 37.0))

    return res

# https://iquilezles.org/articles/boxfunctions


@func
def iBox(ro, rd, rad):
    m = 1.0/rd
    n = m*ro
    k = abs(m)*rad
    t1 = -n - k
    t2 = -n + k
    return float2(max(max(t1.x, t1.y), t1.z), min(min(t2.x, t2.y), t2.z))


@func
def raycast(ro, rd):
    res = float2(-1.0, -1.0)

    tmin = 1.0
    tmax = 20.0

    # raytrace floor plane
    tp1 = (0.0-ro.y)/rd.y
    if (tp1 > 0.0):
        tmax = min(tmax, tp1)
        res = float2(tp1, 1.0)

    # raymarch primitives
    tb = iBox(ro-float3(0.0, 0.4, -0.5), rd, float3(2.5, 0.41, 3.0))
    if (tb.x < tb.y and tb.y > 0.0 and tb.x < tmax):
        tmin = max(tb.x, tmin)
        tmax = min(tb.y, tmax)

        t = tmin
        for i in range(70):
            if t >= tmax:
                break
            h = map(ro+rd*t)
            if (abs(h.x) < (0.0001*t)):
                res = float2(t, h.y)
                break
            t += h.x

    return res


@func
def calcSoftshadow(ro, rd, mint, tmax):
    tp = (0.8-ro.y)/rd.y
    if (tp > 0.0):
        tmax = min(tmax, tp)

    res = 1.0
    t = mint
    for i in range(24):
        h = map(ro + rd*t).x
        s = clamp(8.0*h/t, 0.0, 1.0)
        res = min(res, s)
        t += clamp(h, 0.01, 0.2)
        if (res < 0.004 or t > tmax):
            break
    res = clamp(res, 0.0, 1.0)
    return res*res*(3.0-2.0*res)


@func
def calcNormal(pos):
    n = float3(0.0)
    for i in range(4):
        e = 0.5773*(2.0*float3((((i+3) >> 1) & 1),
                    ((i >> 1) & 1), (i & 1))-1.0)
        n += e*map(pos+0.0005*e).x
    return normalize(n)


@func
def calcAO(pos, nor):
    occ = 0.
    sca = 1.
    for i in range(5):
        h = 0.01 + 0.12*float(i)/4.0
        d = map(pos + h*nor).x
        occ += (h-d)*sca
        sca *= 0.95
        if (occ > 0.35):
            break
    return clamp(1.0 - 3.0*occ, 0.0, 1.0) * (0.5+0.5*nor.y)


@func
def checkersGradBox(p, dpdx, dpdy):
    w = abs(dpdx)+abs(dpdy) + 0.001
    i = 2.0*(abs(fract((p-0.5*w)*0.5)-0.5)-abs(fract((p+0.5*w)*0.5)-0.5))/w
    return 0.5 - 0.5*i.x*i.y


@func
def render(ro, rd, rdx, rdy):
    # background
    col = float3(0.7, 0.7, 0.9) - max(rd.y, 0.0)*0.3

    # raycast scene
    res = raycast(ro, rd)
    t = res.x
    m = res.y
    if (m > -0.5):
        pos = ro + t*rd
        nor = ite((m < 1.5), float3(0.0, 1.0, 0.0), calcNormal(pos))
        ref = reflect(rd, nor)

        # material
        col = 0.2 + 0.2*sin(m*2.0 + float3(0.0, 1.0, 2.0))
        ks = 1.0

        if (m < 1.5):
            # project pixel footprint into the plane
            dpdx = ro.y*(rd/rd.y-rdx/rdx.y)
            dpdy = ro.y*(rd/rd.y-rdy/rdy.y)

            f = checkersGradBox(3.0*pos.xz, 3.0*dpdx.xz, 3.0*dpdy.xz)
            col = 0.15 + f*float3(0.05)
            ks = 0.4

        # lighting
        occ = calcAO(pos, nor)

        lin = float3(0.0)

        # sun
        lig = normalize(float3(-0.5, 0.4, -0.6))
        hal = normalize(lig-rd)
        dif = clamp(dot(nor, lig), 0.0, 1.0)
        # if( dif>0.0001 )
        dif *= calcSoftshadow(pos, lig, 0.02, 2.5)
        spe = pow(clamp(dot(nor, hal), 0.0, 1.0), 16.0)
        spe *= dif
        spe *= 0.04+0.96*pow(clamp(1.0-dot(hal, lig), 0.0, 1.0), 5.0)
        # spe *= 0.04+0.96*pow(clamp(1.0-sqrt(0.5*(1.0-dot(rd,lig))),0.0,1.0),5.0)
        lin += col*2.20*dif*float3(1.30, 1.00, 0.70)
        lin += 5.00*spe*float3(1.30, 1.00, 0.70)*ks
        # sky
        dif = sqrt(clamp(0.5+0.5*nor.y, 0.0, 1.0))
        dif *= occ
        spe = smoothstep(-0.2, 0.2, ref.y)
        spe *= dif
        spe *= 0.04+0.96*pow(clamp(1.0+dot(nor, rd), 0.0, 1.0), 5.0)
        # if( spe>0.001 )
        spe *= calcSoftshadow(pos, ref, 0.02, 2.5)
        lin += col*0.60*dif*float3(0.40, 0.60, 1.15)
        lin += 2.00*spe*float3(0.40, 0.60, 1.30)*ks
        # back
        dif = clamp(dot(nor, normalize(float3(0.5, 0.0, 0.6))),
                    0.0, 1.0)*clamp(1.0-pos.y, 0.0, 1.0)
        dif *= occ
        lin += col*0.55*dif*float3(0.25, 0.25, 0.25)
        # sss
        dif = pow(clamp(1.0+dot(nor, rd), 0.0, 1.0), 2.0)
        dif *= occ
        lin += col*0.25*dif*float3(1.00, 1.00, 1.00)
        col = lin

        col = lerp(col, float3(0.7, 0.7, 0.9), 1.0-exp(-0.0001*t*t*t))

    return float3(clamp(col, 0.0, 1.0))


@func
def setCamera(ro, ta, cr):
    cw = normalize(ta-ro)
    cp = float3(sin(cr), cos(cr), 0.0)
    cu = normalize(cross(cw, cp))
    cv = (cross(cu, cw))
    return float3x3(cu, cv, cw)


@func
def kernel(mouse, time, img):
    set_block_size(16, 16, 1)
    fragCoord = float2(dispatch_id().xy)
    resolution = float2(dispatch_size().xy)
    fragCoord.y = resolution.x - fragCoord.y - 1
    mo = float2(mouse)
    ta = float3(0.25, -4.75, -0.75)
    ro = ta + float3(5.5*cos(0.1*time + 7.0*mo.x),
                     6.2,
                     5.5*sin(0.1*time + 7.0*mo.x))
    # camera-to-world transformation
    ca = setCamera(ro, ta, 0.0)
    tot = float3()
    p = (2.0*fragCoord-resolution.xy)/resolution.y
    fl = 2.5
    rd = ca * normalize(float3(p, fl))

    # ray differentials
    px = (2.0*(fragCoord+float2(1.0, 0.0))-resolution.xy)/resolution.y
    py = (2.0*(fragCoord+float2(0.0, 1.0))-resolution.xy)/resolution.y
    rdx = ca * normalize(float3(px, fl))
    rdy = ca * normalize(float3(py, fl))

    # render
    col = render(ro, rd, rdx, rdy)

    # gain
    # col = col*3.0/(2.5+col)

    # gamma
    col = pow(col, float3(0.4545))

    tot += col
    img.write(dispatch_id().xy, float4(tot, 1.))


@func
def clear_kernel(image):
    set_block_size(16, 16, 1)
    coord = dispatch_id().xy
    image.write(coord, float4(0.3, 0.4, 0.5, 1.))


super_sampling = 4
GroupArray = SharedArrayType(super_sampling * super_sampling, float3)


@func
def downsample_tex(image, out_image):
    set_block_size(super_sampling, super_sampling, 1)
    arr = GroupArray()
    local_coord = thread_id().xy
    col = image.read(dispatch_id().xy).xyz
    value = super_sampling
    while value > 1:
        next_value = value // 2
        if all(local_coord < uint2(value)):
            arr[value * local_coord.y + local_coord.x] = col
        sync_block()
        if all(local_coord < uint2(next_value)):
            last_coord = local_coord * 2
            col = arr[value * last_coord.y + last_coord.x] + \
                arr[value * (last_coord.y + 1) + last_coord.x] + \
                arr[value * last_coord.y + (last_coord.x + 1)] + \
                arr[value * (last_coord.y + 1) + (last_coord.x + 1)]
            col *= 0.25
            pass
        value = next_value
    if all(local_coord == uint2(0)):
        out_image.write(block_id().xy, float4(col, 1.))


res = 1280, 720
super_res = res[0] * super_sampling, res[1] * super_sampling
render_image = Texture2D(*super_res, 4, float, storage="BYTE")
image = Texture2D(*res, 4, float, storage="BYTE")
gui = GUI("Test raymarch", res)
clear_kernel(render_image, dispatch_size=(*super_res, 1))
time = 0.0
mouse_pos = float2(0.5)
while gui.running():
    # left mouse
    if gui.is_mouse_pressed(0):
        mouse_pos = gui.cursor_pos()
    kernel(mouse_pos, time, render_image, dispatch_size=(*super_res, 1))
    downsample_tex(render_image, image, dispatch_size=(*super_res, 1))
    gui.set_image(image)
    # use seconds
    time += gui.show() / 1000.0
synchronize()
