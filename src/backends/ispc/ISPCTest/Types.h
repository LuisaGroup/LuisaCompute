#pragma once
// Texture

#define MAXLOD 20
// need to make sure the layout is same across ISPC & C++
struct Texture2D {
    uint width;
    uint height;
    uint lodLevel;
    float* lods[MAXLOD];

#ifdef __cplusplus // host code

    Texture2D(uint width, uint height, uint lodLevel):
        width(width),
        height(height),
        lodLevel(lodLevel)
    {
        if (lodLevel > MAXLOD) {
            throw "maximal LoD exceeded";
        }
        // mipmap allocate
        int offset[MAXLOD+1];
        offset[0] = 0;
        for (int i=1, w=width, h=height; i<=lodLevel; ++i)
        {
            offset[i] = offset[i-1] + w*h*4;
            w = std::max(w/2, 1);
            h = std::max(h/2, 1);
        }
        lods[0] = new float[offset[lodLevel]*4];
        for (int i=1; i<lodLevel; ++i)
            lods[i] = lods[0] + offset[i];
    }

    ~Texture2D()
    {
        delete lods[0];
    }
#endif
};

#ifndef __cplusplus // ISPC code

#include "lib.h"

// typedef enum TEXTURE_FILTER {
//     POINT,
//     BILINEAR,
//     TRILINEAR
// };
// typedef enum TEXTURE_BIT_COUNT{
//     BIT_8 = 1,
//     BIT_16 = 2,
//     BIT_32 = 4
// };
// typedef enum TEXTURE_DATA_TYPE{
//     FLOAT,
//     UNORM,
//     SNORM,
//     UINT,
//     SINT
// };
// typedef enum TEXTURE_ADDRESS_MODE {
//     TEXTURE_ADDRESS_WRAP,
//     TEXTURE_ADDRESS_MIRROR,
//     TEXTURE_ADDRESS_CLAMP
// };

// struct SamplerState {
//     TEXTURE_FILTER filter;
//     TEXTURE_ADDRESS_MODE address;
//     //TEXTURE_ADDRESS_MODE addressW;
//     //FLOAT MipLODBias;
//     //UINT MaxAnisotropy;
//     //D3D11_COMPARISON_FUNC ComparisonFunc;
//     //FLOAT MinLOD;
//     //FLOAT MaxLOD;
// };



    // Texture2D(uint width, uint height, uint lodLevel):
    //     width(width),
    //     height(height),
    //     lodLevel(lodLevel)
    // {
    //     if (lodLevel != 1) throw "unimplemented";
    //     data = new float[width * height * 4];
    // }

// struct Texture3D {
//     uint width;
//     uint height;
//     uint depth;
//     uint numComponents;
//     uint lodLevel;
//     float** pData;
// };


struct TextureView {
    Texture2D* tex;
    uint level;
};

void texture_write(Texture2D *tex, uint2 p, uint level, float4 value)
{
    if (p.x >= tex->width || p.y >= tex->height)
        // throw "texture write out of bound";
        print("texture write out of bound %u %u, %u %u\n", p.x, p.y, tex->width, tex->height);
    print("TEX WRITE %u %u %f %f %f %f\n", p.x, p.y, value.x, value.y, value.z, value.w);
    tex->lods[level][(p.y * tex->width + p.x) * 4 + 0] = value.x;
    tex->lods[level][(p.y * tex->width + p.x) * 4 + 1] = value.y;
    tex->lods[level][(p.y * tex->width + p.x) * 4 + 2] = value.z;
    tex->lods[level][(p.y * tex->width + p.x) * 4 + 3] = value.w;
}

float4 texture_read(Texture2D *tex, uint2 p, uint level)
{
    float4 value;
    value.x = tex->lods[level][(p.y * tex->width + p.x) * 4 + 0];
    value.y = tex->lods[level][(p.y * tex->width + p.x) * 4 + 1];
    value.z = tex->lods[level][(p.y * tex->width + p.x) * 4 + 2];
    value.w = tex->lods[level][(p.y * tex->width + p.x) * 4 + 3];
    return value;
}

void texture_view_write(TextureView view, uint2 p, float4 value)
{
    texture_write(view.tex, p, view.level, value);
}
float4 texture_view_read(TextureView view, uint2 p)
{
    return texture_read(view.tex, p, view.level);
}


struct Ray {
    float<3> v0; // origin
    float v1; // t_min
    float<3> v2; // direction
    float v3; // t_max
};

struct Hit {
    uint v0; // inst
    uint v1; // prim
    float<2> v2; // uv
    // float4x4 v3; // object_to_world
};


float4 texture_sample_tmp(Texture2D *tex, float2 u, uint level)
{
    if (u.x<0 || u.x>1 || u.y<0 || u.y>1)
        return _float4(0.f);
    // bilinear
    uint w = max(tex->width>>level, 1u);
    uint h = max(tex->height>>level, 1u);
    float x = u.x * w - 0.5f;
    float y = u.y * h - 0.5f;
    float fx = frac(x);
    float fy = frac(y);
    uint x0 = (uint)max((int)0, (int)x);
    uint x1 = (uint)min((int)w-1, (int)x+1);
    uint y0 = (uint)max((int)0, (int)y);
    uint y1 = (uint)min((int)h-1, (int)y+1);
    return
    (1-fx)*(1-fy)*texture_read(tex, _uint2(x0,y0), level) +
    (1-fx)*(fy)*texture_read(tex, _uint2(x0,y1), level) +
    (fx)*(1-fy)*texture_read(tex, _uint2(x1,y0), level) +
    (fx)*(fy)*texture_read(tex, _uint2(x1,y1), level);
}


// float4 texture_sample(Texture2D *tex, float2 u, uint level)
// {
//     switch (addr) {
//         case TEXTURE_ADDRESS_WRAP:
//             u = fmod(u, 1.0f);
//             break;
//         case TEXTURE_ADDRESS_CLAMP:
//             u = clamp(u, _float3(0.0f), _float3(1.0f));
//             break;
//         case TEXTURE_ADDRESS_MIRROR:
//             {
//                 int2 mulOne = _int2(trunc(u)) % 2;
//                 if (mulOne.x)
//                     u.x = frac(u.x);
//                 else
//                     u.x = 1.0f - frac(u.x);
//                 if (mulOne.y)
//                     u.y = frac(u.y);
//                 else
//                     u.y = 1.0f - frac(u.y);
//             }
//             break;
//         case TEXTURE_ADDRESS_ZERO:
//             if (u.x<0 || u.x>1 || u.y<0 || u.y>1)
//                 return float4(0);
//             break;
//     }
//     // bilinear
//     uint w = max(tex->width>>iLevel, 1u);
//     uint h = max(tex->height>>iLevel, 1u);
//     float x = u.x * w - 0.5f;
//     float y = u.y * h - 0.5f;
//     float fx = frac(x);
//     float fy = frac(y);
//     uint x0 = (uint)max(0, int(x));
//     uint x1 = (uint)min(w-1, int(x)+1);
//     uint y0 = (uint)max(0, int(y));
//     uint y1 = (uint)min(h-1, int(y)+1);
//     return
//     (1-fx)*(1-fy)*texture_read(tex, uint2(x0,y0), level) +
//     (1-fx)*(fy)*texture_read(tex, uint2(x0,y1), level) +
//     (fx)*(1-fy)*texture_read(tex, uint2(x1,y0), level) +
//     (fx)*(fy)*texture_read(tex, uint2(x1,y1), level);
// }

// float4 texture_sample_level(Texture2D *tex, float2 u, float fLevel) {
//     if (tex->lodLevel == 1)
//         return texture_sample(tex, u, 0);
//     fLevel = clamp(fLevel, 0, tex->lodLevel-2);
//     uint iLevel = (uint)fLevel;
//     float t = fLevel-iLevel;
//     return (1-t)*texture_sample(tex, u, iLevel) +
//         t*texture_sample(tex, u, iLevel+1);
// }

#endif

