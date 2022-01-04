#pragma once
// Texture

const uint MAXLOD = 20;
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

void texture_write(Texture2D *tex, uint2 p, float4 value)
{
    if (p.x >= tex->width || p.y >= tex->height)
        // throw "texture write out of bound";
        print("texture write out of bound %u %u, %u %u\n", p.x, p.y, tex->width, tex->height);
    print("TEX WRITE %u %u %f %f %f %f\n", p.x, p.y, value.x, value.y, value.z, value.w);
    tex->data[(p.y * tex->width + p.x) * 4 + 0] = value.x;
    tex->data[(p.y * tex->width + p.x) * 4 + 1] = value.y;
    tex->data[(p.y * tex->width + p.x) * 4 + 2] = value.z;
    tex->data[(p.y * tex->width + p.x) * 4 + 3] = value.w;
}

float4 texture_read(Texture2D *tex, uint2 p)
{
    float4 value;
    value.x = tex->data[(p.y * tex->width + p.x) * 4 + 0];
    value.y = tex->data[(p.y * tex->width + p.x) * 4 + 1];
    value.z = tex->data[(p.y * tex->width + p.x) * 4 + 2];
    value.w = tex->data[(p.y * tex->width + p.x) * 4 + 3];
    return value;
}

#endif
