#pragma once
// Texture
typedef enum TEXTURE_FILTER {
    POINT,
    BILINEAR,
    TRILINEAR
};
typedef enum TEXTURE_BIT_COUNT{
    BIT_8 = 1,
    BIT_16 = 2,
    BIT_32 = 4
};
typedef enum TEXTURE_DATA_TYPE{
    FLOAT,
    UNORM,
    SNORM,
    UINT,
    SINT
};
typedef enum TEXTURE_ADDRESS_MODE {
    TEXTURE_ADDRESS_WRAP,
    TEXTURE_ADDRESS_MIRROR,
    TEXTURE_ADDRESS_CLAMP
};

struct SamplerState {
    TEXTURE_FILTER filter;
    TEXTURE_ADDRESS_MODE address;
    //TEXTURE_ADDRESS_MODE addressW;
    //FLOAT MipLODBias;
    //UINT MaxAnisotropy;
    //D3D11_COMPARISON_FUNC ComparisonFunc;
    //FLOAT MinLOD;
    //FLOAT MaxLOD;
};

struct Texture2D{
    uint width;
    uint height;
    uint numComponents;
    uint lodLevel;
    float** pData;
};

struct Texture3D {
    uint width;
    uint height;
    uint depth;
    uint numComponents;
    uint lodLevel;
    float** pData;
};