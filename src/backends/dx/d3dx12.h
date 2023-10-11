#pragma once
//*********************************************************
//
// Copyright (c) Microsoft. All rights reserved.
// This code is licensed under the MIT License (MIT).
// THIS CODE IS PROVIDED *AS IS* WITHOUT WARRANTY OF
// ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING ANY
// IMPLIED WARRANTIES OF FITNESS FOR A PARTICULAR
// PURPOSE, MERCHANTABILITY, OR NON-INFRINGEMENT.
//
//*********************************************************
#include <luisa/vstl/common.h>
#include <Windows.h>
#include <d3d12.h>
#include <comdef.h>
#include <luisa/vstl/functional.h>
#include <dxgi.h>
#include <luisa/core/basic_types.h>
#include <luisa/core/logging.h>
#ifdef UNICODE
using lcdx_pchar = LPCWSTR;
#else
using lcdx_pchar = LPCSTR;
#endif
namespace lc::dx {
#define LUISA_MAKE_VECTOR_TYPES(T) \
    using T##2 = luisa::T##2;      \
    using T##3 = luisa::T##3;      \
    using T##4 = luisa::T##4;

LUISA_MAKE_VECTOR_TYPES(bool)
LUISA_MAKE_VECTOR_TYPES(float)
LUISA_MAKE_VECTOR_TYPES(int)
LUISA_MAKE_VECTOR_TYPES(uint)
using float2x2 = luisa::Matrix<2>;
using float3x3 = luisa::Matrix<3>;
using float4x4 = luisa::Matrix<4>;
enum class TextureDimension : uint8_t {
    None,
    Tex1D,
    Tex2D,
    Tex3D,
    Cubemap,
    Tex2DArray,
};
}// namespace lc::dx
enum GFXFormat {
    GFXFormat_Unknown = DXGI_FORMAT_UNKNOWN,
    GFXFormat_R32G32B32A32_Typeless = DXGI_FORMAT_R32G32B32A32_TYPELESS,
    GFXFormat_R32G32B32A32_Float = DXGI_FORMAT_R32G32B32A32_FLOAT,
    GFXFormat_R32G32B32A32_UInt = DXGI_FORMAT_R32G32B32A32_UINT,
    GFXFormat_R32G32B32A32_SInt = DXGI_FORMAT_R32G32B32A32_SINT,
    GFXFormat_R32G32B32_Typeless = DXGI_FORMAT_R32G32B32_TYPELESS,
    GFXFormat_R32G32B32_Float = DXGI_FORMAT_R32G32B32_FLOAT,
    GFXFormat_R32G32B32_UInt = DXGI_FORMAT_R32G32B32_UINT,
    GFXFormat_R32G32B32_SInt = DXGI_FORMAT_R32G32B32_SINT,
    GFXFormat_R16G16B16A16_Typeless = DXGI_FORMAT_R16G16B16A16_TYPELESS,
    GFXFormat_R16G16B16A16_Float = DXGI_FORMAT_R16G16B16A16_FLOAT,
    GFXFormat_R16G16B16A16_UNorm = DXGI_FORMAT_R16G16B16A16_UNORM,
    GFXFormat_R16G16B16A16_UInt = DXGI_FORMAT_R16G16B16A16_UINT,
    GFXFormat_R16G16B16A16_SNorm = DXGI_FORMAT_R16G16B16A16_SNORM,
    GFXFormat_R16G16B16A16_SInt = DXGI_FORMAT_R16G16B16A16_SINT,
    GFXFormat_R32G32_Typeless = DXGI_FORMAT_R32G32_TYPELESS,
    GFXFormat_R32G32_Float = DXGI_FORMAT_R32G32_FLOAT,
    GFXFormat_R32G32_UInt = DXGI_FORMAT_R32G32_UINT,
    GFXFormat_R32G32_SInt = DXGI_FORMAT_R32G32_SINT,
    GFXFormat_R32G8X24_Typeless = DXGI_FORMAT_R32G8X24_TYPELESS,
    GFXFormat_D32_Float_S8X24_UInt = DXGI_FORMAT_D32_FLOAT_S8X24_UINT,
    GFXFormat_R32_Float_X8X24_Typeless = DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS,
    GFXFormat_X32_Typeless_G8X24_UInt = DXGI_FORMAT_X32_TYPELESS_G8X24_UINT,
    GFXFormat_R10G10B10A2_Typeless = DXGI_FORMAT_R10G10B10A2_TYPELESS,
    GFXFormat_R10G10B10A2_UNorm = DXGI_FORMAT_R10G10B10A2_UNORM,
    GFXFormat_R10G10B10A2_UInt = DXGI_FORMAT_R10G10B10A2_UINT,
    GFXFormat_R11G11B10_Float = DXGI_FORMAT_R11G11B10_FLOAT,
    GFXFormat_R8G8B8A8_Typeless = DXGI_FORMAT_R8G8B8A8_TYPELESS,
    GFXFormat_R8G8B8A8_UNorm = DXGI_FORMAT_R8G8B8A8_UNORM,
    GFXFormat_R8G8B8A8_UNorm_SRGB = DXGI_FORMAT_R8G8B8A8_UNORM_SRGB,
    GFXFormat_R8G8B8A8_UInt = DXGI_FORMAT_R8G8B8A8_UINT,
    GFXFormat_R8G8B8A8_SNorm = DXGI_FORMAT_R8G8B8A8_SNORM,
    GFXFormat_R8G8B8A8_SInt = DXGI_FORMAT_R8G8B8A8_SINT,
    GFXFormat_R16G16_Typeless = DXGI_FORMAT_R16G16_TYPELESS,
    GFXFormat_R16G16_Float = DXGI_FORMAT_R16G16_FLOAT,
    GFXFormat_R16G16_UNorm = DXGI_FORMAT_R16G16_UNORM,
    GFXFormat_R16G16_UInt = DXGI_FORMAT_R16G16_UINT,
    GFXFormat_R16G16_SNorm = DXGI_FORMAT_R16G16_SNORM,
    GFXFormat_R16G16_SInt = DXGI_FORMAT_R16G16_SINT,
    GFXFormat_R32_Typeless = DXGI_FORMAT_R32_TYPELESS,
    GFXFormat_D32_Float = DXGI_FORMAT_D32_FLOAT,
    GFXFormat_R32_Float = DXGI_FORMAT_R32_FLOAT,
    GFXFormat_R32_UInt = DXGI_FORMAT_R32_UINT,
    GFXFormat_R32_SInt = DXGI_FORMAT_R32_SINT,
    GFXFormat_R24G8_Typeless = DXGI_FORMAT_R24G8_TYPELESS,
    GFXFormat_D24_UNorm_S8_UInt = DXGI_FORMAT_D24_UNORM_S8_UINT,
    GFXFormat_R24_UNorm_X8_Typeless = DXGI_FORMAT_R24_UNORM_X8_TYPELESS,
    GFXFormat_X24_Typeless_G8_UInt = DXGI_FORMAT_X24_TYPELESS_G8_UINT,
    GFXFormat_R8G8_Typeless = DXGI_FORMAT_R8G8_TYPELESS,
    GFXFormat_R8G8_UNorm = DXGI_FORMAT_R8G8_UNORM,
    GFXFormat_R8G8_UInt = DXGI_FORMAT_R8G8_UINT,
    GFXFormat_R8G8_SNorm = DXGI_FORMAT_R8G8_SNORM,
    GFXFormat_R8G8_SInt = DXGI_FORMAT_R8G8_SINT,
    GFXFormat_R16_Typeless = DXGI_FORMAT_R16_TYPELESS,
    GFXFormat_R16_Float = DXGI_FORMAT_R16_FLOAT,
    GFXFormat_D16_UNorm = DXGI_FORMAT_D16_UNORM,
    GFXFormat_R16_UNorm = DXGI_FORMAT_R16_UNORM,
    GFXFormat_R16_UInt = DXGI_FORMAT_R16_UINT,
    GFXFormat_R16_SNorm = DXGI_FORMAT_R16_SNORM,
    GFXFormat_R16_SInt = DXGI_FORMAT_R16_SINT,
    GFXFormat_R8_Typeless = DXGI_FORMAT_R8_TYPELESS,
    GFXFormat_R8_UNorm = DXGI_FORMAT_R8_UNORM,
    GFXFormat_R8_UInt = DXGI_FORMAT_R8_UINT,
    GFXFormat_R8_SNorm = DXGI_FORMAT_R8_SNORM,
    GFXFormat_R8_SInt = DXGI_FORMAT_R8_SINT,
    GFXFormat_A8_UNorm = DXGI_FORMAT_A8_UNORM,
    GFXFormat_R1_UNorm = DXGI_FORMAT_R1_UNORM,
    GFXFormat_R9G9B9E5_SharedExp = DXGI_FORMAT_R9G9B9E5_SHAREDEXP,
    GFXFormat_R8G8_B8G8_UNorm = DXGI_FORMAT_R8G8_B8G8_UNORM,
    GFXFormat_G8R8_G8B8_UNorm = DXGI_FORMAT_G8R8_G8B8_UNORM,
    GFXFormat_BC1_Typeless = DXGI_FORMAT_BC1_TYPELESS,
    GFXFormat_BC1_UNorm = DXGI_FORMAT_BC1_UNORM,
    GFXFormat_BC1_UNorm_SRGB = DXGI_FORMAT_BC1_UNORM_SRGB,
    GFXFormat_BC2_Typeless = DXGI_FORMAT_BC2_TYPELESS,
    GFXFormat_BC2_UNorm = DXGI_FORMAT_BC2_UNORM,
    GFXFormat_BC2_UNorm_SRGB = DXGI_FORMAT_BC2_UNORM_SRGB,
    GFXFormat_BC3_Typeless = DXGI_FORMAT_BC3_TYPELESS,
    GFXFormat_BC3_UNorm = DXGI_FORMAT_BC3_UNORM,
    GFXFormat_BC3_UNorm_SRGB = DXGI_FORMAT_BC3_UNORM_SRGB,
    GFXFormat_BC4_Typeless = DXGI_FORMAT_BC4_TYPELESS,
    GFXFormat_BC4_UNorm = DXGI_FORMAT_BC4_UNORM,
    GFXFormat_BC4_SNorm = DXGI_FORMAT_BC4_SNORM,
    GFXFormat_BC5_Typeless = DXGI_FORMAT_BC5_TYPELESS,
    GFXFormat_BC5_UNorm = DXGI_FORMAT_BC5_UNORM,
    GFXFormat_BC5_SNorm = DXGI_FORMAT_BC5_SNORM,
    GFXFormat_B5G6R5_UNorm = DXGI_FORMAT_B5G6R5_UNORM,
    GFXFormat_B5G5R5A1_UNorm = DXGI_FORMAT_B5G5R5A1_UNORM,
    GFXFormat_B8G8R8A8_UNorm = DXGI_FORMAT_B8G8R8A8_UNORM,
    GFXFormat_B8G8R8X8_UNorm = DXGI_FORMAT_B8G8R8X8_UNORM,
    GFXFormat_R10G10B10_XR_BIAS_A2_UNorm = DXGI_FORMAT_R10G10B10_XR_BIAS_A2_UNORM,
    GFXFormat_B8G8R8A8_Typeless = DXGI_FORMAT_B8G8R8A8_TYPELESS,
    GFXFormat_B8G8R8A8_UNorm_SRGB = DXGI_FORMAT_B8G8R8A8_UNORM_SRGB,
    GFXFormat_B8G8R8X8_Typeless = DXGI_FORMAT_B8G8R8X8_TYPELESS,
    GFXFormat_B8G8R8X8_UNorm_SRGB = DXGI_FORMAT_B8G8R8X8_UNORM_SRGB,
    GFXFormat_BC6H_Typeless = DXGI_FORMAT_BC6H_TYPELESS,
    GFXFormat_BC6H_UF16 = DXGI_FORMAT_BC6H_UF16,
    GFXFormat_BC6H_SF16 = DXGI_FORMAT_BC6H_SF16,
    GFXFormat_BC7_Typeless = DXGI_FORMAT_BC7_TYPELESS,
    GFXFormat_BC7_UNorm = DXGI_FORMAT_BC7_UNORM,
    GFXFormat_BC7_UNorm_SRGB = DXGI_FORMAT_BC7_UNORM_SRGB,
    GFXFormat_AYUV = DXGI_FORMAT_AYUV,
    GFXFormat_Y410 = DXGI_FORMAT_Y410,
    GFXFormat_Y416 = DXGI_FORMAT_Y416,
    GFXFormat_NV12 = DXGI_FORMAT_NV12,
    GFXFormat_P010 = DXGI_FORMAT_P010,
    GFXFormat_P016 = DXGI_FORMAT_P016,
    GFXFormat_420_OPAQUE = DXGI_FORMAT_420_OPAQUE,
    GFXFormat_YUY2 = DXGI_FORMAT_YUY2,
    GFXFormat_Y210 = DXGI_FORMAT_Y210,
    GFXFormat_Y216 = DXGI_FORMAT_Y216,
    GFXFormat_NV11 = DXGI_FORMAT_NV11,
    GFXFormat_AI44 = DXGI_FORMAT_AI44,
    GFXFormat_IA44 = DXGI_FORMAT_IA44,
    GFXFormat_P8 = DXGI_FORMAT_P8,
    GFXFormat_A8P8 = DXGI_FORMAT_A8P8,
    GFXFormat_B4G4R4A4_UNorm = DXGI_FORMAT_B4G4R4A4_UNORM,
    GFXFormat_P208 = DXGI_FORMAT_P208,
    GFXFormat_V208 = DXGI_FORMAT_V208,
    GFXFormat_V408 = DXGI_FORMAT_V408,
    GFXFormat_Sampler_FeedBack_Min_Mip_Opaque = DXGI_FORMAT_SAMPLER_FEEDBACK_MIN_MIP_OPAQUE,
    GFXFormat_Sampler_FeedBack_Mip_region_Used_Opaque = DXGI_FORMAT_SAMPLER_FEEDBACK_MIP_REGION_USED_OPAQUE,
    GFXFormat_Force_UInt = DXGI_FORMAT_FORCE_UINT,

};
namespace lc::dx {

inline uint64 CalcAlign(uint64 value, uint64 align) {
    return (value + (align - 1)) & ~(align - 1);
}
inline uint64 CalcConstantBufferByteSize(uint64 byteSize) {
    // Constant buffers must be a multiple of the minimum hardware
    // allocation size (usually 256 bytes).  So round up to nearest
    // multiple of 256.  We do this by adding 255 and then masking off
    // the lower 2 bytes which store all bits < 256.
    // Example: Suppose byteSize = 300.
    // (300 + 255) & ~255
    // 555 & ~255
    // 0x022B & ~0x00ff
    // 0x022B & 0xff00
    // 0x0200
    // 512
    return (byteSize + (D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT - 1)) & ~(D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT - 1);
}
inline uint64 CalcPlacedOffsetAlignment(uint64 offset) {
    return (offset + (D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT - 1)) & ~(D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT - 1);
}
}// namespace lc::dx

#if defined(__cplusplus)

struct CD3DX12_DEFAULT {};
extern const DECLSPEC_SELECTANY CD3DX12_DEFAULT D3D12_DEFAULT;

//------------------------------------------------------------------------------------------------
inline bool operator==(const D3D12_VIEWPORT &l, const D3D12_VIEWPORT &r) noexcept {
    return l.TopLeftX == r.TopLeftX && l.TopLeftY == r.TopLeftY && l.Width == r.Width && l.Height == r.Height && l.MinDepth == r.MinDepth && l.MaxDepth == r.MaxDepth;
}

//------------------------------------------------------------------------------------------------
inline bool operator!=(const D3D12_VIEWPORT &l, const D3D12_VIEWPORT &r) noexcept { return !(l == r); }

//------------------------------------------------------------------------------------------------
struct CD3DX12_RECT : public D3D12_RECT {
    CD3DX12_RECT() = default;
    explicit CD3DX12_RECT(const D3D12_RECT &o) noexcept : D3D12_RECT(o) {}
    explicit CD3DX12_RECT(
        LONG Left,
        LONG Top,
        LONG Right,
        LONG Bottom) noexcept {
        left = Left;
        top = Top;
        right = Right;
        bottom = Bottom;
    }
};

//------------------------------------------------------------------------------------------------
struct CD3DX12_VIEWPORT : public D3D12_VIEWPORT {
    CD3DX12_VIEWPORT() = default;
    explicit CD3DX12_VIEWPORT(const D3D12_VIEWPORT &o) noexcept : D3D12_VIEWPORT(o) {}
    explicit CD3DX12_VIEWPORT(
        FLOAT topLeftX,
        FLOAT topLeftY,
        FLOAT width,
        FLOAT height,
        FLOAT minDepth = D3D12_MIN_DEPTH,
        FLOAT maxDepth = D3D12_MAX_DEPTH) noexcept {
        TopLeftX = topLeftX;
        TopLeftY = topLeftY;
        Width = width;
        Height = height;
        MinDepth = minDepth;
        MaxDepth = maxDepth;
    }
    explicit CD3DX12_VIEWPORT(
        _In_ ID3D12Resource *pResource,
        uint mipSlice = 0,
        FLOAT topLeftX = 0.0f,
        FLOAT topLeftY = 0.0f,
        FLOAT minDepth = D3D12_MIN_DEPTH,
        FLOAT maxDepth = D3D12_MAX_DEPTH) noexcept {
        auto Desc = pResource->GetDesc();
        const UINT64 SubresourceWidth = Desc.Width >> mipSlice;
        const UINT64 SubresourceHeight = Desc.Height >> mipSlice;
        switch (Desc.Dimension) {
            case D3D12_RESOURCE_DIMENSION_BUFFER:
                TopLeftX = topLeftX;
                TopLeftY = 0.0f;
                Width = float(Desc.Width) - topLeftX;
                Height = 1.0f;
                break;
            case D3D12_RESOURCE_DIMENSION_TEXTURE1D:
                TopLeftX = topLeftX;
                TopLeftY = 0.0f;
                Width = (SubresourceWidth ? float(SubresourceWidth) : 1.0f) - topLeftX;
                Height = 1.0f;
                break;
            case D3D12_RESOURCE_DIMENSION_TEXTURE2D:
            case D3D12_RESOURCE_DIMENSION_TEXTURE3D:
                TopLeftX = topLeftX;
                TopLeftY = topLeftY;
                Width = (SubresourceWidth ? float(SubresourceWidth) : 1.0f) - topLeftX;
                Height = (SubresourceHeight ? float(SubresourceHeight) : 1.0f) - topLeftY;
                break;
            default: break;
        }

        MinDepth = minDepth;
        MaxDepth = maxDepth;
    }
};

//------------------------------------------------------------------------------------------------
struct CD3DX12_BOX : public D3D12_BOX {
    CD3DX12_BOX() = default;
    explicit CD3DX12_BOX(const D3D12_BOX &o) noexcept : D3D12_BOX(o) {}
    explicit CD3DX12_BOX(
        LONG Left,
        LONG Right) noexcept {
        left = static_cast<uint>(Left);
        top = 0;
        front = 0;
        right = static_cast<uint>(Right);
        bottom = 1;
        back = 1;
    }
    explicit CD3DX12_BOX(
        LONG Left,
        LONG Top,
        LONG Right,
        LONG Bottom) noexcept {
        left = static_cast<uint>(Left);
        top = static_cast<uint>(Top);
        front = 0;
        right = static_cast<uint>(Right);
        bottom = static_cast<uint>(Bottom);
        back = 1;
    }
    explicit CD3DX12_BOX(
        LONG Left,
        LONG Top,
        LONG Front,
        LONG Right,
        LONG Bottom,
        LONG Back) noexcept {
        left = static_cast<uint>(Left);
        top = static_cast<uint>(Top);
        front = static_cast<uint>(Front);
        right = static_cast<uint>(Right);
        bottom = static_cast<uint>(Bottom);
        back = static_cast<uint>(Back);
    }
};
inline bool operator==(const D3D12_BOX &l, const D3D12_BOX &r) noexcept {
    return l.left == r.left && l.top == r.top && l.front == r.front && l.right == r.right && l.bottom == r.bottom && l.back == r.back;
}
inline bool operator!=(const D3D12_BOX &l, const D3D12_BOX &r) noexcept { return !(l == r); }

//------------------------------------------------------------------------------------------------
struct CD3DX12_DEPTH_STENCIL_DESC : public D3D12_DEPTH_STENCIL_DESC {
    CD3DX12_DEPTH_STENCIL_DESC() = default;
    explicit CD3DX12_DEPTH_STENCIL_DESC(const D3D12_DEPTH_STENCIL_DESC &o) noexcept : D3D12_DEPTH_STENCIL_DESC(o) {}
    explicit CD3DX12_DEPTH_STENCIL_DESC(CD3DX12_DEFAULT) noexcept {
        DepthEnable = TRUE;
        DepthWriteMask = D3D12_DEPTH_WRITE_MASK_ALL;
        DepthFunc = D3D12_COMPARISON_FUNC_LESS;
        StencilEnable = FALSE;
        StencilReadMask = D3D12_DEFAULT_STENCIL_READ_MASK;
        StencilWriteMask = D3D12_DEFAULT_STENCIL_WRITE_MASK;
        const D3D12_DEPTH_STENCILOP_DESC defaultStencilOp =
            {D3D12_STENCIL_OP_KEEP, D3D12_STENCIL_OP_KEEP, D3D12_STENCIL_OP_KEEP, D3D12_COMPARISON_FUNC_ALWAYS};
        FrontFace = defaultStencilOp;
        BackFace = defaultStencilOp;
    }
    explicit CD3DX12_DEPTH_STENCIL_DESC(
        BOOL depthEnable,
        D3D12_DEPTH_WRITE_MASK depthWriteMask,
        D3D12_COMPARISON_FUNC depthFunc,
        BOOL stencilEnable,
        UINT8 stencilReadMask,
        UINT8 stencilWriteMask,
        D3D12_STENCIL_OP frontStencilFailOp,
        D3D12_STENCIL_OP frontStencilDepthFailOp,
        D3D12_STENCIL_OP frontStencilPassOp,
        D3D12_COMPARISON_FUNC frontStencilFunc,
        D3D12_STENCIL_OP backStencilFailOp,
        D3D12_STENCIL_OP backStencilDepthFailOp,
        D3D12_STENCIL_OP backStencilPassOp,
        D3D12_COMPARISON_FUNC backStencilFunc) noexcept {
        DepthEnable = depthEnable;
        DepthWriteMask = depthWriteMask;
        DepthFunc = depthFunc;
        StencilEnable = stencilEnable;
        StencilReadMask = stencilReadMask;
        StencilWriteMask = stencilWriteMask;
        FrontFace.StencilFailOp = frontStencilFailOp;
        FrontFace.StencilDepthFailOp = frontStencilDepthFailOp;
        FrontFace.StencilPassOp = frontStencilPassOp;
        FrontFace.StencilFunc = frontStencilFunc;
        BackFace.StencilFailOp = backStencilFailOp;
        BackFace.StencilDepthFailOp = backStencilDepthFailOp;
        BackFace.StencilPassOp = backStencilPassOp;
        BackFace.StencilFunc = backStencilFunc;
    }
};

//------------------------------------------------------------------------------------------------
struct CD3DX12_DEPTH_STENCIL_DESC1 : public D3D12_DEPTH_STENCIL_DESC1 {
    CD3DX12_DEPTH_STENCIL_DESC1() = default;
    explicit CD3DX12_DEPTH_STENCIL_DESC1(const D3D12_DEPTH_STENCIL_DESC1 &o) noexcept : D3D12_DEPTH_STENCIL_DESC1(o) {}
    explicit CD3DX12_DEPTH_STENCIL_DESC1(const D3D12_DEPTH_STENCIL_DESC &o) noexcept {
        DepthEnable = o.DepthEnable;
        DepthWriteMask = o.DepthWriteMask;
        DepthFunc = o.DepthFunc;
        StencilEnable = o.StencilEnable;
        StencilReadMask = o.StencilReadMask;
        StencilWriteMask = o.StencilWriteMask;
        FrontFace.StencilFailOp = o.FrontFace.StencilFailOp;
        FrontFace.StencilDepthFailOp = o.FrontFace.StencilDepthFailOp;
        FrontFace.StencilPassOp = o.FrontFace.StencilPassOp;
        FrontFace.StencilFunc = o.FrontFace.StencilFunc;
        BackFace.StencilFailOp = o.BackFace.StencilFailOp;
        BackFace.StencilDepthFailOp = o.BackFace.StencilDepthFailOp;
        BackFace.StencilPassOp = o.BackFace.StencilPassOp;
        BackFace.StencilFunc = o.BackFace.StencilFunc;
        DepthBoundsTestEnable = FALSE;
    }
    explicit CD3DX12_DEPTH_STENCIL_DESC1(CD3DX12_DEFAULT) noexcept {
        DepthEnable = TRUE;
        DepthWriteMask = D3D12_DEPTH_WRITE_MASK_ALL;
        DepthFunc = D3D12_COMPARISON_FUNC_LESS;
        StencilEnable = FALSE;
        StencilReadMask = D3D12_DEFAULT_STENCIL_READ_MASK;
        StencilWriteMask = D3D12_DEFAULT_STENCIL_WRITE_MASK;
        const D3D12_DEPTH_STENCILOP_DESC defaultStencilOp =
            {D3D12_STENCIL_OP_KEEP, D3D12_STENCIL_OP_KEEP, D3D12_STENCIL_OP_KEEP, D3D12_COMPARISON_FUNC_ALWAYS};
        FrontFace = defaultStencilOp;
        BackFace = defaultStencilOp;
        DepthBoundsTestEnable = FALSE;
    }
    explicit CD3DX12_DEPTH_STENCIL_DESC1(
        BOOL depthEnable,
        D3D12_DEPTH_WRITE_MASK depthWriteMask,
        D3D12_COMPARISON_FUNC depthFunc,
        BOOL stencilEnable,
        UINT8 stencilReadMask,
        UINT8 stencilWriteMask,
        D3D12_STENCIL_OP frontStencilFailOp,
        D3D12_STENCIL_OP frontStencilDepthFailOp,
        D3D12_STENCIL_OP frontStencilPassOp,
        D3D12_COMPARISON_FUNC frontStencilFunc,
        D3D12_STENCIL_OP backStencilFailOp,
        D3D12_STENCIL_OP backStencilDepthFailOp,
        D3D12_STENCIL_OP backStencilPassOp,
        D3D12_COMPARISON_FUNC backStencilFunc,
        BOOL depthBoundsTestEnable) noexcept {
        DepthEnable = depthEnable;
        DepthWriteMask = depthWriteMask;
        DepthFunc = depthFunc;
        StencilEnable = stencilEnable;
        StencilReadMask = stencilReadMask;
        StencilWriteMask = stencilWriteMask;
        FrontFace.StencilFailOp = frontStencilFailOp;
        FrontFace.StencilDepthFailOp = frontStencilDepthFailOp;
        FrontFace.StencilPassOp = frontStencilPassOp;
        FrontFace.StencilFunc = frontStencilFunc;
        BackFace.StencilFailOp = backStencilFailOp;
        BackFace.StencilDepthFailOp = backStencilDepthFailOp;
        BackFace.StencilPassOp = backStencilPassOp;
        BackFace.StencilFunc = backStencilFunc;
        DepthBoundsTestEnable = depthBoundsTestEnable;
    }
    operator D3D12_DEPTH_STENCIL_DESC() const noexcept {
        D3D12_DEPTH_STENCIL_DESC D;
        D.DepthEnable = DepthEnable;
        D.DepthWriteMask = DepthWriteMask;
        D.DepthFunc = DepthFunc;
        D.StencilEnable = StencilEnable;
        D.StencilReadMask = StencilReadMask;
        D.StencilWriteMask = StencilWriteMask;
        D.FrontFace.StencilFailOp = FrontFace.StencilFailOp;
        D.FrontFace.StencilDepthFailOp = FrontFace.StencilDepthFailOp;
        D.FrontFace.StencilPassOp = FrontFace.StencilPassOp;
        D.FrontFace.StencilFunc = FrontFace.StencilFunc;
        D.BackFace.StencilFailOp = BackFace.StencilFailOp;
        D.BackFace.StencilDepthFailOp = BackFace.StencilDepthFailOp;
        D.BackFace.StencilPassOp = BackFace.StencilPassOp;
        D.BackFace.StencilFunc = BackFace.StencilFunc;
        return D;
    }
};

//------------------------------------------------------------------------------------------------
struct CD3DX12_BLEND_DESC : public D3D12_BLEND_DESC {
    CD3DX12_BLEND_DESC() = default;
    explicit CD3DX12_BLEND_DESC(const D3D12_BLEND_DESC &o) noexcept : D3D12_BLEND_DESC(o) {}
    explicit CD3DX12_BLEND_DESC(CD3DX12_DEFAULT) noexcept {
        AlphaToCoverageEnable = FALSE;
        IndependentBlendEnable = FALSE;
        const D3D12_RENDER_TARGET_BLEND_DESC defaultRenderTargetBlendDesc =
            {
                FALSE,
                FALSE,
                D3D12_BLEND_ONE,
                D3D12_BLEND_ZERO,
                D3D12_BLEND_OP_ADD,
                D3D12_BLEND_ONE,
                D3D12_BLEND_ZERO,
                D3D12_BLEND_OP_ADD,
                D3D12_LOGIC_OP_NOOP,
                D3D12_COLOR_WRITE_ENABLE_ALL,
            };
        for (uint i = 0; i < D3D12_SIMULTANEOUS_RENDER_TARGET_COUNT; ++i)
            RenderTarget[i] = defaultRenderTargetBlendDesc;
    }
};

//------------------------------------------------------------------------------------------------
struct CD3DX12_RASTERIZER_DESC : public D3D12_RASTERIZER_DESC {
    CD3DX12_RASTERIZER_DESC() = default;
    explicit CD3DX12_RASTERIZER_DESC(const D3D12_RASTERIZER_DESC &o) noexcept : D3D12_RASTERIZER_DESC(o) {}
    explicit CD3DX12_RASTERIZER_DESC(CD3DX12_DEFAULT) noexcept {
        FillMode = D3D12_FILL_MODE_SOLID;
        CullMode = D3D12_CULL_MODE_BACK;
        FrontCounterClockwise = FALSE;
        DepthBias = D3D12_DEFAULT_DEPTH_BIAS;
        DepthBiasClamp = D3D12_DEFAULT_DEPTH_BIAS_CLAMP;
        SlopeScaledDepthBias = D3D12_DEFAULT_SLOPE_SCALED_DEPTH_BIAS;
        DepthClipEnable = TRUE;
        MultisampleEnable = FALSE;
        AntialiasedLineEnable = FALSE;
        ForcedSampleCount = 0;
        ConservativeRaster = D3D12_CONSERVATIVE_RASTERIZATION_MODE_OFF;
    }
    explicit CD3DX12_RASTERIZER_DESC(
        D3D12_FILL_MODE fillMode,
        D3D12_CULL_MODE cullMode,
        BOOL frontCounterClockwise,
        INT depthBias,
        FLOAT depthBiasClamp,
        FLOAT slopeScaledDepthBias,
        BOOL depthClipEnable,
        BOOL multisampleEnable,
        BOOL antialiasedLineEnable,
        uint forcedSampleCount,
        D3D12_CONSERVATIVE_RASTERIZATION_MODE conservativeRaster) noexcept {
        FillMode = fillMode;
        CullMode = cullMode;
        FrontCounterClockwise = frontCounterClockwise;
        DepthBias = depthBias;
        DepthBiasClamp = depthBiasClamp;
        SlopeScaledDepthBias = slopeScaledDepthBias;
        DepthClipEnable = depthClipEnable;
        MultisampleEnable = multisampleEnable;
        AntialiasedLineEnable = antialiasedLineEnable;
        ForcedSampleCount = forcedSampleCount;
        ConservativeRaster = conservativeRaster;
    }
};

//------------------------------------------------------------------------------------------------
struct CD3DX12_RESOURCE_ALLOCATION_INFO : public D3D12_RESOURCE_ALLOCATION_INFO {
    CD3DX12_RESOURCE_ALLOCATION_INFO() = default;
    explicit CD3DX12_RESOURCE_ALLOCATION_INFO(const D3D12_RESOURCE_ALLOCATION_INFO &o) noexcept : D3D12_RESOURCE_ALLOCATION_INFO(o) {}
    CD3DX12_RESOURCE_ALLOCATION_INFO(
        UINT64 size,
        UINT64 alignment) noexcept {
        SizeInBytes = size;
        Alignment = alignment;
    }
};

//------------------------------------------------------------------------------------------------
struct CD3DX12_HEAP_PROPERTIES : public D3D12_HEAP_PROPERTIES {
    CD3DX12_HEAP_PROPERTIES() = default;
    explicit CD3DX12_HEAP_PROPERTIES(const D3D12_HEAP_PROPERTIES &o) noexcept : D3D12_HEAP_PROPERTIES(o) {}
    CD3DX12_HEAP_PROPERTIES(
        D3D12_CPU_PAGE_PROPERTY cpuPageProperty,
        D3D12_MEMORY_POOL memoryPoolPreference,
        uint creationNodeMask = 1,
        uint nodeMask = 1) noexcept {
        Type = D3D12_HEAP_TYPE_CUSTOM;
        CPUPageProperty = cpuPageProperty;
        MemoryPoolPreference = memoryPoolPreference;
        CreationNodeMask = creationNodeMask;
        VisibleNodeMask = nodeMask;
    }
    explicit CD3DX12_HEAP_PROPERTIES(
        D3D12_HEAP_TYPE type,
        uint creationNodeMask = 1,
        uint nodeMask = 1) noexcept {
        Type = type;
        CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
        MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
        CreationNodeMask = creationNodeMask;
        VisibleNodeMask = nodeMask;
    }
    bool IsCPUAccessible() const noexcept {
        return Type == D3D12_HEAP_TYPE_UPLOAD || Type == D3D12_HEAP_TYPE_READBACK || (Type == D3D12_HEAP_TYPE_CUSTOM && (CPUPageProperty == D3D12_CPU_PAGE_PROPERTY_WRITE_COMBINE || CPUPageProperty == D3D12_CPU_PAGE_PROPERTY_WRITE_BACK));
    }
};
inline bool operator==(const D3D12_HEAP_PROPERTIES &l, const D3D12_HEAP_PROPERTIES &r) noexcept {
    return l.Type == r.Type && l.CPUPageProperty == r.CPUPageProperty && l.MemoryPoolPreference == r.MemoryPoolPreference && l.CreationNodeMask == r.CreationNodeMask && l.VisibleNodeMask == r.VisibleNodeMask;
}
inline bool operator!=(const D3D12_HEAP_PROPERTIES &l, const D3D12_HEAP_PROPERTIES &r) noexcept { return !(l == r); }

//------------------------------------------------------------------------------------------------
struct CD3DX12_HEAP_DESC : public D3D12_HEAP_DESC {
    CD3DX12_HEAP_DESC() = default;
    explicit CD3DX12_HEAP_DESC(const D3D12_HEAP_DESC &o) noexcept : D3D12_HEAP_DESC(o) {}
    CD3DX12_HEAP_DESC(
        UINT64 size,
        D3D12_HEAP_PROPERTIES properties,
        UINT64 alignment = 0,
        D3D12_HEAP_FLAGS flags = D3D12_HEAP_FLAG_NONE) noexcept {
        SizeInBytes = size;
        Properties = properties;
        Alignment = alignment;
        Flags = flags;
    }
    CD3DX12_HEAP_DESC(
        UINT64 size,
        D3D12_HEAP_TYPE type,
        UINT64 alignment = 0,
        D3D12_HEAP_FLAGS flags = D3D12_HEAP_FLAG_NONE) noexcept {
        SizeInBytes = size;
        Properties = CD3DX12_HEAP_PROPERTIES(type);
        Alignment = alignment;
        Flags = flags;
    }
    CD3DX12_HEAP_DESC(
        UINT64 size,
        D3D12_CPU_PAGE_PROPERTY cpuPageProperty,
        D3D12_MEMORY_POOL memoryPoolPreference,
        UINT64 alignment = 0,
        D3D12_HEAP_FLAGS flags = D3D12_HEAP_FLAG_NONE) noexcept {
        SizeInBytes = size;
        Properties = CD3DX12_HEAP_PROPERTIES(cpuPageProperty, memoryPoolPreference);
        Alignment = alignment;
        Flags = flags;
    }
    CD3DX12_HEAP_DESC(
        const D3D12_RESOURCE_ALLOCATION_INFO &resAllocInfo,
        D3D12_HEAP_PROPERTIES properties,
        D3D12_HEAP_FLAGS flags = D3D12_HEAP_FLAG_NONE) noexcept {
        SizeInBytes = resAllocInfo.SizeInBytes;
        Properties = properties;
        Alignment = resAllocInfo.Alignment;
        Flags = flags;
    }
    CD3DX12_HEAP_DESC(
        const D3D12_RESOURCE_ALLOCATION_INFO &resAllocInfo,
        D3D12_HEAP_TYPE type,
        D3D12_HEAP_FLAGS flags = D3D12_HEAP_FLAG_NONE) noexcept {
        SizeInBytes = resAllocInfo.SizeInBytes;
        Properties = CD3DX12_HEAP_PROPERTIES(type);
        Alignment = resAllocInfo.Alignment;
        Flags = flags;
    }
    CD3DX12_HEAP_DESC(
        const D3D12_RESOURCE_ALLOCATION_INFO &resAllocInfo,
        D3D12_CPU_PAGE_PROPERTY cpuPageProperty,
        D3D12_MEMORY_POOL memoryPoolPreference,
        D3D12_HEAP_FLAGS flags = D3D12_HEAP_FLAG_NONE) noexcept {
        SizeInBytes = resAllocInfo.SizeInBytes;
        Properties = CD3DX12_HEAP_PROPERTIES(cpuPageProperty, memoryPoolPreference);
        Alignment = resAllocInfo.Alignment;
        Flags = flags;
    }
    bool IsCPUAccessible() const noexcept { return static_cast<const CD3DX12_HEAP_PROPERTIES *>(&Properties)->IsCPUAccessible(); }
};
inline bool operator==(const D3D12_HEAP_DESC &l, const D3D12_HEAP_DESC &r) noexcept {
    return l.SizeInBytes == r.SizeInBytes && l.Properties == r.Properties && l.Alignment == r.Alignment && l.Flags == r.Flags;
}
inline bool operator!=(const D3D12_HEAP_DESC &l, const D3D12_HEAP_DESC &r) noexcept { return !(l == r); }

//------------------------------------------------------------------------------------------------
struct CD3DX12_CLEAR_VALUE : public D3D12_CLEAR_VALUE {
    CD3DX12_CLEAR_VALUE() = default;
    explicit CD3DX12_CLEAR_VALUE(const D3D12_CLEAR_VALUE &o) noexcept : D3D12_CLEAR_VALUE(o) {}
    CD3DX12_CLEAR_VALUE(
        DXGI_FORMAT format,
        const FLOAT color[4]) noexcept {
        Format = format;
        memcpy(Color, color, sizeof(Color));
    }
    CD3DX12_CLEAR_VALUE(
        DXGI_FORMAT format,
        FLOAT depth,
        UINT8 stencil) noexcept {
        Format = format;
        memset(&Color, 0, sizeof(Color));
        /* Use memcpy to preserve NAN values */
        memcpy(&DepthStencil.Depth, &depth, sizeof(depth));
        DepthStencil.Stencil = stencil;
    }
};

//------------------------------------------------------------------------------------------------
struct CD3DX12_RANGE : public D3D12_RANGE {
    CD3DX12_RANGE() = default;
    explicit CD3DX12_RANGE(const D3D12_RANGE &o) noexcept : D3D12_RANGE(o) {}
    CD3DX12_RANGE(
        SIZE_T begin,
        SIZE_T end) noexcept {
        Begin = begin;
        End = end;
    }
};

//------------------------------------------------------------------------------------------------
struct CD3DX12_RANGE_UInt64 : public D3D12_RANGE_UINT64 {
    CD3DX12_RANGE_UInt64() = default;
    explicit CD3DX12_RANGE_UInt64(const D3D12_RANGE_UINT64 &o) noexcept : D3D12_RANGE_UINT64(o) {}
    CD3DX12_RANGE_UInt64(
        UINT64 begin,
        UINT64 end) noexcept {
        Begin = begin;
        End = end;
    }
};

//------------------------------------------------------------------------------------------------
struct CD3DX12_SUBRESOURCE_RANGE_UInt64 : public D3D12_SUBRESOURCE_RANGE_UINT64 {
    CD3DX12_SUBRESOURCE_RANGE_UInt64() = default;
    explicit CD3DX12_SUBRESOURCE_RANGE_UInt64(const D3D12_SUBRESOURCE_RANGE_UINT64 &o) noexcept : D3D12_SUBRESOURCE_RANGE_UINT64(o) {}
    CD3DX12_SUBRESOURCE_RANGE_UInt64(
        uint subresource,
        const D3D12_RANGE_UINT64 &range) noexcept {
        Subresource = subresource;
        Range = range;
    }
    CD3DX12_SUBRESOURCE_RANGE_UInt64(
        uint subresource,
        UINT64 begin,
        UINT64 end) noexcept {
        Subresource = subresource;
        Range.Begin = begin;
        Range.End = end;
    }
};

//------------------------------------------------------------------------------------------------
struct CD3DX12_SHADER_BYTECODE : public D3D12_SHADER_BYTECODE {
    CD3DX12_SHADER_BYTECODE() = default;
    explicit CD3DX12_SHADER_BYTECODE(const D3D12_SHADER_BYTECODE &o) noexcept : D3D12_SHADER_BYTECODE(o) {}
    CD3DX12_SHADER_BYTECODE(
        _In_ ID3DBlob *pShaderBlob) noexcept {
        pShaderBytecode = pShaderBlob->GetBufferPointer();
        BytecodeLength = pShaderBlob->GetBufferSize();
    }
    CD3DX12_SHADER_BYTECODE(
        const void *_pShaderBytecode,
        SIZE_T bytecodeLength) noexcept {
        pShaderBytecode = _pShaderBytecode;
        BytecodeLength = bytecodeLength;
    }
};

//------------------------------------------------------------------------------------------------
struct CD3DX12_TILED_RESOURCE_COORDINATE : public D3D12_TILED_RESOURCE_COORDINATE {
    CD3DX12_TILED_RESOURCE_COORDINATE() = default;
    explicit CD3DX12_TILED_RESOURCE_COORDINATE(const D3D12_TILED_RESOURCE_COORDINATE &o) noexcept : D3D12_TILED_RESOURCE_COORDINATE(o) {}
    CD3DX12_TILED_RESOURCE_COORDINATE(
        uint x,
        uint y,
        uint z,
        uint subresource) noexcept {
        X = x;
        Y = y;
        Z = z;
        Subresource = subresource;
    }
};

//------------------------------------------------------------------------------------------------
struct CD3DX12_TILE_REGION_SIZE : public D3D12_TILE_REGION_SIZE {
    CD3DX12_TILE_REGION_SIZE() = default;
    explicit CD3DX12_TILE_REGION_SIZE(const D3D12_TILE_REGION_SIZE &o) noexcept : D3D12_TILE_REGION_SIZE(o) {}
    CD3DX12_TILE_REGION_SIZE(
        uint numTiles,
        BOOL useBox,
        uint width,
        UINT16 height,
        UINT16 depth) noexcept {
        NumTiles = numTiles;
        UseBox = useBox;
        Width = width;
        Height = height;
        Depth = depth;
    }
};

//------------------------------------------------------------------------------------------------
struct CD3DX12_SUBRESOURCE_TILING : public D3D12_SUBRESOURCE_TILING {
    CD3DX12_SUBRESOURCE_TILING() = default;
    explicit CD3DX12_SUBRESOURCE_TILING(const D3D12_SUBRESOURCE_TILING &o) noexcept : D3D12_SUBRESOURCE_TILING(o) {}
    CD3DX12_SUBRESOURCE_TILING(
        uint widthInTiles,
        UINT16 heightInTiles,
        UINT16 depthInTiles,
        uint startTileIndexInOverallResource) noexcept {
        WidthInTiles = widthInTiles;
        HeightInTiles = heightInTiles;
        DepthInTiles = depthInTiles;
        StartTileIndexInOverallResource = startTileIndexInOverallResource;
    }
};

//------------------------------------------------------------------------------------------------
struct CD3DX12_TILE_SHAPE : public D3D12_TILE_SHAPE {
    CD3DX12_TILE_SHAPE() = default;
    explicit CD3DX12_TILE_SHAPE(const D3D12_TILE_SHAPE &o) noexcept : D3D12_TILE_SHAPE(o) {}
    CD3DX12_TILE_SHAPE(
        uint widthInTexels,
        uint heightInTexels,
        uint depthInTexels) noexcept {
        WidthInTexels = widthInTexels;
        HeightInTexels = heightInTexels;
        DepthInTexels = depthInTexels;
    }
};
//------------------------------------------------------------------------------------------------
struct CD3DX12_RESOURCE_BARRIER : public D3D12_RESOURCE_BARRIER {
    CD3DX12_RESOURCE_BARRIER() = default;
    explicit CD3DX12_RESOURCE_BARRIER(const D3D12_RESOURCE_BARRIER &o) noexcept : D3D12_RESOURCE_BARRIER(o) {}
    static inline CD3DX12_RESOURCE_BARRIER Transition(
        _In_ ID3D12Resource *pResource,
        D3D12_RESOURCE_STATES stateBefore,
        D3D12_RESOURCE_STATES stateAfter,
        uint subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
        D3D12_RESOURCE_BARRIER_FLAGS flags = D3D12_RESOURCE_BARRIER_FLAG_NONE) noexcept {
        CD3DX12_RESOURCE_BARRIER result = {};
        D3D12_RESOURCE_BARRIER &barrier = result;
        result.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        result.Flags = flags;
        barrier.Transition.pResource = pResource;
        barrier.Transition.StateBefore = stateBefore;
        barrier.Transition.StateAfter = stateAfter;
        barrier.Transition.Subresource = subresource;
        return result;
    }
    static inline CD3DX12_RESOURCE_BARRIER Aliasing(
        _In_ ID3D12Resource *pResourceBefore,
        _In_ ID3D12Resource *pResourceAfter) noexcept {
        CD3DX12_RESOURCE_BARRIER result = {};
        D3D12_RESOURCE_BARRIER &barrier = result;
        result.Type = D3D12_RESOURCE_BARRIER_TYPE_ALIASING;
        barrier.Aliasing.pResourceBefore = pResourceBefore;
        barrier.Aliasing.pResourceAfter = pResourceAfter;
        return result;
    }
    static inline CD3DX12_RESOURCE_BARRIER UAV(
        _In_ ID3D12Resource *pResource) noexcept {
        CD3DX12_RESOURCE_BARRIER result = {};
        D3D12_RESOURCE_BARRIER &barrier = result;
        result.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
        barrier.UAV.pResource = pResource;
        return result;
    }
};

//------------------------------------------------------------------------------------------------
struct CD3DX12_PACKED_MIP_INFO : public D3D12_PACKED_MIP_INFO {
    CD3DX12_PACKED_MIP_INFO() = default;
    explicit CD3DX12_PACKED_MIP_INFO(const D3D12_PACKED_MIP_INFO &o) noexcept : D3D12_PACKED_MIP_INFO(o) {}
    CD3DX12_PACKED_MIP_INFO(
        UINT8 numStandardMips,
        UINT8 numPackedMips,
        uint numTilesForPackedMips,
        uint startTileIndexInOverallResource) noexcept {
        NumStandardMips = numStandardMips;
        NumPackedMips = numPackedMips;
        NumTilesForPackedMips = numTilesForPackedMips;
        StartTileIndexInOverallResource = startTileIndexInOverallResource;
    }
};

//------------------------------------------------------------------------------------------------
struct CD3DX12_SUBRESOURCE_FOOTPRINT : public D3D12_SUBRESOURCE_FOOTPRINT {
    CD3DX12_SUBRESOURCE_FOOTPRINT() = default;
    explicit CD3DX12_SUBRESOURCE_FOOTPRINT(const D3D12_SUBRESOURCE_FOOTPRINT &o) noexcept : D3D12_SUBRESOURCE_FOOTPRINT(o) {}
    CD3DX12_SUBRESOURCE_FOOTPRINT(
        DXGI_FORMAT format,
        uint width,
        uint height,
        uint depth,
        uint rowPitch) noexcept {
        Format = format;
        Width = width;
        Height = height;
        Depth = depth;
        RowPitch = rowPitch;
    }
    explicit CD3DX12_SUBRESOURCE_FOOTPRINT(
        const D3D12_RESOURCE_DESC &resDesc,
        uint rowPitch) noexcept {
        Format = resDesc.Format;
        Width = uint(resDesc.Width);
        Height = resDesc.Height;
        Depth = (resDesc.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D ? resDesc.DepthOrArraySize : 1);
        RowPitch = rowPitch;
    }
};

//------------------------------------------------------------------------------------------------
struct CD3DX12_TEXTURE_COPY_LOCATION : public D3D12_TEXTURE_COPY_LOCATION {
    CD3DX12_TEXTURE_COPY_LOCATION() = default;
    explicit CD3DX12_TEXTURE_COPY_LOCATION(const D3D12_TEXTURE_COPY_LOCATION &o) noexcept : D3D12_TEXTURE_COPY_LOCATION(o) {}
    CD3DX12_TEXTURE_COPY_LOCATION(_In_ ID3D12Resource *pRes) noexcept {
        pResource = pRes;
        Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        PlacedFootprint = {};
    }
    CD3DX12_TEXTURE_COPY_LOCATION(_In_ ID3D12Resource *pRes, D3D12_PLACED_SUBRESOURCE_FOOTPRINT const &Footprint) noexcept {
        pResource = pRes;
        Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        PlacedFootprint = Footprint;
    }
    CD3DX12_TEXTURE_COPY_LOCATION(_In_ ID3D12Resource *pRes, uint Sub) noexcept {
        pResource = pRes;
        Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        PlacedFootprint = {};
        SubresourceIndex = Sub;
    }
};

//------------------------------------------------------------------------------------------------
struct CD3DX12_DESCRIPTOR_RANGE : public D3D12_DESCRIPTOR_RANGE {
    CD3DX12_DESCRIPTOR_RANGE() = default;
    explicit CD3DX12_DESCRIPTOR_RANGE(const D3D12_DESCRIPTOR_RANGE &o) noexcept : D3D12_DESCRIPTOR_RANGE(o) {}
    CD3DX12_DESCRIPTOR_RANGE(
        D3D12_DESCRIPTOR_RANGE_TYPE rangeType,
        uint numDescriptors,
        uint baseShaderRegister,
        uint registerSpace = 0,
        uint offsetInDescriptorsFromTableStart =
            D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND) noexcept {
        Init(rangeType, numDescriptors, baseShaderRegister, registerSpace, offsetInDescriptorsFromTableStart);
    }

    inline void Init(
        D3D12_DESCRIPTOR_RANGE_TYPE rangeType,
        uint numDescriptors,
        uint baseShaderRegister,
        uint registerSpace = 0,
        uint offsetInDescriptorsFromTableStart =
            D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND) noexcept {
        Init(*this, rangeType, numDescriptors, baseShaderRegister, registerSpace, offsetInDescriptorsFromTableStart);
    }

    static inline void Init(
        _Out_ D3D12_DESCRIPTOR_RANGE &range,
        D3D12_DESCRIPTOR_RANGE_TYPE rangeType,
        uint numDescriptors,
        uint baseShaderRegister,
        uint registerSpace = 0,
        uint offsetInDescriptorsFromTableStart =
            D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND) noexcept {
        range.RangeType = rangeType;
        range.NumDescriptors = numDescriptors;
        range.BaseShaderRegister = baseShaderRegister;
        range.RegisterSpace = registerSpace;
        range.OffsetInDescriptorsFromTableStart = offsetInDescriptorsFromTableStart;
    }
};

//------------------------------------------------------------------------------------------------
struct CD3DX12_ROOT_DESCRIPTOR_TABLE : public D3D12_ROOT_DESCRIPTOR_TABLE {
    CD3DX12_ROOT_DESCRIPTOR_TABLE() = default;
    explicit CD3DX12_ROOT_DESCRIPTOR_TABLE(const D3D12_ROOT_DESCRIPTOR_TABLE &o) noexcept : D3D12_ROOT_DESCRIPTOR_TABLE(o) {}
    CD3DX12_ROOT_DESCRIPTOR_TABLE(
        uint numDescriptorRanges,
        _In_reads_opt_(numDescriptorRanges) const D3D12_DESCRIPTOR_RANGE *_pDescriptorRanges) noexcept {
        Init(numDescriptorRanges, _pDescriptorRanges);
    }

    inline void Init(
        uint numDescriptorRanges,
        _In_reads_opt_(numDescriptorRanges) const D3D12_DESCRIPTOR_RANGE *_pDescriptorRanges) noexcept {
        Init(*this, numDescriptorRanges, _pDescriptorRanges);
    }

    static inline void Init(
        _Out_ D3D12_ROOT_DESCRIPTOR_TABLE &rootDescriptorTable,
        uint numDescriptorRanges,
        _In_reads_opt_(numDescriptorRanges) const D3D12_DESCRIPTOR_RANGE *_pDescriptorRanges) noexcept {
        rootDescriptorTable.NumDescriptorRanges = numDescriptorRanges;
        rootDescriptorTable.pDescriptorRanges = _pDescriptorRanges;
    }
};

//------------------------------------------------------------------------------------------------
struct CD3DX12_ROOT_CONSTANTS : public D3D12_ROOT_CONSTANTS {
    CD3DX12_ROOT_CONSTANTS() = default;
    explicit CD3DX12_ROOT_CONSTANTS(const D3D12_ROOT_CONSTANTS &o) noexcept : D3D12_ROOT_CONSTANTS(o) {}
    CD3DX12_ROOT_CONSTANTS(
        uint num32BitValues,
        uint shaderRegister,
        uint registerSpace = 0) noexcept {
        Init(num32BitValues, shaderRegister, registerSpace);
    }

    inline void Init(
        uint num32BitValues,
        uint shaderRegister,
        uint registerSpace = 0) noexcept {
        Init(*this, num32BitValues, shaderRegister, registerSpace);
    }

    static inline void Init(
        _Out_ D3D12_ROOT_CONSTANTS &rootConstants,
        uint num32BitValues,
        uint shaderRegister,
        uint registerSpace = 0) noexcept {
        rootConstants.Num32BitValues = num32BitValues;
        rootConstants.ShaderRegister = shaderRegister;
        rootConstants.RegisterSpace = registerSpace;
    }
};

//------------------------------------------------------------------------------------------------
struct CD3DX12_ROOT_DESCRIPTOR : public D3D12_ROOT_DESCRIPTOR {
    CD3DX12_ROOT_DESCRIPTOR() = default;
    explicit CD3DX12_ROOT_DESCRIPTOR(const D3D12_ROOT_DESCRIPTOR &o) noexcept : D3D12_ROOT_DESCRIPTOR(o) {}
    CD3DX12_ROOT_DESCRIPTOR(
        uint shaderRegister,
        uint registerSpace = 0) noexcept {
        Init(shaderRegister, registerSpace);
    }

    inline void Init(
        uint shaderRegister,
        uint registerSpace = 0) noexcept {
        Init(*this, shaderRegister, registerSpace);
    }

    static inline void Init(_Out_ D3D12_ROOT_DESCRIPTOR &table, uint shaderRegister, uint registerSpace = 0) noexcept {
        table.ShaderRegister = shaderRegister;
        table.RegisterSpace = registerSpace;
    }
};

//------------------------------------------------------------------------------------------------
struct CD3DX12_ROOT_PARAMETER : public D3D12_ROOT_PARAMETER {
    CD3DX12_ROOT_PARAMETER() = default;
    explicit CD3DX12_ROOT_PARAMETER(const D3D12_ROOT_PARAMETER &o) noexcept : D3D12_ROOT_PARAMETER(o) {}

    static inline void InitAsDescriptorTable(
        _Out_ D3D12_ROOT_PARAMETER &rootParam,
        uint numDescriptorRanges,
        _In_reads_(numDescriptorRanges) const D3D12_DESCRIPTOR_RANGE *pDescriptorRanges,
        D3D12_SHADER_VISIBILITY visibility = D3D12_SHADER_VISIBILITY_ALL) noexcept {
        rootParam.ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        rootParam.ShaderVisibility = visibility;
        CD3DX12_ROOT_DESCRIPTOR_TABLE::Init(rootParam.DescriptorTable, numDescriptorRanges, pDescriptorRanges);
    }

    static inline void InitAsConstants(
        _Out_ D3D12_ROOT_PARAMETER &rootParam,
        uint num32BitValues,
        uint shaderRegister,
        uint registerSpace = 0,
        D3D12_SHADER_VISIBILITY visibility = D3D12_SHADER_VISIBILITY_ALL) noexcept {
        rootParam.ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
        rootParam.ShaderVisibility = visibility;
        CD3DX12_ROOT_CONSTANTS::Init(rootParam.Constants, num32BitValues, shaderRegister, registerSpace);
    }

    static inline void InitAsConstantBufferView(
        _Out_ D3D12_ROOT_PARAMETER &rootParam,
        uint shaderRegister,
        uint registerSpace = 0,
        D3D12_SHADER_VISIBILITY visibility = D3D12_SHADER_VISIBILITY_ALL) noexcept {
        rootParam.ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
        rootParam.ShaderVisibility = visibility;
        CD3DX12_ROOT_DESCRIPTOR::Init(rootParam.Descriptor, shaderRegister, registerSpace);
    }

    static inline void InitAsShaderResourceView(
        _Out_ D3D12_ROOT_PARAMETER &rootParam,
        uint shaderRegister,
        uint registerSpace = 0,
        D3D12_SHADER_VISIBILITY visibility = D3D12_SHADER_VISIBILITY_ALL) noexcept {
        rootParam.ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
        rootParam.ShaderVisibility = visibility;
        CD3DX12_ROOT_DESCRIPTOR::Init(rootParam.Descriptor, shaderRegister, registerSpace);
    }

    static inline void InitAsUnorderedAccessView(
        _Out_ D3D12_ROOT_PARAMETER &rootParam,
        uint shaderRegister,
        uint registerSpace = 0,
        D3D12_SHADER_VISIBILITY visibility = D3D12_SHADER_VISIBILITY_ALL) noexcept {
        rootParam.ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
        rootParam.ShaderVisibility = visibility;
        CD3DX12_ROOT_DESCRIPTOR::Init(rootParam.Descriptor, shaderRegister, registerSpace);
    }

    inline void InitAsDescriptorTable(
        uint numDescriptorRanges,
        _In_reads_(numDescriptorRanges) const D3D12_DESCRIPTOR_RANGE *pDescriptorRanges,
        D3D12_SHADER_VISIBILITY visibility = D3D12_SHADER_VISIBILITY_ALL) noexcept {
        InitAsDescriptorTable(*this, numDescriptorRanges, pDescriptorRanges, visibility);
    }

    inline void InitAsConstants(
        uint num32BitValues,
        uint shaderRegister,
        uint registerSpace = 0,
        D3D12_SHADER_VISIBILITY visibility = D3D12_SHADER_VISIBILITY_ALL) noexcept {
        InitAsConstants(*this, num32BitValues, shaderRegister, registerSpace, visibility);
    }

    inline void InitAsConstantBufferView(
        uint shaderRegister,
        uint registerSpace = 0,
        D3D12_SHADER_VISIBILITY visibility = D3D12_SHADER_VISIBILITY_ALL) noexcept {
        InitAsConstantBufferView(*this, shaderRegister, registerSpace, visibility);
    }

    inline void InitAsShaderResourceView(
        uint shaderRegister,
        uint registerSpace = 0,
        D3D12_SHADER_VISIBILITY visibility = D3D12_SHADER_VISIBILITY_ALL) noexcept {
        InitAsShaderResourceView(*this, shaderRegister, registerSpace, visibility);
    }

    inline void InitAsUnorderedAccessView(
        uint shaderRegister,
        uint registerSpace = 0,
        D3D12_SHADER_VISIBILITY visibility = D3D12_SHADER_VISIBILITY_ALL) noexcept {
        InitAsUnorderedAccessView(*this, shaderRegister, registerSpace, visibility);
    }
};

//------------------------------------------------------------------------------------------------
struct CD3DX12_STATIC_SAMPLER_DESC : public D3D12_STATIC_SAMPLER_DESC {
    CD3DX12_STATIC_SAMPLER_DESC() = default;
    explicit CD3DX12_STATIC_SAMPLER_DESC(const D3D12_STATIC_SAMPLER_DESC &o) noexcept : D3D12_STATIC_SAMPLER_DESC(o) {}
    CD3DX12_STATIC_SAMPLER_DESC(
        uint shaderRegister,
        D3D12_FILTER filter = D3D12_FILTER_ANISOTROPIC,
        D3D12_TEXTURE_ADDRESS_MODE addressU = D3D12_TEXTURE_ADDRESS_MODE_WRAP,
        D3D12_TEXTURE_ADDRESS_MODE addressV = D3D12_TEXTURE_ADDRESS_MODE_WRAP,
        D3D12_TEXTURE_ADDRESS_MODE addressW = D3D12_TEXTURE_ADDRESS_MODE_WRAP,
        FLOAT mipLODBias = 0,
        uint maxAnisotropy = 16,
        D3D12_COMPARISON_FUNC comparisonFunc = D3D12_COMPARISON_FUNC_LESS_EQUAL,
        D3D12_STATIC_BORDER_COLOR borderColor = D3D12_STATIC_BORDER_COLOR_OPAQUE_WHITE,
        FLOAT minLOD = 0.f,
        FLOAT maxLOD = D3D12_FLOAT32_MAX,
        D3D12_SHADER_VISIBILITY shaderVisibility = D3D12_SHADER_VISIBILITY_ALL,
        uint registerSpace = 0) noexcept {
        Init(
            shaderRegister,
            filter,
            addressU,
            addressV,
            addressW,
            mipLODBias,
            maxAnisotropy,
            comparisonFunc,
            borderColor,
            minLOD,
            maxLOD,
            shaderVisibility,
            registerSpace);
    }

    static inline void Init(
        _Out_ D3D12_STATIC_SAMPLER_DESC &samplerDesc,
        uint shaderRegister,
        D3D12_FILTER filter = D3D12_FILTER_ANISOTROPIC,
        D3D12_TEXTURE_ADDRESS_MODE addressU = D3D12_TEXTURE_ADDRESS_MODE_WRAP,
        D3D12_TEXTURE_ADDRESS_MODE addressV = D3D12_TEXTURE_ADDRESS_MODE_WRAP,
        D3D12_TEXTURE_ADDRESS_MODE addressW = D3D12_TEXTURE_ADDRESS_MODE_WRAP,
        FLOAT mipLODBias = 0,
        uint maxAnisotropy = 16,
        D3D12_COMPARISON_FUNC comparisonFunc = D3D12_COMPARISON_FUNC_LESS_EQUAL,
        D3D12_STATIC_BORDER_COLOR borderColor = D3D12_STATIC_BORDER_COLOR_OPAQUE_WHITE,
        FLOAT minLOD = 0.f,
        FLOAT maxLOD = D3D12_FLOAT32_MAX,
        D3D12_SHADER_VISIBILITY shaderVisibility = D3D12_SHADER_VISIBILITY_ALL,
        uint registerSpace = 0) noexcept {
        samplerDesc.ShaderRegister = shaderRegister;
        samplerDesc.Filter = filter;
        samplerDesc.AddressU = addressU;
        samplerDesc.AddressV = addressV;
        samplerDesc.AddressW = addressW;
        samplerDesc.MipLODBias = mipLODBias;
        samplerDesc.MaxAnisotropy = maxAnisotropy;
        samplerDesc.ComparisonFunc = comparisonFunc;
        samplerDesc.BorderColor = borderColor;
        samplerDesc.MinLOD = minLOD;
        samplerDesc.MaxLOD = maxLOD;
        samplerDesc.ShaderVisibility = shaderVisibility;
        samplerDesc.RegisterSpace = registerSpace;
    }
    inline void Init(
        uint shaderRegister,
        D3D12_FILTER filter = D3D12_FILTER_ANISOTROPIC,
        D3D12_TEXTURE_ADDRESS_MODE addressU = D3D12_TEXTURE_ADDRESS_MODE_WRAP,
        D3D12_TEXTURE_ADDRESS_MODE addressV = D3D12_TEXTURE_ADDRESS_MODE_WRAP,
        D3D12_TEXTURE_ADDRESS_MODE addressW = D3D12_TEXTURE_ADDRESS_MODE_WRAP,
        FLOAT mipLODBias = 0,
        uint maxAnisotropy = 16,
        D3D12_COMPARISON_FUNC comparisonFunc = D3D12_COMPARISON_FUNC_LESS_EQUAL,
        D3D12_STATIC_BORDER_COLOR borderColor = D3D12_STATIC_BORDER_COLOR_OPAQUE_WHITE,
        FLOAT minLOD = 0.f,
        FLOAT maxLOD = D3D12_FLOAT32_MAX,
        D3D12_SHADER_VISIBILITY shaderVisibility = D3D12_SHADER_VISIBILITY_ALL,
        uint registerSpace = 0) noexcept {
        Init(
            *this,
            shaderRegister,
            filter,
            addressU,
            addressV,
            addressW,
            mipLODBias,
            maxAnisotropy,
            comparisonFunc,
            borderColor,
            minLOD,
            maxLOD,
            shaderVisibility,
            registerSpace);
    }
};

//------------------------------------------------------------------------------------------------
struct CD3DX12_ROOT_SIGNATURE_DESC : public D3D12_ROOT_SIGNATURE_DESC {
    CD3DX12_ROOT_SIGNATURE_DESC() = default;
    explicit CD3DX12_ROOT_SIGNATURE_DESC(const D3D12_ROOT_SIGNATURE_DESC &o) noexcept : D3D12_ROOT_SIGNATURE_DESC(o) {}
    CD3DX12_ROOT_SIGNATURE_DESC(
        uint numParameters,
        _In_reads_opt_(numParameters) const D3D12_ROOT_PARAMETER *_pParameters,
        uint numStaticSamplers = 0,
        _In_reads_opt_(numStaticSamplers) const D3D12_STATIC_SAMPLER_DESC *_pStaticSamplers = nullptr,
        D3D12_ROOT_SIGNATURE_FLAGS flags = D3D12_ROOT_SIGNATURE_FLAG_NONE) noexcept {
        Init(numParameters, _pParameters, numStaticSamplers, _pStaticSamplers, flags);
    }
    CD3DX12_ROOT_SIGNATURE_DESC(CD3DX12_DEFAULT) noexcept {
        Init(0, nullptr, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_NONE);
    }

    inline void Init(
        uint numParameters,
        _In_reads_opt_(numParameters) const D3D12_ROOT_PARAMETER *_pParameters,
        uint numStaticSamplers = 0,
        _In_reads_opt_(numStaticSamplers) const D3D12_STATIC_SAMPLER_DESC *_pStaticSamplers = nullptr,
        D3D12_ROOT_SIGNATURE_FLAGS flags = D3D12_ROOT_SIGNATURE_FLAG_NONE) noexcept {
        Init(*this, numParameters, _pParameters, numStaticSamplers, _pStaticSamplers, flags);
    }

    static inline void Init(
        _Out_ D3D12_ROOT_SIGNATURE_DESC &desc,
        uint numParameters,
        _In_reads_opt_(numParameters) const D3D12_ROOT_PARAMETER *_pParameters,
        uint numStaticSamplers = 0,
        _In_reads_opt_(numStaticSamplers) const D3D12_STATIC_SAMPLER_DESC *_pStaticSamplers = nullptr,
        D3D12_ROOT_SIGNATURE_FLAGS flags = D3D12_ROOT_SIGNATURE_FLAG_NONE) noexcept {
        desc.NumParameters = numParameters;
        desc.pParameters = _pParameters;
        desc.NumStaticSamplers = numStaticSamplers;
        desc.pStaticSamplers = _pStaticSamplers;
        desc.Flags = flags;
    }
};

//------------------------------------------------------------------------------------------------
struct CD3DX12_DESCRIPTOR_RANGE1 : public D3D12_DESCRIPTOR_RANGE1 {
    CD3DX12_DESCRIPTOR_RANGE1() = default;
    explicit CD3DX12_DESCRIPTOR_RANGE1(const D3D12_DESCRIPTOR_RANGE1 &o) noexcept : D3D12_DESCRIPTOR_RANGE1(o) {}
    CD3DX12_DESCRIPTOR_RANGE1(
        D3D12_DESCRIPTOR_RANGE_TYPE rangeType,
        uint numDescriptors,
        uint baseShaderRegister,
        uint registerSpace = 0,
        D3D12_DESCRIPTOR_RANGE_FLAGS flags = D3D12_DESCRIPTOR_RANGE_FLAG_NONE,
        uint offsetInDescriptorsFromTableStart =
            D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND) noexcept {
        Init(rangeType, numDescriptors, baseShaderRegister, registerSpace, flags, offsetInDescriptorsFromTableStart);
    }

    inline void Init(
        D3D12_DESCRIPTOR_RANGE_TYPE rangeType,
        uint numDescriptors,
        uint baseShaderRegister,
        uint registerSpace = 0,
        D3D12_DESCRIPTOR_RANGE_FLAGS flags = D3D12_DESCRIPTOR_RANGE_FLAG_NONE,
        uint offsetInDescriptorsFromTableStart =
            D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND) noexcept {
        Init(*this, rangeType, numDescriptors, baseShaderRegister, registerSpace, flags, offsetInDescriptorsFromTableStart);
    }

    static inline void Init(
        _Out_ D3D12_DESCRIPTOR_RANGE1 &range,
        D3D12_DESCRIPTOR_RANGE_TYPE rangeType,
        uint numDescriptors,
        uint baseShaderRegister,
        uint registerSpace = 0,
        D3D12_DESCRIPTOR_RANGE_FLAGS flags = D3D12_DESCRIPTOR_RANGE_FLAG_NONE,
        uint offsetInDescriptorsFromTableStart =
            D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND) noexcept {
        range.RangeType = rangeType;
        range.NumDescriptors = numDescriptors;
        range.BaseShaderRegister = baseShaderRegister;
        range.RegisterSpace = registerSpace;
        range.Flags = flags;
        range.OffsetInDescriptorsFromTableStart = offsetInDescriptorsFromTableStart;
    }
};

//------------------------------------------------------------------------------------------------
struct CD3DX12_ROOT_DESCRIPTOR_TABLE1 : public D3D12_ROOT_DESCRIPTOR_TABLE1 {
    CD3DX12_ROOT_DESCRIPTOR_TABLE1() = default;
    explicit CD3DX12_ROOT_DESCRIPTOR_TABLE1(const D3D12_ROOT_DESCRIPTOR_TABLE1 &o) noexcept : D3D12_ROOT_DESCRIPTOR_TABLE1(o) {}
    CD3DX12_ROOT_DESCRIPTOR_TABLE1(
        uint numDescriptorRanges,
        _In_reads_opt_(numDescriptorRanges) const D3D12_DESCRIPTOR_RANGE1 *_pDescriptorRanges) noexcept {
        Init(numDescriptorRanges, _pDescriptorRanges);
    }

    inline void Init(
        uint numDescriptorRanges,
        _In_reads_opt_(numDescriptorRanges) const D3D12_DESCRIPTOR_RANGE1 *_pDescriptorRanges) noexcept {
        Init(*this, numDescriptorRanges, _pDescriptorRanges);
    }

    static inline void Init(
        _Out_ D3D12_ROOT_DESCRIPTOR_TABLE1 &rootDescriptorTable,
        uint numDescriptorRanges,
        _In_reads_opt_(numDescriptorRanges) const D3D12_DESCRIPTOR_RANGE1 *_pDescriptorRanges) noexcept {
        rootDescriptorTable.NumDescriptorRanges = numDescriptorRanges;
        rootDescriptorTable.pDescriptorRanges = _pDescriptorRanges;
    }
};

//------------------------------------------------------------------------------------------------
struct CD3DX12_ROOT_DESCRIPTOR1 : public D3D12_ROOT_DESCRIPTOR1 {
    CD3DX12_ROOT_DESCRIPTOR1() = default;
    explicit CD3DX12_ROOT_DESCRIPTOR1(const D3D12_ROOT_DESCRIPTOR1 &o) noexcept : D3D12_ROOT_DESCRIPTOR1(o) {}
    CD3DX12_ROOT_DESCRIPTOR1(
        uint shaderRegister,
        uint registerSpace = 0,
        D3D12_ROOT_DESCRIPTOR_FLAGS flags = D3D12_ROOT_DESCRIPTOR_FLAG_NONE) noexcept {
        Init(shaderRegister, registerSpace, flags);
    }

    inline void Init(
        uint shaderRegister,
        uint registerSpace = 0,
        D3D12_ROOT_DESCRIPTOR_FLAGS flags = D3D12_ROOT_DESCRIPTOR_FLAG_NONE) noexcept {
        Init(*this, shaderRegister, registerSpace, flags);
    }

    static inline void Init(
        _Out_ D3D12_ROOT_DESCRIPTOR1 &table,
        uint shaderRegister,
        uint registerSpace = 0,
        D3D12_ROOT_DESCRIPTOR_FLAGS flags = D3D12_ROOT_DESCRIPTOR_FLAG_NONE) noexcept {
        table.ShaderRegister = shaderRegister;
        table.RegisterSpace = registerSpace;
        table.Flags = flags;
    }
};

//------------------------------------------------------------------------------------------------
struct CD3DX12_ROOT_PARAMETER1 : public D3D12_ROOT_PARAMETER1 {
    CD3DX12_ROOT_PARAMETER1() = default;
    explicit CD3DX12_ROOT_PARAMETER1(const D3D12_ROOT_PARAMETER1 &o) noexcept : D3D12_ROOT_PARAMETER1(o) {}

    static inline void InitAsDescriptorTable(
        _Out_ D3D12_ROOT_PARAMETER1 &rootParam,
        uint numDescriptorRanges,
        _In_reads_(numDescriptorRanges) const D3D12_DESCRIPTOR_RANGE1 *pDescriptorRanges,
        D3D12_SHADER_VISIBILITY visibility = D3D12_SHADER_VISIBILITY_ALL) noexcept {
        rootParam.ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        rootParam.ShaderVisibility = visibility;
        CD3DX12_ROOT_DESCRIPTOR_TABLE1::Init(rootParam.DescriptorTable, numDescriptorRanges, pDescriptorRanges);
    }

    static inline void InitAsConstants(
        _Out_ D3D12_ROOT_PARAMETER1 &rootParam,
        uint num32BitValues,
        uint shaderRegister,
        uint registerSpace = 0,
        D3D12_SHADER_VISIBILITY visibility = D3D12_SHADER_VISIBILITY_ALL) noexcept {
        rootParam.ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
        rootParam.ShaderVisibility = visibility;
        CD3DX12_ROOT_CONSTANTS::Init(rootParam.Constants, num32BitValues, shaderRegister, registerSpace);
    }

    static inline void InitAsConstantBufferView(
        _Out_ D3D12_ROOT_PARAMETER1 &rootParam,
        uint shaderRegister,
        uint registerSpace = 0,
        D3D12_ROOT_DESCRIPTOR_FLAGS flags = D3D12_ROOT_DESCRIPTOR_FLAG_NONE,
        D3D12_SHADER_VISIBILITY visibility = D3D12_SHADER_VISIBILITY_ALL) noexcept {
        rootParam.ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
        rootParam.ShaderVisibility = visibility;
        CD3DX12_ROOT_DESCRIPTOR1::Init(rootParam.Descriptor, shaderRegister, registerSpace, flags);
    }

    static inline void InitAsShaderResourceView(
        _Out_ D3D12_ROOT_PARAMETER1 &rootParam,
        uint shaderRegister,
        uint registerSpace = 0,
        D3D12_ROOT_DESCRIPTOR_FLAGS flags = D3D12_ROOT_DESCRIPTOR_FLAG_NONE,
        D3D12_SHADER_VISIBILITY visibility = D3D12_SHADER_VISIBILITY_ALL) noexcept {
        rootParam.ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
        rootParam.ShaderVisibility = visibility;
        CD3DX12_ROOT_DESCRIPTOR1::Init(rootParam.Descriptor, shaderRegister, registerSpace, flags);
    }

    static inline void InitAsUnorderedAccessView(
        _Out_ D3D12_ROOT_PARAMETER1 &rootParam,
        uint shaderRegister,
        uint registerSpace = 0,
        D3D12_ROOT_DESCRIPTOR_FLAGS flags = D3D12_ROOT_DESCRIPTOR_FLAG_NONE,
        D3D12_SHADER_VISIBILITY visibility = D3D12_SHADER_VISIBILITY_ALL) noexcept {
        rootParam.ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
        rootParam.ShaderVisibility = visibility;
        CD3DX12_ROOT_DESCRIPTOR1::Init(rootParam.Descriptor, shaderRegister, registerSpace, flags);
    }

    inline void InitAsDescriptorTable(
        uint numDescriptorRanges,
        _In_reads_(numDescriptorRanges) const D3D12_DESCRIPTOR_RANGE1 *pDescriptorRanges,
        D3D12_SHADER_VISIBILITY visibility = D3D12_SHADER_VISIBILITY_ALL) noexcept {
        InitAsDescriptorTable(*this, numDescriptorRanges, pDescriptorRanges, visibility);
    }

    inline void InitAsConstants(
        uint num32BitValues,
        uint shaderRegister,
        uint registerSpace = 0,
        D3D12_SHADER_VISIBILITY visibility = D3D12_SHADER_VISIBILITY_ALL) noexcept {
        InitAsConstants(*this, num32BitValues, shaderRegister, registerSpace, visibility);
    }

    inline void InitAsConstantBufferView(
        uint shaderRegister,
        uint registerSpace = 0,
        D3D12_ROOT_DESCRIPTOR_FLAGS flags = D3D12_ROOT_DESCRIPTOR_FLAG_NONE,
        D3D12_SHADER_VISIBILITY visibility = D3D12_SHADER_VISIBILITY_ALL) noexcept {
        InitAsConstantBufferView(*this, shaderRegister, registerSpace, flags, visibility);
    }

    inline void InitAsShaderResourceView(
        uint shaderRegister,
        uint registerSpace = 0,
        D3D12_ROOT_DESCRIPTOR_FLAGS flags = D3D12_ROOT_DESCRIPTOR_FLAG_NONE,
        D3D12_SHADER_VISIBILITY visibility = D3D12_SHADER_VISIBILITY_ALL) noexcept {
        InitAsShaderResourceView(*this, shaderRegister, registerSpace, flags, visibility);
    }

    inline void InitAsUnorderedAccessView(
        uint shaderRegister,
        uint registerSpace = 0,
        D3D12_ROOT_DESCRIPTOR_FLAGS flags = D3D12_ROOT_DESCRIPTOR_FLAG_NONE,
        D3D12_SHADER_VISIBILITY visibility = D3D12_SHADER_VISIBILITY_ALL) noexcept {
        InitAsUnorderedAccessView(*this, shaderRegister, registerSpace, flags, visibility);
    }
};

//------------------------------------------------------------------------------------------------
struct CD3DX12_VERSIONED_ROOT_SIGNATURE_DESC : public D3D12_VERSIONED_ROOT_SIGNATURE_DESC {
    CD3DX12_VERSIONED_ROOT_SIGNATURE_DESC() = default;
    explicit CD3DX12_VERSIONED_ROOT_SIGNATURE_DESC(const D3D12_VERSIONED_ROOT_SIGNATURE_DESC &o) noexcept : D3D12_VERSIONED_ROOT_SIGNATURE_DESC(o) {}
    explicit CD3DX12_VERSIONED_ROOT_SIGNATURE_DESC(const D3D12_ROOT_SIGNATURE_DESC &o) noexcept {
        Version = D3D_ROOT_SIGNATURE_VERSION_1_0;
        Desc_1_0 = o;
    }
    explicit CD3DX12_VERSIONED_ROOT_SIGNATURE_DESC(const D3D12_ROOT_SIGNATURE_DESC1 &o) noexcept {
        Version = D3D_ROOT_SIGNATURE_VERSION_1_1;
        Desc_1_1 = o;
    }
    CD3DX12_VERSIONED_ROOT_SIGNATURE_DESC(
        uint numParameters,
        _In_reads_opt_(numParameters) const D3D12_ROOT_PARAMETER *_pParameters,
        uint numStaticSamplers = 0,
        _In_reads_opt_(numStaticSamplers) const D3D12_STATIC_SAMPLER_DESC *_pStaticSamplers = nullptr,
        D3D12_ROOT_SIGNATURE_FLAGS flags = D3D12_ROOT_SIGNATURE_FLAG_NONE) noexcept {
        Init_1_0(numParameters, _pParameters, numStaticSamplers, _pStaticSamplers, flags);
    }
    CD3DX12_VERSIONED_ROOT_SIGNATURE_DESC(
        uint numParameters,
        _In_reads_opt_(numParameters) const D3D12_ROOT_PARAMETER1 *_pParameters,
        uint numStaticSamplers = 0,
        _In_reads_opt_(numStaticSamplers) const D3D12_STATIC_SAMPLER_DESC *_pStaticSamplers = nullptr,
        D3D12_ROOT_SIGNATURE_FLAGS flags = D3D12_ROOT_SIGNATURE_FLAG_NONE) noexcept {
        Init_1_1(numParameters, _pParameters, numStaticSamplers, _pStaticSamplers, flags);
    }
    CD3DX12_VERSIONED_ROOT_SIGNATURE_DESC(CD3DX12_DEFAULT) noexcept {
        Init_1_1(0, nullptr, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_NONE);
    }

    inline void Init_1_0(
        uint numParameters,
        _In_reads_opt_(numParameters) const D3D12_ROOT_PARAMETER *_pParameters,
        uint numStaticSamplers = 0,
        _In_reads_opt_(numStaticSamplers) const D3D12_STATIC_SAMPLER_DESC *_pStaticSamplers = nullptr,
        D3D12_ROOT_SIGNATURE_FLAGS flags = D3D12_ROOT_SIGNATURE_FLAG_NONE) noexcept {
        Init_1_0(*this, numParameters, _pParameters, numStaticSamplers, _pStaticSamplers, flags);
    }

    static inline void Init_1_0(
        _Out_ D3D12_VERSIONED_ROOT_SIGNATURE_DESC &desc,
        uint numParameters,
        _In_reads_opt_(numParameters) const D3D12_ROOT_PARAMETER *_pParameters,
        uint numStaticSamplers = 0,
        _In_reads_opt_(numStaticSamplers) const D3D12_STATIC_SAMPLER_DESC *_pStaticSamplers = nullptr,
        D3D12_ROOT_SIGNATURE_FLAGS flags = D3D12_ROOT_SIGNATURE_FLAG_NONE) noexcept {
        desc.Version = D3D_ROOT_SIGNATURE_VERSION_1_0;
        desc.Desc_1_0.NumParameters = numParameters;
        desc.Desc_1_0.pParameters = _pParameters;
        desc.Desc_1_0.NumStaticSamplers = numStaticSamplers;
        desc.Desc_1_0.pStaticSamplers = _pStaticSamplers;
        desc.Desc_1_0.Flags = flags;
    }

    inline void Init_1_1(
        uint numParameters,
        _In_reads_opt_(numParameters) const D3D12_ROOT_PARAMETER1 *_pParameters,
        uint numStaticSamplers = 0,
        _In_reads_opt_(numStaticSamplers) const D3D12_STATIC_SAMPLER_DESC *_pStaticSamplers = nullptr,
        D3D12_ROOT_SIGNATURE_FLAGS flags = D3D12_ROOT_SIGNATURE_FLAG_NONE) noexcept {
        Init_1_1(*this, numParameters, _pParameters, numStaticSamplers, _pStaticSamplers, flags);
    }

    static inline void Init_1_1(
        _Out_ D3D12_VERSIONED_ROOT_SIGNATURE_DESC &desc,
        uint numParameters,
        _In_reads_opt_(numParameters) const D3D12_ROOT_PARAMETER1 *_pParameters,
        uint numStaticSamplers = 0,
        _In_reads_opt_(numStaticSamplers) const D3D12_STATIC_SAMPLER_DESC *_pStaticSamplers = nullptr,
        D3D12_ROOT_SIGNATURE_FLAGS flags = D3D12_ROOT_SIGNATURE_FLAG_NONE) noexcept {
        desc.Version = D3D_ROOT_SIGNATURE_VERSION_1_1;
        desc.Desc_1_1.NumParameters = numParameters;
        desc.Desc_1_1.pParameters = _pParameters;
        desc.Desc_1_1.NumStaticSamplers = numStaticSamplers;
        desc.Desc_1_1.pStaticSamplers = _pStaticSamplers;
        desc.Desc_1_1.Flags = flags;
    }
};

//------------------------------------------------------------------------------------------------
struct CD3DX12_CPU_DESCRIPTOR_HANDLE : public D3D12_CPU_DESCRIPTOR_HANDLE {
    CD3DX12_CPU_DESCRIPTOR_HANDLE() = default;
    explicit CD3DX12_CPU_DESCRIPTOR_HANDLE(const D3D12_CPU_DESCRIPTOR_HANDLE &o) noexcept : D3D12_CPU_DESCRIPTOR_HANDLE(o) {}
    CD3DX12_CPU_DESCRIPTOR_HANDLE(CD3DX12_DEFAULT) noexcept { ptr = 0; }
    CD3DX12_CPU_DESCRIPTOR_HANDLE(_In_ const D3D12_CPU_DESCRIPTOR_HANDLE &other, INT offsetScaledByIncrementSize) noexcept {
        InitOffsetted(other, offsetScaledByIncrementSize);
    }
    CD3DX12_CPU_DESCRIPTOR_HANDLE(_In_ const D3D12_CPU_DESCRIPTOR_HANDLE &other, INT offsetInDescriptors, uint descriptorIncrementSize) noexcept {
        InitOffsetted(other, offsetInDescriptors, descriptorIncrementSize);
    }
    CD3DX12_CPU_DESCRIPTOR_HANDLE &Offset(INT offsetInDescriptors, uint descriptorIncrementSize) noexcept {
        ptr = SIZE_T(INT64(ptr) + INT64(offsetInDescriptors) * INT64(descriptorIncrementSize));
        return *this;
    }
    CD3DX12_CPU_DESCRIPTOR_HANDLE &Offset(INT offsetScaledByIncrementSize) noexcept {
        ptr = SIZE_T(INT64(ptr) + INT64(offsetScaledByIncrementSize));
        return *this;
    }
    bool operator==(_In_ const D3D12_CPU_DESCRIPTOR_HANDLE &other) const noexcept {
        return (ptr == other.ptr);
    }
    bool operator!=(_In_ const D3D12_CPU_DESCRIPTOR_HANDLE &other) const noexcept {
        return (ptr != other.ptr);
    }
    CD3DX12_CPU_DESCRIPTOR_HANDLE &operator=(const D3D12_CPU_DESCRIPTOR_HANDLE &other) noexcept {
        ptr = other.ptr;
        return *this;
    }

    inline void InitOffsetted(_In_ const D3D12_CPU_DESCRIPTOR_HANDLE &base, INT offsetScaledByIncrementSize) noexcept {
        InitOffsetted(*this, base, offsetScaledByIncrementSize);
    }

    inline void InitOffsetted(_In_ const D3D12_CPU_DESCRIPTOR_HANDLE &base, INT offsetInDescriptors, uint descriptorIncrementSize) noexcept {
        InitOffsetted(*this, base, offsetInDescriptors, descriptorIncrementSize);
    }

    static inline void InitOffsetted(_Out_ D3D12_CPU_DESCRIPTOR_HANDLE &handle, _In_ const D3D12_CPU_DESCRIPTOR_HANDLE &base, INT offsetScaledByIncrementSize) noexcept {
        handle.ptr = SIZE_T(INT64(base.ptr) + INT64(offsetScaledByIncrementSize));
    }

    static inline void InitOffsetted(_Out_ D3D12_CPU_DESCRIPTOR_HANDLE &handle, _In_ const D3D12_CPU_DESCRIPTOR_HANDLE &base, INT offsetInDescriptors, uint descriptorIncrementSize) noexcept {
        handle.ptr = SIZE_T(INT64(base.ptr) + INT64(offsetInDescriptors) * INT64(descriptorIncrementSize));
    }
};

//------------------------------------------------------------------------------------------------
struct CD3DX12_GPU_DESCRIPTOR_HANDLE : public D3D12_GPU_DESCRIPTOR_HANDLE {
    CD3DX12_GPU_DESCRIPTOR_HANDLE() = default;
    explicit CD3DX12_GPU_DESCRIPTOR_HANDLE(const D3D12_GPU_DESCRIPTOR_HANDLE &o) noexcept : D3D12_GPU_DESCRIPTOR_HANDLE(o) {}
    CD3DX12_GPU_DESCRIPTOR_HANDLE(CD3DX12_DEFAULT) noexcept { ptr = 0; }
    CD3DX12_GPU_DESCRIPTOR_HANDLE(_In_ const D3D12_GPU_DESCRIPTOR_HANDLE &other, INT offsetScaledByIncrementSize) noexcept {
        InitOffsetted(other, offsetScaledByIncrementSize);
    }
    CD3DX12_GPU_DESCRIPTOR_HANDLE(_In_ const D3D12_GPU_DESCRIPTOR_HANDLE &other, INT offsetInDescriptors, uint descriptorIncrementSize) noexcept {
        InitOffsetted(other, offsetInDescriptors, descriptorIncrementSize);
    }
    CD3DX12_GPU_DESCRIPTOR_HANDLE &Offset(INT offsetInDescriptors, uint descriptorIncrementSize) noexcept {
        ptr = UINT64(INT64(ptr) + INT64(offsetInDescriptors) * INT64(descriptorIncrementSize));
        return *this;
    }
    CD3DX12_GPU_DESCRIPTOR_HANDLE &Offset(INT offsetScaledByIncrementSize) noexcept {
        ptr = UINT64(INT64(ptr) + INT64(offsetScaledByIncrementSize));
        return *this;
    }
    inline bool operator==(_In_ const D3D12_GPU_DESCRIPTOR_HANDLE &other) const noexcept {
        return (ptr == other.ptr);
    }
    inline bool operator!=(_In_ const D3D12_GPU_DESCRIPTOR_HANDLE &other) const noexcept {
        return (ptr != other.ptr);
    }
    CD3DX12_GPU_DESCRIPTOR_HANDLE &operator=(const D3D12_GPU_DESCRIPTOR_HANDLE &other) noexcept {
        ptr = other.ptr;
        return *this;
    }

    inline void InitOffsetted(_In_ const D3D12_GPU_DESCRIPTOR_HANDLE &base, INT offsetScaledByIncrementSize) noexcept {
        InitOffsetted(*this, base, offsetScaledByIncrementSize);
    }

    inline void InitOffsetted(_In_ const D3D12_GPU_DESCRIPTOR_HANDLE &base, INT offsetInDescriptors, uint descriptorIncrementSize) noexcept {
        InitOffsetted(*this, base, offsetInDescriptors, descriptorIncrementSize);
    }

    static inline void InitOffsetted(_Out_ D3D12_GPU_DESCRIPTOR_HANDLE &handle, _In_ const D3D12_GPU_DESCRIPTOR_HANDLE &base, INT offsetScaledByIncrementSize) noexcept {
        handle.ptr = UINT64(INT64(base.ptr) + INT64(offsetScaledByIncrementSize));
    }

    static inline void InitOffsetted(_Out_ D3D12_GPU_DESCRIPTOR_HANDLE &handle, _In_ const D3D12_GPU_DESCRIPTOR_HANDLE &base, INT offsetInDescriptors, uint descriptorIncrementSize) noexcept {
        handle.ptr = UINT64(INT64(base.ptr) + INT64(offsetInDescriptors) * INT64(descriptorIncrementSize));
    }
};

//------------------------------------------------------------------------------------------------
constexpr uint D3D12CalcSubresource(uint MipSlice, uint ArraySlice, uint PlaneSlice, uint MipLevels, uint ArraySize) noexcept {
    return MipSlice + ArraySlice * MipLevels + PlaneSlice * MipLevels * ArraySize;
}

//------------------------------------------------------------------------------------------------
template<typename T, typename U, typename V>
inline void D3D12DecomposeSubresource(uint Subresource, uint MipLevels, uint ArraySize, _Out_ T &MipSlice, _Out_ U &ArraySlice, _Out_ V &PlaneSlice) noexcept {
    MipSlice = static_cast<T>(Subresource % MipLevels);
    ArraySlice = static_cast<U>((Subresource / MipLevels) % ArraySize);
    PlaneSlice = static_cast<V>(Subresource / (MipLevels * ArraySize));
}

//------------------------------------------------------------------------------------------------
inline UINT8 D3D12GetFormatPlaneCount(
    _In_ ID3D12Device *pDevice,
    DXGI_FORMAT Format) noexcept {
    D3D12_FEATURE_DATA_FORMAT_INFO formatInfo = {Format, 0};
    if (FAILED(pDevice->CheckFeatureSupport(D3D12_FEATURE_FORMAT_INFO, &formatInfo, sizeof(formatInfo)))) {
        return 0;
    }
    return formatInfo.PlaneCount;
}

//------------------------------------------------------------------------------------------------
struct CD3DX12_RESOURCE_DESC : public D3D12_RESOURCE_DESC {
    CD3DX12_RESOURCE_DESC() = default;
    explicit CD3DX12_RESOURCE_DESC(const D3D12_RESOURCE_DESC &o) noexcept : D3D12_RESOURCE_DESC(o) {}
    CD3DX12_RESOURCE_DESC(
        D3D12_RESOURCE_DIMENSION dimension,
        UINT64 alignment,
        UINT64 width,
        uint height,
        UINT16 depthOrArraySize,
        UINT16 mipLevels,
        DXGI_FORMAT format,
        uint sampleCount,
        uint sampleQuality,
        D3D12_TEXTURE_LAYOUT layout,
        D3D12_RESOURCE_FLAGS flags) noexcept {
        Dimension = dimension;
        Alignment = alignment;
        Width = width;
        Height = height;
        DepthOrArraySize = depthOrArraySize;
        MipLevels = mipLevels;
        Format = format;
        SampleDesc.Count = sampleCount;
        SampleDesc.Quality = sampleQuality;
        Layout = layout;
        Flags = flags;
    }
    static inline CD3DX12_RESOURCE_DESC Buffer(
        const D3D12_RESOURCE_ALLOCATION_INFO &resAllocInfo,
        D3D12_RESOURCE_FLAGS flags = D3D12_RESOURCE_FLAG_NONE) noexcept {
        return CD3DX12_RESOURCE_DESC(D3D12_RESOURCE_DIMENSION_BUFFER, resAllocInfo.Alignment, resAllocInfo.SizeInBytes,
                                     1, 1, 1, DXGI_FORMAT_UNKNOWN, 1, 0, D3D12_TEXTURE_LAYOUT_ROW_MAJOR, flags);
    }
    static inline CD3DX12_RESOURCE_DESC Buffer(
        UINT64 width,
        D3D12_RESOURCE_FLAGS flags = D3D12_RESOURCE_FLAG_NONE,
        UINT64 alignment = 0) noexcept {
        return CD3DX12_RESOURCE_DESC(D3D12_RESOURCE_DIMENSION_BUFFER, alignment, width, 1, 1, 1,
                                     DXGI_FORMAT_UNKNOWN, 1, 0, D3D12_TEXTURE_LAYOUT_ROW_MAJOR, flags);
    }
    static inline CD3DX12_RESOURCE_DESC Tex1D(
        DXGI_FORMAT format,
        UINT64 width,
        UINT16 arraySize = 1,
        UINT16 mipLevels = 0,
        D3D12_RESOURCE_FLAGS flags = D3D12_RESOURCE_FLAG_NONE,
        D3D12_TEXTURE_LAYOUT layout = D3D12_TEXTURE_LAYOUT_UNKNOWN,
        UINT64 alignment = 0) noexcept {
        return CD3DX12_RESOURCE_DESC(D3D12_RESOURCE_DIMENSION_TEXTURE1D, alignment, width, 1, arraySize,
                                     mipLevels, format, 1, 0, layout, flags);
    }
    static inline CD3DX12_RESOURCE_DESC Tex2D(
        DXGI_FORMAT format,
        UINT64 width,
        uint height,
        UINT16 arraySize = 1,
        UINT16 mipLevels = 0,
        uint sampleCount = 1,
        uint sampleQuality = 0,
        D3D12_RESOURCE_FLAGS flags = D3D12_RESOURCE_FLAG_NONE,
        D3D12_TEXTURE_LAYOUT layout = D3D12_TEXTURE_LAYOUT_UNKNOWN,
        UINT64 alignment = 0) noexcept {
        return CD3DX12_RESOURCE_DESC(D3D12_RESOURCE_DIMENSION_TEXTURE2D, alignment, width, height, arraySize,
                                     mipLevels, format, sampleCount, sampleQuality, layout, flags);
    }
    static inline CD3DX12_RESOURCE_DESC Tex3D(
        DXGI_FORMAT format,
        UINT64 width,
        uint height,
        UINT16 depth,
        UINT16 mipLevels = 0,
        D3D12_RESOURCE_FLAGS flags = D3D12_RESOURCE_FLAG_NONE,
        D3D12_TEXTURE_LAYOUT layout = D3D12_TEXTURE_LAYOUT_UNKNOWN,
        UINT64 alignment = 0) noexcept {
        return CD3DX12_RESOURCE_DESC(D3D12_RESOURCE_DIMENSION_TEXTURE3D, alignment, width, height, depth,
                                     mipLevels, format, 1, 0, layout, flags);
    }
    inline UINT16 Depth() const noexcept { return (Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D ? DepthOrArraySize : 1); }
    inline UINT16 ArraySize() const noexcept { return (Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE3D ? DepthOrArraySize : 1); }
    inline UINT8 PlaneCount(_In_ ID3D12Device *pDevice) const noexcept { return D3D12GetFormatPlaneCount(pDevice, Format); }
    inline uint Subresources(_In_ ID3D12Device *pDevice) const noexcept { return MipLevels * ArraySize() * PlaneCount(pDevice); }
    inline uint CalcSubresource(uint MipSlice, uint ArraySlice, uint PlaneSlice) noexcept { return D3D12CalcSubresource(MipSlice, ArraySlice, PlaneSlice, MipLevels, ArraySize()); }
};
inline bool operator==(const D3D12_RESOURCE_DESC &l, const D3D12_RESOURCE_DESC &r) noexcept {
    return l.Dimension == r.Dimension && l.Alignment == r.Alignment && l.Width == r.Width && l.Height == r.Height && l.DepthOrArraySize == r.DepthOrArraySize && l.MipLevels == r.MipLevels && l.Format == r.Format && l.SampleDesc.Count == r.SampleDesc.Count && l.SampleDesc.Quality == r.SampleDesc.Quality && l.Layout == r.Layout && l.Flags == r.Flags;
}
inline bool operator!=(const D3D12_RESOURCE_DESC &l, const D3D12_RESOURCE_DESC &r) noexcept { return !(l == r); }

//------------------------------------------------------------------------------------------------
struct CD3DX12_RESOURCE_DESC1 : public D3D12_RESOURCE_DESC1 {
    CD3DX12_RESOURCE_DESC1() = default;
    explicit CD3DX12_RESOURCE_DESC1(const D3D12_RESOURCE_DESC1 &o) noexcept : D3D12_RESOURCE_DESC1(o) {}
    CD3DX12_RESOURCE_DESC1(
        D3D12_RESOURCE_DIMENSION dimension,
        UINT64 alignment,
        UINT64 width,
        uint height,
        UINT16 depthOrArraySize,
        UINT16 mipLevels,
        DXGI_FORMAT format,
        uint sampleCount,
        uint sampleQuality,
        D3D12_TEXTURE_LAYOUT layout,
        D3D12_RESOURCE_FLAGS flags,
        uint samplerFeedbackMipRegionWidth = 0,
        uint samplerFeedbackMipRegionHeight = 0,
        uint samplerFeedbackMipRegionDepth = 0) noexcept {
        Dimension = dimension;
        Alignment = alignment;
        Width = width;
        Height = height;
        DepthOrArraySize = depthOrArraySize;
        MipLevels = mipLevels;
        Format = format;
        SampleDesc.Count = sampleCount;
        SampleDesc.Quality = sampleQuality;
        Layout = layout;
        Flags = flags;
        SamplerFeedbackMipRegion.Width = samplerFeedbackMipRegionWidth;
        SamplerFeedbackMipRegion.Height = samplerFeedbackMipRegionHeight;
        SamplerFeedbackMipRegion.Depth = samplerFeedbackMipRegionDepth;
    }
    static inline CD3DX12_RESOURCE_DESC1 Buffer(
        const D3D12_RESOURCE_ALLOCATION_INFO &resAllocInfo,
        D3D12_RESOURCE_FLAGS flags = D3D12_RESOURCE_FLAG_NONE) noexcept {
        return CD3DX12_RESOURCE_DESC1(D3D12_RESOURCE_DIMENSION_BUFFER, resAllocInfo.Alignment, resAllocInfo.SizeInBytes,
                                      1, 1, 1, DXGI_FORMAT_UNKNOWN, 1, 0, D3D12_TEXTURE_LAYOUT_ROW_MAJOR, flags, 0, 0, 0);
    }
    static inline CD3DX12_RESOURCE_DESC1 Buffer(
        UINT64 width,
        D3D12_RESOURCE_FLAGS flags = D3D12_RESOURCE_FLAG_NONE,
        UINT64 alignment = 0) noexcept {
        return CD3DX12_RESOURCE_DESC1(D3D12_RESOURCE_DIMENSION_BUFFER, alignment, width, 1, 1, 1,
                                      DXGI_FORMAT_UNKNOWN, 1, 0, D3D12_TEXTURE_LAYOUT_ROW_MAJOR, flags, 0, 0, 0);
    }
    static inline CD3DX12_RESOURCE_DESC1 Tex1D(
        DXGI_FORMAT format,
        UINT64 width,
        UINT16 arraySize = 1,
        UINT16 mipLevels = 0,
        D3D12_RESOURCE_FLAGS flags = D3D12_RESOURCE_FLAG_NONE,
        D3D12_TEXTURE_LAYOUT layout = D3D12_TEXTURE_LAYOUT_UNKNOWN,
        UINT64 alignment = 0) noexcept {
        return CD3DX12_RESOURCE_DESC1(D3D12_RESOURCE_DIMENSION_TEXTURE1D, alignment, width, 1, arraySize,
                                      mipLevels, format, 1, 0, layout, flags, 0, 0, 0);
    }
    static inline CD3DX12_RESOURCE_DESC1 Tex2D(
        DXGI_FORMAT format,
        UINT64 width,
        uint height,
        UINT16 arraySize = 1,
        UINT16 mipLevels = 0,
        uint sampleCount = 1,
        uint sampleQuality = 0,
        D3D12_RESOURCE_FLAGS flags = D3D12_RESOURCE_FLAG_NONE,
        D3D12_TEXTURE_LAYOUT layout = D3D12_TEXTURE_LAYOUT_UNKNOWN,
        UINT64 alignment = 0,
        uint samplerFeedbackMipRegionWidth = 0,
        uint samplerFeedbackMipRegionHeight = 0,
        uint samplerFeedbackMipRegionDepth = 0) noexcept {
        return CD3DX12_RESOURCE_DESC1(D3D12_RESOURCE_DIMENSION_TEXTURE2D, alignment, width, height, arraySize,
                                      mipLevels, format, sampleCount, sampleQuality, layout, flags, samplerFeedbackMipRegionWidth,
                                      samplerFeedbackMipRegionHeight, samplerFeedbackMipRegionDepth);
    }
    static inline CD3DX12_RESOURCE_DESC1 Tex3D(
        DXGI_FORMAT format,
        UINT64 width,
        uint height,
        UINT16 depth,
        UINT16 mipLevels = 0,
        D3D12_RESOURCE_FLAGS flags = D3D12_RESOURCE_FLAG_NONE,
        D3D12_TEXTURE_LAYOUT layout = D3D12_TEXTURE_LAYOUT_UNKNOWN,
        UINT64 alignment = 0) noexcept {
        return CD3DX12_RESOURCE_DESC1(D3D12_RESOURCE_DIMENSION_TEXTURE3D, alignment, width, height, depth,
                                      mipLevels, format, 1, 0, layout, flags, 0, 0, 0);
    }
    inline UINT16 Depth() const noexcept { return (Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D ? DepthOrArraySize : 1); }
    inline UINT16 ArraySize() const noexcept { return (Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE3D ? DepthOrArraySize : 1); }
    inline UINT8 PlaneCount(_In_ ID3D12Device *pDevice) const noexcept { return D3D12GetFormatPlaneCount(pDevice, Format); }
    inline uint Subresources(_In_ ID3D12Device *pDevice) const noexcept { return MipLevels * ArraySize() * PlaneCount(pDevice); }
    inline uint CalcSubresource(uint MipSlice, uint ArraySlice, uint PlaneSlice) noexcept { return D3D12CalcSubresource(MipSlice, ArraySlice, PlaneSlice, MipLevels, ArraySize()); }
};
inline bool operator==(const D3D12_RESOURCE_DESC1 &l, const D3D12_RESOURCE_DESC1 &r) noexcept {
    return l.Dimension == r.Dimension && l.Alignment == r.Alignment && l.Width == r.Width && l.Height == r.Height && l.DepthOrArraySize == r.DepthOrArraySize && l.MipLevels == r.MipLevels && l.Format == r.Format && l.SampleDesc.Count == r.SampleDesc.Count && l.SampleDesc.Quality == r.SampleDesc.Quality && l.Layout == r.Layout && l.Flags == r.Flags && l.SamplerFeedbackMipRegion.Width == r.SamplerFeedbackMipRegion.Width && l.SamplerFeedbackMipRegion.Height == r.SamplerFeedbackMipRegion.Height && l.SamplerFeedbackMipRegion.Depth == r.SamplerFeedbackMipRegion.Depth;
}
inline bool operator!=(const D3D12_RESOURCE_DESC1 &l, const D3D12_RESOURCE_DESC1 &r) noexcept { return !(l == r); }

//------------------------------------------------------------------------------------------------
struct CD3DX12_VIEW_INSTANCING_DESC : public D3D12_VIEW_INSTANCING_DESC {
    CD3DX12_VIEW_INSTANCING_DESC() = default;
    explicit CD3DX12_VIEW_INSTANCING_DESC(const D3D12_VIEW_INSTANCING_DESC &o) noexcept : D3D12_VIEW_INSTANCING_DESC(o) {}
    explicit CD3DX12_VIEW_INSTANCING_DESC(CD3DX12_DEFAULT) noexcept {
        ViewInstanceCount = 0;
        pViewInstanceLocations = nullptr;
        Flags = D3D12_VIEW_INSTANCING_FLAG_NONE;
    }
    explicit CD3DX12_VIEW_INSTANCING_DESC(
        uint InViewInstanceCount,
        const D3D12_VIEW_INSTANCE_LOCATION *InViewInstanceLocations,
        D3D12_VIEW_INSTANCING_FLAGS InFlags) noexcept {
        ViewInstanceCount = InViewInstanceCount;
        pViewInstanceLocations = InViewInstanceLocations;
        Flags = InFlags;
    }
};

//------------------------------------------------------------------------------------------------
// Row-by-row memcpy
void MemcpySubresource(
    _In_ const D3D12_MEMCPY_DEST *pDest,
    _In_ const D3D12_SUBRESOURCE_DATA *pSrc,
    SIZE_T RowSizeInBytes,
    uint NumRows,
    uint NumSlices) noexcept;

//------------------------------------------------------------------------------------------------
// Row-by-row memcpy
void MemcpySubresource(
    _In_ const D3D12_MEMCPY_DEST *pDest,
    _In_ const void *pResourceData,
    _In_ const D3D12_SUBRESOURCE_INFO *pSrc,
    SIZE_T RowSizeInBytes,
    uint NumRows,
    uint NumSlices);

//------------------------------------------------------------------------------------------------
// Returns required size of a buffer to be used for data upload
UINT64 GetRequiredIntermediateSize(
    _In_ ID3D12Resource *pDestinationResource,
    _In_range_(0, D3D12_REQ_SUBRESOURCES) uint FirstSubresource,
    _In_range_(0, D3D12_REQ_SUBRESOURCES - FirstSubresource) uint NumSubresources) noexcept;

//------------------------------------------------------------------------------------------------
// All arrays must be populated (e.g. by calling GetCopyableFootprints)
UINT64 UpdateSubresources(
    _In_ ID3D12GraphicsCommandList *pCmdList,
    _In_ ID3D12Resource *pDestinationResource,
    _In_ ID3D12Resource *pIntermediate,
    _In_range_(0, D3D12_REQ_SUBRESOURCES) uint FirstSubresource,
    _In_range_(0, D3D12_REQ_SUBRESOURCES - FirstSubresource) uint NumSubresources,
    UINT64 RequiredSize,
    _In_reads_(NumSubresources) const D3D12_PLACED_SUBRESOURCE_FOOTPRINT *pLayouts,
    _In_reads_(NumSubresources) const uint *pNumRows,
    _In_reads_(NumSubresources) const UINT64 *pRowSizesInBytes,
    _In_reads_(NumSubresources) const D3D12_SUBRESOURCE_DATA *pSrcData) noexcept;

//------------------------------------------------------------------------------------------------
// All arrays must be populated (e.g. by calling GetCopyableFootprints)
UINT64 UpdateSubresources(
    _In_ ID3D12GraphicsCommandList *pCmdList,
    _In_ ID3D12Resource *pDestinationResource,
    _In_ ID3D12Resource *pIntermediate,
    _In_range_(0, D3D12_REQ_SUBRESOURCES) uint FirstSubresource,
    _In_range_(0, D3D12_REQ_SUBRESOURCES - FirstSubresource) uint NumSubresources,
    UINT64 RequiredSize,
    _In_reads_(NumSubresources) const D3D12_PLACED_SUBRESOURCE_FOOTPRINT *pLayouts,
    _In_reads_(NumSubresources) const uint *pNumRows,
    _In_reads_(NumSubresources) const UINT64 *pRowSizesInBytes,
    _In_ const void *pResourceData,
    _In_reads_(NumSubresources) const D3D12_SUBRESOURCE_INFO *pSrcData);

//------------------------------------------------------------------------------------------------
// Heap-allocating UpdateSubresources implementation
UINT64 UpdateSubresources(
    _In_ ID3D12GraphicsCommandList *pCmdList,
    _In_ ID3D12Resource *pDestinationResource,
    _In_ ID3D12Resource *pIntermediate,
    UINT64 IntermediateOffset,
    _In_range_(0, D3D12_REQ_SUBRESOURCES) uint FirstSubresource,
    _In_range_(0, D3D12_REQ_SUBRESOURCES - FirstSubresource) uint NumSubresources,
    _In_reads_(NumSubresources) const D3D12_SUBRESOURCE_DATA *pSrcData) noexcept;

//------------------------------------------------------------------------------------------------
// Heap-allocating UpdateSubresources implementation
UINT64 UpdateSubresources(
    _In_ ID3D12GraphicsCommandList *pCmdList,
    _In_ ID3D12Resource *pDestinationResource,
    _In_ ID3D12Resource *pIntermediate,
    UINT64 IntermediateOffset,
    _In_range_(0, D3D12_REQ_SUBRESOURCES) uint FirstSubresource,
    _In_range_(0, D3D12_REQ_SUBRESOURCES - FirstSubresource) uint NumSubresources,
    _In_ const void *pResourceData,
    _In_reads_(NumSubresources) D3D12_SUBRESOURCE_INFO *pSrcData);

//------------------------------------------------------------------------------------------------
// Stack-allocating UpdateSubresources implementation
template<uint MaxSubresources>
inline UINT64 UpdateSubresources(
    _In_ ID3D12GraphicsCommandList *pCmdList,
    _In_ ID3D12Resource *pDestinationResource,
    _In_ ID3D12Resource *pIntermediate,
    UINT64 IntermediateOffset,
    _In_range_(0, MaxSubresources) uint FirstSubresource,
    _In_range_(1, MaxSubresources - FirstSubresource) uint NumSubresources,
    _In_reads_(NumSubresources) const D3D12_SUBRESOURCE_DATA *pSrcData) noexcept {
    UINT64 RequiredSize = 0;
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT Layouts[MaxSubresources];
    uint NumRows[MaxSubresources];
    UINT64 RowSizesInBytes[MaxSubresources];

    auto Desc = pDestinationResource->GetDesc();
    ID3D12Device *pDevice = nullptr;
    pDestinationResource->GetDevice(IID_PPV_ARGS(&pDevice));
    pDevice->GetCopyableFootprints(&Desc, FirstSubresource, NumSubresources, IntermediateOffset, Layouts, NumRows, RowSizesInBytes, &RequiredSize);
    pDevice->Release();

    return UpdateSubresources(pCmdList, pDestinationResource, pIntermediate, FirstSubresource, NumSubresources, RequiredSize, Layouts, NumRows, RowSizesInBytes, pSrcData);
}

//------------------------------------------------------------------------------------------------
// Stack-allocating UpdateSubresources implementation
template<uint MaxSubresources>
inline UINT64 UpdateSubresources(
    _In_ ID3D12GraphicsCommandList *pCmdList,
    _In_ ID3D12Resource *pDestinationResource,
    _In_ ID3D12Resource *pIntermediate,
    UINT64 IntermediateOffset,
    _In_range_(0, MaxSubresources) uint FirstSubresource,
    _In_range_(1, MaxSubresources - FirstSubresource) uint NumSubresources,
    _In_ const void *pResourceData,
    _In_reads_(NumSubresources) D3D12_SUBRESOURCE_INFO *pSrcData) {
    UINT64 RequiredSize = 0;
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT Layouts[MaxSubresources];
    uint NumRows[MaxSubresources];
    UINT64 RowSizesInBytes[MaxSubresources];

    auto Desc = pDestinationResource->GetDesc();
    ID3D12Device *pDevice = nullptr;
    pDestinationResource->GetDevice(IID_PPV_ARGS(&pDevice));
    pDevice->GetCopyableFootprints(&Desc, FirstSubresource, NumSubresources, IntermediateOffset, Layouts, NumRows, RowSizesInBytes, &RequiredSize);
    pDevice->Release();

    return UpdateSubresources(pCmdList, pDestinationResource, pIntermediate, FirstSubresource, NumSubresources, RequiredSize, Layouts, NumRows, RowSizesInBytes, pResourceData, pSrcData);
}

//------------------------------------------------------------------------------------------------
constexpr bool D3D12IsLayoutOpaque(D3D12_TEXTURE_LAYOUT Layout) noexcept { return Layout == D3D12_TEXTURE_LAYOUT_UNKNOWN || Layout == D3D12_TEXTURE_LAYOUT_64KB_UNDEFINED_SWIZZLE; }

//------------------------------------------------------------------------------------------------
template<typename t_CommandListType>
inline ID3D12CommandList *const *CommandListCast(t_CommandListType *const *pp) noexcept {
    // This cast is useful for passing strongly typed command list pointers into
    // ExecuteCommandLists.
    // This cast is valid as long as the const-ness is respected. D3D12 APIs do
    // respect the const-ness of their arguments.
    return reinterpret_cast<ID3D12CommandList *const *>(pp);
}

//------------------------------------------------------------------------------------------------
// D3D12 exports a new method for serializing root signatures in the Windows 10 Anniversary Update.
// To help enable root signature 1.1 features when they are available and not require maintaining
// two code paths for building root signatures, this helper method reconstructs a 1.0 signature when
// 1.1 is not supported.
HRESULT D3DX12SerializeVersionedRootSignature(
    _In_ const D3D12_VERSIONED_ROOT_SIGNATURE_DESC *pRootSignatureDesc,
    D3D_ROOT_SIGNATURE_VERSION MaxVersion,
    _Outptr_ ID3DBlob **ppBlob,
    _Always_(_Outptr_opt_result_maybenull_) ID3DBlob **ppErrorBlob) noexcept;

//------------------------------------------------------------------------------------------------
struct CD3DX12_RT_FORMAT_ARRAY : public D3D12_RT_FORMAT_ARRAY {
    CD3DX12_RT_FORMAT_ARRAY() = default;
    explicit CD3DX12_RT_FORMAT_ARRAY(const D3D12_RT_FORMAT_ARRAY &o) noexcept
        : D3D12_RT_FORMAT_ARRAY(o) {}
    explicit CD3DX12_RT_FORMAT_ARRAY(_In_reads_(NumFormats) const DXGI_FORMAT *pFormats, uint NumFormats) noexcept {
        NumRenderTargets = NumFormats;
        memcpy(RTFormats, pFormats, sizeof(RTFormats));
        // assumes ARRAY_SIZE(pFormats) == ARRAY_SIZE(RTFormats)
    }
};

//------------------------------------------------------------------------------------------------
// Pipeline State Stream Helpers
//------------------------------------------------------------------------------------------------

//------------------------------------------------------------------------------------------------
// Stream Subobjects, i.e. elements of a stream

struct DefaultSampleMask {
    operator uint() noexcept { return UINT_MAX; }
};
struct DefaultSampleDesc {
    operator DXGI_SAMPLE_DESC() noexcept { return DXGI_SAMPLE_DESC{1, 0}; }
};

#pragma warning(push)
#pragma warning(disable : 4324)
template<typename InnerStructType, D3D12_PIPELINE_STATE_SUBOBJECT_TYPE Type, typename DefaultArg = InnerStructType>
class alignas(void *) CD3DX12_PIPELINE_STATE_STREAM_SUBOBJECT {
private:
    D3D12_PIPELINE_STATE_SUBOBJECT_TYPE _Type;
    InnerStructType _Inner;

public:
    CD3DX12_PIPELINE_STATE_STREAM_SUBOBJECT() noexcept : _Type(Type), _Inner(DefaultArg()) {}
    CD3DX12_PIPELINE_STATE_STREAM_SUBOBJECT(InnerStructType const &i) noexcept : _Type(Type), _Inner(i) {}
    CD3DX12_PIPELINE_STATE_STREAM_SUBOBJECT &operator=(InnerStructType const &i) noexcept {
        _Type = Type;
        _Inner = i;
        return *this;
    }
    operator InnerStructType const &() const noexcept { return _Inner; }
    operator InnerStructType &() noexcept { return _Inner; }
    InnerStructType *operator&() noexcept { return &_Inner; }
    InnerStructType const *operator&() const noexcept { return &_Inner; }
};
#pragma warning(pop)
typedef CD3DX12_PIPELINE_STATE_STREAM_SUBOBJECT<D3D12_PIPELINE_STATE_FLAGS, D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_FLAGS> CD3DX12_PIPELINE_STATE_STREAM_FLAGS;
typedef CD3DX12_PIPELINE_STATE_STREAM_SUBOBJECT<uint, D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_NODE_MASK> CD3DX12_PIPELINE_STATE_STREAM_NODE_MASK;
typedef CD3DX12_PIPELINE_STATE_STREAM_SUBOBJECT<ID3D12RootSignature *, D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_ROOT_SIGNATURE> CD3DX12_PIPELINE_STATE_STREAM_ROOT_SIGNATURE;
typedef CD3DX12_PIPELINE_STATE_STREAM_SUBOBJECT<D3D12_INPUT_LAYOUT_DESC, D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_INPUT_LAYOUT> CD3DX12_PIPELINE_STATE_STREAM_INPUT_LAYOUT;
typedef CD3DX12_PIPELINE_STATE_STREAM_SUBOBJECT<D3D12_INDEX_BUFFER_STRIP_CUT_VALUE, D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_IB_STRIP_CUT_VALUE> CD3DX12_PIPELINE_STATE_STREAM_IB_STRIP_CUT_VALUE;
typedef CD3DX12_PIPELINE_STATE_STREAM_SUBOBJECT<D3D12_PRIMITIVE_TOPOLOGY_TYPE, D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PRIMITIVE_TOPOLOGY> CD3DX12_PIPELINE_STATE_STREAM_PRIMITIVE_TOPOLOGY;
typedef CD3DX12_PIPELINE_STATE_STREAM_SUBOBJECT<D3D12_SHADER_BYTECODE, D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_VS> CD3DX12_PIPELINE_STATE_STREAM_VS;
typedef CD3DX12_PIPELINE_STATE_STREAM_SUBOBJECT<D3D12_SHADER_BYTECODE, D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_GS> CD3DX12_PIPELINE_STATE_STREAM_GS;
typedef CD3DX12_PIPELINE_STATE_STREAM_SUBOBJECT<D3D12_STREAM_OUTPUT_DESC, D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_STREAM_OUTPUT> CD3DX12_PIPELINE_STATE_STREAM_STREAM_OUTPUT;
typedef CD3DX12_PIPELINE_STATE_STREAM_SUBOBJECT<D3D12_SHADER_BYTECODE, D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_HS> CD3DX12_PIPELINE_STATE_STREAM_HS;
typedef CD3DX12_PIPELINE_STATE_STREAM_SUBOBJECT<D3D12_SHADER_BYTECODE, D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DS> CD3DX12_PIPELINE_STATE_STREAM_DS;
typedef CD3DX12_PIPELINE_STATE_STREAM_SUBOBJECT<D3D12_SHADER_BYTECODE, D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PS> CD3DX12_PIPELINE_STATE_STREAM_PS;
typedef CD3DX12_PIPELINE_STATE_STREAM_SUBOBJECT<D3D12_SHADER_BYTECODE, D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_AS> CD3DX12_PIPELINE_STATE_STREAM_AS;
typedef CD3DX12_PIPELINE_STATE_STREAM_SUBOBJECT<D3D12_SHADER_BYTECODE, D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_MS> CD3DX12_PIPELINE_STATE_STREAM_MS;
typedef CD3DX12_PIPELINE_STATE_STREAM_SUBOBJECT<D3D12_SHADER_BYTECODE, D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_CS> CD3DX12_PIPELINE_STATE_STREAM_CS;
typedef CD3DX12_PIPELINE_STATE_STREAM_SUBOBJECT<CD3DX12_BLEND_DESC, D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_BLEND, CD3DX12_DEFAULT> CD3DX12_PIPELINE_STATE_STREAM_BLEND_DESC;
typedef CD3DX12_PIPELINE_STATE_STREAM_SUBOBJECT<CD3DX12_DEPTH_STENCIL_DESC, D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL, CD3DX12_DEFAULT> CD3DX12_PIPELINE_STATE_STREAM_DEPTH_STENCIL;
typedef CD3DX12_PIPELINE_STATE_STREAM_SUBOBJECT<CD3DX12_DEPTH_STENCIL_DESC1, D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL1, CD3DX12_DEFAULT> CD3DX12_PIPELINE_STATE_STREAM_DEPTH_STENCIL1;
typedef CD3DX12_PIPELINE_STATE_STREAM_SUBOBJECT<DXGI_FORMAT, D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL_FORMAT> CD3DX12_PIPELINE_STATE_STREAM_DEPTH_STENCIL_FORMAT;
typedef CD3DX12_PIPELINE_STATE_STREAM_SUBOBJECT<CD3DX12_RASTERIZER_DESC, D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RASTERIZER, CD3DX12_DEFAULT> CD3DX12_PIPELINE_STATE_STREAM_RASTERIZER;
typedef CD3DX12_PIPELINE_STATE_STREAM_SUBOBJECT<D3D12_RT_FORMAT_ARRAY, D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RENDER_TARGET_FORMATS> CD3DX12_PIPELINE_STATE_STREAM_RENDER_TARGET_FORMATS;
typedef CD3DX12_PIPELINE_STATE_STREAM_SUBOBJECT<DXGI_SAMPLE_DESC, D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_SAMPLE_DESC, DefaultSampleDesc> CD3DX12_PIPELINE_STATE_STREAM_SAMPLE_DESC;
typedef CD3DX12_PIPELINE_STATE_STREAM_SUBOBJECT<uint, D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_SAMPLE_MASK, DefaultSampleMask> CD3DX12_PIPELINE_STATE_STREAM_SAMPLE_MASK;
typedef CD3DX12_PIPELINE_STATE_STREAM_SUBOBJECT<D3D12_CACHED_PIPELINE_STATE, D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_CACHED_PSO> CD3DX12_PIPELINE_STATE_STREAM_CACHED_PSO;
typedef CD3DX12_PIPELINE_STATE_STREAM_SUBOBJECT<CD3DX12_VIEW_INSTANCING_DESC, D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_VIEW_INSTANCING, CD3DX12_DEFAULT> CD3DX12_PIPELINE_STATE_STREAM_VIEW_INSTANCING;

//------------------------------------------------------------------------------------------------
// Stream Parser Helpers

struct ID3DX12PipelineParserCallbacks {
    // Subobject Callbacks
    virtual void FlagsCb(D3D12_PIPELINE_STATE_FLAGS) {}
    virtual void NodeMaskCb(uint) {}
    virtual void RootSignatureCb(ID3D12RootSignature *) {}
    virtual void InputLayoutCb(const D3D12_INPUT_LAYOUT_DESC &) {}
    virtual void IBStripCutValueCb(D3D12_INDEX_BUFFER_STRIP_CUT_VALUE) {}
    virtual void PrimitiveTopologyTypeCb(D3D12_PRIMITIVE_TOPOLOGY_TYPE) {}
    virtual void VSCb(const D3D12_SHADER_BYTECODE &) {}
    virtual void GSCb(const D3D12_SHADER_BYTECODE &) {}
    virtual void StreamOutputCb(const D3D12_STREAM_OUTPUT_DESC &) {}
    virtual void HSCb(const D3D12_SHADER_BYTECODE &) {}
    virtual void DSCb(const D3D12_SHADER_BYTECODE &) {}
    virtual void PSCb(const D3D12_SHADER_BYTECODE &) {}
    virtual void CSCb(const D3D12_SHADER_BYTECODE &) {}
    virtual void ASCb(const D3D12_SHADER_BYTECODE &) {}
    virtual void MSCb(const D3D12_SHADER_BYTECODE &) {}
    virtual void BlendStateCb(const D3D12_BLEND_DESC &) {}
    virtual void DepthStencilStateCb(const D3D12_DEPTH_STENCIL_DESC &) {}
    virtual void DepthStencilState1Cb(const D3D12_DEPTH_STENCIL_DESC1 &) {}
    virtual void DSVFormatCb(DXGI_FORMAT) {}
    virtual void RasterizerStateCb(const D3D12_RASTERIZER_DESC &) {}
    virtual void RTVFormatsCb(const D3D12_RT_FORMAT_ARRAY &) {}
    virtual void SampleDescCb(const DXGI_SAMPLE_DESC &) {}
    virtual void SampleMaskCb(uint) {}
    virtual void ViewInstancingCb(const D3D12_VIEW_INSTANCING_DESC &) {}
    virtual void CachedPSOCb(const D3D12_CACHED_PIPELINE_STATE &) {}

    // Error Callbacks
    virtual void ErrorBadInputParameter(uint /*ParameterIndex*/) {}
    virtual void ErrorDuplicateSubobject(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE /*DuplicateType*/) {}
    virtual void ErrorUnknownSubobject(uint /*UnknownTypeValue*/) {}

    virtual ~ID3DX12PipelineParserCallbacks() = default;
};

struct D3DX12_MESH_SHADER_PIPELINE_STATE_DESC {
    ID3D12RootSignature *pRootSignature;
    D3D12_SHADER_BYTECODE AS;
    D3D12_SHADER_BYTECODE MS;
    D3D12_SHADER_BYTECODE PS;
    D3D12_BLEND_DESC BlendState;
    uint SampleMask;
    D3D12_RASTERIZER_DESC RasterizerState;
    D3D12_DEPTH_STENCIL_DESC DepthStencilState;
    D3D12_PRIMITIVE_TOPOLOGY_TYPE PrimitiveTopologyType;
    uint NumRenderTargets;
    DXGI_FORMAT RTVFormats[D3D12_SIMULTANEOUS_RENDER_TARGET_COUNT];
    DXGI_FORMAT DSVFormat;
    DXGI_SAMPLE_DESC SampleDesc;
    uint NodeMask;
    D3D12_CACHED_PIPELINE_STATE CachedPSO;
    D3D12_PIPELINE_STATE_FLAGS Flags;
};

// CD3DX12_PIPELINE_STATE_STREAM2 Works on OS Build 19041+ (where there is a new mesh shader pipeline).
// Use CD3DX12_PIPELINE_STATE_STREAM1 for OS Build 16299+ (where there is a new view instancing subobject).
// Use CD3DX12_PIPELINE_STATE_STREAM for OS Build 15063+ support.
struct CD3DX12_PIPELINE_STATE_STREAM2 {
    CD3DX12_PIPELINE_STATE_STREAM2() = default;
    // Mesh and amplification shaders must be set manually, since they do not have representation in D3D12_GRAPHICS_PIPELINE_STATE_DESC
    CD3DX12_PIPELINE_STATE_STREAM2(const D3D12_GRAPHICS_PIPELINE_STATE_DESC &Desc) noexcept
        : Flags(Desc.Flags), NodeMask(Desc.NodeMask), pRootSignature(Desc.pRootSignature), InputLayout(Desc.InputLayout), IBStripCutValue(Desc.IBStripCutValue), PrimitiveTopologyType(Desc.PrimitiveTopologyType), VS(Desc.VS), GS(Desc.GS), StreamOutput(Desc.StreamOutput), HS(Desc.HS), DS(Desc.DS), PS(Desc.PS), BlendState(CD3DX12_BLEND_DESC(Desc.BlendState)), DepthStencilState(CD3DX12_DEPTH_STENCIL_DESC1(Desc.DepthStencilState)), DSVFormat(Desc.DSVFormat), RasterizerState(CD3DX12_RASTERIZER_DESC(Desc.RasterizerState)), RTVFormats(CD3DX12_RT_FORMAT_ARRAY(Desc.RTVFormats, Desc.NumRenderTargets)), SampleDesc(Desc.SampleDesc), SampleMask(Desc.SampleMask), CachedPSO(Desc.CachedPSO), ViewInstancingDesc(CD3DX12_VIEW_INSTANCING_DESC(CD3DX12_DEFAULT())) {}
    CD3DX12_PIPELINE_STATE_STREAM2(const D3DX12_MESH_SHADER_PIPELINE_STATE_DESC &Desc) noexcept
        : Flags(Desc.Flags), NodeMask(Desc.NodeMask), pRootSignature(Desc.pRootSignature), PrimitiveTopologyType(Desc.PrimitiveTopologyType), PS(Desc.PS), AS(Desc.AS), MS(Desc.MS), BlendState(CD3DX12_BLEND_DESC(Desc.BlendState)), DepthStencilState(CD3DX12_DEPTH_STENCIL_DESC1(Desc.DepthStencilState)), DSVFormat(Desc.DSVFormat), RasterizerState(CD3DX12_RASTERIZER_DESC(Desc.RasterizerState)), RTVFormats(CD3DX12_RT_FORMAT_ARRAY(Desc.RTVFormats, Desc.NumRenderTargets)), SampleDesc(Desc.SampleDesc), SampleMask(Desc.SampleMask), CachedPSO(Desc.CachedPSO), ViewInstancingDesc(CD3DX12_VIEW_INSTANCING_DESC(CD3DX12_DEFAULT())) {}
    CD3DX12_PIPELINE_STATE_STREAM2(const D3D12_COMPUTE_PIPELINE_STATE_DESC &Desc) noexcept
        : Flags(Desc.Flags), NodeMask(Desc.NodeMask), pRootSignature(Desc.pRootSignature), CS(CD3DX12_SHADER_BYTECODE(Desc.CS)), CachedPSO(Desc.CachedPSO) {
        static_cast<D3D12_DEPTH_STENCIL_DESC1 &>(DepthStencilState).DepthEnable = false;
    }
    CD3DX12_PIPELINE_STATE_STREAM_FLAGS Flags;
    CD3DX12_PIPELINE_STATE_STREAM_NODE_MASK NodeMask;
    CD3DX12_PIPELINE_STATE_STREAM_ROOT_SIGNATURE pRootSignature;
    CD3DX12_PIPELINE_STATE_STREAM_INPUT_LAYOUT InputLayout;
    CD3DX12_PIPELINE_STATE_STREAM_IB_STRIP_CUT_VALUE IBStripCutValue;
    CD3DX12_PIPELINE_STATE_STREAM_PRIMITIVE_TOPOLOGY PrimitiveTopologyType;
    CD3DX12_PIPELINE_STATE_STREAM_VS VS;
    CD3DX12_PIPELINE_STATE_STREAM_GS GS;
    CD3DX12_PIPELINE_STATE_STREAM_STREAM_OUTPUT StreamOutput;
    CD3DX12_PIPELINE_STATE_STREAM_HS HS;
    CD3DX12_PIPELINE_STATE_STREAM_DS DS;
    CD3DX12_PIPELINE_STATE_STREAM_PS PS;
    CD3DX12_PIPELINE_STATE_STREAM_AS AS;
    CD3DX12_PIPELINE_STATE_STREAM_MS MS;
    CD3DX12_PIPELINE_STATE_STREAM_CS CS;
    CD3DX12_PIPELINE_STATE_STREAM_BLEND_DESC BlendState;
    CD3DX12_PIPELINE_STATE_STREAM_DEPTH_STENCIL1 DepthStencilState;
    CD3DX12_PIPELINE_STATE_STREAM_DEPTH_STENCIL_FORMAT DSVFormat;
    CD3DX12_PIPELINE_STATE_STREAM_RASTERIZER RasterizerState;
    CD3DX12_PIPELINE_STATE_STREAM_RENDER_TARGET_FORMATS RTVFormats;
    CD3DX12_PIPELINE_STATE_STREAM_SAMPLE_DESC SampleDesc;
    CD3DX12_PIPELINE_STATE_STREAM_SAMPLE_MASK SampleMask;
    CD3DX12_PIPELINE_STATE_STREAM_CACHED_PSO CachedPSO;
    CD3DX12_PIPELINE_STATE_STREAM_VIEW_INSTANCING ViewInstancingDesc;
    D3D12_GRAPHICS_PIPELINE_STATE_DESC GraphicsDescV0() const noexcept {
        D3D12_GRAPHICS_PIPELINE_STATE_DESC D;
        D.Flags = this->Flags;
        D.NodeMask = this->NodeMask;
        D.pRootSignature = this->pRootSignature;
        D.InputLayout = this->InputLayout;
        D.IBStripCutValue = this->IBStripCutValue;
        D.PrimitiveTopologyType = this->PrimitiveTopologyType;
        D.VS = this->VS;
        D.GS = this->GS;
        D.StreamOutput = this->StreamOutput;
        D.HS = this->HS;
        D.DS = this->DS;
        D.PS = this->PS;
        D.BlendState = this->BlendState;
        D.DepthStencilState = CD3DX12_DEPTH_STENCIL_DESC1(D3D12_DEPTH_STENCIL_DESC1(this->DepthStencilState));
        D.DSVFormat = this->DSVFormat;
        D.RasterizerState = this->RasterizerState;
        D.NumRenderTargets = D3D12_RT_FORMAT_ARRAY(this->RTVFormats).NumRenderTargets;
        memcpy(D.RTVFormats, D3D12_RT_FORMAT_ARRAY(this->RTVFormats).RTFormats, sizeof(D.RTVFormats));
        D.SampleDesc = this->SampleDesc;
        D.SampleMask = this->SampleMask;
        D.CachedPSO = this->CachedPSO;
        return D;
    }
    D3D12_COMPUTE_PIPELINE_STATE_DESC ComputeDescV0() const noexcept {
        D3D12_COMPUTE_PIPELINE_STATE_DESC D;
        D.Flags = this->Flags;
        D.NodeMask = this->NodeMask;
        D.pRootSignature = this->pRootSignature;
        D.CS = this->CS;
        D.CachedPSO = this->CachedPSO;
        return D;
    }
};

// CD3DX12_PIPELINE_STATE_STREAM1 Works on OS Build 16299+ (where there is a new view instancing subobject).
// Use CD3DX12_PIPELINE_STATE_STREAM for OS Build 15063+ support.
struct CD3DX12_PIPELINE_STATE_STREAM1 {
    CD3DX12_PIPELINE_STATE_STREAM1() = default;
    // Mesh and amplification shaders must be set manually, since they do not have representation in D3D12_GRAPHICS_PIPELINE_STATE_DESC
    CD3DX12_PIPELINE_STATE_STREAM1(const D3D12_GRAPHICS_PIPELINE_STATE_DESC &Desc) noexcept
        : Flags(Desc.Flags), NodeMask(Desc.NodeMask), pRootSignature(Desc.pRootSignature), InputLayout(Desc.InputLayout), IBStripCutValue(Desc.IBStripCutValue), PrimitiveTopologyType(Desc.PrimitiveTopologyType), VS(Desc.VS), GS(Desc.GS), StreamOutput(Desc.StreamOutput), HS(Desc.HS), DS(Desc.DS), PS(Desc.PS), BlendState(CD3DX12_BLEND_DESC(Desc.BlendState)), DepthStencilState(CD3DX12_DEPTH_STENCIL_DESC1(Desc.DepthStencilState)), DSVFormat(Desc.DSVFormat), RasterizerState(CD3DX12_RASTERIZER_DESC(Desc.RasterizerState)), RTVFormats(CD3DX12_RT_FORMAT_ARRAY(Desc.RTVFormats, Desc.NumRenderTargets)), SampleDesc(Desc.SampleDesc), SampleMask(Desc.SampleMask), CachedPSO(Desc.CachedPSO), ViewInstancingDesc(CD3DX12_VIEW_INSTANCING_DESC(CD3DX12_DEFAULT())) {}
    CD3DX12_PIPELINE_STATE_STREAM1(const D3DX12_MESH_SHADER_PIPELINE_STATE_DESC &Desc) noexcept
        : Flags(Desc.Flags), NodeMask(Desc.NodeMask), pRootSignature(Desc.pRootSignature), PrimitiveTopologyType(Desc.PrimitiveTopologyType), PS(Desc.PS), BlendState(CD3DX12_BLEND_DESC(Desc.BlendState)), DepthStencilState(CD3DX12_DEPTH_STENCIL_DESC1(Desc.DepthStencilState)), DSVFormat(Desc.DSVFormat), RasterizerState(CD3DX12_RASTERIZER_DESC(Desc.RasterizerState)), RTVFormats(CD3DX12_RT_FORMAT_ARRAY(Desc.RTVFormats, Desc.NumRenderTargets)), SampleDesc(Desc.SampleDesc), SampleMask(Desc.SampleMask), CachedPSO(Desc.CachedPSO), ViewInstancingDesc(CD3DX12_VIEW_INSTANCING_DESC(CD3DX12_DEFAULT())) {}
    CD3DX12_PIPELINE_STATE_STREAM1(const D3D12_COMPUTE_PIPELINE_STATE_DESC &Desc) noexcept
        : Flags(Desc.Flags), NodeMask(Desc.NodeMask), pRootSignature(Desc.pRootSignature), CS(CD3DX12_SHADER_BYTECODE(Desc.CS)), CachedPSO(Desc.CachedPSO) {
        static_cast<D3D12_DEPTH_STENCIL_DESC1 &>(DepthStencilState).DepthEnable = false;
    }
    CD3DX12_PIPELINE_STATE_STREAM_FLAGS Flags;
    CD3DX12_PIPELINE_STATE_STREAM_NODE_MASK NodeMask;
    CD3DX12_PIPELINE_STATE_STREAM_ROOT_SIGNATURE pRootSignature;
    CD3DX12_PIPELINE_STATE_STREAM_INPUT_LAYOUT InputLayout;
    CD3DX12_PIPELINE_STATE_STREAM_IB_STRIP_CUT_VALUE IBStripCutValue;
    CD3DX12_PIPELINE_STATE_STREAM_PRIMITIVE_TOPOLOGY PrimitiveTopologyType;
    CD3DX12_PIPELINE_STATE_STREAM_VS VS;
    CD3DX12_PIPELINE_STATE_STREAM_GS GS;
    CD3DX12_PIPELINE_STATE_STREAM_STREAM_OUTPUT StreamOutput;
    CD3DX12_PIPELINE_STATE_STREAM_HS HS;
    CD3DX12_PIPELINE_STATE_STREAM_DS DS;
    CD3DX12_PIPELINE_STATE_STREAM_PS PS;
    CD3DX12_PIPELINE_STATE_STREAM_CS CS;
    CD3DX12_PIPELINE_STATE_STREAM_BLEND_DESC BlendState;
    CD3DX12_PIPELINE_STATE_STREAM_DEPTH_STENCIL1 DepthStencilState;
    CD3DX12_PIPELINE_STATE_STREAM_DEPTH_STENCIL_FORMAT DSVFormat;
    CD3DX12_PIPELINE_STATE_STREAM_RASTERIZER RasterizerState;
    CD3DX12_PIPELINE_STATE_STREAM_RENDER_TARGET_FORMATS RTVFormats;
    CD3DX12_PIPELINE_STATE_STREAM_SAMPLE_DESC SampleDesc;
    CD3DX12_PIPELINE_STATE_STREAM_SAMPLE_MASK SampleMask;
    CD3DX12_PIPELINE_STATE_STREAM_CACHED_PSO CachedPSO;
    CD3DX12_PIPELINE_STATE_STREAM_VIEW_INSTANCING ViewInstancingDesc;
    D3D12_GRAPHICS_PIPELINE_STATE_DESC GraphicsDescV0() const noexcept {
        D3D12_GRAPHICS_PIPELINE_STATE_DESC D;
        D.Flags = this->Flags;
        D.NodeMask = this->NodeMask;
        D.pRootSignature = this->pRootSignature;
        D.InputLayout = this->InputLayout;
        D.IBStripCutValue = this->IBStripCutValue;
        D.PrimitiveTopologyType = this->PrimitiveTopologyType;
        D.VS = this->VS;
        D.GS = this->GS;
        D.StreamOutput = this->StreamOutput;
        D.HS = this->HS;
        D.DS = this->DS;
        D.PS = this->PS;
        D.BlendState = this->BlendState;
        D.DepthStencilState = CD3DX12_DEPTH_STENCIL_DESC1(D3D12_DEPTH_STENCIL_DESC1(this->DepthStencilState));
        D.DSVFormat = this->DSVFormat;
        D.RasterizerState = this->RasterizerState;
        D.NumRenderTargets = D3D12_RT_FORMAT_ARRAY(this->RTVFormats).NumRenderTargets;
        memcpy(D.RTVFormats, D3D12_RT_FORMAT_ARRAY(this->RTVFormats).RTFormats, sizeof(D.RTVFormats));
        D.SampleDesc = this->SampleDesc;
        D.SampleMask = this->SampleMask;
        D.CachedPSO = this->CachedPSO;
        return D;
    }
    D3D12_COMPUTE_PIPELINE_STATE_DESC ComputeDescV0() const noexcept {
        D3D12_COMPUTE_PIPELINE_STATE_DESC D;
        D.Flags = this->Flags;
        D.NodeMask = this->NodeMask;
        D.pRootSignature = this->pRootSignature;
        D.CS = this->CS;
        D.CachedPSO = this->CachedPSO;
        return D;
    }
};

struct CD3DX12_PIPELINE_MESH_STATE_STREAM {
    CD3DX12_PIPELINE_MESH_STATE_STREAM() = default;
    CD3DX12_PIPELINE_MESH_STATE_STREAM(const D3DX12_MESH_SHADER_PIPELINE_STATE_DESC &Desc) noexcept
        : Flags(Desc.Flags), NodeMask(Desc.NodeMask), pRootSignature(Desc.pRootSignature), PS(Desc.PS), AS(Desc.AS), MS(Desc.MS), BlendState(CD3DX12_BLEND_DESC(Desc.BlendState)), DepthStencilState(CD3DX12_DEPTH_STENCIL_DESC1(Desc.DepthStencilState)), DSVFormat(Desc.DSVFormat), RasterizerState(CD3DX12_RASTERIZER_DESC(Desc.RasterizerState)), RTVFormats(CD3DX12_RT_FORMAT_ARRAY(Desc.RTVFormats, Desc.NumRenderTargets)), SampleDesc(Desc.SampleDesc), SampleMask(Desc.SampleMask), CachedPSO(Desc.CachedPSO), ViewInstancingDesc(CD3DX12_VIEW_INSTANCING_DESC(CD3DX12_DEFAULT())) {}
    CD3DX12_PIPELINE_STATE_STREAM_FLAGS Flags;
    CD3DX12_PIPELINE_STATE_STREAM_NODE_MASK NodeMask;
    CD3DX12_PIPELINE_STATE_STREAM_ROOT_SIGNATURE pRootSignature;
    CD3DX12_PIPELINE_STATE_STREAM_PS PS;
    CD3DX12_PIPELINE_STATE_STREAM_AS AS;
    CD3DX12_PIPELINE_STATE_STREAM_MS MS;
    CD3DX12_PIPELINE_STATE_STREAM_BLEND_DESC BlendState;
    CD3DX12_PIPELINE_STATE_STREAM_DEPTH_STENCIL1 DepthStencilState;
    CD3DX12_PIPELINE_STATE_STREAM_DEPTH_STENCIL_FORMAT DSVFormat;
    CD3DX12_PIPELINE_STATE_STREAM_RASTERIZER RasterizerState;
    CD3DX12_PIPELINE_STATE_STREAM_RENDER_TARGET_FORMATS RTVFormats;
    CD3DX12_PIPELINE_STATE_STREAM_SAMPLE_DESC SampleDesc;
    CD3DX12_PIPELINE_STATE_STREAM_SAMPLE_MASK SampleMask;
    CD3DX12_PIPELINE_STATE_STREAM_CACHED_PSO CachedPSO;
    CD3DX12_PIPELINE_STATE_STREAM_VIEW_INSTANCING ViewInstancingDesc;
    D3DX12_MESH_SHADER_PIPELINE_STATE_DESC MeshShaderDescV0() const noexcept {
        D3DX12_MESH_SHADER_PIPELINE_STATE_DESC D;
        D.Flags = this->Flags;
        D.NodeMask = this->NodeMask;
        D.pRootSignature = this->pRootSignature;
        D.PS = this->PS;
        D.AS = this->AS;
        D.MS = this->MS;
        D.BlendState = this->BlendState;
        D.DepthStencilState = CD3DX12_DEPTH_STENCIL_DESC1(D3D12_DEPTH_STENCIL_DESC1(this->DepthStencilState));
        D.DSVFormat = this->DSVFormat;
        D.RasterizerState = this->RasterizerState;
        D.NumRenderTargets = D3D12_RT_FORMAT_ARRAY(this->RTVFormats).NumRenderTargets;
        memcpy(D.RTVFormats, D3D12_RT_FORMAT_ARRAY(this->RTVFormats).RTFormats, sizeof(D.RTVFormats));
        D.SampleDesc = this->SampleDesc;
        D.SampleMask = this->SampleMask;
        D.CachedPSO = this->CachedPSO;
        return D;
    }
};

// CD3DX12_PIPELINE_STATE_STREAM works on OS Build 15063+ but does not support new subobject(s) added in OS Build 16299+.
// See CD3DX12_PIPELINE_STATE_STREAM1 for instance.
struct CD3DX12_PIPELINE_STATE_STREAM {
    CD3DX12_PIPELINE_STATE_STREAM() = default;
    CD3DX12_PIPELINE_STATE_STREAM(const D3D12_GRAPHICS_PIPELINE_STATE_DESC &Desc) noexcept
        : Flags(Desc.Flags), NodeMask(Desc.NodeMask), pRootSignature(Desc.pRootSignature), InputLayout(Desc.InputLayout), IBStripCutValue(Desc.IBStripCutValue), PrimitiveTopologyType(Desc.PrimitiveTopologyType), VS(Desc.VS), GS(Desc.GS), StreamOutput(Desc.StreamOutput), HS(Desc.HS), DS(Desc.DS), PS(Desc.PS), BlendState(CD3DX12_BLEND_DESC(Desc.BlendState)), DepthStencilState(CD3DX12_DEPTH_STENCIL_DESC1(Desc.DepthStencilState)), DSVFormat(Desc.DSVFormat), RasterizerState(CD3DX12_RASTERIZER_DESC(Desc.RasterizerState)), RTVFormats(CD3DX12_RT_FORMAT_ARRAY(Desc.RTVFormats, Desc.NumRenderTargets)), SampleDesc(Desc.SampleDesc), SampleMask(Desc.SampleMask), CachedPSO(Desc.CachedPSO) {}
    CD3DX12_PIPELINE_STATE_STREAM(const D3D12_COMPUTE_PIPELINE_STATE_DESC &Desc) noexcept
        : Flags(Desc.Flags), NodeMask(Desc.NodeMask), pRootSignature(Desc.pRootSignature), CS(CD3DX12_SHADER_BYTECODE(Desc.CS)), CachedPSO(Desc.CachedPSO) {}
    CD3DX12_PIPELINE_STATE_STREAM_FLAGS Flags;
    CD3DX12_PIPELINE_STATE_STREAM_NODE_MASK NodeMask;
    CD3DX12_PIPELINE_STATE_STREAM_ROOT_SIGNATURE pRootSignature;
    CD3DX12_PIPELINE_STATE_STREAM_INPUT_LAYOUT InputLayout;
    CD3DX12_PIPELINE_STATE_STREAM_IB_STRIP_CUT_VALUE IBStripCutValue;
    CD3DX12_PIPELINE_STATE_STREAM_PRIMITIVE_TOPOLOGY PrimitiveTopologyType;
    CD3DX12_PIPELINE_STATE_STREAM_VS VS;
    CD3DX12_PIPELINE_STATE_STREAM_GS GS;
    CD3DX12_PIPELINE_STATE_STREAM_STREAM_OUTPUT StreamOutput;
    CD3DX12_PIPELINE_STATE_STREAM_HS HS;
    CD3DX12_PIPELINE_STATE_STREAM_DS DS;
    CD3DX12_PIPELINE_STATE_STREAM_PS PS;
    CD3DX12_PIPELINE_STATE_STREAM_CS CS;
    CD3DX12_PIPELINE_STATE_STREAM_BLEND_DESC BlendState;
    CD3DX12_PIPELINE_STATE_STREAM_DEPTH_STENCIL1 DepthStencilState;
    CD3DX12_PIPELINE_STATE_STREAM_DEPTH_STENCIL_FORMAT DSVFormat;
    CD3DX12_PIPELINE_STATE_STREAM_RASTERIZER RasterizerState;
    CD3DX12_PIPELINE_STATE_STREAM_RENDER_TARGET_FORMATS RTVFormats;
    CD3DX12_PIPELINE_STATE_STREAM_SAMPLE_DESC SampleDesc;
    CD3DX12_PIPELINE_STATE_STREAM_SAMPLE_MASK SampleMask;
    CD3DX12_PIPELINE_STATE_STREAM_CACHED_PSO CachedPSO;
    D3D12_GRAPHICS_PIPELINE_STATE_DESC GraphicsDescV0() const noexcept {
        D3D12_GRAPHICS_PIPELINE_STATE_DESC D;
        D.Flags = this->Flags;
        D.NodeMask = this->NodeMask;
        D.pRootSignature = this->pRootSignature;
        D.InputLayout = this->InputLayout;
        D.IBStripCutValue = this->IBStripCutValue;
        D.PrimitiveTopologyType = this->PrimitiveTopologyType;
        D.VS = this->VS;
        D.GS = this->GS;
        D.StreamOutput = this->StreamOutput;
        D.HS = this->HS;
        D.DS = this->DS;
        D.PS = this->PS;
        D.BlendState = this->BlendState;
        D.DepthStencilState = CD3DX12_DEPTH_STENCIL_DESC1(D3D12_DEPTH_STENCIL_DESC1(this->DepthStencilState));
        D.DSVFormat = this->DSVFormat;
        D.RasterizerState = this->RasterizerState;
        D.NumRenderTargets = D3D12_RT_FORMAT_ARRAY(this->RTVFormats).NumRenderTargets;
        memcpy(D.RTVFormats, D3D12_RT_FORMAT_ARRAY(this->RTVFormats).RTFormats, sizeof(D.RTVFormats));
        D.SampleDesc = this->SampleDesc;
        D.SampleMask = this->SampleMask;
        D.CachedPSO = this->CachedPSO;
        return D;
    }
    D3D12_COMPUTE_PIPELINE_STATE_DESC ComputeDescV0() const noexcept {
        D3D12_COMPUTE_PIPELINE_STATE_DESC D;
        D.Flags = this->Flags;
        D.NodeMask = this->NodeMask;
        D.pRootSignature = this->pRootSignature;
        D.CS = this->CS;
        D.CachedPSO = this->CachedPSO;
        return D;
    }
};

struct CD3DX12_PIPELINE_STATE_STREAM2_PARSE_HELPER : public ID3DX12PipelineParserCallbacks {
    CD3DX12_PIPELINE_STATE_STREAM2 PipelineStream;
    CD3DX12_PIPELINE_STATE_STREAM2_PARSE_HELPER() noexcept
        : SeenDSS(false) {
        // Adjust defaults to account for absent members.
        PipelineStream.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;

        // Depth disabled if no DSV format specified.
        static_cast<D3D12_DEPTH_STENCIL_DESC1 &>(PipelineStream.DepthStencilState).DepthEnable = false;
    }

    // ID3DX12PipelineParserCallbacks
    void FlagsCb(D3D12_PIPELINE_STATE_FLAGS Flags) override { PipelineStream.Flags = Flags; }
    void NodeMaskCb(uint NodeMask) override { PipelineStream.NodeMask = NodeMask; }
    void RootSignatureCb(ID3D12RootSignature *pRootSignature) override { PipelineStream.pRootSignature = pRootSignature; }
    void InputLayoutCb(const D3D12_INPUT_LAYOUT_DESC &InputLayout) override { PipelineStream.InputLayout = InputLayout; }
    void IBStripCutValueCb(D3D12_INDEX_BUFFER_STRIP_CUT_VALUE IBStripCutValue) override { PipelineStream.IBStripCutValue = IBStripCutValue; }
    void PrimitiveTopologyTypeCb(D3D12_PRIMITIVE_TOPOLOGY_TYPE PrimitiveTopologyType) override { PipelineStream.PrimitiveTopologyType = PrimitiveTopologyType; }
    void VSCb(const D3D12_SHADER_BYTECODE &VS) override { PipelineStream.VS = VS; }
    void GSCb(const D3D12_SHADER_BYTECODE &GS) override { PipelineStream.GS = GS; }
    void StreamOutputCb(const D3D12_STREAM_OUTPUT_DESC &StreamOutput) override { PipelineStream.StreamOutput = StreamOutput; }
    void HSCb(const D3D12_SHADER_BYTECODE &HS) override { PipelineStream.HS = HS; }
    void DSCb(const D3D12_SHADER_BYTECODE &DS) override { PipelineStream.DS = DS; }
    void PSCb(const D3D12_SHADER_BYTECODE &PS) override { PipelineStream.PS = PS; }
    void CSCb(const D3D12_SHADER_BYTECODE &CS) override { PipelineStream.CS = CS; }
    void ASCb(const D3D12_SHADER_BYTECODE &AS) override { PipelineStream.AS = AS; }
    void MSCb(const D3D12_SHADER_BYTECODE &MS) override { PipelineStream.MS = MS; }
    void BlendStateCb(const D3D12_BLEND_DESC &BlendState) override { PipelineStream.BlendState = CD3DX12_BLEND_DESC(BlendState); }
    void DepthStencilStateCb(const D3D12_DEPTH_STENCIL_DESC &DepthStencilState) override {
        PipelineStream.DepthStencilState = CD3DX12_DEPTH_STENCIL_DESC1(DepthStencilState);
        SeenDSS = true;
    }
    void DepthStencilState1Cb(const D3D12_DEPTH_STENCIL_DESC1 &DepthStencilState) override {
        PipelineStream.DepthStencilState = CD3DX12_DEPTH_STENCIL_DESC1(DepthStencilState);
        SeenDSS = true;
    }
    void DSVFormatCb(DXGI_FORMAT DSVFormat) override {
        PipelineStream.DSVFormat = DSVFormat;
        if (!SeenDSS && DSVFormat != DXGI_FORMAT_UNKNOWN) {
            // Re-enable depth for the default state.
            static_cast<D3D12_DEPTH_STENCIL_DESC1 &>(PipelineStream.DepthStencilState).DepthEnable = true;
        }
    }
    void RasterizerStateCb(const D3D12_RASTERIZER_DESC &RasterizerState) override { PipelineStream.RasterizerState = CD3DX12_RASTERIZER_DESC(RasterizerState); }
    void RTVFormatsCb(const D3D12_RT_FORMAT_ARRAY &RTVFormats) override { PipelineStream.RTVFormats = RTVFormats; }
    void SampleDescCb(const DXGI_SAMPLE_DESC &SampleDesc) override { PipelineStream.SampleDesc = SampleDesc; }
    void SampleMaskCb(uint SampleMask) override { PipelineStream.SampleMask = SampleMask; }
    void ViewInstancingCb(const D3D12_VIEW_INSTANCING_DESC &ViewInstancingDesc) override { PipelineStream.ViewInstancingDesc = CD3DX12_VIEW_INSTANCING_DESC(ViewInstancingDesc); }
    void CachedPSOCb(const D3D12_CACHED_PIPELINE_STATE &CachedPSO) override { PipelineStream.CachedPSO = CachedPSO; }

private:
    bool SeenDSS;
};

struct CD3DX12_PIPELINE_STATE_STREAM_PARSE_HELPER : public ID3DX12PipelineParserCallbacks {
    CD3DX12_PIPELINE_STATE_STREAM1 PipelineStream;
    CD3DX12_PIPELINE_STATE_STREAM_PARSE_HELPER() noexcept
        : SeenDSS(false) {
        // Adjust defaults to account for absent members.
        PipelineStream.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;

        // Depth disabled if no DSV format specified.
        static_cast<D3D12_DEPTH_STENCIL_DESC1 &>(PipelineStream.DepthStencilState).DepthEnable = false;
    }

    // ID3DX12PipelineParserCallbacks
    void FlagsCb(D3D12_PIPELINE_STATE_FLAGS Flags) override { PipelineStream.Flags = Flags; }
    void NodeMaskCb(uint NodeMask) override { PipelineStream.NodeMask = NodeMask; }
    void RootSignatureCb(ID3D12RootSignature *pRootSignature) override { PipelineStream.pRootSignature = pRootSignature; }
    void InputLayoutCb(const D3D12_INPUT_LAYOUT_DESC &InputLayout) override { PipelineStream.InputLayout = InputLayout; }
    void IBStripCutValueCb(D3D12_INDEX_BUFFER_STRIP_CUT_VALUE IBStripCutValue) override { PipelineStream.IBStripCutValue = IBStripCutValue; }
    void PrimitiveTopologyTypeCb(D3D12_PRIMITIVE_TOPOLOGY_TYPE PrimitiveTopologyType) override { PipelineStream.PrimitiveTopologyType = PrimitiveTopologyType; }
    void VSCb(const D3D12_SHADER_BYTECODE &VS) override { PipelineStream.VS = VS; }
    void GSCb(const D3D12_SHADER_BYTECODE &GS) override { PipelineStream.GS = GS; }
    void StreamOutputCb(const D3D12_STREAM_OUTPUT_DESC &StreamOutput) override { PipelineStream.StreamOutput = StreamOutput; }
    void HSCb(const D3D12_SHADER_BYTECODE &HS) override { PipelineStream.HS = HS; }
    void DSCb(const D3D12_SHADER_BYTECODE &DS) override { PipelineStream.DS = DS; }
    void PSCb(const D3D12_SHADER_BYTECODE &PS) override { PipelineStream.PS = PS; }
    void CSCb(const D3D12_SHADER_BYTECODE &CS) override { PipelineStream.CS = CS; }
    void BlendStateCb(const D3D12_BLEND_DESC &BlendState) override { PipelineStream.BlendState = CD3DX12_BLEND_DESC(BlendState); }
    void DepthStencilStateCb(const D3D12_DEPTH_STENCIL_DESC &DepthStencilState) override {
        PipelineStream.DepthStencilState = CD3DX12_DEPTH_STENCIL_DESC1(DepthStencilState);
        SeenDSS = true;
    }
    void DepthStencilState1Cb(const D3D12_DEPTH_STENCIL_DESC1 &DepthStencilState) override {
        PipelineStream.DepthStencilState = CD3DX12_DEPTH_STENCIL_DESC1(DepthStencilState);
        SeenDSS = true;
    }
    void DSVFormatCb(DXGI_FORMAT DSVFormat) override {
        PipelineStream.DSVFormat = DSVFormat;
        if (!SeenDSS && DSVFormat != DXGI_FORMAT_UNKNOWN) {
            // Re-enable depth for the default state.
            static_cast<D3D12_DEPTH_STENCIL_DESC1 &>(PipelineStream.DepthStencilState).DepthEnable = true;
        }
    }
    void RasterizerStateCb(const D3D12_RASTERIZER_DESC &RasterizerState) override { PipelineStream.RasterizerState = CD3DX12_RASTERIZER_DESC(RasterizerState); }
    void RTVFormatsCb(const D3D12_RT_FORMAT_ARRAY &RTVFormats) override { PipelineStream.RTVFormats = RTVFormats; }
    void SampleDescCb(const DXGI_SAMPLE_DESC &SampleDesc) override { PipelineStream.SampleDesc = SampleDesc; }
    void SampleMaskCb(uint SampleMask) override { PipelineStream.SampleMask = SampleMask; }
    void ViewInstancingCb(const D3D12_VIEW_INSTANCING_DESC &ViewInstancingDesc) override { PipelineStream.ViewInstancingDesc = CD3DX12_VIEW_INSTANCING_DESC(ViewInstancingDesc); }
    void CachedPSOCb(const D3D12_CACHED_PIPELINE_STATE &CachedPSO) override { PipelineStream.CachedPSO = CachedPSO; }

private:
    bool SeenDSS;
};

D3D12_PIPELINE_STATE_SUBOBJECT_TYPE D3DX12GetBaseSubobjectType(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE SubobjectType) noexcept;

HRESULT D3DX12ParsePipelineStream(const D3D12_PIPELINE_STATE_STREAM_DESC &Desc, ID3DX12PipelineParserCallbacks *pCallbacks);

//------------------------------------------------------------------------------------------------
inline bool operator==(const D3D12_CLEAR_VALUE &a, const D3D12_CLEAR_VALUE &b) noexcept {
    if (a.Format != b.Format) return false;
    if (a.Format == DXGI_FORMAT_D24_UNORM_S8_UINT || a.Format == DXGI_FORMAT_D16_UNORM || a.Format == DXGI_FORMAT_D32_FLOAT || a.Format == DXGI_FORMAT_D32_FLOAT_S8X24_UINT) {
        return (a.DepthStencil.Depth == b.DepthStencil.Depth) && (a.DepthStencil.Stencil == b.DepthStencil.Stencil);
    } else {
        return (a.Color[0] == b.Color[0]) && (a.Color[1] == b.Color[1]) && (a.Color[2] == b.Color[2]) && (a.Color[3] == b.Color[3]);
    }
}
inline bool operator==(const D3D12_RENDER_PASS_BEGINNING_ACCESS_CLEAR_PARAMETERS &a, const D3D12_RENDER_PASS_BEGINNING_ACCESS_CLEAR_PARAMETERS &b) noexcept {
    return a.ClearValue == b.ClearValue;
}
inline bool operator==(const D3D12_RENDER_PASS_ENDING_ACCESS_RESOLVE_PARAMETERS &a, const D3D12_RENDER_PASS_ENDING_ACCESS_RESOLVE_PARAMETERS &b) noexcept {
    if (a.pSrcResource != b.pSrcResource) return false;
    if (a.pDstResource != b.pDstResource) return false;
    if (a.SubresourceCount != b.SubresourceCount) return false;
    if (a.Format != b.Format) return false;
    if (a.ResolveMode != b.ResolveMode) return false;
    if (a.PreserveResolveSource != b.PreserveResolveSource) return false;
    return true;
}
inline bool operator==(const D3D12_RENDER_PASS_BEGINNING_ACCESS &a, const D3D12_RENDER_PASS_BEGINNING_ACCESS &b) noexcept {
    if (a.Type != b.Type) return false;
    if (a.Type == D3D12_RENDER_PASS_BEGINNING_ACCESS_TYPE_CLEAR && !(a.Clear == b.Clear)) return false;
    return true;
}
inline bool operator==(const D3D12_RENDER_PASS_ENDING_ACCESS &a, const D3D12_RENDER_PASS_ENDING_ACCESS &b) noexcept {
    if (a.Type != b.Type) return false;
    if (a.Type == D3D12_RENDER_PASS_ENDING_ACCESS_TYPE_RESOLVE && !(a.Resolve == b.Resolve)) return false;
    return true;
}
inline bool operator==(const D3D12_RENDER_PASS_RENDER_TARGET_DESC &a, const D3D12_RENDER_PASS_RENDER_TARGET_DESC &b) noexcept {
    if (a.cpuDescriptor.ptr != b.cpuDescriptor.ptr) return false;
    if (!(a.BeginningAccess == b.BeginningAccess)) return false;
    if (!(a.EndingAccess == b.EndingAccess)) return false;
    return true;
}
inline bool operator==(const D3D12_RENDER_PASS_DEPTH_STENCIL_DESC &a, const D3D12_RENDER_PASS_DEPTH_STENCIL_DESC &b) noexcept {
    if (a.cpuDescriptor.ptr != b.cpuDescriptor.ptr) return false;
    if (!(a.DepthBeginningAccess == b.DepthBeginningAccess)) return false;
    if (!(a.StencilBeginningAccess == b.StencilBeginningAccess)) return false;
    if (!(a.DepthEndingAccess == b.DepthEndingAccess)) return false;
    if (!(a.StencilEndingAccess == b.StencilEndingAccess)) return false;
    return true;
}

#ifndef D3DX12_NO_STATE_OBJECT_HELPERS

//================================================================================================
// D3DX12 State Object Creation Helpers
//
// Helper classes for creating new style state objects out of an arbitrary set of subobjects.
// Uses STL
//
// Start by instantiating CD3DX12_STATE_OBJECT_DESC (see it's public methods).
// One of its methods is CreateSubobject(), which has a comment showing a couple of options for
// defining subobjects using the helper classes for each subobject (CD3DX12_DXIL_LIBRARY_SUBOBJECT
// etc.). The subobject helpers each have methods specific to the subobject for configuring it's
// contents.
//
//================================================================================================
#include <memory>
#ifndef D3DX12_USE_ATL
#include <wrl/client.h>
#define D3DX12_COM_PTR Microsoft::WRL::ComPtr
#define D3DX12_COM_PTR_GET(x) x.Get()
#define D3DX12_COM_PTR_ADDRESSOF(x) x.GetAddressOf()
#else
#include <atlbase.h>
#define D3DX12_COM_PTR ATL::CComPtr
#define D3DX12_COM_PTR_GET(x) x.p
#define D3DX12_COM_PTR_ADDRESSOF(x) &x.p
#endif

//------------------------------------------------------------------------------------------------
class CD3DX12_STATE_OBJECT_DESC {
public:
    CD3DX12_STATE_OBJECT_DESC() noexcept {
        Init(D3D12_STATE_OBJECT_TYPE_COLLECTION);
    }
    CD3DX12_STATE_OBJECT_DESC(D3D12_STATE_OBJECT_TYPE Type) noexcept {
        Init(Type);
    }
    void SetStateObjectType(D3D12_STATE_OBJECT_TYPE Type) noexcept { m_Desc.Type = Type; }
    operator const D3D12_STATE_OBJECT_DESC &() {
        // Do final preparation work
        m_RepointedAssociations.clear();
        m_SubobjectArray.clear();
        m_SubobjectArray.reserve(m_Desc.NumSubobjects);
        // Flatten subobjects into an array (each flattened subobject still has a
        // member that's a pointer to it's desc that's not flattened)
        for (auto Iter = m_SubobjectList.begin();
             Iter != m_SubobjectList.end(); Iter++) {
            m_SubobjectArray.push_back(*Iter);
            // Store new location in array so we can redirect pointers contained in subobjects
            Iter->pSubobjectArrayLocation = &(*(m_SubobjectArray.end() - 1));
        }
        // For subobjects with pointer fields, create a new copy of those subobject definitions
        // with fixed pointers
        for (uint i = 0; i < m_Desc.NumSubobjects; i++) {
            if (m_SubobjectArray[i].Type == D3D12_STATE_SUBOBJECT_TYPE_SUBOBJECT_TO_EXPORTS_ASSOCIATION) {
                auto pOriginalSubobjectAssociation =
                    static_cast<const D3D12_SUBOBJECT_TO_EXPORTS_ASSOCIATION *>(m_SubobjectArray[i].pDesc);
                D3D12_SUBOBJECT_TO_EXPORTS_ASSOCIATION Repointed = *pOriginalSubobjectAssociation;
                auto pWrapper =
                    static_cast<const SUBOBJECT_WRAPPER *>(pOriginalSubobjectAssociation->pSubobjectToAssociate);
                Repointed.pSubobjectToAssociate = pWrapper->pSubobjectArrayLocation;
                m_RepointedAssociations.push_back(Repointed);
                m_SubobjectArray[i].pDesc = &(*(m_RepointedAssociations.end() - 1));
            }
        }
        // Below: using ugly way to get pointer in case .data() is not defined
        m_Desc.pSubobjects = m_Desc.NumSubobjects ? &m_SubobjectArray[0] : nullptr;
        return m_Desc;
    }
    operator const D3D12_STATE_OBJECT_DESC *() {
        // Cast calls the above final preparation work
        return &static_cast<const D3D12_STATE_OBJECT_DESC &>(*this);
    }

    // CreateSubobject creates a sububject helper (e.g. CD3DX12_HIT_GROUP_SUBOBJECT)
    // whose lifetime is owned by this class.
    // e.g.
    //
    //	CD3DX12_STATE_OBJECT_DESC Collection1(D3D12_STATE_OBJECT_TYPE_COLLECTION);
    //	auto Lib0 = Collection1.CreateSubobject<CD3DX12_DXIL_LIBRARY_SUBOBJECT>();
    //	Lib0->SetDXILLibrary(&pMyAppDxilLibs[0]);
    //	Lib0->DefineExport(L"rayGenShader0"_sv); // in practice these export listings might be
    //										  // data/engine driven
    //	etc.
    //
    // Alternatively, users can instantiate sububject helpers explicitly, such as via local
    // variables instead, passing the state object desc that should point to it into the helper
    // constructor (or call mySubobjectHelper.AddToStateObject(Collection1)).
    // In this alternative scenario, the user must keep the subobject alive as long as the state
    // object it is associated with is alive, else it's pointer references will be stale.
    // e.g.
    //
    //	CD3DX12_STATE_OBJECT_DESC RaytracingState2(D3D12_STATE_OBJECT_TYPE_RAYTRACING_PIPELINE);
    //	CD3DX12_DXIL_LIBRARY_SUBOBJECT LibA(RaytracingState2);
    //	LibA.SetDXILLibrary(&pMyAppDxilLibs[4]); // not manually specifying exports
    //											 // - meaning all exports in the libraries
    //											 // are exported
    //	etc.

    template<typename T>
    T *CreateSubobject() {
        T *pSubobject = new T(*this);
        m_OwnedSubobjectHelpers.emplace_back(pSubobject);
        return pSubobject;
    }

private:
    D3D12_STATE_SUBOBJECT *TrackSubobject(D3D12_STATE_SUBOBJECT_TYPE Type, void *pDesc) {
        SUBOBJECT_WRAPPER Subobject;
        Subobject.pSubobjectArrayLocation = nullptr;
        Subobject.Type = Type;
        Subobject.pDesc = pDesc;
        m_SubobjectList.push_back(Subobject);
        m_Desc.NumSubobjects++;
        return &(*(m_SubobjectList.end() - 1));
    }
    void Init(D3D12_STATE_OBJECT_TYPE Type) noexcept {
        SetStateObjectType(Type);
        m_Desc.pSubobjects = nullptr;
        m_Desc.NumSubobjects = 0;
        m_SubobjectList.clear();
        m_SubobjectArray.clear();
        m_RepointedAssociations.clear();
    }
    typedef struct SUBOBJECT_WRAPPER : public D3D12_STATE_SUBOBJECT {
        D3D12_STATE_SUBOBJECT *pSubobjectArrayLocation;// new location when flattened into array
                                                       // for repointing pointers in subobjects
    } SUBOBJECT_WRAPPER;
    D3D12_STATE_OBJECT_DESC m_Desc;
    vstd::vector<SUBOBJECT_WRAPPER> m_SubobjectList;     // Pointers to list nodes handed out so
                                                         // these can be edited live
    vstd::vector<D3D12_STATE_SUBOBJECT> m_SubobjectArray;// Built at the end, copying list contents

    vstd::vector<D3D12_SUBOBJECT_TO_EXPORTS_ASSOCIATION>
        m_RepointedAssociations;// subobject type that contains pointers to other subobjects,
                                // repointed to flattened array

    class StringContainer {
    public:
        LPCWSTR LocalCopy(LPCWSTR string, bool bSingleString = false) {
            if (string) {
                if (bSingleString) {
                    m_Strings.clear();
                    m_Strings.push_back(string);
                } else {
                    m_Strings.push_back(string);
                }
                return (m_Strings.end() - 1)->c_str();
            } else {
                return nullptr;
            }
        }
        void clear() noexcept { m_Strings.clear(); }

    private:
        vstd::vector<vstd::wstring> m_Strings;
    };

    class SUBOBJECT_HELPER_BASE {
    public:
        SUBOBJECT_HELPER_BASE() noexcept { Init(); }
        virtual ~SUBOBJECT_HELPER_BASE() = default;
        virtual D3D12_STATE_SUBOBJECT_TYPE Type() const noexcept = 0;
        void AddToStateObject(CD3DX12_STATE_OBJECT_DESC &ContainingStateObject) {
            m_pSubobject = ContainingStateObject.TrackSubobject(Type(), Data());
        }

    protected:
        virtual void *Data() noexcept = 0;
        void Init() noexcept { m_pSubobject = nullptr; }
        D3D12_STATE_SUBOBJECT *m_pSubobject;
    };

#if (__cplusplus >= 201103L)
    vstd::vector<std::unique_ptr<const SUBOBJECT_HELPER_BASE>> m_OwnedSubobjectHelpers;
#else
    class OWNED_HELPER {
    public:
        OWNED_HELPER(const SUBOBJECT_HELPER_BASE *pHelper) noexcept { m_pHelper = pHelper; }
        ~OWNED_HELPER() { delete m_pHelper; }
        const SUBOBJECT_HELPER_BASE *m_pHelper;
    };

    vstd::vector<OWNED_HELPER> m_OwnedSubobjectHelpers;
#endif

    friend class CD3DX12_DXIL_LIBRARY_SUBOBJECT;
    friend class CD3DX12_EXISTING_COLLECTION_SUBOBJECT;
    friend class CD3DX12_SUBOBJECT_TO_EXPORTS_ASSOCIATION_SUBOBJECT;
    friend class CD3DX12_DXIL_SUBOBJECT_TO_EXPORTS_ASSOCIATION;
    friend class CD3DX12_HIT_GROUP_SUBOBJECT;
    friend class CD3DX12_RAYTRACING_SHADER_CONFIG_SUBOBJECT;
    friend class CD3DX12_RAYTRACING_PIPELINE_CONFIG_SUBOBJECT;
    friend class CD3DX12_RAYTRACING_PIPELINE_CONFIG1_SUBOBJECT;
    friend class CD3DX12_GLOBAL_ROOT_SIGNATURE_SUBOBJECT;
    friend class CD3DX12_LOCAL_ROOT_SIGNATURE_SUBOBJECT;
    friend class CD3DX12_STATE_OBJECT_CONFIG_SUBOBJECT;
    friend class CD3DX12_NODE_MASK_SUBOBJECT;
};

//------------------------------------------------------------------------------------------------
class CD3DX12_DXIL_LIBRARY_SUBOBJECT
    : public CD3DX12_STATE_OBJECT_DESC::SUBOBJECT_HELPER_BASE {
public:
    CD3DX12_DXIL_LIBRARY_SUBOBJECT() noexcept {
        Init();
    }
    CD3DX12_DXIL_LIBRARY_SUBOBJECT(CD3DX12_STATE_OBJECT_DESC &ContainingStateObject) {
        Init();
        AddToStateObject(ContainingStateObject);
    }
    void SetDXILLibrary(const D3D12_SHADER_BYTECODE *pCode) noexcept {
        static const D3D12_SHADER_BYTECODE Default = {};
        m_Desc.DXILLibrary = pCode ? *pCode : Default;
    }
    void DefineExport(
        LPCWSTR Name,
        LPCWSTR ExportToRename = nullptr,
        D3D12_EXPORT_FLAGS Flags = D3D12_EXPORT_FLAG_NONE) {
        D3D12_EXPORT_DESC Export;
        Export.Name = m_Strings.LocalCopy(Name, false);
        Export.ExportToRename = m_Strings.LocalCopy(ExportToRename, false);
        Export.Flags = Flags;
        m_Exports.push_back(Export);
        m_Desc.pExports = &m_Exports[0];// using ugly way to get pointer in case .data() is not defined
        m_Desc.NumExports = static_cast<uint>(m_Exports.size());
    }
    template<size_t N>
    void DefineExports(LPCWSTR (&Exports)[N]) {
        for (uint i = 0; i < N; i++) {
            DefineExport(Exports[i]);
        }
    }
    void DefineExports(const LPCWSTR *Exports, uint N) {
        for (uint i = 0; i < N; i++) {
            DefineExport(Exports[i]);
        }
    }
    D3D12_STATE_SUBOBJECT_TYPE Type() const noexcept override {
        return D3D12_STATE_SUBOBJECT_TYPE_DXIL_LIBRARY;
    }
    operator const D3D12_STATE_SUBOBJECT &() const noexcept { return *m_pSubobject; }
    operator const D3D12_DXIL_LIBRARY_DESC &() const noexcept { return m_Desc; }

private:
    void Init() noexcept {
        SUBOBJECT_HELPER_BASE::Init();
        m_Desc = {};
        m_Strings.clear();
        m_Exports.clear();
    }
    void *Data() noexcept override { return &m_Desc; }
    D3D12_DXIL_LIBRARY_DESC m_Desc;
    CD3DX12_STATE_OBJECT_DESC::StringContainer m_Strings;
    vstd::vector<D3D12_EXPORT_DESC> m_Exports;
};

//------------------------------------------------------------------------------------------------
class CD3DX12_EXISTING_COLLECTION_SUBOBJECT
    : public CD3DX12_STATE_OBJECT_DESC::SUBOBJECT_HELPER_BASE {
public:
    CD3DX12_EXISTING_COLLECTION_SUBOBJECT() noexcept {
        Init();
    }
    CD3DX12_EXISTING_COLLECTION_SUBOBJECT(CD3DX12_STATE_OBJECT_DESC &ContainingStateObject) {
        Init();
        AddToStateObject(ContainingStateObject);
    }
    void SetExistingCollection(ID3D12StateObject *pExistingCollection) noexcept {
        m_Desc.pExistingCollection = pExistingCollection;
        m_CollectionRef = pExistingCollection;
    }
    void DefineExport(
        LPCWSTR Name,
        LPCWSTR ExportToRename = nullptr,
        D3D12_EXPORT_FLAGS Flags = D3D12_EXPORT_FLAG_NONE) {
        D3D12_EXPORT_DESC Export;
        Export.Name = m_Strings.LocalCopy(Name, false);
        Export.ExportToRename = m_Strings.LocalCopy(ExportToRename, false);
        Export.Flags = Flags;
        m_Exports.push_back(Export);
        m_Desc.pExports = &m_Exports[0];// using ugly way to get pointer in case .data() is not defined
        m_Desc.NumExports = static_cast<uint>(m_Exports.size());
    }
    template<size_t N>
    void DefineExports(LPCWSTR (&Exports)[N]) {
        for (uint i = 0; i < N; i++) {
            DefineExport(Exports[i]);
        }
    }
    void DefineExports(const LPCWSTR *Exports, uint N) {
        for (uint i = 0; i < N; i++) {
            DefineExport(Exports[i]);
        }
    }
    D3D12_STATE_SUBOBJECT_TYPE Type() const noexcept override {
        return D3D12_STATE_SUBOBJECT_TYPE_EXISTING_COLLECTION;
    }
    operator const D3D12_STATE_SUBOBJECT &() const noexcept { return *m_pSubobject; }
    operator const D3D12_EXISTING_COLLECTION_DESC &() const noexcept { return m_Desc; }

private:
    void Init() noexcept {
        SUBOBJECT_HELPER_BASE::Init();
        m_Desc = {};
        m_CollectionRef = nullptr;
        m_Strings.clear();
        m_Exports.clear();
    }
    void *Data() noexcept override { return &m_Desc; }
    D3D12_EXISTING_COLLECTION_DESC m_Desc;
    D3DX12_COM_PTR<ID3D12StateObject> m_CollectionRef;
    CD3DX12_STATE_OBJECT_DESC::StringContainer m_Strings;
    vstd::vector<D3D12_EXPORT_DESC> m_Exports;
};

//------------------------------------------------------------------------------------------------
class CD3DX12_SUBOBJECT_TO_EXPORTS_ASSOCIATION_SUBOBJECT
    : public CD3DX12_STATE_OBJECT_DESC::SUBOBJECT_HELPER_BASE {
public:
    CD3DX12_SUBOBJECT_TO_EXPORTS_ASSOCIATION_SUBOBJECT() noexcept {
        Init();
    }
    CD3DX12_SUBOBJECT_TO_EXPORTS_ASSOCIATION_SUBOBJECT(CD3DX12_STATE_OBJECT_DESC &ContainingStateObject) {
        Init();
        AddToStateObject(ContainingStateObject);
    }
    void SetSubobjectToAssociate(const D3D12_STATE_SUBOBJECT &SubobjectToAssociate) noexcept {
        m_Desc.pSubobjectToAssociate = &SubobjectToAssociate;
    }
    void AddExport(LPCWSTR Export) {
        m_Desc.NumExports++;
        m_Exports.push_back(m_Strings.LocalCopy(Export, false));
        m_Desc.pExports = &m_Exports[0];// using ugly way to get pointer in case .data() is not defined
    }
    template<size_t N>
    void AddExports(LPCWSTR (&Exports)[N]) {
        for (uint i = 0; i < N; i++) {
            AddExport(Exports[i]);
        }
    }
    void AddExports(const LPCWSTR *Exports, uint N) {
        for (uint i = 0; i < N; i++) {
            AddExport(Exports[i]);
        }
    }
    D3D12_STATE_SUBOBJECT_TYPE Type() const noexcept override {
        return D3D12_STATE_SUBOBJECT_TYPE_SUBOBJECT_TO_EXPORTS_ASSOCIATION;
    }
    operator const D3D12_STATE_SUBOBJECT &() const noexcept { return *m_pSubobject; }
    operator const D3D12_SUBOBJECT_TO_EXPORTS_ASSOCIATION &() const noexcept { return m_Desc; }

private:
    void Init() noexcept {
        SUBOBJECT_HELPER_BASE::Init();
        m_Desc = {};
        m_Strings.clear();
        m_Exports.clear();
    }
    void *Data() noexcept override { return &m_Desc; }
    D3D12_SUBOBJECT_TO_EXPORTS_ASSOCIATION m_Desc;
    CD3DX12_STATE_OBJECT_DESC::StringContainer m_Strings;
    vstd::vector<LPCWSTR> m_Exports;
};

//------------------------------------------------------------------------------------------------
class CD3DX12_DXIL_SUBOBJECT_TO_EXPORTS_ASSOCIATION
    : public CD3DX12_STATE_OBJECT_DESC::SUBOBJECT_HELPER_BASE {
public:
    CD3DX12_DXIL_SUBOBJECT_TO_EXPORTS_ASSOCIATION() noexcept {
        Init();
    }
    CD3DX12_DXIL_SUBOBJECT_TO_EXPORTS_ASSOCIATION(CD3DX12_STATE_OBJECT_DESC &ContainingStateObject) {
        Init();
        AddToStateObject(ContainingStateObject);
    }
    void SetSubobjectNameToAssociate(LPCWSTR SubobjectToAssociate) {
        m_Desc.SubobjectToAssociate = m_SubobjectName.LocalCopy(SubobjectToAssociate, true);
    }
    void AddExport(LPCWSTR Export) {
        m_Desc.NumExports++;
        m_Exports.push_back(m_Strings.LocalCopy(Export, false));
        m_Desc.pExports = &m_Exports[0];// using ugly way to get pointer in case .data() is not defined
    }
    template<size_t N>
    void AddExports(LPCWSTR (&Exports)[N]) {
        for (uint i = 0; i < N; i++) {
            AddExport(Exports[i]);
        }
    }
    void AddExports(const LPCWSTR *Exports, uint N) {
        for (uint i = 0; i < N; i++) {
            AddExport(Exports[i]);
        }
    }
    D3D12_STATE_SUBOBJECT_TYPE Type() const noexcept override {
        return D3D12_STATE_SUBOBJECT_TYPE_DXIL_SUBOBJECT_TO_EXPORTS_ASSOCIATION;
    }
    operator const D3D12_STATE_SUBOBJECT &() const noexcept { return *m_pSubobject; }
    operator const D3D12_DXIL_SUBOBJECT_TO_EXPORTS_ASSOCIATION &() const noexcept { return m_Desc; }

private:
    void Init() noexcept {
        SUBOBJECT_HELPER_BASE::Init();
        m_Desc = {};
        m_Strings.clear();
        m_SubobjectName.clear();
        m_Exports.clear();
    }
    void *Data() noexcept override { return &m_Desc; }
    D3D12_DXIL_SUBOBJECT_TO_EXPORTS_ASSOCIATION m_Desc;
    CD3DX12_STATE_OBJECT_DESC::StringContainer m_Strings;
    CD3DX12_STATE_OBJECT_DESC::StringContainer m_SubobjectName;
    vstd::vector<LPCWSTR> m_Exports;
};

//------------------------------------------------------------------------------------------------
class CD3DX12_HIT_GROUP_SUBOBJECT
    : public CD3DX12_STATE_OBJECT_DESC::SUBOBJECT_HELPER_BASE {
public:
    CD3DX12_HIT_GROUP_SUBOBJECT() noexcept {
        Init();
    }
    CD3DX12_HIT_GROUP_SUBOBJECT(CD3DX12_STATE_OBJECT_DESC &ContainingStateObject) {
        Init();
        AddToStateObject(ContainingStateObject);
    }
    void SetHitGroupExport(LPCWSTR exportName) {
        m_Desc.HitGroupExport = m_Strings[0].LocalCopy(exportName, true);
    }
    void SetHitGroupType(D3D12_HIT_GROUP_TYPE Type) noexcept { m_Desc.Type = Type; }
    void SetAnyHitShaderImport(LPCWSTR importName) {
        m_Desc.AnyHitShaderImport = m_Strings[1].LocalCopy(importName, true);
    }
    void SetClosestHitShaderImport(LPCWSTR importName) {
        m_Desc.ClosestHitShaderImport = m_Strings[2].LocalCopy(importName, true);
    }
    void SetIntersectionShaderImport(LPCWSTR importName) {
        m_Desc.IntersectionShaderImport = m_Strings[3].LocalCopy(importName, true);
    }
    D3D12_STATE_SUBOBJECT_TYPE Type() const noexcept override {
        return D3D12_STATE_SUBOBJECT_TYPE_HIT_GROUP;
    }
    operator const D3D12_STATE_SUBOBJECT &() const noexcept { return *m_pSubobject; }
    operator const D3D12_HIT_GROUP_DESC &() const noexcept { return m_Desc; }

private:
    void Init() noexcept {
        SUBOBJECT_HELPER_BASE::Init();
        m_Desc = {};
        for (uint i = 0; i < m_NumStrings; i++) {
            m_Strings[i].clear();
        }
    }
    void *Data() noexcept override { return &m_Desc; }
    D3D12_HIT_GROUP_DESC m_Desc;
    static const uint m_NumStrings = 4;
    CD3DX12_STATE_OBJECT_DESC::StringContainer
        m_Strings[m_NumStrings];// one string for every entrypoint name
};

//------------------------------------------------------------------------------------------------
class CD3DX12_RAYTRACING_SHADER_CONFIG_SUBOBJECT
    : public CD3DX12_STATE_OBJECT_DESC::SUBOBJECT_HELPER_BASE {
public:
    CD3DX12_RAYTRACING_SHADER_CONFIG_SUBOBJECT() noexcept {
        Init();
    }
    CD3DX12_RAYTRACING_SHADER_CONFIG_SUBOBJECT(CD3DX12_STATE_OBJECT_DESC &ContainingStateObject) {
        Init();
        AddToStateObject(ContainingStateObject);
    }
    void Config(uint MaxPayloadSizeInBytes, uint MaxAttributeSizeInBytes) noexcept {
        m_Desc.MaxPayloadSizeInBytes = MaxPayloadSizeInBytes;
        m_Desc.MaxAttributeSizeInBytes = MaxAttributeSizeInBytes;
    }
    D3D12_STATE_SUBOBJECT_TYPE Type() const noexcept override {
        return D3D12_STATE_SUBOBJECT_TYPE_RAYTRACING_SHADER_CONFIG;
    }
    operator const D3D12_STATE_SUBOBJECT &() const noexcept { return *m_pSubobject; }
    operator const D3D12_RAYTRACING_SHADER_CONFIG &() const noexcept { return m_Desc; }

private:
    void Init() noexcept {
        SUBOBJECT_HELPER_BASE::Init();
        m_Desc = {};
    }
    void *Data() noexcept override { return &m_Desc; }
    D3D12_RAYTRACING_SHADER_CONFIG m_Desc;
};

//------------------------------------------------------------------------------------------------
class CD3DX12_RAYTRACING_PIPELINE_CONFIG_SUBOBJECT
    : public CD3DX12_STATE_OBJECT_DESC::SUBOBJECT_HELPER_BASE {
public:
    CD3DX12_RAYTRACING_PIPELINE_CONFIG_SUBOBJECT() noexcept {
        Init();
    }
    CD3DX12_RAYTRACING_PIPELINE_CONFIG_SUBOBJECT(CD3DX12_STATE_OBJECT_DESC &ContainingStateObject) {
        Init();
        AddToStateObject(ContainingStateObject);
    }
    void Config(uint MaxTraceRecursionDepth) noexcept {
        m_Desc.MaxTraceRecursionDepth = MaxTraceRecursionDepth;
    }
    D3D12_STATE_SUBOBJECT_TYPE Type() const noexcept override {
        return D3D12_STATE_SUBOBJECT_TYPE_RAYTRACING_PIPELINE_CONFIG;
    }
    operator const D3D12_STATE_SUBOBJECT &() const noexcept { return *m_pSubobject; }
    operator const D3D12_RAYTRACING_PIPELINE_CONFIG &() const noexcept { return m_Desc; }

private:
    void Init() noexcept {
        SUBOBJECT_HELPER_BASE::Init();
        m_Desc = {};
    }
    void *Data() noexcept override { return &m_Desc; }
    D3D12_RAYTRACING_PIPELINE_CONFIG m_Desc;
};

//------------------------------------------------------------------------------------------------
class CD3DX12_RAYTRACING_PIPELINE_CONFIG1_SUBOBJECT
    : public CD3DX12_STATE_OBJECT_DESC::SUBOBJECT_HELPER_BASE {
public:
    CD3DX12_RAYTRACING_PIPELINE_CONFIG1_SUBOBJECT() noexcept {
        Init();
    }
    CD3DX12_RAYTRACING_PIPELINE_CONFIG1_SUBOBJECT(CD3DX12_STATE_OBJECT_DESC &ContainingStateObject) {
        Init();
        AddToStateObject(ContainingStateObject);
    }
    void Config(uint MaxTraceRecursionDepth, D3D12_RAYTRACING_PIPELINE_FLAGS Flags) noexcept {
        m_Desc.MaxTraceRecursionDepth = MaxTraceRecursionDepth;
        m_Desc.Flags = Flags;
    }
    D3D12_STATE_SUBOBJECT_TYPE Type() const noexcept override {
        return D3D12_STATE_SUBOBJECT_TYPE_RAYTRACING_PIPELINE_CONFIG1;
    }
    operator const D3D12_STATE_SUBOBJECT &() const noexcept { return *m_pSubobject; }
    operator const D3D12_RAYTRACING_PIPELINE_CONFIG1 &() const noexcept { return m_Desc; }

private:
    void Init() noexcept {
        SUBOBJECT_HELPER_BASE::Init();
        m_Desc = {};
    }
    void *Data() noexcept override { return &m_Desc; }
    D3D12_RAYTRACING_PIPELINE_CONFIG1 m_Desc;
};

//------------------------------------------------------------------------------------------------
class CD3DX12_GLOBAL_ROOT_SIGNATURE_SUBOBJECT
    : public CD3DX12_STATE_OBJECT_DESC::SUBOBJECT_HELPER_BASE {
public:
    CD3DX12_GLOBAL_ROOT_SIGNATURE_SUBOBJECT() noexcept {
        Init();
    }
    CD3DX12_GLOBAL_ROOT_SIGNATURE_SUBOBJECT(CD3DX12_STATE_OBJECT_DESC &ContainingStateObject) {
        Init();
        AddToStateObject(ContainingStateObject);
    }
    void SetRootSignature(ID3D12RootSignature *pRootSig) noexcept {
        m_pRootSig = pRootSig;
    }
    D3D12_STATE_SUBOBJECT_TYPE Type() const noexcept override {
        return D3D12_STATE_SUBOBJECT_TYPE_GLOBAL_ROOT_SIGNATURE;
    }
    operator const D3D12_STATE_SUBOBJECT &() const noexcept { return *m_pSubobject; }
    operator ID3D12RootSignature *() const noexcept { return D3DX12_COM_PTR_GET(m_pRootSig); }

private:
    void Init() noexcept {
        SUBOBJECT_HELPER_BASE::Init();
        m_pRootSig = nullptr;
    }
    void *Data() noexcept override { return D3DX12_COM_PTR_ADDRESSOF(m_pRootSig); }
    D3DX12_COM_PTR<ID3D12RootSignature> m_pRootSig;
};

//------------------------------------------------------------------------------------------------
class CD3DX12_LOCAL_ROOT_SIGNATURE_SUBOBJECT
    : public CD3DX12_STATE_OBJECT_DESC::SUBOBJECT_HELPER_BASE {
public:
    CD3DX12_LOCAL_ROOT_SIGNATURE_SUBOBJECT() noexcept {
        Init();
    }
    CD3DX12_LOCAL_ROOT_SIGNATURE_SUBOBJECT(CD3DX12_STATE_OBJECT_DESC &ContainingStateObject) {
        Init();
        AddToStateObject(ContainingStateObject);
    }
    void SetRootSignature(ID3D12RootSignature *pRootSig) noexcept {
        m_pRootSig = pRootSig;
    }
    D3D12_STATE_SUBOBJECT_TYPE Type() const noexcept override {
        return D3D12_STATE_SUBOBJECT_TYPE_LOCAL_ROOT_SIGNATURE;
    }
    operator const D3D12_STATE_SUBOBJECT &() const noexcept { return *m_pSubobject; }
    operator ID3D12RootSignature *() const noexcept { return D3DX12_COM_PTR_GET(m_pRootSig); }

private:
    void Init() noexcept {
        SUBOBJECT_HELPER_BASE::Init();
        m_pRootSig = nullptr;
    }
    void *Data() noexcept override { return D3DX12_COM_PTR_ADDRESSOF(m_pRootSig); }
    D3DX12_COM_PTR<ID3D12RootSignature> m_pRootSig;
};

//------------------------------------------------------------------------------------------------
class CD3DX12_STATE_OBJECT_CONFIG_SUBOBJECT
    : public CD3DX12_STATE_OBJECT_DESC::SUBOBJECT_HELPER_BASE {
public:
    CD3DX12_STATE_OBJECT_CONFIG_SUBOBJECT() noexcept {
        Init();
    }
    CD3DX12_STATE_OBJECT_CONFIG_SUBOBJECT(CD3DX12_STATE_OBJECT_DESC &ContainingStateObject) {
        Init();
        AddToStateObject(ContainingStateObject);
    }
    void SetFlags(D3D12_STATE_OBJECT_FLAGS Flags) noexcept {
        m_Desc.Flags = Flags;
    }
    D3D12_STATE_SUBOBJECT_TYPE Type() const noexcept override {
        return D3D12_STATE_SUBOBJECT_TYPE_STATE_OBJECT_CONFIG;
    }
    operator const D3D12_STATE_SUBOBJECT &() const noexcept { return *m_pSubobject; }
    operator const D3D12_STATE_OBJECT_CONFIG &() const noexcept { return m_Desc; }

private:
    void Init() noexcept {
        SUBOBJECT_HELPER_BASE::Init();
        m_Desc = {};
    }
    void *Data() noexcept override { return &m_Desc; }
    D3D12_STATE_OBJECT_CONFIG m_Desc;
};

//------------------------------------------------------------------------------------------------
class CD3DX12_NODE_MASK_SUBOBJECT
    : public CD3DX12_STATE_OBJECT_DESC::SUBOBJECT_HELPER_BASE {
public:
    CD3DX12_NODE_MASK_SUBOBJECT() noexcept {
        Init();
    }
    CD3DX12_NODE_MASK_SUBOBJECT(CD3DX12_STATE_OBJECT_DESC &ContainingStateObject) {
        Init();
        AddToStateObject(ContainingStateObject);
    }
    void SetNodeMask(uint NodeMask) noexcept {
        m_Desc.NodeMask = NodeMask;
    }
    D3D12_STATE_SUBOBJECT_TYPE Type() const noexcept override {
        return D3D12_STATE_SUBOBJECT_TYPE_NODE_MASK;
    }
    operator const D3D12_STATE_SUBOBJECT &() const noexcept { return *m_pSubobject; }
    operator const D3D12_NODE_MASK &() const noexcept { return m_Desc; }

private:
    void Init() noexcept {
        SUBOBJECT_HELPER_BASE::Init();
        m_Desc = {};
    }
    void *Data() noexcept override { return &m_Desc; }
    D3D12_NODE_MASK m_Desc;
};

#undef D3DX12_COM_PTR
#undef D3DX12_COM_PTR_GET
#undef D3DX12_COM_PTR_ADDRESSOF
#endif// #ifndef D3DX12_NO_STATE_OBJECT_HELPERS

#endif// defined( __cplusplus )
inline vstd::wstring AnsiToWString(const vstd::string &str) {
    WCHAR buffer[512];
    MultiByteToWideChar(CP_ACP, 0, str.c_str(), -1, buffer, 512);
    return vstd::wstring(buffer);
}

inline const char *d3d12_error_name(HRESULT hr) {
    switch (hr) {
        case D3D12_ERROR_ADAPTER_NOT_FOUND: return "D3D12_ERROR_ADAPTER_NOT_FOUND";
        case D3D12_ERROR_DRIVER_VERSION_MISMATCH: return "D3D12_ERROR_DRIVER_VERSION_MISMATCH";
        case DXGI_ERROR_ACCESS_DENIED: return "DXGI_ERROR_ACCESS_DENIED";
        case DXGI_ERROR_ACCESS_LOST: return "DXGI_ERROR_ACCESS_LOST";
        case DXGI_ERROR_ALREADY_EXISTS: return "DXGI_ERROR_ALREADY_EXISTS";
        case DXGI_ERROR_CANNOT_PROTECT_CONTENT: return "DXGI_ERROR_CANNOT_PROTECT_CONTENT";
        case DXGI_ERROR_DEVICE_HUNG: return "DXGI_ERROR_DEVICE_HUNG";
        case DXGI_ERROR_DEVICE_REMOVED: return "DXGI_ERROR_DEVICE_REMOVED";
        case DXGI_ERROR_DEVICE_RESET: return "DXGI_ERROR_DEVICE_RESET";
        case DXGI_ERROR_DRIVER_INTERNAL_ERROR: return "DXGI_ERROR_DRIVER_INTERNAL_ERROR";
        case DXGI_ERROR_GRAPHICS_VIDPN_SOURCE_IN_USE: return "DXGI_ERROR_GRAPHICS_VIDPN_SOURCE_IN_USE";
        case DXGI_ERROR_FRAME_STATISTICS_DISJOINT: return "DXGI_ERROR_FRAME_STATISTICS_DISJOINT";
        case DXGI_ERROR_INVALID_CALL: return "DXGI_ERROR_INVALID_CALL";
        case DXGI_ERROR_MORE_DATA: return "DXGI_ERROR_MORE_DATA";
        case DXGI_ERROR_NAME_ALREADY_EXISTS: return "DXGI_ERROR_NAME_ALREADY_EXISTS";
        case DXGI_ERROR_NONEXCLUSIVE: return "DXGI_ERROR_NONEXCLUSIVE";
        case DXGI_ERROR_NOT_CURRENTLY_AVAILABLE: return "DXGI_ERROR_NOT_CURRENTLY_AVAILABLE";
        case DXGI_ERROR_NOT_FOUND: return "DXGI_ERROR_NOT_FOUND";
        case DXGI_ERROR_REMOTE_CLIENT_DISCONNECTED: return "DXGI_ERROR_REMOTE_CLIENT_DISCONNECTED";
        case DXGI_ERROR_REMOTE_OUTOFMEMORY: return "DXGI_ERROR_REMOTE_OUTOFMEMORY";
        case DXGI_ERROR_RESTRICT_TO_OUTPUT_STALE: return "DXGI_ERROR_RESTRICT_TO_OUTPUT_STALE";
        case DXGI_ERROR_SDK_COMPONENT_MISSING: return "DXGI_ERROR_SDK_COMPONENT_MISSING";
        case DXGI_ERROR_SESSION_DISCONNECTED: return "DXGI_ERROR_SESSION_DISCONNECTED";
        case DXGI_ERROR_UNSUPPORTED: return "DXGI_ERROR_UNSUPPORTED";
        case DXGI_ERROR_WAIT_TIMEOUT: return "DXGI_ERROR_WAIT_TIMEOUT";
        case DXGI_ERROR_WAS_STILL_DRAWING: return "DXGI_ERROR_WAS_STILL_DRAWING";
        case E_FAIL: return "E_FAIL";
        case E_INVALIDARG: return "E_INVALIDARG";
        case E_OUTOFMEMORY: return "E_OUTOFMEMORY";
        case E_NOTIMPL: return "E_NOTIMPL";
        case S_FALSE: return "S_FALSE";
        case S_OK: return "S_OK";
        default: break;
    }
    return "Unknown error";
}

#ifndef ThrowIfFailed
#define ThrowIfFailed(x)                                                          \
    do {                                                                          \
        HRESULT hr_ = (x);                                                        \
        if (hr_ != S_OK) {                                                        \
            LUISA_ERROR_WITH_LOCATION("D3D12 call '{}' failed with "              \
                                      "error {} (code = {}).",                    \
                                      #x, d3d12_error_name(hr_), (long long)hr_); \
            abort();                                                              \
        }                                                                         \
    } while (false)
#endif
#include <luisa/vstl/unique_ptr.h>
namespace vstd {
template<typename T>
struct com_deleter {
    void operator()(T *ptr) const noexcept {
        if constexpr (std::is_base_of_v<IUnknown, T>) {
            ptr->Release();
        } else {
            unique_ptr_deleter().operator()<T>(ptr);
        }
    }
};
template<typename T>
using ComUniquePtr = std::unique_ptr<T, com_deleter<T>>;
template<typename T>
ComUniquePtr<T> create_comptr(
    vstd::function<HRESULT(T **)> const &func) {
    T *ptr = nullptr;
    ThrowIfFailed(func(&ptr));
    return ComUniquePtr<T>(ptr);
}
}// namespace vstd
