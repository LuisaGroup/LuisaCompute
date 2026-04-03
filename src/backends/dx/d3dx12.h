#pragma once

#include <Windows.h>
#include <LCAgilitySDK/d3d12.h>
#include <comdef.h>
#include <dxgi.h>
#include <LCAgilitySDK/d3dx12/d3dx12.h>

// below is utility / helper functions for dx backend
#include <luisa/vstl/common.h>
#include <luisa/vstl/functional.h>
#include <luisa/vstl/unique_ptr.h>
#include <luisa/core/basic_types.h>
#include <luisa/core/logging.h>

#ifdef UNICODE
using lcdx_pchar = LPCWSTR;
#else
using lcdx_pchar = LPCSTR;
#endif

namespace lc::dx {
void process_dxgi_error(HRESULT hr);

#define LUISA_MAKE_VECTOR_TYPES(T) \
    using T##2 = luisa::T##2;      \
    using T##3 = luisa::T##3;      \
    using T##4 = luisa::T##4;

LUISA_MAKE_VECTOR_TYPES(bool)
LUISA_MAKE_VECTOR_TYPES(float)
LUISA_MAKE_VECTOR_TYPES(int)
LUISA_MAKE_VECTOR_TYPES(uint)
using float2x2 = luisa::float2x2;
using float3x3 = luisa::float3x3;
using float4x4 = luisa::float4x4;
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

inline vstd::wstring AnsiToWString(const vstd::string &str) {
    WCHAR buffer[512];
    MultiByteToWideChar(CP_ACP, 0, str.c_str(), -1, buffer, 512);
    return vstd::wstring(buffer);
}
inline const char *d3d12_error_name(HRESULT hr) {
    lc::dx::process_dxgi_error(hr);
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
        if (hr_ != S_OK) [[unlikely]] {                                           \
            LUISA_ERROR_WITH_LOCATION("D3D12 call '{}' failed with "              \
                                      "error {} (code = {}).",                    \
                                      #x, d3d12_error_name(hr_), (long long)hr_); \
            abort();                                                              \
        }                                                                         \
    } while (false)
#endif

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
