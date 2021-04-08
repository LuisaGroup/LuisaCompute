//***************************************************************************************
// GFXUtil.h by Frank Luna (C) 2015 All Rights Reserved.
//
// General helper code.
//***************************************************************************************
#pragma once
#include <VEngineConfig.h>
#include <windows.h>
#include <wrl.h>
#include <dxgi1_4.h>
#include <d3d12.h>
#include <D3Dcompiler.h>
#include <DirectXColors.h>
#include <DirectXCollision.h>
#include "Common.h"
#include "../Struct/CameraRenderPath.h"
using GFXCommandList = ID3D12GraphicsCommandList;
using GFXDevice = ID3D12Device;
using GFXCommandQueue = ID3D12CommandQueue;
using GFXCommandAllocator = ID3D12CommandAllocator;
using GFXResource = ID3D12Resource;
using GFXPipelineState = ID3D12PipelineState;
using GFXVertexBufferView = D3D12_VERTEX_BUFFER_VIEW;
using GFXIndexBufferView = D3D12_INDEX_BUFFER_VIEW;
#ifdef _DEBUG
#include <comdef.h>

class DxException {
public:
	DxException() = default;
	DxException(HRESULT hr, const vengine::string& functionName, const vengine::wstring& filename, int32_t lineNumber)
		: ErrorCode(hr),
		  FunctionName(functionName),
		  Filename(filename),
		  LineNumber(lineNumber) {}

	vengine::string ToString() const {
		// Get the string description of the error code.
		_com_error err(ErrorCode);
		vengine::wstring msg = err.ErrorMessage();
		vengine::string strMsg;
		strMsg.resize(msg.size());
		for (size_t i = 0; i < msg.size(); ++i) {
			strMsg[i] = msg[i];
		}
		vengine::string strFileName;
		strFileName.resize(Filename.size());
		for (size_t i = 0; i < Filename.size(); ++i) {
			strFileName[i] = Filename[i];
		}
		return FunctionName + " failed in " + strFileName + "; line " + vengine::to_string(LineNumber) + "; error: " + strMsg;
	}
	HRESULT ErrorCode = S_OK;
	vengine::string FunctionName;
	vengine::wstring Filename;
	int32_t LineNumber = -1;
};
#endif
struct GpuAddress {
	uint64 address;
};
enum GPUResourceState {
	GPUResourceState_Common = D3D12_RESOURCE_STATE_COMMON,
	GPUResourceState_VertBuffer = D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER,
	GPUResourceState_IndexBuffer = D3D12_RESOURCE_STATE_INDEX_BUFFER,
	GPUResourceState_ConstBuffer = D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER,
	GPUResourceState_RenderTarget = D3D12_RESOURCE_STATE_RENDER_TARGET,
	GPUResourceState_UnorderedAccess = D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
	GPUResourceState_NonPixelShaderRes = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
	GPUResourceState_IndirectArg = D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
	GPUResourceState_CopyDest = D3D12_RESOURCE_STATE_COPY_DEST,
	GPUResourceState_CopySource = D3D12_RESOURCE_STATE_COPY_SOURCE,
	GPUResourceState_RayTracingStruct = D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE,
	GPUResourceState_ShadingRateSource = D3D12_RESOURCE_STATE_SHADING_RATE_SOURCE,
	GPUResourceState_GenericRead = D3D12_RESOURCE_STATE_GENERIC_READ,
	GPUResourceState_Present = D3D12_RESOURCE_STATE_PRESENT

	//GFXReseourceState_PixelRead = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
	//GPUResourceState_StreamOut = D3D12_RESOURCE_STATE_STREAM_OUT,
	//GPUResourceState_DepthWrite = D3D12_RESOURCE_STATE_DEPTH_WRITE,
	//GPUResourceState_DepthRead = D3D12_RESOURCE_STATE_DEPTH_READ,
	//GPUResourceState_ResolveDest = D3D12_RESOURCE_STATE_RESOLVE_DEST,
	//GPUResourceState_ResolveSource = D3D12_RESOURCE_STATE_RESOLVE_SOURCE,
	//GPUResourceState_Predication = D3D12_RESOURCE_STATE_PREDICATION
};
using GFXResourceState = D3D12_RESOURCE_STATES;
enum PrimitiveTopologyType {
	PrimitiveTopologyType_UnDefined = D3D12_PRIMITIVE_TOPOLOGY_TYPE_UNDEFINED,
	PrimitiveTopologyType_Point = D3D12_PRIMITIVE_TOPOLOGY_TYPE_POINT,
	PrimitiveTopologyType_Line = D3D12_PRIMITIVE_TOPOLOGY_TYPE_LINE,
	PrimitiveTopologyType_Triangle = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE
};
static constexpr D3D_PRIMITIVE_TOPOLOGY GetD3DTopology(PrimitiveTopologyType type) {
	switch (type) {
		case PrimitiveTopologyType_Point:
			return D3D_PRIMITIVE_TOPOLOGY_POINTLIST;
		case PrimitiveTopologyType_Line:
			return D3D_PRIMITIVE_TOPOLOGY_LINELIST;
		case PrimitiveTopologyType_Triangle:
			return D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
		default:
			return D3D_PRIMITIVE_TOPOLOGY_UNDEFINED;
	}
}
enum GFXCommandListType {
	GFXCommandListType_Direct = D3D12_COMMAND_LIST_TYPE_DIRECT,
	GFXCommandListType_Copy = D3D12_COMMAND_LIST_TYPE_COPY,
	GFXCommandListType_Compute = D3D12_COMMAND_LIST_TYPE_COMPUTE
};
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
#include "d3dx12.h"

//////////Configures

class Camera;

INLINE void d3dSetDebugName(IDXGIObject* obj, const char* name) {
	if (obj) {
		obj->SetPrivateData(WKPDID_D3DDebugObjectName, lstrlenA(name), name);
	}
}
INLINE void d3dSetDebugName(GFXDevice* obj, const char* name) {
	if (obj) {
		obj->SetPrivateData(WKPDID_D3DDebugObjectName, lstrlenA(name), name);
	}
}
INLINE void d3dSetDebugName(ID3D12DeviceChild* obj, const char* name) {
	if (obj) {
		obj->SetPrivateData(WKPDID_D3DDebugObjectName, lstrlenA(name), name);
	}
}

INLINE vengine::wstring AnsiToWString(const vengine::string& str) {
	WCHAR buffer[512];
	MultiByteToWideChar(CP_ACP, 0, str.c_str(), -1, buffer, 512);
	return vengine::wstring(buffer);
}

/*
#if defined(_DEBUG)
	#ifndef Assert
	#define Assert(x, description)                                  \
	{                                                               \
		static bool ignoreAssert = false;                           \
		if(!ignoreAssert && !(x))                                   \
		{                                                           \
			Debug::AssertResult result = Debug::ShowAssertDialog(   \
			(L#x), description, AnsiToWString(__FILE__), __LINE__); \
		if(result == Debug::AssertIgnore)                           \
		{                                                           \
			ignoreAssert = true;                                    \
		}                                                           \
					else if(result == Debug::AssertBreak)           \
		{                                                           \
			__debugbreak();                                         \
		}                                                           \
		}                                                           \
	}
	#endif
#else
	#ifndef Assert
	#define Assert(x, description)
	#endif
#endif
	*/

class VENGINE_DLL_RENDERER GFXUtil {
public:
	static constexpr uint64 CalcAlign(uint64 value, uint64 align) {
		return (value + (align - 1)) & ~(align - 1);
	}
	static uint CalcConstantBufferByteSize(uint byteSize) {
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

	static uint64 CalcPlacedOffsetAlignment(uint64 offset) {
		return (offset + (D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT - 1)) & ~(D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT - 1);
	}
	static constexpr uint STATIC_SAMPLER_COUNT = 13;
	static std::array<const CD3DX12_STATIC_SAMPLER_DESC, STATIC_SAMPLER_COUNT> const& GetStaticSamplers();
};

#ifndef ThrowIfFailed
#ifdef NDEBUG
#define ThrowIfFailed(x) [&] { x; }()
#else
#define ThrowIfFailed(x)                                                  \
	[&] {                                                                 \
		HRESULT hr__ = x;                                                 \
		vengine::wstring wfn = AnsiToWString(__FILE__);                   \
		if (FAILED(hr__)) { throw DxException(hr__, #x, wfn, __LINE__); } \
	}()
#endif
#endif
#ifndef ThrowHResult
#define ThrowHResult(hr__, x)                                             \
	[&] {                                                                 \
		vengine::wstring wfn = AnsiToWString(__FILE__);                   \
		if (FAILED(hr__)) { throw DxException(hr__, #x, wfn, __LINE__); } \
	}()

#endif

#ifndef ReleaseCom
#define ReleaseCom(x)     \
	[&] {                 \
		if (x) {          \
			x->Release(); \
			x = 0;        \
		}                 \
	}()
#endif
