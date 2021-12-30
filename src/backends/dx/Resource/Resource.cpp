#pragma vengine_package vengine_directx
#include <Resource/Resource.h>
namespace toolhub::directx {
uint64 Resource::GetTextureSize(
	Device* device,
	uint width,
	uint height, 
	GFXFormat Format, 
	TextureDimension type, 
	uint depthCount, 
	uint mipCount) {
    mipCount = std::max<uint>(mipCount, 1);
	uint arraySize;
	switch (type) {
		case TextureDimension::Cubemap:
			arraySize = 6;
			break;
		case TextureDimension::Tex2D:
		case TextureDimension::Tex1D:
			arraySize = 1;
			break;
		default:
			arraySize = std::max<uint>(1, depthCount);
			break;
	}
	D3D12_RESOURCE_DESC texDesc;
	memset(&texDesc, 0, sizeof(D3D12_RESOURCE_DESC));
	texDesc.Dimension = (type == TextureDimension::Tex3D) ? D3D12_RESOURCE_DIMENSION_TEXTURE3D : D3D12_RESOURCE_DIMENSION_TEXTURE2D;
	texDesc.Alignment = 0;
	texDesc.Width = width;
	texDesc.Height = height;
	texDesc.DepthOrArraySize = arraySize;
	texDesc.MipLevels = mipCount;
	texDesc.Format = (DXGI_FORMAT)Format;
	texDesc.SampleDesc.Count = 1;
	texDesc.SampleDesc.Quality = 0;
	texDesc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
	texDesc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
	auto allocateInfo = device->device->GetResourceAllocationInfo(
		0, 1, &texDesc);
	return allocateInfo.SizeInBytes;
}
uint64 Resource::GetTexturePixelSize(GFXFormat format) {
	switch (format) {
		case GFXFormat_R32G32B32A32_Typeless:
		case GFXFormat_R32G32B32A32_Float:
		case GFXFormat_R32G32B32A32_UInt:
		case GFXFormat_R32G32B32A32_SInt:
			return 16;
		case GFXFormat_R32G32_Typeless:
		case GFXFormat_R32G32_Float:
		case GFXFormat_R32G32_UInt:
		case GFXFormat_R32G32_SInt:
			return 8;
		case GFXFormat_R32_Typeless:
		case GFXFormat_R32_Float:
		case GFXFormat_R32_UInt:
		case GFXFormat_R32_SInt:
			return 4;
		case GFXFormat_R16G16B16A16_Typeless:
		case GFXFormat_R16G16B16A16_Float:
		case GFXFormat_R16G16B16A16_UNorm:
		case GFXFormat_R16G16B16A16_SNorm:
		case GFXFormat_R16G16B16A16_UInt:
		case GFXFormat_R16G16B16A16_SInt:
			return 8;
		case GFXFormat_R16G16_Typeless:
		case GFXFormat_R16G16_Float:
		case GFXFormat_R16G16_UNorm:
		case GFXFormat_R16G16_SNorm:
		case GFXFormat_R16G16_UInt:
		case GFXFormat_R16G16_SInt:
			return 4;
		case GFXFormat_R16_Typeless:
		case GFXFormat_R16_Float:
		case GFXFormat_R16_UNorm:
		case GFXFormat_R16_SNorm:
		case GFXFormat_R16_UInt:
		case GFXFormat_R16_SInt:
			return 2;
		case GFXFormat_R8G8B8A8_Typeless:
		case GFXFormat_R8G8B8A8_UInt:
		case GFXFormat_R8G8B8A8_UNorm:
		case GFXFormat_R8G8B8A8_SNorm:
		case GFXFormat_R8G8B8A8_SInt:
			return 4;
		case GFXFormat_R8G8_Typeless:
		case GFXFormat_R8G8_UInt:
		case GFXFormat_R8G8_UNorm:
		case GFXFormat_R8G8_SNorm:
		case GFXFormat_R8G8_SInt:
			return 2;
		case GFXFormat_R8_Typeless:
		case GFXFormat_R8_UInt:
		case GFXFormat_R8_UNorm:
		case GFXFormat_R8_SNorm:
		case GFXFormat_R8_SInt:
			return 1;
		case GFXFormat_R10G10B10A2_Typeless:
		case GFXFormat_R10G10B10A2_UNorm:
		case GFXFormat_R10G10B10A2_UInt:
		case GFXFormat_R11G11B10_Float:
			return 4;
		case GFXFormat_BC6H_Typeless:
		case GFXFormat_BC6H_UF16:
		case GFXFormat_BC6H_SF16:
		case GFXFormat_BC7_Typeless:
		case GFXFormat_BC7_UNorm:
		case GFXFormat_BC7_UNorm_SRGB:
		case GFXFormat_BC5_Typeless:
		case GFXFormat_BC5_UNorm:
		case GFXFormat_BC5_SNorm:
			return 4;
		case GFXFormat_BC4_Typeless:
		case GFXFormat_BC4_UNorm:
		case GFXFormat_BC4_SNorm:
			return 2;
		default: return 0;
	}
}
}// namespace toolhub::directx