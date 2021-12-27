#pragma vengine_package vengine_directx
#include <Resource/RenderTexture.h>
namespace toolhub::directx {
RenderTexture::RenderTexture(
	Device* device,
	uint width,
	uint height,
	GFXFormat format,
	TextureDimension dimension,
	uint depth,
	uint mip,
	bool allowUav,
	IGpuAllocator* allocator)
	: TextureBase(device, width, height, format, dimension, depth, mip),
	  allocHandle(allocator),
	  allowUav(allowUav) {
	D3D12_RESOURCE_DESC texDesc;
	memset(&texDesc, 0, sizeof(D3D12_RESOURCE_DESC));
	switch (dimension) {
		case TextureDimension::Cubemap:
		case TextureDimension::Tex2DArray:
		case TextureDimension::Tex2D:
			texDesc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
			break;
		case TextureDimension::Tex3D:
			texDesc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE3D;
			break;
		case TextureDimension::Tex1D:
			texDesc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE1D;
			break;
	}
	texDesc.Alignment = 0;
	texDesc.Width = this->width;
	texDesc.Height = this->height;
	texDesc.DepthOrArraySize = this->depth;
	texDesc.MipLevels = mip;
	texDesc.Format = (DXGI_FORMAT)format;
	texDesc.SampleDesc.Count = 1;
	texDesc.SampleDesc.Quality = 0;
	texDesc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
	texDesc.Flags = allowUav ? D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS : D3D12_RESOURCE_FLAG_NONE;

	if (!allocator) {
		auto prop = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT);
		D3D12_HEAP_PROPERTIES const* propPtr = &prop;
		ThrowIfFailed(device->device->CreateCommittedResource(
			propPtr,
			D3D12_HEAP_FLAG_NONE,
			&texDesc,
			D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
			nullptr,
			IID_PPV_ARGS(&allocHandle.resource)));
	} else {
		ID3D12Heap* heap;
		uint64 offset;
		allocHandle.allocateHandle = allocator->AllocateTextureHeap(
			device,
			format,
			width,
			height,
			depth,
			dimension,
			mip,
			&heap,
			&offset,
			true);
		ThrowIfFailed(device->device->CreatePlacedResource(
			heap,
			offset,
			&texDesc,
			D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
			nullptr,
			IID_PPV_ARGS(&allocHandle.resource)));
	}
}
RenderTexture ::~RenderTexture() {
}
D3D12_SHADER_RESOURCE_VIEW_DESC RenderTexture::GetColorSrvDesc() const {
	D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc;
	srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
	auto format = allocHandle.resource->GetDesc();
	switch (format.Format) {
		case GFXFormat_D16_UNorm:
			srvDesc.Format = (DXGI_FORMAT)GFXFormat_R16_UNorm;
			break;
		case GFXFormat_D32_Float:
			srvDesc.Format = (DXGI_FORMAT)GFXFormat_R32_Float;
			break;
		default:
			srvDesc.Format = format.Format;
	}
	switch (dimension) {
		case TextureDimension::Cubemap:
			srvDesc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURECUBE;
			srvDesc.TextureCube.MostDetailedMip = 0;
			srvDesc.TextureCube.MipLevels = format.MipLevels;
			srvDesc.TextureCube.ResourceMinLODClamp = 0.0f;
			break;
		case TextureDimension::Tex2D:
			srvDesc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
			srvDesc.Texture2D.MostDetailedMip = 0;
			srvDesc.Texture2D.MipLevels = format.MipLevels;
			srvDesc.Texture2D.PlaneSlice = 0;
			srvDesc.Texture2D.ResourceMinLODClamp = 0.0f;
			break;
		case TextureDimension::Tex1D:
			srvDesc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE1D;
			srvDesc.Texture1D.MipLevels = format.MipLevels;
			srvDesc.Texture1D.MostDetailedMip = 0;
			srvDesc.Texture1D.ResourceMinLODClamp = 0.0f;
			break;
		case TextureDimension::Tex2DArray:
			srvDesc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2DARRAY;
			srvDesc.Texture2DArray.MostDetailedMip = 0;
			srvDesc.Texture2DArray.MipLevels = format.MipLevels;
			srvDesc.Texture2DArray.PlaneSlice = 0;
			srvDesc.Texture2DArray.ResourceMinLODClamp = 0.0f;
			srvDesc.Texture2DArray.ArraySize = depth;
			srvDesc.Texture2DArray.FirstArraySlice = 0;
			break;
		case TextureDimension::Tex3D:
			srvDesc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE3D;
			srvDesc.Texture3D.MipLevels = format.MipLevels;
			srvDesc.Texture3D.MostDetailedMip = 0;
			srvDesc.Texture3D.ResourceMinLODClamp = 0.0f;
			break;
	}
	return srvDesc;
}
D3D12_UNORDERED_ACCESS_VIEW_DESC RenderTexture::GetColorUavDesc(uint targetMipLevel) const {
	D3D12_UNORDERED_ACCESS_VIEW_DESC uavDesc;
	auto desc = allocHandle.resource->GetDesc();
	uint maxLevel = desc.MipLevels - 1;
	targetMipLevel = std::min(targetMipLevel, maxLevel);
	uavDesc.Format = desc.Format;
	switch (dimension) {
		case TextureDimension::Tex2D:
			uavDesc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
			uavDesc.Texture2D.MipSlice = targetMipLevel;
			uavDesc.Texture2D.PlaneSlice = 0;
			break;
		case TextureDimension::Tex1D:
			uavDesc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE1D;
			uavDesc.Texture1D.MipSlice = targetMipLevel;
			break;
		case TextureDimension::Tex3D:
			uavDesc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE3D;
			uavDesc.Texture3D.FirstWSlice = 0;
			uavDesc.Texture3D.MipSlice = targetMipLevel;
			uavDesc.Texture3D.WSize = depth >> targetMipLevel;
			break;
		default:
			uavDesc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2DARRAY;
			uavDesc.Texture2DArray.ArraySize = depth;
			uavDesc.Texture2DArray.FirstArraySlice = 0;
			uavDesc.Texture2DArray.MipSlice = targetMipLevel;
			uavDesc.Texture2DArray.PlaneSlice = 0;
			break;
	}
	return uavDesc;
}
}// namespace toolhub::directx