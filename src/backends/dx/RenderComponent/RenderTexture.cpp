#include <RenderComponent/RenderTexture.h>
//#endif
#include <RenderComponent/RenderTexture.h>
#include <Singleton/Graphics.h>
#include <RenderComponent/TextureHeap.h>
#include <PipelineComponent/ThreadCommand.h>
#include <RenderComponent/Utility/ITextureAllocator.h>
void RenderTexture::ClearRenderTarget(ThreadCommand* commandList, uint slice, uint mip) const {
#ifndef NDEBUG
	if (dimension == TextureDimension::Tex3D) {
		throw "Tex3D can not be cleaned!";
	}
#endif
	commandList->ExecuteResBarrier();
	if (usage == RenderTextureUsage::ColorBuffer) {
		float colors[4] = {clearColor, clearColor, clearColor, clearColor};
		commandList->GetCmdList()->ClearRenderTargetView(rtvHeap.hCPU(slice * mipCount + mip), colors, 0, nullptr);
	} else
		commandList->GetCmdList()->ClearDepthStencilView(rtvHeap.hCPU(slice * mipCount + mip), D3D12_CLEAR_FLAG_DEPTH | D3D12_CLEAR_FLAG_STENCIL, clearColor, 0, 0, nullptr);
}
void RenderTexture::SetViewport(ThreadCommand* commandList, uint mipCount) const {
	uint currentWidth = mWidth >> mipCount;
	uint currentHeight = mHeight >> mipCount;
	D3D12_VIEWPORT Viewport({0.0f, 0.0f, (float)(currentWidth), (float)(currentHeight), 0.0f, 1.0f});
	D3D12_RECT ScissorRect({0, 0, (LONG)currentWidth, (LONG)currentHeight});
	commandList->GetCmdList()->RSSetViewports(1, &Viewport);
	commandList->GetCmdList()->RSSetScissorRects(1, &ScissorRect);
}
void RenderTexture::GetColorUAVDesc(D3D12_UNORDERED_ACCESS_VIEW_DESC& uavDesc, uint targetMipLevel) const {
	uint maxLevel = Resource->GetDesc().MipLevels - 1;
	targetMipLevel = Min(targetMipLevel, maxLevel);
	uavDesc.Format = Resource->GetDesc().Format;
	switch (dimension) {
		case TextureDimension::Tex2D:
			uavDesc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
			uavDesc.Texture2D.MipSlice = targetMipLevel;
			uavDesc.Texture2D.PlaneSlice = 0;
			break;
		case TextureDimension::Tex3D:
			uavDesc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE3D;
			uavDesc.Texture3D.FirstWSlice = 0;
			uavDesc.Texture3D.MipSlice = targetMipLevel;
			uavDesc.Texture3D.WSize = depthSlice >> targetMipLevel;
			break;
		default:
			uavDesc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2DARRAY;
			uavDesc.Texture2DArray.ArraySize = depthSlice;
			uavDesc.Texture2DArray.FirstArraySlice = 0;
			uavDesc.Texture2DArray.MipSlice = targetMipLevel;
			uavDesc.Texture2DArray.PlaneSlice = 0;
			break;
	}
}
D3D12_RESOURCE_STATES RenderTexture::GetGFXResourceState(GPUResourceState gfxState) const {
	switch (gfxState) {
		case GPUResourceState_RenderTarget:
			if (static_cast<bool>(usage)) {
				return D3D12_RESOURCE_STATE_DEPTH_WRITE;
			} else {
				return D3D12_RESOURCE_STATE_RENDER_TARGET;
			}
		case GPUResourceState_NonPixelShaderRes:
			if (static_cast<bool>(usage)) {
				return D3D12_RESOURCE_STATE_DEPTH_READ;
			} else {
				return D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
			}
		case GPUResourceState_GenericRead:
			if (static_cast<bool>(usage)) {
				return D3D12_RESOURCE_STATE_DEPTH_READ;
			} else {
				return D3D12_RESOURCE_STATE_GENERIC_READ;
			}
			break;
		default:
			return (D3D12_RESOURCE_STATES)gfxState;
	}
}
void RenderTexture::GetColorViewDesc(D3D12_SHADER_RESOURCE_VIEW_DESC& srvDesc) const {
	srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
	auto format = Resource->GetDesc();
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
		case TextureDimension::Tex2DArray:
			srvDesc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2DARRAY;
			srvDesc.Texture2DArray.MostDetailedMip = 0;
			srvDesc.Texture2DArray.MipLevels = format.MipLevels;
			srvDesc.Texture2DArray.PlaneSlice = 0;
			srvDesc.Texture2DArray.ResourceMinLODClamp = 0.0f;
			srvDesc.Texture2DArray.ArraySize = depthSlice;
			srvDesc.Texture2DArray.FirstArraySlice = 0;
			break;
		case TextureDimension::Tex3D:
			srvDesc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE3D;
			srvDesc.Texture3D.MipLevels = format.MipLevels;
			srvDesc.Texture3D.MostDetailedMip = 0;
			srvDesc.Texture3D.ResourceMinLODClamp = 0.0f;
			break;
	}
}
D3D12_CPU_DESCRIPTOR_HANDLE RenderTexture::GetColorDescriptor(uint slice, uint mip) const {
#ifndef NDEBUG
	if (dimension == TextureDimension::Tex3D) {
		throw "Tex3D have no rtv heap!";
	}
#endif
	return rtvHeap.hCPU(slice * mipCount + mip);
}
uint64_t RenderTexture::GetSizeFromProperty(
	GFXDevice* device,
	uint width,
	uint height,
	RenderTextureFormat rtFormat,
	TextureDimension type,
	uint depthCount,
	uint mipCount,
	RenderTextureState initState) {
	mipCount = Max<uint>(mipCount, 1);
	uint arraySize;
	switch (type) {
		case TextureDimension::Cubemap:
			arraySize = 6;
			break;
		case TextureDimension::Tex2D:
			arraySize = 1;
			break;
		default:
			arraySize = Max<uint>(1, depthCount);
			break;
	}
	if (rtFormat.usage == RenderTextureUsage::ColorBuffer) {
		D3D12_RESOURCE_DESC texDesc;
		ZeroMemory(&texDesc, sizeof(D3D12_RESOURCE_DESC));
		texDesc.Dimension = (type == TextureDimension::Tex3D) ? D3D12_RESOURCE_DIMENSION_TEXTURE3D : D3D12_RESOURCE_DIMENSION_TEXTURE2D;
		texDesc.Alignment = 0;
		texDesc.Width = width;
		texDesc.Height = height;
		texDesc.DepthOrArraySize = arraySize;
		texDesc.MipLevels = mipCount;
		texDesc.Format = (DXGI_FORMAT)rtFormat.colorFormat;
		texDesc.SampleDesc.Count = 1;
		texDesc.SampleDesc.Quality = 0;
		texDesc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
		texDesc.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET | D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
		auto allocateInfo = device->device()->GetResourceAllocationInfo(
			0, 1, &texDesc);
		return allocateInfo.SizeInBytes;
	} else {
		GFXFormat mDepthFormat;
		GFXFormat mFormat;
		switch (rtFormat.depthFormat) {
			case RenderTextureDepthSettings_Depth32:
				mFormat = GFXFormat_D32_Float;
				mDepthFormat = GFXFormat_D32_Float;
				break;
			case RenderTextureDepthSettings_Depth16:
				mFormat = GFXFormat_D16_UNorm;
				mDepthFormat = GFXFormat_D16_UNorm;
				break;
			case RenderTextureDepthSettings_DepthStencil:
				mFormat = GFXFormat_R24G8_Typeless;
				mDepthFormat = GFXFormat_D24_UNorm_S8_UInt;
				break;
			default:
				mFormat = GFXFormat_Unknown;
				mDepthFormat = GFXFormat_Unknown;
				break;
		}
		if (mFormat != GFXFormat_Unknown) {
			D3D12_RESOURCE_DESC depthStencilDesc;
			depthStencilDesc.Dimension = (type == TextureDimension::Tex3D) ? D3D12_RESOURCE_DIMENSION_TEXTURE3D : D3D12_RESOURCE_DIMENSION_TEXTURE2D;
			depthStencilDesc.Alignment = 0;
			depthStencilDesc.Width = width;
			depthStencilDesc.Height = height;
			depthStencilDesc.DepthOrArraySize = arraySize;
			depthStencilDesc.MipLevels = 1;
			depthStencilDesc.Format = (DXGI_FORMAT)mFormat;
			mFormat = mDepthFormat;
			depthStencilDesc.SampleDesc.Count = 1;
			depthStencilDesc.SampleDesc.Quality = 0;
			depthStencilDesc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
			depthStencilDesc.Flags = D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;
			return device->device()->GetResourceAllocationInfo(
							 0, 1, &depthStencilDesc)
				.SizeInBytes;
		}
	}
	return 0;
}
RenderTexture::RenderTexture(
	GFXDevice* device,
	ITextureAllocator* allocator,
	uint width,
	uint height,
	RenderTextureFormat rtFormat,
	TextureDimension type,
	uint depthCount,
	uint mipCount,
	RenderTextureState initState,
	float clearColor)
	: TextureBase(),
	  usage(rtFormat.usage),
	  clearColor(clearColor) {
	mipCount = Max<uint>(mipCount, 1);
	dimension = type;
	mWidth = width;
	mHeight = height;
	uint arraySize;
	switch (type) {
		case TextureDimension::Cubemap:
			arraySize = 6;
			break;
		case TextureDimension::Tex2D:
			arraySize = 1;
			break;
		default:
			arraySize = Max<uint>(1, depthCount);
			break;
	}
	if (rtFormat.usage == RenderTextureUsage::ColorBuffer) {
		mFormat = rtFormat.colorFormat;
		this->mipCount = mipCount;
		depthSlice = arraySize;
		D3D12_RESOURCE_DESC texDesc;
		ZeroMemory(&texDesc, sizeof(D3D12_RESOURCE_DESC));
		texDesc.Dimension = (type == TextureDimension::Tex3D) ? D3D12_RESOURCE_DIMENSION_TEXTURE3D : D3D12_RESOURCE_DIMENSION_TEXTURE2D;
		texDesc.Alignment = 0;
		texDesc.Width = mWidth;
		texDesc.Height = mHeight;
		texDesc.DepthOrArraySize = arraySize;
		texDesc.MipLevels = mipCount;
		texDesc.Format = (DXGI_FORMAT)mFormat;
		texDesc.SampleDesc.Count = 1;
		texDesc.SampleDesc.Quality = 0;
		texDesc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
		texDesc.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET | D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
		D3D12_CLEAR_VALUE clearValue;
		clearValue.Format = (DXGI_FORMAT)mFormat;
		clearValue.Color[0] = clearColor;
		clearValue.Color[1] = clearColor;
		clearValue.Color[2] = clearColor;
		clearValue.Color[3] = clearColor;
		switch (initState) {
			case RenderTextureState::Unordered_Access:
				this->initState = GPUResourceState_UnorderedAccess;
				break;
			case RenderTextureState::Render_Target:
				this->initState = GPUResourceState_RenderTarget;
				break;
			case RenderTextureState::Common:
				this->initState = GPUResourceState_Common;
				break;
			case RenderTextureState::Generic_Read:
				this->initState = GPUResourceState_GenericRead;
				break;
			case RenderTextureState::Non_Pixel_SRV:
				this->initState = GPUResourceState_NonPixelShaderRes;
				break;
			default:
				if (type == TextureDimension::Tex3D)
					this->initState = GPUResourceState_UnorderedAccess;
				else
					this->initState = GPUResourceState_RenderTarget;
				break;
		}
		resourceSize = device->device()->GetResourceAllocationInfo(
								 0, 1, &texDesc)
						   .SizeInBytes;
		if (!allocator) {
			auto prop = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT);
			ThrowIfFailed(device->device()->CreateCommittedResource(
				&prop,
				D3D12_HEAP_FLAG_NONE,
				&texDesc,
				GetGFXResourceState(this->initState),
				&clearValue,
				IID_PPV_ARGS(&Resource)));
		} else {
			ID3D12Heap* heap;
			uint64 offset;
			allocator->AllocateTextureHeap(
				device,
				mFormat,
				width,
				height,
				arraySize,
				type,
				mipCount,
				&heap,
				&offset,
				true,
				this);
			ThrowIfFailed(device->device()->CreatePlacedResource(
				heap,
				offset,
				&texDesc,
				GetGFXResourceState(this->initState),
				&clearValue,
				IID_PPV_ARGS(&Resource)));
		}
		if (type != TextureDimension::Tex3D)
			rtvHeap.Create(device, D3D12_DESCRIPTOR_HEAP_TYPE_RTV, arraySize * mipCount, false);
		D3D12_RENDER_TARGET_VIEW_DESC rtvDesc;
		switch (type) {
			case TextureDimension::Tex2D:
				rtvDesc.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2D;
				rtvDesc.Format = (DXGI_FORMAT)mFormat;
				rtvDesc.Texture2D.PlaneSlice = 0;
				for (uint i = 0; i < mipCount; ++i) {
					rtvDesc.Texture2D.MipSlice = i;
					rtvHeap.CreateRTV(device, this, 0, i, &rtvDesc, i);
				}
				break;
			case TextureDimension::Tex3D:
				break;
			default:
				rtvDesc.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2DARRAY;
				rtvDesc.Format = (DXGI_FORMAT)mFormat;
				rtvDesc.Texture2DArray.PlaneSlice = 0;
				rtvDesc.Texture2DArray.ArraySize = 1;
				for (int32_t i = 0, count = 0; i < arraySize; ++i) {
					for (uint j = 0; j < mipCount; ++j) {
						// Render target to ith element.
						rtvDesc.Texture2DArray.FirstArraySlice = i;
						rtvDesc.Texture2DArray.MipSlice = j;
						// Create RTV to ith cubemap face.
						rtvHeap.CreateRTV(device, this, i, j, &rtvDesc, count);
						count++;
					}
				}
				break;
		}
		uavDescIndices.resize(mipCount);
		for (uint i = 0; i < mipCount; ++i) {
			uavDescIndices[i] = Graphics::GetDescHeapIndexFromPool();
			BindUAVToHeap(Graphics::GetGlobalDescHeapNonConst(), uavDescIndices[i], device, i);
		}
	} else {
		GFXFormat mDepthFormat;
		switch (rtFormat.depthFormat) {
			case RenderTextureDepthSettings_Depth32:
				mFormat = GFXFormat_D32_Float;
				mDepthFormat = GFXFormat_D32_Float;
				break;
			case RenderTextureDepthSettings_Depth16:
				mFormat = GFXFormat_D16_UNorm;
				mDepthFormat = GFXFormat_D16_UNorm;
				break;
			case RenderTextureDepthSettings_DepthStencil:
				mFormat = GFXFormat_R24G8_Typeless;
				mDepthFormat = GFXFormat_D24_UNorm_S8_UInt;
				break;
			default:
				mFormat = GFXFormat_Unknown;
				mDepthFormat = GFXFormat_Unknown;
				break;
		}
		if (mFormat != GFXFormat_Unknown) {
			rtvHeap.Create(device, D3D12_DESCRIPTOR_HEAP_TYPE_DSV, arraySize * mipCount, false);
			D3D12_RESOURCE_DESC depthStencilDesc;
			depthStencilDesc.Dimension = (type == TextureDimension::Tex3D) ? D3D12_RESOURCE_DIMENSION_TEXTURE3D : D3D12_RESOURCE_DIMENSION_TEXTURE2D;
			depthStencilDesc.Alignment = 0;
			depthStencilDesc.Width = width;
			depthStencilDesc.Height = height;
			depthStencilDesc.DepthOrArraySize = arraySize;
			depthStencilDesc.MipLevels = 1;
			depthStencilDesc.Format = (DXGI_FORMAT)mFormat;
			mFormat = mDepthFormat;
			depthStencilDesc.SampleDesc.Count = 1;
			depthStencilDesc.SampleDesc.Quality = 0;
			depthStencilDesc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
			depthStencilDesc.Flags = D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;
			D3D12_CLEAR_VALUE depthClearValue;
			depthClearValue.Format = (DXGI_FORMAT)mFormat;
			depthClearValue.DepthStencil.Depth = clearColor;
			depthClearValue.DepthStencil.Stencil = 0;
			switch (initState) {
				case RenderTextureState::Unordered_Access:
					this->initState = GPUResourceState_UnorderedAccess;
					break;
				case RenderTextureState::Generic_Read:
					this->initState = GPUResourceState_GenericRead;
					break;
				case RenderTextureState::Common:
					this->initState = GPUResourceState_Common;
					break;
				case RenderTextureState::Non_Pixel_SRV:
					this->initState = GPUResourceState_NonPixelShaderRes;
					break;
				default:
					this->initState = GPUResourceState_RenderTarget;
					break;
			}
			resourceSize = device->device()->GetResourceAllocationInfo(
									 0, 1, &depthStencilDesc)
							   .SizeInBytes;
			if (!allocator) {
				auto heap = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT);
				ThrowIfFailed(device->device()->CreateCommittedResource(
					&heap,
					D3D12_HEAP_FLAG_NONE,
					&depthStencilDesc,
					GetGFXResourceState(this->initState),
					&depthClearValue,
					IID_PPV_ARGS(&Resource)));
			} else {
				ID3D12Heap* heap;
				uint64 offset;
				allocator->AllocateTextureHeap(
					device,
					mFormat,
					width,
					height,
					arraySize,
					type,
					mipCount,
					&heap,
					&offset,
					true,
					this);
				ThrowIfFailed(device->device()->CreatePlacedResource(
					heap,
					offset,
					&depthStencilDesc,
					GetGFXResourceState(this->initState),
					&depthClearValue,
					IID_PPV_ARGS(&Resource)));
			}
			D3D12_DEPTH_STENCIL_VIEW_DESC dsvDesc;
			switch (type) {
				case TextureDimension::Tex2D:
					dsvDesc.Flags = D3D12_DSV_FLAG_NONE;
					dsvDesc.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2D;
					dsvDesc.Format = (DXGI_FORMAT)mDepthFormat;
					for (uint i = 0; i < mipCount; ++i) {
						dsvDesc.Texture2D.MipSlice = i;
						rtvHeap.CreateDSV(device, this, 0, i, &dsvDesc, i);
					}
					break;
				default:
					dsvDesc.Flags = D3D12_DSV_FLAG_NONE;
					dsvDesc.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2DARRAY;
					dsvDesc.Texture2DArray.ArraySize = 1;
					for (int32_t i = 0, count = 0; i < arraySize; ++i) {
						for (uint j = 0; j < mipCount; ++j) {
							dsvDesc.Texture2DArray.MipSlice = j;
							dsvDesc.Texture2DArray.FirstArraySlice = i;
							rtvHeap.CreateDSV(device, this, i, j, &dsvDesc, count);
							count++;
						}
					}
					break;
			}
		}
	}
	BindSRVToHeap(Graphics::GetGlobalDescHeapNonConst(), GetGlobalDescIndex(), device);
}
RenderTexture::RenderTexture(
	GFXDevice* device,
	uint width,
	uint height,
	RenderTextureFormat rtFormat,
	TextureDimension type,
	uint depthCount,
	uint mipCount,
	RenderTextureState initState,
	TextureHeap* targetHeap,
	uint64_t placedOffset,
	float clearColor)
	: TextureBase(),
	  usage(rtFormat.usage),
	  clearColor(clearColor) {

	mipCount = Max<uint>(mipCount, 1);
	dimension = type;
	mWidth = width;
	mHeight = height;
	uint arraySize;
	switch (type) {
		case TextureDimension::Cubemap:
			arraySize = 6;
			break;
		case TextureDimension::Tex2D:
			arraySize = 1;
			break;
		default:
			arraySize = Max<uint>(1, depthCount);
			break;
	}
	if (rtFormat.usage == RenderTextureUsage::ColorBuffer) {
		mFormat = rtFormat.colorFormat;
		this->mipCount = mipCount;
		depthSlice = arraySize;
		D3D12_RESOURCE_DESC texDesc;
		ZeroMemory(&texDesc, sizeof(D3D12_RESOURCE_DESC));
		texDesc.Dimension = (type == TextureDimension::Tex3D) ? D3D12_RESOURCE_DIMENSION_TEXTURE3D : D3D12_RESOURCE_DIMENSION_TEXTURE2D;
		texDesc.Alignment = 0;
		texDesc.Width = mWidth;
		texDesc.Height = mHeight;
		texDesc.DepthOrArraySize = arraySize;
		texDesc.MipLevels = mipCount;
		texDesc.Format = (DXGI_FORMAT)mFormat;
		texDesc.SampleDesc.Count = 1;
		texDesc.SampleDesc.Quality = 0;
		texDesc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
		texDesc.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET | D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
		D3D12_CLEAR_VALUE clearValue;
		clearValue.Format = (DXGI_FORMAT)mFormat;
		clearValue.Color[0] = clearColor;
		clearValue.Color[1] = clearColor;
		clearValue.Color[2] = clearColor;
		clearValue.Color[3] = clearColor;
		switch (initState) {
			case RenderTextureState::Unordered_Access:
				this->initState = GPUResourceState_UnorderedAccess;
				break;
			case RenderTextureState::Render_Target:
				this->initState = GPUResourceState_RenderTarget;
				break;
			case RenderTextureState::Common:
				this->initState = GPUResourceState_Common;
				break;
			case RenderTextureState::Generic_Read:
				this->initState = GPUResourceState_GenericRead;
				break;
			case RenderTextureState::Non_Pixel_SRV:
				this->initState = GPUResourceState_NonPixelShaderRes;
				break;
			default:
				if (type == TextureDimension::Tex3D)
					this->initState = GPUResourceState_UnorderedAccess;
				else
					this->initState = GPUResourceState_RenderTarget;
				break;
		}
		resourceSize = device->device()->GetResourceAllocationInfo(
								 0, 1, &texDesc)
						   .SizeInBytes;
		if (!targetHeap) {
			auto prop = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT);
			ThrowIfFailed(device->device()->CreateCommittedResource(
				&prop,
				D3D12_HEAP_FLAG_NONE,
				&texDesc,
				GetGFXResourceState(this->initState),
				&clearValue,
				IID_PPV_ARGS(&Resource)));
		} else {
			ThrowIfFailed(device->device()->CreatePlacedResource(
				targetHeap->GetHeap(),
				placedOffset,
				&texDesc,
				GetGFXResourceState(this->initState),
				&clearValue,
				IID_PPV_ARGS(&Resource)));
		}
		if (type != TextureDimension::Tex3D)
			rtvHeap.Create(device, D3D12_DESCRIPTOR_HEAP_TYPE_RTV, arraySize * mipCount, false);
		D3D12_RENDER_TARGET_VIEW_DESC rtvDesc;
		switch (type) {
			case TextureDimension::Tex2D:
				rtvDesc.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2D;
				rtvDesc.Format = (DXGI_FORMAT)mFormat;
				rtvDesc.Texture2D.PlaneSlice = 0;
				for (uint i = 0; i < mipCount; ++i) {
					rtvDesc.Texture2D.MipSlice = i;
					rtvHeap.CreateRTV(device, this, 0, i, &rtvDesc, i);
				}
				break;
			case TextureDimension::Tex3D:
				break;
			default:
				rtvDesc.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2DARRAY;
				rtvDesc.Format = (DXGI_FORMAT)mFormat;
				rtvDesc.Texture2DArray.PlaneSlice = 0;
				rtvDesc.Texture2DArray.ArraySize = 1;
				for (int32_t i = 0, count = 0; i < arraySize; ++i) {
					for (uint j = 0; j < mipCount; ++j) {
						// Render target to ith element.
						rtvDesc.Texture2DArray.FirstArraySlice = i;
						rtvDesc.Texture2DArray.MipSlice = j;
						// Create RTV to ith cubemap face.
						rtvHeap.CreateRTV(device, this, i, j, &rtvDesc, count);
						count++;
					}
				}
				break;
		}
		uavDescIndices.resize(mipCount);
		for (uint i = 0; i < mipCount; ++i) {
			uavDescIndices[i] = Graphics::GetDescHeapIndexFromPool();
			BindUAVToHeap(Graphics::GetGlobalDescHeapNonConst(), uavDescIndices[i], device, i);
		}
	} else {
		GFXFormat mDepthFormat;
		switch (rtFormat.depthFormat) {
			case RenderTextureDepthSettings_Depth32:
				mFormat = GFXFormat_D32_Float;
				mDepthFormat = GFXFormat_D32_Float;
				break;
			case RenderTextureDepthSettings_Depth16:
				mFormat = GFXFormat_D16_UNorm;
				mDepthFormat = GFXFormat_D16_UNorm;
				break;
			case RenderTextureDepthSettings_DepthStencil:
				mFormat = GFXFormat_R24G8_Typeless;
				mDepthFormat = GFXFormat_D24_UNorm_S8_UInt;
				break;
			default:
				mFormat = GFXFormat_Unknown;
				mDepthFormat = GFXFormat_Unknown;
				break;
		}
		if (mFormat != GFXFormat_Unknown) {
			rtvHeap.Create(device, D3D12_DESCRIPTOR_HEAP_TYPE_DSV, arraySize * mipCount, false);
			D3D12_RESOURCE_DESC depthStencilDesc;
			depthStencilDesc.Dimension = (type == TextureDimension::Tex3D) ? D3D12_RESOURCE_DIMENSION_TEXTURE3D : D3D12_RESOURCE_DIMENSION_TEXTURE2D;
			depthStencilDesc.Alignment = 0;
			depthStencilDesc.Width = width;
			depthStencilDesc.Height = height;
			depthStencilDesc.DepthOrArraySize = arraySize;
			depthStencilDesc.MipLevels = 1;
			depthStencilDesc.Format = (DXGI_FORMAT)mFormat;
			mFormat = mDepthFormat;
			depthStencilDesc.SampleDesc.Count = 1;
			depthStencilDesc.SampleDesc.Quality = 0;
			depthStencilDesc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
			depthStencilDesc.Flags = D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;
			D3D12_CLEAR_VALUE depthClearValue;
			depthClearValue.Format = (DXGI_FORMAT)mFormat;
			depthClearValue.DepthStencil.Depth = clearColor;
			depthClearValue.DepthStencil.Stencil = 0;
			switch (initState) {
				case RenderTextureState::Unordered_Access:
					this->initState = GPUResourceState_UnorderedAccess;
					break;
				case RenderTextureState::Generic_Read:
					this->initState = GPUResourceState_GenericRead;
					break;
				case RenderTextureState::Common:
					this->initState = GPUResourceState_Common;
					break;
				case RenderTextureState::Non_Pixel_SRV:
					this->initState = GPUResourceState_NonPixelShaderRes;
					break;
				default:
					this->initState = GPUResourceState_RenderTarget;
					break;
			}
			resourceSize = device->device()->GetResourceAllocationInfo(
									 0, 1, &depthStencilDesc)
							   .SizeInBytes;
			if (!targetHeap) {
				auto heap = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT);
				ThrowIfFailed(device->device()->CreateCommittedResource(
					&heap,
					D3D12_HEAP_FLAG_NONE,
					&depthStencilDesc,
					GetGFXResourceState(this->initState),
					&depthClearValue,
					IID_PPV_ARGS(&Resource)));
			} else {
				ThrowIfFailed(device->device()->CreatePlacedResource(
					targetHeap->GetHeap(),
					placedOffset,
					&depthStencilDesc,
					GetGFXResourceState(this->initState),
					&depthClearValue,
					IID_PPV_ARGS(&Resource)));
			}
			D3D12_DEPTH_STENCIL_VIEW_DESC dsvDesc;
			switch (type) {
				case TextureDimension::Tex2D:
					dsvDesc.Flags = D3D12_DSV_FLAG_NONE;
					dsvDesc.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2D;
					dsvDesc.Format = (DXGI_FORMAT)mDepthFormat;
					for (uint i = 0; i < mipCount; ++i) {
						dsvDesc.Texture2D.MipSlice = i;
						rtvHeap.CreateDSV(device, this, 0, i, &dsvDesc, i);
					}
					break;
				default:
					dsvDesc.Flags = D3D12_DSV_FLAG_NONE;
					dsvDesc.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2DARRAY;
					dsvDesc.Texture2DArray.ArraySize = 1;
					for (int32_t i = 0, count = 0; i < arraySize; ++i) {
						for (uint j = 0; j < mipCount; ++j) {
							dsvDesc.Texture2DArray.MipSlice = j;
							dsvDesc.Texture2DArray.FirstArraySlice = i;
							rtvHeap.CreateDSV(device, this, i, j, &dsvDesc, count);
							count++;
						}
					}
					break;
			}
		}
	}
	BindSRVToHeap(Graphics::GetGlobalDescHeapNonConst(), GetGlobalDescIndex(), device);
}
void RenderTexture::BindRTVToHeap(DescriptorHeap* targetHeap, uint index, GFXDevice* device, uint slice, uint mip) const {
	if (usage == RenderTextureUsage::ColorBuffer) {
		D3D12_RENDER_TARGET_VIEW_DESC rtvDesc;
		switch (dimension) {
			case TextureDimension::Tex2D:
				rtvDesc.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2D;
				rtvDesc.Format = (DXGI_FORMAT)mFormat;
				rtvDesc.Texture2D.MipSlice = mip;
				rtvDesc.Texture2D.PlaneSlice = 0;
				targetHeap->CreateRTV(device, this, 0, mip, &rtvDesc, index * mipCount + mip);
				break;
			case TextureDimension::Tex3D:
				break;
			default:
				rtvDesc.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2DARRAY;
				rtvDesc.Format = (DXGI_FORMAT)mFormat;
				rtvDesc.Texture2DArray.MipSlice = mip;
				rtvDesc.Texture2DArray.PlaneSlice = 0;
				rtvDesc.Texture2DArray.ArraySize = 1;
				rtvDesc.Texture2DArray.FirstArraySlice = slice;
				targetHeap->CreateRTV(device, this, slice, mip, &rtvDesc, index * mipCount + mip);
				break;
		}
	} else {
		D3D12_DEPTH_STENCIL_VIEW_DESC dsvDesc;
		switch (dimension) {
			case TextureDimension::Tex2D:
				dsvDesc.Flags = D3D12_DSV_FLAG_NONE;
				dsvDesc.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2D;
				dsvDesc.Format = (DXGI_FORMAT)mFormat;
				dsvDesc.Texture2D.MipSlice = 0;
				break;
			default:
				dsvDesc.Flags = D3D12_DSV_FLAG_NONE;
				dsvDesc.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2DARRAY;
				dsvDesc.Texture2DArray.ArraySize = 1;
				dsvDesc.Texture2DArray.MipSlice = 0;
				dsvDesc.Texture2DArray.FirstArraySlice = slice;
				break;
		}
		targetHeap->CreateDSV(device, this, 0, 0, &dsvDesc, index);
	}
}
void RenderTexture::BindSRVToHeap(DescriptorHeap* targetHeap, uint index, GFXDevice* device) const {
	D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
	GetColorViewDesc(srvDesc);
	targetHeap->CreateSRV(device, this, &srvDesc, index);
}
void RenderTexture::BindUAVToHeap(DescriptorHeap* targetHeap, uint index, GFXDevice* device, uint targetMipLevel) const {
	D3D12_UNORDERED_ACCESS_VIEW_DESC uavDesc = {};
	GetColorUAVDesc(uavDesc, targetMipLevel);
	targetHeap->CreateUAV(device, this, &uavDesc, index);
}
RenderTexture::~RenderTexture() {
	for (auto ite = uavDescIndices.begin(); ite != uavDescIndices.end(); ++ite) {
		Graphics::ReturnDescHeapIndexToPool(*ite);
	}
}
uint RenderTexture::GetGlobalUAVDescIndex(uint mipLevel) const {
	if (usage == RenderTextureUsage::ColorBuffer) {
		mipLevel = Min<uint>(mipLevel, mipCount - 1);
		return uavDescIndices[mipLevel];
	} else {
#if defined(DEBUG) || defined(_DEBUG)
		throw "Depth UAV Not Allowed!";
#else
		return 0;
#endif
	}
}
