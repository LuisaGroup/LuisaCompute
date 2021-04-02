#include "GFXUtil.h"
#include <comdef.h>
#include <fstream>
#include "Camera.h"
#include "../PipelineComponent/IPipelineResource.h"
#pragma comment(lib, "d3dcompiler.lib")
#pragma comment(lib, "D3D12.lib")
using Microsoft::WRL::ComPtr;
std::array<const CD3DX12_STATIC_SAMPLER_DESC, GFXUtil::STATIC_SAMPLER_COUNT> const& GFXUtil::GetStaticSamplers() {
	// Applications usually only need a handful of samplers.  So just define them all up front
	// and keep them available as part of the root signature.
	static const CD3DX12_STATIC_SAMPLER_DESC pointWrap(
		0,								 // shaderRegister
		D3D12_FILTER_MIN_MAG_MIP_POINT,	 // filter
		D3D12_TEXTURE_ADDRESS_MODE_WRAP, // addressU
		D3D12_TEXTURE_ADDRESS_MODE_WRAP, // addressV
		D3D12_TEXTURE_ADDRESS_MODE_WRAP);// addressW
	static const CD3DX12_STATIC_SAMPLER_DESC pointClamp(
		1,								  // shaderRegister
		D3D12_FILTER_MIN_MAG_MIP_POINT,	  // filter
		D3D12_TEXTURE_ADDRESS_MODE_CLAMP, // addressU
		D3D12_TEXTURE_ADDRESS_MODE_CLAMP, // addressV
		D3D12_TEXTURE_ADDRESS_MODE_CLAMP);// addressW
	static const CD3DX12_STATIC_SAMPLER_DESC bilinearWrap(
		2,									  // shaderRegister
		D3D12_FILTER_MIN_MAG_LINEAR_MIP_POINT,// filter
		D3D12_TEXTURE_ADDRESS_MODE_WRAP,	  // addressU
		D3D12_TEXTURE_ADDRESS_MODE_WRAP,	  // addressV
		D3D12_TEXTURE_ADDRESS_MODE_WRAP);	  // addressW
	static const CD3DX12_STATIC_SAMPLER_DESC bilinearClamp(
		3,									  // shaderRegister
		D3D12_FILTER_MIN_MAG_LINEAR_MIP_POINT,// filter
		D3D12_TEXTURE_ADDRESS_MODE_CLAMP,	  // addressU
		D3D12_TEXTURE_ADDRESS_MODE_CLAMP,	  // addressV
		D3D12_TEXTURE_ADDRESS_MODE_CLAMP);	  // addressW
	static const CD3DX12_STATIC_SAMPLER_DESC trilinearWrap(
		4,								 // shaderRegister
		D3D12_FILTER_MIN_MAG_MIP_LINEAR, // filter
		D3D12_TEXTURE_ADDRESS_MODE_WRAP, // addressU
		D3D12_TEXTURE_ADDRESS_MODE_WRAP, // addressV
		D3D12_TEXTURE_ADDRESS_MODE_WRAP);// addressW
	static const CD3DX12_STATIC_SAMPLER_DESC trilinearClamp(
		5,								  // shaderRegister
		D3D12_FILTER_MIN_MAG_MIP_LINEAR,  // filter
		D3D12_TEXTURE_ADDRESS_MODE_CLAMP, // addressU
		D3D12_TEXTURE_ADDRESS_MODE_CLAMP, // addressV
		D3D12_TEXTURE_ADDRESS_MODE_CLAMP);// addressW
	static const CD3DX12_STATIC_SAMPLER_DESC anisotropicWrap(
		6,								// shaderRegister
		D3D12_FILTER_ANISOTROPIC,		// filter
		D3D12_TEXTURE_ADDRESS_MODE_WRAP,// addressU
		D3D12_TEXTURE_ADDRESS_MODE_WRAP,// addressV
		D3D12_TEXTURE_ADDRESS_MODE_WRAP,// addressW
		0.0f,							// mipLODBias
		16);							// maxAnisotropy
	static const CD3DX12_STATIC_SAMPLER_DESC anisotropicClamp(
		7,								 // shaderRegister
		D3D12_FILTER_ANISOTROPIC,		 // filter
		D3D12_TEXTURE_ADDRESS_MODE_CLAMP,// addressU
		D3D12_TEXTURE_ADDRESS_MODE_CLAMP,// addressV
		D3D12_TEXTURE_ADDRESS_MODE_CLAMP,// addressW
		0.0f,							 // mipLODBias
		16);							 // maxAnisotropy
	static const CD3DX12_STATIC_SAMPLER_DESC linearShadowClamp(
		8,
		D3D12_FILTER_COMPARISON_MIN_MAG_LINEAR_MIP_POINT,
		D3D12_TEXTURE_ADDRESS_MODE_CLAMP,// addressU
		D3D12_TEXTURE_ADDRESS_MODE_CLAMP,// addressV
		D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
		0,
		16,
		D3D12_COMPARISON_FUNC_GREATER);// addressW
	static const CD3DX12_STATIC_SAMPLER_DESC linearCubemapShadowClamp(
		9,
		D3D12_FILTER_COMPARISON_MIN_MAG_LINEAR_MIP_POINT,
		D3D12_TEXTURE_ADDRESS_MODE_WRAP,// addressU
		D3D12_TEXTURE_ADDRESS_MODE_WRAP,// addressV
		D3D12_TEXTURE_ADDRESS_MODE_WRAP,
		0,
		16,
		D3D12_COMPARISON_FUNC_LESS);// addressW
	static const CD3DX12_STATIC_SAMPLER_DESC mipLinearPointClamp(
		10,
		D3D12_FILTER_MIN_MAG_POINT_MIP_LINEAR,
		D3D12_TEXTURE_ADDRESS_MODE_CLAMP,// addressU
		D3D12_TEXTURE_ADDRESS_MODE_CLAMP,// addressV
		D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
		0,
		16);
	static const CD3DX12_STATIC_SAMPLER_DESC mipLinearPointWrap(
		11,
		D3D12_FILTER_MIN_MAG_POINT_MIP_LINEAR,
		D3D12_TEXTURE_ADDRESS_MODE_WRAP,// addressU
		D3D12_TEXTURE_ADDRESS_MODE_WRAP,// addressV
		D3D12_TEXTURE_ADDRESS_MODE_WRAP,
		0,
		16);
	static const CD3DX12_STATIC_SAMPLER_DESC pointShadowClamp(
		12,
		D3D12_FILTER_COMPARISON_MIN_MAG_MIP_POINT,
		D3D12_TEXTURE_ADDRESS_MODE_CLAMP,// addressU
		D3D12_TEXTURE_ADDRESS_MODE_CLAMP,// addressV
		D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
		0,
		16,
		D3D12_COMPARISON_FUNC_GREATER);// addressW
	static const std::array<const CD3DX12_STATIC_SAMPLER_DESC, GFXUtil::STATIC_SAMPLER_COUNT> arr = {
		pointWrap,
		pointClamp,
		bilinearWrap,
		bilinearClamp,
		trilinearWrap,
		trilinearClamp,
		anisotropicWrap,
		anisotropicClamp,
		linearShadowClamp,
		linearCubemapShadowClamp,
		mipLinearPointClamp,
		mipLinearPointWrap,
		pointShadowClamp};
	return arr;
}
void MemcpySubresource(
	_In_ const D3D12_MEMCPY_DEST* pDest,
	_In_ const D3D12_SUBRESOURCE_DATA* pSrc,
	SIZE_T RowSizeInBytes,
	uint NumRows,
	uint NumSlices) noexcept {
	for (uint z = 0; z < NumSlices; ++z) {
		auto pDestSlice = static_cast<BYTE*>(pDest->pData) + pDest->SlicePitch * z;
		auto pSrcSlice = static_cast<const BYTE*>(pSrc->pData) + pSrc->SlicePitch * LONG_PTR(z);
		for (uint y = 0; y < NumRows; ++y) {
			memcpy(pDestSlice + pDest->RowPitch * y,
				   pSrcSlice + pSrc->RowPitch * LONG_PTR(y),
				   RowSizeInBytes);
		}
	}
}
//------------------------------------------------------------------------------------------------
// Row-by-row memcpy
void MemcpySubresource(
	_In_ const D3D12_MEMCPY_DEST* pDest,
	_In_ const void* pResourceData,
	_In_ const D3D12_SUBRESOURCE_INFO* pSrc,
	SIZE_T RowSizeInBytes,
	uint NumRows,
	uint NumSlices) {
	for (uint z = 0; z < NumSlices; ++z) {
		auto pDestSlice = reinterpret_cast<BYTE*>(pDest->pData) + pDest->SlicePitch * z;
		auto pSrcSlice = (reinterpret_cast<const BYTE*>(pResourceData) + pSrc->Offset) + pSrc->DepthPitch * LONG_PTR(z);
		for (uint y = 0; y < NumRows; ++y) {
			memcpy(pDestSlice + pDest->RowPitch * y,
				   pSrcSlice + pSrc->RowPitch * LONG_PTR(y),
				   RowSizeInBytes);
		}
	}
}
//------------------------------------------------------------------------------------------------
// Returns required size of a buffer to be used for data upload
UINT64 GetRequiredIntermediateSize(
	_In_ GFXResource* pDestinationResource,
	_In_range_(0, D3D12_REQ_SUBRESOURCES) uint FirstSubresource,
	_In_range_(0, D3D12_REQ_SUBRESOURCES - FirstSubresource) uint NumSubresources) noexcept {
	auto Desc = pDestinationResource->GetDesc();
	UINT64 RequiredSize = 0;
	GFXDevice* pDevice = nullptr;
	pDestinationResource->GetDevice(IID_PPV_ARGS(&pDevice));
	pDevice->GetCopyableFootprints(&Desc, FirstSubresource, NumSubresources, 0, nullptr, nullptr, nullptr, &RequiredSize);
	pDevice->Release();
	return RequiredSize;
}
//------------------------------------------------------------------------------------------------
// All arrays must be populated (e.g. by calling GetCopyableFootprints)
UINT64 UpdateSubresources(
	_In_ GFXCommandList* pCmdList,
	_In_ GFXResource* pDestinationResource,
	_In_ GFXResource* pIntermediate,
	_In_range_(0, D3D12_REQ_SUBRESOURCES) uint FirstSubresource,
	_In_range_(0, D3D12_REQ_SUBRESOURCES - FirstSubresource) uint NumSubresources,
	UINT64 RequiredSize,
	_In_reads_(NumSubresources) const D3D12_PLACED_SUBRESOURCE_FOOTPRINT* pLayouts,
	_In_reads_(NumSubresources) const uint* pNumRows,
	_In_reads_(NumSubresources) const UINT64* pRowSizesInBytes,
	_In_reads_(NumSubresources) const D3D12_SUBRESOURCE_DATA* pSrcData) noexcept {
	// Minor validation
	auto IntermediateDesc = pIntermediate->GetDesc();
	auto DestinationDesc = pDestinationResource->GetDesc();
	if (IntermediateDesc.Dimension != D3D12_RESOURCE_DIMENSION_BUFFER || IntermediateDesc.Width < RequiredSize + pLayouts[0].Offset || RequiredSize > SIZE_T(-1) || (DestinationDesc.Dimension == D3D12_RESOURCE_DIMENSION_BUFFER && (FirstSubresource != 0 || NumSubresources != 1))) {
		return 0;
	}
	BYTE* pData;
	HRESULT hr = pIntermediate->Map(0, nullptr, reinterpret_cast<void**>(&pData));
	if (FAILED(hr)) {
		return 0;
	}
	for (uint i = 0; i < NumSubresources; ++i) {
		if (pRowSizesInBytes[i] > SIZE_T(-1)) return 0;
		D3D12_MEMCPY_DEST DestData = {pData + pLayouts[i].Offset, pLayouts[i].Footprint.RowPitch, SIZE_T(pLayouts[i].Footprint.RowPitch) * SIZE_T(pNumRows[i])};
		MemcpySubresource(&DestData, &pSrcData[i], static_cast<SIZE_T>(pRowSizesInBytes[i]), pNumRows[i], pLayouts[i].Footprint.Depth);
	}
	pIntermediate->Unmap(0, nullptr);
	if (DestinationDesc.Dimension == D3D12_RESOURCE_DIMENSION_BUFFER) {
		pCmdList->CopyBufferRegion(
			pDestinationResource, 0, pIntermediate, pLayouts[0].Offset, pLayouts[0].Footprint.Width);
	} else {
		for (uint i = 0; i < NumSubresources; ++i) {
			CD3DX12_TEXTURE_COPY_LOCATION Dst(pDestinationResource, i + FirstSubresource);
			CD3DX12_TEXTURE_COPY_LOCATION Src(pIntermediate, pLayouts[i]);
			pCmdList->CopyTextureRegion(&Dst, 0, 0, 0, &Src, nullptr);
		}
	}
	return RequiredSize;
}
//------------------------------------------------------------------------------------------------
// All arrays must be populated (e.g. by calling GetCopyableFootprints)
UINT64 UpdateSubresources(
	_In_ GFXCommandList* pCmdList,
	_In_ GFXResource* pDestinationResource,
	_In_ GFXResource* pIntermediate,
	_In_range_(0, D3D12_REQ_SUBRESOURCES) uint FirstSubresource,
	_In_range_(0, D3D12_REQ_SUBRESOURCES - FirstSubresource) uint NumSubresources,
	UINT64 RequiredSize,
	_In_reads_(NumSubresources) const D3D12_PLACED_SUBRESOURCE_FOOTPRINT* pLayouts,
	_In_reads_(NumSubresources) const uint* pNumRows,
	_In_reads_(NumSubresources) const UINT64* pRowSizesInBytes,
	_In_ const void* pResourceData,
	_In_reads_(NumSubresources) const D3D12_SUBRESOURCE_INFO* pSrcData) {
	// Minor validation
	auto IntermediateDesc = pIntermediate->GetDesc();
	auto DestinationDesc = pDestinationResource->GetDesc();
	if (IntermediateDesc.Dimension != D3D12_RESOURCE_DIMENSION_BUFFER || IntermediateDesc.Width < RequiredSize + pLayouts[0].Offset || RequiredSize > SIZE_T(-1) || (DestinationDesc.Dimension == D3D12_RESOURCE_DIMENSION_BUFFER && (FirstSubresource != 0 || NumSubresources != 1))) {
		return 0;
	}
	BYTE* pData;
	HRESULT hr = pIntermediate->Map(0, nullptr, reinterpret_cast<void**>(&pData));
	if (FAILED(hr)) {
		return 0;
	}
	for (uint i = 0; i < NumSubresources; ++i) {
		if (pRowSizesInBytes[i] > SIZE_T(-1)) return 0;
		D3D12_MEMCPY_DEST DestData = {pData + pLayouts[i].Offset, pLayouts[i].Footprint.RowPitch, SIZE_T(pLayouts[i].Footprint.RowPitch) * SIZE_T(pNumRows[i])};
		MemcpySubresource(&DestData, pResourceData, &pSrcData[i], static_cast<SIZE_T>(pRowSizesInBytes[i]), pNumRows[i], pLayouts[i].Footprint.Depth);
	}
	pIntermediate->Unmap(0, nullptr);
	if (DestinationDesc.Dimension == D3D12_RESOURCE_DIMENSION_BUFFER) {
		pCmdList->CopyBufferRegion(
			pDestinationResource, 0, pIntermediate, pLayouts[0].Offset, pLayouts[0].Footprint.Width);
	} else {
		for (uint i = 0; i < NumSubresources; ++i) {
			CD3DX12_TEXTURE_COPY_LOCATION Dst(pDestinationResource, i + FirstSubresource);
			CD3DX12_TEXTURE_COPY_LOCATION Src(pIntermediate, pLayouts[i]);
			pCmdList->CopyTextureRegion(&Dst, 0, 0, 0, &Src, nullptr);
		}
	}
	return RequiredSize;
}
//------------------------------------------------------------------------------------------------
// Heap-allocating UpdateSubresources implementation
UINT64 UpdateSubresources(
	_In_ GFXCommandList* pCmdList,
	_In_ GFXResource* pDestinationResource,
	_In_ GFXResource* pIntermediate,
	UINT64 IntermediateOffset,
	_In_range_(0, D3D12_REQ_SUBRESOURCES) uint FirstSubresource,
	_In_range_(0, D3D12_REQ_SUBRESOURCES - FirstSubresource) uint NumSubresources,
	_In_reads_(NumSubresources) const D3D12_SUBRESOURCE_DATA* pSrcData) noexcept {
	UINT64 RequiredSize = 0;
	UINT64 MemToAlloc = static_cast<UINT64>(sizeof(D3D12_PLACED_SUBRESOURCE_FOOTPRINT) + sizeof(uint) + sizeof(UINT64)) * NumSubresources;
	if (MemToAlloc > SIZE_MAX) {
		return 0;
	}
	void* pMem = HeapAlloc(GetProcessHeap(), 0, static_cast<SIZE_T>(MemToAlloc));
	if (pMem == nullptr) {
		return 0;
	}
	auto pLayouts = static_cast<D3D12_PLACED_SUBRESOURCE_FOOTPRINT*>(pMem);
	UINT64* pRowSizesInBytes = reinterpret_cast<UINT64*>(pLayouts + NumSubresources);
	uint* pNumRows = reinterpret_cast<uint*>(pRowSizesInBytes + NumSubresources);
	auto Desc = pDestinationResource->GetDesc();
	GFXDevice* pDevice = nullptr;
	pDestinationResource->GetDevice(IID_PPV_ARGS(&pDevice));
	pDevice->GetCopyableFootprints(&Desc, FirstSubresource, NumSubresources, IntermediateOffset, pLayouts, pNumRows, pRowSizesInBytes, &RequiredSize);
	pDevice->Release();
	UINT64 Result = UpdateSubresources(pCmdList, pDestinationResource, pIntermediate, FirstSubresource, NumSubresources, RequiredSize, pLayouts, pNumRows, pRowSizesInBytes, pSrcData);
	HeapFree(GetProcessHeap(), 0, pMem);
	return Result;
}
HRESULT D3DX12SerializeVersionedRootSignature(
	_In_ const D3D12_VERSIONED_ROOT_SIGNATURE_DESC* pRootSignatureDesc,
	D3D_ROOT_SIGNATURE_VERSION MaxVersion,
	_Outptr_ ID3DBlob** ppBlob,
	_Always_(_Outptr_opt_result_maybenull_) ID3DBlob** ppErrorBlob) noexcept {
	if (ppErrorBlob != nullptr) {
		*ppErrorBlob = nullptr;
	}
	switch (MaxVersion) {
		case D3D_ROOT_SIGNATURE_VERSION_1_0:
			switch (pRootSignatureDesc->Version) {
				case D3D_ROOT_SIGNATURE_VERSION_1_0:
					return D3D12SerializeRootSignature(&pRootSignatureDesc->Desc_1_0, D3D_ROOT_SIGNATURE_VERSION_1, ppBlob, ppErrorBlob);
				case D3D_ROOT_SIGNATURE_VERSION_1_1: {
					HRESULT hr = S_OK;
					const D3D12_ROOT_SIGNATURE_DESC1& desc_1_1 = pRootSignatureDesc->Desc_1_1;
					const SIZE_T ParametersSize = sizeof(D3D12_ROOT_PARAMETER) * desc_1_1.NumParameters;
					void* pParameters = (ParametersSize > 0) ? HeapAlloc(GetProcessHeap(), 0, ParametersSize) : nullptr;
					if (ParametersSize > 0 && pParameters == nullptr) {
						hr = E_OUTOFMEMORY;
					}
					auto pParameters_1_0 = static_cast<D3D12_ROOT_PARAMETER*>(pParameters);
					if (SUCCEEDED(hr)) {
						for (uint n = 0; n < desc_1_1.NumParameters; n++) {
							__analysis_assume(ParametersSize == sizeof(D3D12_ROOT_PARAMETER) * desc_1_1.NumParameters);
							pParameters_1_0[n].ParameterType = desc_1_1.pParameters[n].ParameterType;
							pParameters_1_0[n].ShaderVisibility = desc_1_1.pParameters[n].ShaderVisibility;
							switch (desc_1_1.pParameters[n].ParameterType) {
								case D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS:
									pParameters_1_0[n].Constants.Num32BitValues = desc_1_1.pParameters[n].Constants.Num32BitValues;
									pParameters_1_0[n].Constants.RegisterSpace = desc_1_1.pParameters[n].Constants.RegisterSpace;
									pParameters_1_0[n].Constants.ShaderRegister = desc_1_1.pParameters[n].Constants.ShaderRegister;
									break;
								case D3D12_ROOT_PARAMETER_TYPE_CBV:
								case D3D12_ROOT_PARAMETER_TYPE_SRV:
								case D3D12_ROOT_PARAMETER_TYPE_UAV:
									pParameters_1_0[n].Descriptor.RegisterSpace = desc_1_1.pParameters[n].Descriptor.RegisterSpace;
									pParameters_1_0[n].Descriptor.ShaderRegister = desc_1_1.pParameters[n].Descriptor.ShaderRegister;
									break;
								case D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE:
									const D3D12_ROOT_DESCRIPTOR_TABLE1& table_1_1 = desc_1_1.pParameters[n].DescriptorTable;
									const SIZE_T DescriptorRangesSize = sizeof(D3D12_DESCRIPTOR_RANGE) * table_1_1.NumDescriptorRanges;
									void* pDescriptorRanges = (DescriptorRangesSize > 0 && SUCCEEDED(hr)) ? HeapAlloc(GetProcessHeap(), 0, DescriptorRangesSize) : nullptr;
									if (DescriptorRangesSize > 0 && pDescriptorRanges == nullptr) {
										hr = E_OUTOFMEMORY;
									}
									auto pDescriptorRanges_1_0 = static_cast<D3D12_DESCRIPTOR_RANGE*>(pDescriptorRanges);
									if (SUCCEEDED(hr)) {
										for (uint x = 0; x < table_1_1.NumDescriptorRanges; x++) {
											__analysis_assume(DescriptorRangesSize == sizeof(D3D12_DESCRIPTOR_RANGE) * table_1_1.NumDescriptorRanges);
											pDescriptorRanges_1_0[x].BaseShaderRegister = table_1_1.pDescriptorRanges[x].BaseShaderRegister;
											pDescriptorRanges_1_0[x].NumDescriptors = table_1_1.pDescriptorRanges[x].NumDescriptors;
											pDescriptorRanges_1_0[x].OffsetInDescriptorsFromTableStart = table_1_1.pDescriptorRanges[x].OffsetInDescriptorsFromTableStart;
											pDescriptorRanges_1_0[x].RangeType = table_1_1.pDescriptorRanges[x].RangeType;
											pDescriptorRanges_1_0[x].RegisterSpace = table_1_1.pDescriptorRanges[x].RegisterSpace;
										}
									}
									D3D12_ROOT_DESCRIPTOR_TABLE& table_1_0 = pParameters_1_0[n].DescriptorTable;
									table_1_0.NumDescriptorRanges = table_1_1.NumDescriptorRanges;
									table_1_0.pDescriptorRanges = pDescriptorRanges_1_0;
							}
						}
					}
					if (SUCCEEDED(hr)) {
						CD3DX12_ROOT_SIGNATURE_DESC desc_1_0(desc_1_1.NumParameters, pParameters_1_0, desc_1_1.NumStaticSamplers, desc_1_1.pStaticSamplers, desc_1_1.Flags);
						hr = D3D12SerializeRootSignature(&desc_1_0, D3D_ROOT_SIGNATURE_VERSION_1, ppBlob, ppErrorBlob);
					}
					if (pParameters) {
						for (uint n = 0; n < desc_1_1.NumParameters; n++) {
							if (desc_1_1.pParameters[n].ParameterType == D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE) {
								HeapFree(GetProcessHeap(), 0, reinterpret_cast<void*>(const_cast<D3D12_DESCRIPTOR_RANGE*>(pParameters_1_0[n].DescriptorTable.pDescriptorRanges)));
							}
						}
						HeapFree(GetProcessHeap(), 0, pParameters);
					}
					return hr;
				}
			}
			break;
		case D3D_ROOT_SIGNATURE_VERSION_1_1:
			return D3D12SerializeVersionedRootSignature(pRootSignatureDesc, ppBlob, ppErrorBlob);
	}
	return E_INVALIDARG;
}
//------------------------------------------------------------------------------------------------
// Heap-allocating UpdateSubresources implementation
UINT64 UpdateSubresources(
	_In_ GFXCommandList* pCmdList,
	_In_ GFXResource* pDestinationResource,
	_In_ GFXResource* pIntermediate,
	UINT64 IntermediateOffset,
	_In_range_(0, D3D12_REQ_SUBRESOURCES) uint FirstSubresource,
	_In_range_(0, D3D12_REQ_SUBRESOURCES - FirstSubresource) uint NumSubresources,
	_In_ const void* pResourceData,
	_In_reads_(NumSubresources) D3D12_SUBRESOURCE_INFO* pSrcData) {
	UINT64 RequiredSize = 0;
	UINT64 MemToAlloc = static_cast<UINT64>(sizeof(D3D12_PLACED_SUBRESOURCE_FOOTPRINT) + sizeof(uint) + sizeof(UINT64)) * NumSubresources;
	if (MemToAlloc > SIZE_MAX) {
		return 0;
	}
	void* pMem = HeapAlloc(GetProcessHeap(), 0, static_cast<SIZE_T>(MemToAlloc));
	if (pMem == nullptr) {
		return 0;
	}
	auto pLayouts = reinterpret_cast<D3D12_PLACED_SUBRESOURCE_FOOTPRINT*>(pMem);
	UINT64* pRowSizesInBytes = reinterpret_cast<UINT64*>(pLayouts + NumSubresources);
	uint* pNumRows = reinterpret_cast<uint*>(pRowSizesInBytes + NumSubresources);
	auto Desc = pDestinationResource->GetDesc();
	GFXDevice* pDevice = nullptr;
	pDestinationResource->GetDevice(IID_PPV_ARGS(&pDevice));
	pDevice->GetCopyableFootprints(&Desc, FirstSubresource, NumSubresources, IntermediateOffset, pLayouts, pNumRows, pRowSizesInBytes, &RequiredSize);
	pDevice->Release();
	UINT64 Result = UpdateSubresources(pCmdList, pDestinationResource, pIntermediate, FirstSubresource, NumSubresources, RequiredSize, pLayouts, pNumRows, pRowSizesInBytes, pResourceData, pSrcData);
	HeapFree(GetProcessHeap(), 0, pMem);
	return Result;
}
D3D12_PIPELINE_STATE_SUBOBJECT_TYPE D3DX12GetBaseSubobjectType(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE SubobjectType) noexcept {
	switch (SubobjectType) {
		case D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL1:
			return D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL;
		default:
			return SubobjectType;
	}
}
HRESULT D3DX12ParsePipelineStream(const D3D12_PIPELINE_STATE_STREAM_DESC& Desc, ID3DX12PipelineParserCallbacks* pCallbacks) {
	if (pCallbacks == nullptr) {
		return E_INVALIDARG;
	}
	if (Desc.SizeInBytes == 0 || Desc.pPipelineStateSubobjectStream == nullptr) {
		pCallbacks->ErrorBadInputParameter(1);// first parameter issue
		return E_INVALIDARG;
	}
	bool SubobjectSeen[D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_MAX_VALID] = {};
	for (SIZE_T CurOffset = 0, SizeOfSubobject = 0; CurOffset < Desc.SizeInBytes; CurOffset += SizeOfSubobject) {
		BYTE* pStream = static_cast<BYTE*>(Desc.pPipelineStateSubobjectStream) + CurOffset;
		auto SubobjectType = *reinterpret_cast<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE*>(pStream);
		if (SubobjectType < 0 || SubobjectType >= D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_MAX_VALID) {
			pCallbacks->ErrorUnknownSubobject(SubobjectType);
			return E_INVALIDARG;
		}
		if (SubobjectSeen[D3DX12GetBaseSubobjectType(SubobjectType)]) {
			pCallbacks->ErrorDuplicateSubobject(SubobjectType);
			return E_INVALIDARG;// disallow subobject duplicates in a stream
		}
		SubobjectSeen[SubobjectType] = true;
		switch (SubobjectType) {
			case D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_ROOT_SIGNATURE:
				pCallbacks->RootSignatureCb(*reinterpret_cast<decltype(CD3DX12_PIPELINE_STATE_STREAM::pRootSignature)*>(pStream));
				SizeOfSubobject = sizeof(CD3DX12_PIPELINE_STATE_STREAM::pRootSignature);
				break;
			case D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_VS:
				pCallbacks->VSCb(*reinterpret_cast<decltype(CD3DX12_PIPELINE_STATE_STREAM::VS)*>(pStream));
				SizeOfSubobject = sizeof(CD3DX12_PIPELINE_STATE_STREAM::VS);
				break;
			case D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PS:
				pCallbacks->PSCb(*reinterpret_cast<decltype(CD3DX12_PIPELINE_STATE_STREAM::PS)*>(pStream));
				SizeOfSubobject = sizeof(CD3DX12_PIPELINE_STATE_STREAM::PS);
				break;
			case D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DS:
				pCallbacks->DSCb(*reinterpret_cast<decltype(CD3DX12_PIPELINE_STATE_STREAM::DS)*>(pStream));
				SizeOfSubobject = sizeof(CD3DX12_PIPELINE_STATE_STREAM::DS);
				break;
			case D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_HS:
				pCallbacks->HSCb(*reinterpret_cast<decltype(CD3DX12_PIPELINE_STATE_STREAM::HS)*>(pStream));
				SizeOfSubobject = sizeof(CD3DX12_PIPELINE_STATE_STREAM::HS);
				break;
			case D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_GS:
				pCallbacks->GSCb(*reinterpret_cast<decltype(CD3DX12_PIPELINE_STATE_STREAM::GS)*>(pStream));
				SizeOfSubobject = sizeof(CD3DX12_PIPELINE_STATE_STREAM::GS);
				break;
			case D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_CS:
				pCallbacks->CSCb(*reinterpret_cast<decltype(CD3DX12_PIPELINE_STATE_STREAM::CS)*>(pStream));
				SizeOfSubobject = sizeof(CD3DX12_PIPELINE_STATE_STREAM::CS);
				break;
			case D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_AS:
				pCallbacks->ASCb(*reinterpret_cast<decltype(CD3DX12_PIPELINE_STATE_STREAM2::AS)*>(pStream));
				SizeOfSubobject = sizeof(CD3DX12_PIPELINE_STATE_STREAM2::AS);
				break;
			case D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_MS:
				pCallbacks->MSCb(*reinterpret_cast<decltype(CD3DX12_PIPELINE_STATE_STREAM2::MS)*>(pStream));
				SizeOfSubobject = sizeof(CD3DX12_PIPELINE_STATE_STREAM2::MS);
				break;
			case D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_STREAM_OUTPUT:
				pCallbacks->StreamOutputCb(*reinterpret_cast<decltype(CD3DX12_PIPELINE_STATE_STREAM::StreamOutput)*>(pStream));
				SizeOfSubobject = sizeof(CD3DX12_PIPELINE_STATE_STREAM::StreamOutput);
				break;
			case D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_BLEND:
				pCallbacks->BlendStateCb(*reinterpret_cast<decltype(CD3DX12_PIPELINE_STATE_STREAM::BlendState)*>(pStream));
				SizeOfSubobject = sizeof(CD3DX12_PIPELINE_STATE_STREAM::BlendState);
				break;
			case D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_SAMPLE_MASK:
				pCallbacks->SampleMaskCb(*reinterpret_cast<decltype(CD3DX12_PIPELINE_STATE_STREAM::SampleMask)*>(pStream));
				SizeOfSubobject = sizeof(CD3DX12_PIPELINE_STATE_STREAM::SampleMask);
				break;
			case D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RASTERIZER:
				pCallbacks->RasterizerStateCb(*reinterpret_cast<decltype(CD3DX12_PIPELINE_STATE_STREAM::RasterizerState)*>(pStream));
				SizeOfSubobject = sizeof(CD3DX12_PIPELINE_STATE_STREAM::RasterizerState);
				break;
			case D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL:
				pCallbacks->DepthStencilStateCb(*reinterpret_cast<CD3DX12_PIPELINE_STATE_STREAM_DEPTH_STENCIL*>(pStream));
				SizeOfSubobject = sizeof(CD3DX12_PIPELINE_STATE_STREAM_DEPTH_STENCIL);
				break;
			case D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL1:
				pCallbacks->DepthStencilState1Cb(*reinterpret_cast<decltype(CD3DX12_PIPELINE_STATE_STREAM::DepthStencilState)*>(pStream));
				SizeOfSubobject = sizeof(CD3DX12_PIPELINE_STATE_STREAM::DepthStencilState);
				break;
			case D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_INPUT_LAYOUT:
				pCallbacks->InputLayoutCb(*reinterpret_cast<decltype(CD3DX12_PIPELINE_STATE_STREAM::InputLayout)*>(pStream));
				SizeOfSubobject = sizeof(CD3DX12_PIPELINE_STATE_STREAM::InputLayout);
				break;
			case D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_IB_STRIP_CUT_VALUE:
				pCallbacks->IBStripCutValueCb(*reinterpret_cast<decltype(CD3DX12_PIPELINE_STATE_STREAM::IBStripCutValue)*>(pStream));
				SizeOfSubobject = sizeof(CD3DX12_PIPELINE_STATE_STREAM::IBStripCutValue);
				break;
			case D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PRIMITIVE_TOPOLOGY:
				pCallbacks->PrimitiveTopologyTypeCb(*reinterpret_cast<decltype(CD3DX12_PIPELINE_STATE_STREAM::PrimitiveTopologyType)*>(pStream));
				SizeOfSubobject = sizeof(CD3DX12_PIPELINE_STATE_STREAM::PrimitiveTopologyType);
				break;
			case D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RENDER_TARGET_FORMATS:
				pCallbacks->RTVFormatsCb(*reinterpret_cast<decltype(CD3DX12_PIPELINE_STATE_STREAM::RTVFormats)*>(pStream));
				SizeOfSubobject = sizeof(CD3DX12_PIPELINE_STATE_STREAM::RTVFormats);
				break;
			case D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL_FORMAT:
				pCallbacks->DSVFormatCb(*reinterpret_cast<decltype(CD3DX12_PIPELINE_STATE_STREAM::DSVFormat)*>(pStream));
				SizeOfSubobject = sizeof(CD3DX12_PIPELINE_STATE_STREAM::DSVFormat);
				break;
			case D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_SAMPLE_DESC:
				pCallbacks->SampleDescCb(*reinterpret_cast<decltype(CD3DX12_PIPELINE_STATE_STREAM::SampleDesc)*>(pStream));
				SizeOfSubobject = sizeof(CD3DX12_PIPELINE_STATE_STREAM::SampleDesc);
				break;
			case D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_NODE_MASK:
				pCallbacks->NodeMaskCb(*reinterpret_cast<decltype(CD3DX12_PIPELINE_STATE_STREAM::NodeMask)*>(pStream));
				SizeOfSubobject = sizeof(CD3DX12_PIPELINE_STATE_STREAM::NodeMask);
				break;
			case D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_CACHED_PSO:
				pCallbacks->CachedPSOCb(*reinterpret_cast<decltype(CD3DX12_PIPELINE_STATE_STREAM::CachedPSO)*>(pStream));
				SizeOfSubobject = sizeof(CD3DX12_PIPELINE_STATE_STREAM::CachedPSO);
				break;
			case D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_FLAGS:
				pCallbacks->FlagsCb(*reinterpret_cast<decltype(CD3DX12_PIPELINE_STATE_STREAM::Flags)*>(pStream));
				SizeOfSubobject = sizeof(CD3DX12_PIPELINE_STATE_STREAM::Flags);
				break;
			case D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_VIEW_INSTANCING:
				pCallbacks->ViewInstancingCb(*reinterpret_cast<decltype(CD3DX12_PIPELINE_STATE_STREAM1::ViewInstancingDesc)*>(pStream));
				SizeOfSubobject = sizeof(CD3DX12_PIPELINE_STATE_STREAM1::ViewInstancingDesc);
				break;
			default:
				pCallbacks->ErrorUnknownSubobject(SubobjectType);
				return E_INVALIDARG;
		}
	}
	return S_OK;
}