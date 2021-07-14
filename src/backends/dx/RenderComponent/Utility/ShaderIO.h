#pragma once
#include <Common/GFXUtil.h>
#include <CJsonObject/BinaryJson.h>
#include <RenderComponent/ComputeShader.h>
class BinaryJson;
class ThreadCommand;
class VENGINE_DLL_RENDERER ShaderIO {
public:
	static void DecodeComputeShader(
		const vstd::string& fileName,
		vstd::vector<ShaderVariable>& vars,
		vstd::vector<ComputeKernel>& datas,
		StackObject<BinaryJson, true>& serObj);
	static HRESULT GetRootSignature(
		vstd::vector<ShaderVariable> const& variables,
		Microsoft::WRL::ComPtr<ID3DBlob>& serializedRootSig,
		Microsoft::WRL::ComPtr<ID3DBlob>& errorBlob,
		D3D_ROOT_SIGNATURE_VERSION rootSigVersion);
	static bool SetComputeBufferByAddress(
		HashMap<uint, uint> const& varDict,
		vstd::vector<ShaderVariable> const& varVector,
		ThreadCommand* commandList,
		uint id,
		GpuAddress address);
	static void DecodeDXRShader(
		const vstd::string& fileName,
		vstd::vector<ShaderVariable>& vars,
		DXRHitGroup& hitGroups,
		vstd::vector<char>& binaryData,
		uint64& recursiveCount,
		uint64& raypayloadSize,
		StackObject<BinaryJson, true>& serObj);
	static bool SetComputeResource(IShader const* shader, ThreadCommand* commandList, uint id, DescriptorHeap const* descHeap, uint64 elementOffset);
	static bool SetComputeResource(IShader const* shader, ThreadCommand* commandList, uint id, UploadBuffer const* buffer, uint64 elementOffset);
	static bool SetComputeResource(IShader const* shader, ThreadCommand* commandList, uint id, StructuredBuffer const* buffer, uint64 elementOffset);
	static bool SetComputeResource(IShader const* shader, ThreadCommand* commandList, uint id, TextureBase const* texture);
	static bool SetComputeResource(IShader const* shader, ThreadCommand* commandList, uint id, RenderTexture const* renderTexture, uint64 uavMipLevel);
};