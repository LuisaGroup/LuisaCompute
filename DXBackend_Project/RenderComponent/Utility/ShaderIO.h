#pragma once
#include "../../Common/GFXUtil.h"
#include "../Shader.h"
#include "../../CJsonObject/SerializedObject.h"
#include "../ComputeShader.h"
class SerializedObject;
class ThreadCommand;
class ShaderIO
{
public:
	static void DecodeShader(
		const vengine::string& fileName,
		vengine::vector<ShaderVariable>& vars,
		vengine::vector<ShaderPass>& passes,
		StackObject<SerializedObject, true>& serObj);
	static void DecodeComputeShader(
		const vengine::string& fileName,
		vengine::vector<ShaderVariable>& vars,
		vengine::vector<ComputeKernel>& datas,
		StackObject<SerializedObject, true>& serObj);
	static HRESULT GetRootSignature(
		vengine::vector<ShaderVariable> const& variables,
		Microsoft::WRL::ComPtr<ID3DBlob>& serializedRootSig,
		Microsoft::WRL::ComPtr<ID3DBlob>& errorBlob,
		D3D_ROOT_SIGNATURE_VERSION rootSigVersion
	);
	static bool SetComputeBufferByAddress(
		HashMap<uint, uint> const& varDict,
		vengine::vector<ShaderVariable> const& varVector,
		ThreadCommand* commandList,
		uint id, 
		GpuAddress address);
	static void DecodeDXRShader(
		const vengine::string& fileName,
		vengine::vector<ShaderVariable>& vars,
		DXRHitGroup& hitGroups,
		vengine::vector<char>& binaryData,
		uint64& recursiveCount,
		uint64& raypayloadSize,
		StackObject<SerializedObject, true>& serObj);

	static bool SetComputeShaderResWithoutCheck(
		vengine::vector<ShaderVariable> const& varVector,
		ThreadCommand* commandList,
		HashMap<uint, uint>::Iterator& ite,
		const VObject* targetObj,
		uint64 indexOffset,
		const std::type_info& tyid);
};