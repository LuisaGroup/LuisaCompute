#pragma once
#include "IShader.h"
#include "UploadBuffer.h"

class JobBucket;
class DescriptorHeap;
class StructuredBuffer;
class ComputeShaderCompiler;
class ComputeShaderReader;
class ShaderLoader;
class RayShader final : public IShader
{
private:
	HashMap<uint, uint> mVariablesDict;
	vengine::vector<ShaderVariable> mVariablesVector;
	Microsoft::WRL::ComPtr<ID3D12StateObject> mStateObj;
	Microsoft::WRL::ComPtr<ID3D12RootSignature> mRootSignature;
	DXRHitGroup hitGroups;
	StackObject<SerializedObject, true> serObj;
	StackObject<UploadBuffer, true> identifierBuffer;
	bool SetRes(ThreadCommand* commandList, uint id, const VObject* targetObj, uint64 indexOffset, const std::type_info& tyid) const override;

public:
	vengine::string const& GetName() const {
		return "";
	}
	SerializedObject const* GetJsonObject() const override
	{
		return serObj.Initialized() ? serObj : nullptr;
	}
	size_t VariableLength() const override { return mVariablesVector.size(); }
	int32_t GetPropertyRootSigPos(uint id) const override;
	void BindShader(ThreadCommand* commandList) const override;
	void BindShader(ThreadCommand* commandList, const DescriptorHeap* heap) const override;
	bool SetBufferByAddress(ThreadCommand* commandList, uint id, GpuAddress address) const override;
	template<typename Func>
	void IterateVariables(const Func& f)
	{
		for (int32_t i = 0; i < mVariablesVector.size(); ++i)
		{
			f(mVariablesVector[i]);
		}
	}
	RayShader(GFXDevice* device, vengine::string const& path);
	void DispatchRays(
		ThreadCommand* cmdList,
		uint width,
		uint height,
		uint depth
	) const;
	~RayShader();
	DECLARE_VENGINE_OVERRIDE_OPERATOR_NEW
	KILL_COPY_CONSTRUCT(RayShader)
};