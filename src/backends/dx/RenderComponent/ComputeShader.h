#pragma once
#include "IShader.h"
#include "Utility/CommandSignature.h"
#include <Singleton/ShaderLoader.h>
class JobBucket;
struct ComputeKernel
{
	vengine::string name;
	Microsoft::WRL::ComPtr<ID3DBlob> datas;
};

class VENGINE_DLL_RENDERER ComputeShader final : public IShader
{
	friend class ShaderLoader;
private:
	StackObject<SerializedObject, true> serObj;
	vengine::string name;
	HashMap<vengine::string, uint> kernelNames;
	HashMap<uint, uint> mVariablesDict;
	vengine::vector<ShaderVariable> mVariablesVector;
	vengine::vector<ComputeKernel> csShaders;
	vengine::vector<Microsoft::WRL::ComPtr<GFXPipelineState>> pso;
	Microsoft::WRL::ComPtr<ID3D12RootSignature> mRootSignature;
	CommandSignature cmdSig;
	bool SetRes(ThreadCommand* commandList, uint id, const VObject* targetObj, uint64 indexOffset, ResourceType tyid) const override;
	//	bool TrySetRes(ThreadCommand* commandList, uint id, const VObject* targetObj, uint64 indexOffset, const std::type_info& tyid) const;
	ComputeShader(
		vengine::string const& name,
		const vengine::string& csoFilePath,
		GFXDevice* device);

	~ComputeShader();
public:
	SerializedObject const* GetJsonObject() const
	{
		return serObj.Initialized() ? serObj : nullptr;
	}
	vengine::string const& GetName() const
	{
		return name;
	}
	uint GetKernelIndex(const vengine::string& str) const;
	size_t VariableLength() const override { return mVariablesVector.size(); }
	int32_t GetPropertyRootSigPos(uint id) const override;
	void BindShader(ThreadCommand* commandList) const override;
	void BindShader(ThreadCommand* commandList, const DescriptorHeap* heap) const override;
	bool SetBufferByAddress(ThreadCommand* commandList, uint id, GpuAddress address) const override;
	void Dispatch(ThreadCommand* cList, uint kernel, uint x, uint y, uint z) const;
	void DispatchIndirect(ThreadCommand* cList, uint dispatchKernel, StructuredBuffer* indirectBuffer, uint bufferElement, uint bufferIndex) const;
	template<typename Func>
	void IterateVariables(const Func& f)
	{
		for (int32_t i = 0; i < mVariablesVector.size(); ++i)
		{
			f(mVariablesVector[i]);
		}
	}
	DECLARE_VENGINE_OVERRIDE_OPERATOR_NEW
	KILL_COPY_CONSTRUCT(ComputeShader)
};