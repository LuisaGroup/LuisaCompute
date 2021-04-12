#pragma once
#include <RenderComponent/IShader.h>
#include <RenderComponent/Utility/CommandSignature.h>
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
	vengine::vector<ComputeKernel> csShaders;
	vengine::vector<Microsoft::WRL::ComPtr<GFXPipelineState>> pso;
	CommandSignature cmdSig;
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
	void BindShader(ThreadCommand* commandList) const override;
	void BindShader(ThreadCommand* commandList, const DescriptorHeap* heap) const override;
	bool SetBufferByAddress(ThreadCommand* commandList, uint id, GpuAddress address) const override;
	void Dispatch(ThreadCommand* cList, uint kernel, uint x, uint y, uint z) const;
	void DispatchIndirect(ThreadCommand* cList, uint dispatchKernel, StructuredBuffer* indirectBuffer, uint bufferElement, uint bufferIndex) const;
	bool SetResource(ThreadCommand* commandList, uint id, DescriptorHeap const* descHeap, uint64 elementOffset) const override;
	bool SetResource(ThreadCommand* commandList, uint id, UploadBuffer const* buffer, uint64 elementOffset) const override;
	bool SetResource(ThreadCommand* commandList, uint id, StructuredBuffer const* buffer, uint64 elementOffset) const override;
	bool SetResource(ThreadCommand* commandList, uint id, Mesh const* mesh, uint64 byteOffset) const override;
	bool SetResource(ThreadCommand* commandList, uint id, TextureBase const* texture) const override;
	bool SetResource(ThreadCommand* commandList, uint id, RenderTexture const* renderTexture, uint64 uavMipLevel) const override;
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