#pragma once
#include <RenderComponent/IShader.h>
#include <RenderComponent/Utility/CommandSignature.h>
class JobBucket;
struct ComputeKernel {
	vstd::string name;
	Microsoft::WRL::ComPtr<ID3DBlob> datas;
};

class VENGINE_DLL_RENDERER ComputeShader final : public IShader {
	friend class ShaderLoader;

private:
	StackObject<BinaryJson, true> serObj;
	vstd::string name;
	HashMap<vstd::string, uint> kernelNames;
	vstd::vector<ComputeKernel> csShaders;
	vstd::vector<Microsoft::WRL::ComPtr<GFXPipelineState>> pso;
	CommandSignature cmdSig;
	//	bool TrySetRes(ThreadCommand* commandList, uint id, const VObject* targetObj, uint64 indexOffset, const std::type_info& tyid) const;
	ComputeShader(
		vstd::string const& name,
		const vstd::string& csoFilePath,
		GFXDevice* device);

	~ComputeShader();

public:
	BinaryJson const* GetJsonObject() const {
		return serObj.Initialized() ? static_cast<BinaryJson const*>(serObj) : nullptr;
	}
	vstd::string const& GetName() const {
		return name;
	}
	uint GetKernelIndex(const vstd::string& str) const;
	void BindShader(ThreadCommand* commandList) const override;
	void BindShader(ThreadCommand* commandList, const DescriptorHeap* heap) const override;
	bool SetBufferByAddress(ThreadCommand* commandList, uint id, GpuAddress address) const override;
	void Dispatch(ThreadCommand* cList, uint kernel, uint x, uint y, uint z) const;
	void DispatchIndirect(ThreadCommand* cList, uint dispatchKernel, StructuredBuffer* indirectBuffer, uint bufferElement, uint bufferIndex) const;
	bool SetResource(ThreadCommand* commandList, uint id, DescriptorHeap const* descHeap, uint64 elementOffset) const override;
	bool SetResource(ThreadCommand* commandList, uint id, UploadBuffer const* buffer, uint64 elementOffset) const override;
	bool SetResource(ThreadCommand* commandList, uint id, StructuredBuffer const* buffer, uint64 elementOffset) const override;
	bool SetResource(ThreadCommand* commandList, uint id, TextureBase const* texture) const override;
	bool SetResource(ThreadCommand* commandList, uint id, RenderTexture const* renderTexture, uint64 uavMipLevel) const override;
	template<typename Func>
	void IterateVariables(const Func& f) {
		for (int32_t i = 0; i < mVariablesVector.size(); ++i) {
			f(mVariablesVector[i]);
		}
	}
	VSTL_OVERRIDE_OPERATOR_NEW
	VSTL_DELETE_COPY_CONSTRUCT(ComputeShader)
};