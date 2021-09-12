#pragma once
#include <RenderComponent/IShader.h>
#include <RenderComponent/UploadBuffer.h>

class JobBucket;
class DescriptorHeap;
class StructuredBuffer;
class ShaderLoader;
class VENGINE_DLL_RENDERER RayShader final : public IShader {
private:
	Microsoft::WRL::ComPtr<ID3D12StateObject> mStateObj;
	DXRHitGroup hitGroups;
	StackObject<BinaryJson, true> serObj;
	StackObject<UploadBuffer, true> identifierBuffer;

public:
	bool SetResource(ThreadCommand* commandList, uint id, DescriptorHeap const* descHeap, uint64 elementOffset) const override;
	bool SetResource(ThreadCommand* commandList, uint id, UploadBuffer const* buffer, uint64 elementOffset) const override;
	bool SetResource(ThreadCommand* commandList, uint id, StructuredBuffer const* buffer, uint64 elementOffset) const override;
	bool SetResource(ThreadCommand* commandList, uint id, TextureBase const* texture) const override;
	bool SetResource(ThreadCommand* commandList, uint id, RenderTexture const* renderTexture, uint64 uavMipLevel) const override;

	vstd::string const& GetName() const {
		return "";
	}
	BinaryJson const* GetJsonObject() const override {
		return serObj.Initialized() ? static_cast<BinaryJson const*>(serObj) : nullptr;
	}
	void BindShader(ThreadCommand* commandList) const override;
	void BindShader(ThreadCommand* commandList, const DescriptorHeap* heap) const override;
	bool SetBufferByAddress(ThreadCommand* commandList, uint id, GpuAddress address) const override;
	template<typename Func>
	void IterateVariables(const Func& f) {
		for (int32_t i = 0; i < mVariablesVector.size(); ++i) {
			f(mVariablesVector[i]);
		}
	}
	RayShader(GFXDevice* device, vstd::string const& path);
	void DispatchRays(
		ThreadCommand* cmdList,
		uint width,
		uint height,
		uint depth) const;
	~RayShader();
	VSTL_OVERRIDE_OPERATOR_NEW
	VSTL_DELETE_COPY_CONSTRUCT(RayShader)
};