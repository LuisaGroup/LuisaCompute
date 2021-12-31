#pragma once
#include <Shader/Shader.h>
#include <Resource/DefaultBuffer.h>
namespace toolhub::directx {
class CommandBufferBuilder;
class RTShader final : public Shader {
protected:
	Microsoft::WRL::ComPtr<ID3D12StateObject> stateObj;
	using ByteVector = vstd::vector<vbyte, VEngine_AllocType::VEngine, D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES>;
	ByteVector raygenIdentifier;
	ByteVector missIdentifier;
	ByteVector identifier;
	DefaultBuffer identityBuffer;
	mutable std::atomic_flag finishedUpdate;
	DXRHitGroup GetHitGroup() const;
	void Update(CommandBufferBuilder& builder) const;

public:
	static wchar_t const* GetClosestHitFuncName();
	static wchar_t const* GetRayGenFuncName();
	static wchar_t const* GetIntersectFuncName();
	static wchar_t const* GetAnyHitFuncName();
	static wchar_t const* GetMissFuncName();

	void DispatchRays(
		CommandBufferBuilder& originCmdList,
		uint width,
		uint height,
		uint depth) const;
	RTShader(
		bool closestHit,
		bool anyHit,
		bool intersectHit,
        std::span<std::pair<vstd::string_view, Property>> properties,
        vstd::span<vbyte> binData,
		Device* device);
	~RTShader();
	RTShader(RTShader&& v) = default;
	KILL_COPY_CONSTRUCT(RTShader)
};
}// namespace toolhub::directx