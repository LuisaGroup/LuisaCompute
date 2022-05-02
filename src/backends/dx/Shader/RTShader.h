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
    void Update(CommandBufferBuilder &builder) const;
    void Init(
        bool closestHit,
        bool anyHit,
        bool intersectHit,
        vstd::span<vbyte const> binData,
        Device* device);

public:
    static wchar_t const *GetClosestHitFuncName();
    static wchar_t const *GetRayGenFuncName();
    static wchar_t const *GetIntersectFuncName();
    static wchar_t const *GetAnyHitFuncName();
    static wchar_t const *GetMissFuncName();

    Tag GetTag() const { return Tag::RayTracingShader; }
    void DispatchRays(
        CommandBufferBuilder &originCmdList,
        uint width,
        uint height,
        uint depth) const;
    RTShader(
        bool closestHit,
        bool anyHit,
        bool intersectHit,
        vstd::span<std::pair<vstd::string, Property> const> properties,
        vstd::span<vbyte const> binData,
        Device *device);
    RTShader(
        bool closestHit,
        bool anyHit,
        bool intersectHit,
        vstd::span<std::pair<vstd::string, Property> const> prop,
        ComPtr<ID3D12RootSignature> &&rootSig,
        vstd::span<vbyte const> binData,
        Device *device);

    ~RTShader();
    KILL_COPY_CONSTRUCT(RTShader)
    KILL_MOVE_CONSTRUCT(RTShader)
};
}// namespace toolhub::directx