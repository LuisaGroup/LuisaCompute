#include "Skybox.h"
#include "Mesh.h"
#include "Texture.h"
#include "Shader.h"
#include <mutex>
#include "../Singleton/ShaderLoader.h"
#include "../Singleton/ShaderID.h"
#include "../Common/Camera.h"
#include "../RenderComponent/PSOContainer.h"
#include "DescriptorHeap.h"
#include "../WorldManagement/World.h"
#include "../Common/Camera.h"
#include "../Singleton/Graphics.h"
#include "Shader.h"
#include "CBufferPool.h"
#include "CBufferAllocator.h"
#include "../PipelineComponent/ThreadCommand.h"
std::unique_ptr<Mesh> Skybox::fullScreenMesh = nullptr;
void Skybox::Draw(
	uint targetPass,
	CBufferChunk const& cameraBuffer,
	const RenderPackage& package) const {
	PSODescriptor desc;
	desc.meshLayoutIndex = fullScreenMesh->GetLayoutIndex();
	desc.shaderPass = targetPass;
	desc.shaderPtr = shader;
	GFXPipelineState* pso = package.tCmd->TryGetPSOStateAsync(desc, package.device);
	if (!pso) return;
	auto commandList = package.tCmd->GetCmdList();
	package.tCmd->UpdatePSO(pso);
	shader->BindShader(package.tCmd, Graphics::GetGlobalDescHeap());
	shader->SetResource(package.tCmd, ShaderID::GetMainTex(), Graphics::GetGlobalDescHeap(), skyboxTex->GetGlobalDescIndex());
	shader->SetResource(package.tCmd, SkyboxCBufferID, cameraBuffer.GetBuffer(), cameraBuffer.GetOffset());
	auto vbv = fullScreenMesh->VertexBufferViews();
	commandList->IASetVertexBuffers(0, fullScreenMesh->VertexBufferViewCount(), vbv);
	auto ibv = fullScreenMesh->IndexBufferView();
	commandList->IASetIndexBuffer(&ibv);
	commandList->IASetPrimitiveTopology(GetD3DTopology(desc.topology));
	package.tCmd->ExecuteResBarrier();
	commandList->DrawIndexedInstanced(fullScreenMesh->GetIndexCount(), 1, 0, 0, 0);
}
Skybox::~Skybox() {
	fullScreenMesh = nullptr;
	skyboxTex = nullptr;
}
Skybox::Skybox(
	const ObjectPtr<TextureBase>& tex,
	GFXDevice* device) : skyboxTex(tex) {
	World* world = World::GetInstance();
	SkyboxCBufferID = ShaderID::PropertyToID("SkyboxBuffer");
	ObjectPtr<UploadBuffer> noProperty = nullptr;
	shader = ShaderLoader::GetShader("Skybox");
	if (fullScreenMesh == nullptr) {
		std::array<float3, 3> vertex;
		vertex[0] = {-3, -1, 1};
		vertex[1] = {1, 3, 1};
		vertex[2] = {1, -1, 1};
		std::array<INT16, 3> indices{0, 1, 2};
		fullScreenMesh = std::unique_ptr<Mesh>(new Mesh(
			3,
			vertex.data(),
			nullptr,
			nullptr,
			nullptr,
			nullptr,
			nullptr,
			nullptr,
			nullptr,
			nullptr,
			nullptr,
			device,
			GFXFormat_R16_UInt,
			3,
			indices.data()));
	}
}
