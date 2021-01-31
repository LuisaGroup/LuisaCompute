#pragma once
#include "../Common/GFXUtil.h"
#include "../Common/VObject.h"
#include "../Struct/RenderPackage.h"
class PSOContainer;
class CBufferChunk;
class Shader;
class Mesh;
class TextureBase;
class Skybox : public VObject {
private:
	static std::unique_ptr<Mesh> fullScreenMesh;
	ObjectPtr<TextureBase> skyboxTex;
	const Shader* shader;
	uint SkyboxCBufferID;

public:
	virtual ~Skybox();
	Skybox(
		const ObjectPtr<TextureBase>& tex,
		GFXDevice* device);

	TextureBase* GetTexture() const {
		return skyboxTex;
	}

	void Draw(
		uint targetPass,
		CBufferChunk const& cameraBuffer,
		const RenderPackage& package) const;
};
