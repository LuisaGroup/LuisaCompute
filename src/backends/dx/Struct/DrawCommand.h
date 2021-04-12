#pragma once
#include <Common/GFXUtil.h>
class IMesh;
class IMaterial;
class MeshRenderer;
struct DrawCommand
{
	IMaterial* mat;
	IMesh* mesh;
	uint submeshIndex;
	MeshRenderer* meshRenderer;
	DrawCommand(
		IMaterial* mat,
		IMesh* mesh,
		uint submeshIndex,
		MeshRenderer* meshRenderer
	) : mat(mat), mesh(mesh), submeshIndex(submeshIndex), meshRenderer(meshRenderer) {}
};