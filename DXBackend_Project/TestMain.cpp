#include <iostream>
#include "ShaderCompile/ShaderVarUtility.h"
#include <ast/expression.h>
int main() {
	using namespace LCShader;
	vengine::vengine_init_malloc();
	/*
	float3 a;
	float3 e;
	float4 b;
	float2 s;
	float4x4 d;
	float dfff;
	
	*/
	vengine::vector<VarNode const*> vars;
	vengine::vector<VarNode*> alignVars;
	{
		NumVar* nv = new NumVar();
		nv->rowLength = 3;
		nv->columeLength = 1;
		nv->varType = NumVarType::Float;
		VarNode* vn = new VarNode();
		vn->isConstant = false;
		vn->type = nv;
		vn->valueName = "a";
		vars.push_back(vn);
	}
	{
		NumVar* nv = new NumVar();
		nv->rowLength = 3;
		nv->columeLength = 1;
		nv->varType = NumVarType::Float;
		VarNode* vn = new VarNode();
		vn->isConstant = false;
		vn->type = nv;
		vn->valueName = "e";
		vars.push_back(vn);
	}
	{
		NumVar* nv = new NumVar();
		nv->rowLength = 4;
		nv->columeLength = 1;
		nv->varType = NumVarType::Float;
		VarNode* vn = new VarNode();
		vn->isConstant = false;
		vn->type = nv;
		vn->valueName = "b";
		vars.push_back(vn);
	}
	{
		NumVar* nv = new NumVar();
		nv->rowLength = 2;
		nv->columeLength = 1;
		nv->varType = NumVarType::Float;
		VarNode* vn = new VarNode();
		vn->isConstant = false;
		vn->type = nv;
		vn->valueName = "s";
		vars.push_back(vn);
	}
	{
		NumVar* nv = new NumVar();
		nv->rowLength = 4;
		nv->columeLength = 4;
		nv->varType = NumVarType::Float;
		VarNode* vn = new VarNode();
		vn->isConstant = false;
		vn->type = nv;
		vn->valueName = "d";
		vars.push_back(vn);
	}
	{
		NumVar* nv = new NumVar();
		nv->rowLength = 1;
		nv->columeLength = 1;
		nv->varType = NumVarType::Float;
		VarNode* vn = new VarNode();
		vn->isConstant = false;
		vn->type = nv;
		vn->valueName = "dfff";
		vars.push_back(vn);
	}
	vengine::vector<VarNode const*> resultNode;
	vengine::vector<VarNode*> alignNode;
	vengine::string errorMessage;
	ShaderVarUtility::GenerateCBuffer(
		vars,resultNode,alignNode,
		errorMessage
	);
	for (auto& i : resultNode) {
		ShaderVarUtility::GetVarName(*static_cast<NumVar const*>(i->type), errorMessage);
		std::cout << errorMessage << " " << i->valueName << "\n";
	}
}