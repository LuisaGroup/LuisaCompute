#pragma once
#include <iostream>
#include "../Common/vstring.h"
#include "../Common/vector.h"
enum class ShaderType : uint8_t
{
	ComputeShader,
	VertexShader,
	PixelShader,
	HullShader,
	DomainShader,
	GeometryShader,
	RayTracingShader,
	ShaderTypeNum
};
enum class ShaderFileType : uint8_t
{
	None,
	Shader,
	ComputeShader,
	DXRShader
};
ShaderFileType GetShaderFileType(vengine::string const& str);

struct Command
{
	vengine::string fileName;
	vengine::string propertyFileName;
	ShaderFileType shaderFileType;
};
bool GetShaderFileCommand(vengine::string const& str, Command& cmd);
class ICompileIterator
{
public:
	virtual vengine::vector<Command>& GetCommand() = 0;
	virtual void UpdateCommand() = 0;
	virtual ~ICompileIterator() {}
};