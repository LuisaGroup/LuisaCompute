#pragma once
#include <core/vstl/HashMap.h>
#include <Struct/DrawCommand.h>
#include <Common/BitVector.h>
class Shader;
namespace RenderCull
{
	struct CullBatch
	{
		Shader const* shader;
		uint32_t meshLayout;
	};
	struct CullBatchHash
	{
		size_t operator()(CullBatch const& value) const noexcept
		{
			return Hash::CharArrayHash(reinterpret_cast<char const*>(&value), sizeof(value.shader) + sizeof(value.meshLayout));
		}
	};
	struct CullBatchEqual
	{
		bool operator()(CullBatch const& a, CullBatch const& b) const noexcept
		{
			return a.shader == b.shader && a.meshLayout == b.meshLayout;
		}
	};
	struct CullResult
	{
		BitVector containingMap;
		HashMap<CullBatch, ArrayList<DrawCommand>, CullBatchHash, CullBatchEqual> hashMaps;
	};
}