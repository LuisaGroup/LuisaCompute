#pragma once
#include <Resource/Resource.h>
#include <Resource/DefaultBuffer.h>
#include <vstl/LockFreeArrayQueue.h>
namespace toolhub::directx {
class TextureBase;
class CommandBufferBuilder;
class BindlessArray final : public Resource {
public:
	struct Tex2D {
		TextureBase const* tex;
		template<typename T>
		requires(std::is_constructible_v<TextureBase const*, T&&>)
			Tex2D(T&& t) : tex(std::forward<T>(t)) {}
		bool operator==(Tex2D const& a) const {
			return a.tex == tex;
		}
		bool operator!=(Tex2D const& a) const {
			return !operator==(a);
		}
		bool operator>(Tex2D const& a) const {
			return a.tex > tex;
		}
		bool operator<(Tex2D const& a) const {
			return a.tex < tex;
		}
		template<typename T>
		void operator=(T const& t) {
			new (this) Tex2D(t);
		}
	};
	struct Tex3D {
		TextureBase const* tex;
		template<typename T>
		requires(std::is_constructible_v<TextureBase const*, T&&>)
			Tex3D(T&& t) : tex(std::forward<T>(t)) {}
		template<typename T>
		void operator=(T const& t) {
			new (this) Tex3D(t);
		}
		bool operator==(Tex3D const& a) const {
			return a.tex == tex;
		}
		bool operator!=(Tex3D const& a) const {
			return !operator==(a);
		}
		bool operator>(Tex3D const& a) const {
			return a.tex > tex;
		}
		bool operator<(Tex3D const& a) const {
			return a.tex < tex;
		}
	};
	using Property = vstd::variant<
		BufferView,
		Tex2D,
		Tex3D>;

private:
	using TupleType = std::tuple<
		std::pair<BufferView, uint>,
		std::pair<Tex2D, uint>,
		std::pair<Tex3D, uint>>;
	vstd::vector<TupleType> binded;
	enum class BindType : vbyte {
		Tex2D,
		Tex3D,
		Buffer
	};
	vstd::HashMap<Property, uint> bindedResource;
	vstd::HashMap<uint, uint> updateMap;
	vstd::LockFreeArrayQueue<uint> disposeQueue;
	DefaultBuffer buffer;
	template<typename T>
	void TryReturnBind(T& view);
	template<typename T>
	void RemoveLast(T& pair);
	template<typename T>
	void AddNew(std::pair<T, uint>& pair, T const& newValue, uint index);
	uint GetNewIndex();

public:
	BufferView GetBufferArray() const;
	BufferView GetTex2DArray() const;
	BufferView GetTex3DArray() const;
	void BindBuffer(BufferView prop, uint index);
	void BindTex2D(Tex2D prop, uint index);
	void BindTex3D(Tex3D prop, uint index);
	void UnBind(Property prop);
	vstd::optional<uint> PropertyIdx(Property prop) const;
	void Update(
		CommandBufferBuilder& builder);
	Tag GetTag() const override { return Tag::BindlessArray; }
	BindlessArray(
		Device* device,
		uint arraySize);
	~BindlessArray();
	VSTD_SELF_PTR
};
}// namespace toolhub::directx