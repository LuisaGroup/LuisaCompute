#pragma once

#include <string>
#include <array>
#include <vstl/MetaLib.h>

namespace vstd {
static constexpr size_t MD5_SIZE = 16;

LUISA_DLL std::array<uint8_t, MD5_SIZE> GetMD5FromString(std::string const& str);
LUISA_DLL std::array<uint8_t, MD5_SIZE> GetMD5FromArray(std::span<uint8_t const> data);

//Used for unity
class LUISA_DLL MD5 {
public:
	struct MD5Data {
		uint64 data0;
		uint64 data1;
	};

private:
	MD5Data data;

public:
	MD5Data const& ToBinary() const { return data; }
	MD5(std::string const& str);
	MD5(std::string_view str);
	MD5(std::span<uint8_t const> bin);
	MD5(MD5 const&) = default;
	MD5(MD5&&) = default;
	MD5(MD5Data const& data);
	std::string ToString(bool upper = true) const;
	template<typename T>
	MD5& operator=(T&& t) {
		this->~MD5();
		new (this) MD5(std::forward<T>(t));
		return *this;
	}
	~MD5() = default;
	bool operator==(MD5 const& m) const;
	bool operator!=(MD5 const& m) const;
};
template<>
struct hash<MD5::MD5Data> {
	size_t operator()(MD5::MD5Data const& m) const {
		uint const* ptr = reinterpret_cast<uint const*>(&m.data0);
		auto endPtr = ptr + sizeof(MD5::MD5Data) / sizeof(uint);
		return Hash::Int32ArrayHash(ptr, endPtr);
	}
};
template<>
struct hash<MD5> {
	size_t operator()(MD5 const& m) const {
		static hash<MD5::MD5Data> dataHasher;
		return dataHasher(m.ToBinary());
	}
};
}// namespace vstd