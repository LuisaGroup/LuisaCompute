#pragma once
#include <util/vstl_config.h>
#include <util/MD5.h>
namespace vstd {
class VENGINE_DLL_COMMON Guid {
public:
	struct GuidData {
		uint64 data0;
		uint64 data1;
	};

private:
	GuidData data;

public:
	friend VENGINE_DLL_COMMON std::ostream& operator<<(std::ostream& out, const Guid& obj) noexcept;

	explicit Guid(bool generate);
	Guid(std::string_view strv);
	Guid(std::span<uint8_t> data);
	Guid(MD5 const& md5) {
		auto&& bin = md5.ToBinary();
		data.data0 = bin.data0;
		data.data1 = bin.data1;
	}
	operator MD5() const {
		return MD5(MD5::MD5Data{data.data0, data.data1});
	}
	Guid(std::array<uint8_t, sizeof(GuidData)> const& data);
	Guid(GuidData const& d);
	void ReGenerate();
	Guid(Guid const&) = default;
	Guid(Guid&&) = default;
	void Reset() {
		data.data0 = 0;
		data.data1 = 0;
	}
	GuidData const& ToBinary() const { return data; }
	std::array<uint8_t, sizeof(GuidData)> ToArray() const;
	std::string ToString(bool upper = true) const;
	void ToString(char* result, bool upper = true) const;
	std::string ToCompressedString() const;
	void ToCompressedString(char* result) const;
	inline bool operator==(Guid const& d) const {
		return data.data0 == d.data.data0 && data.data1 == d.data.data1;
	}
	inline bool operator!=(Guid const& d) const {
		return !operator==(d);
	}
	inline Guid& operator=(Guid const& d) {
		this->~Guid();
		new (this) Guid(d);
		return *this;
	}
	inline Guid& operator=(GuidData const& d) {
		this->~Guid();
		new (this) Guid(d);
		return *this;
	}
	inline operator bool() const {
		return data.data0 != 0 && data.data1 != 0;
	}
	inline bool operator!() const {
		return !(operator bool());
	}
};
template<>
struct hash<Guid> {
	size_t operator()(Guid const& guid) const {
		uint const* ptr = reinterpret_cast<uint const*>(&guid.ToBinary().data0);
		return Hash::Int32ArrayHash(
			ptr,
			ptr + sizeof(Guid::GuidData) / sizeof(uint));
	}
};
}// namespace vstd