#pragma once
#include <vstl/common.h>
#include <vstl/md5.h>
namespace vstd {
class LC_VSTL_API Guid {
	friend class StackObject<Guid, false>;

public:
	struct GuidData {
		uint64 data0;
		uint64 data1;
	};

private:
	GuidData data;
	Guid() {}

public:

	explicit Guid(bool generate);
	Guid(std::string_view strv);
	static optional<Guid> TryParseGuid(std::string_view strv);
	Guid(span<uint8_t> data);
	Guid(MD5 const& md5) {
		auto&& bin = md5.to_binary();
		data.data0 = bin.data0;
		data.data1 = bin.data1;
	}
	operator MD5() const {
		return MD5(MD5::MD5Data{data.data0, data.data1});
	}
	Guid(std::array<uint8_t, sizeof(GuidData)> const& data);
	Guid(GuidData const& d);
	void remake();
	Guid(Guid const&) = default;
	Guid(Guid&&) = default;
	void reset() {
		data.data0 = 0;
		data.data1 = 0;
	}
	GuidData const& to_binary() const { return data; }
	std::array<uint8_t, sizeof(GuidData)> ToArray() const;
	string to_string(bool upper = true) const;
	void to_string(char* result, bool upper = true) const;
	string to_base64() const;
	void to_base64(char* result) const;

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
		return Hash::binary_hash(
			&guid.to_binary().data0,
			sizeof(Guid::GuidData));
	}
};
template<>
struct compare<Guid> {
	int32 operator()(Guid const& a, Guid const& b) const{
		if (a.to_binary().data0 > b.to_binary().data0) return 1;
		if (a.to_binary().data0 < b.to_binary().data0) return -1;
		if (a.to_binary().data1 > b.to_binary().data1) return 1;
		if (a.to_binary().data1 < b.to_binary().data1) return -1;
		return 0;
	}
};

}// namespace vstd