#pragma once

#include <vstl/VGuid.h>
#include <serialize/Common.h>
#include <serialize/SimpleParser.h>

namespace toolhub::db {

class IJsonDict;
class IJsonArray;

struct Disposer {
	void operator()(vstd::IDisposable* d) {
        d->Dispose();
	}
};

template <typename T>
using UniquePtr = std::unique_ptr<T, Disposer>;

template<typename T>
[[nodiscard]] inline auto MakeUnique(T* ptr) {
    return UniquePtr<T>(ptr, Disposer());
}

class IJsonDatabase : public vstd::IDisposable {
protected:
	~IJsonDatabase() = default;

public:
	virtual std::vector<uint8_t> Serialize() = 0;
	virtual bool Read(
		std::span<uint8_t const> data,
		bool clearLast) = 0;
	virtual std::string Print() = 0;
	virtual IJsonDict* GetRootNode() = 0;
	virtual UniquePtr<IJsonDict> CreateDict() = 0;
	virtual UniquePtr<IJsonArray> CreateArray() = 0;
	virtual vstd::optional<ParsingException> Parse(
		std::string_view str,
		bool clearLast) = 0;
	virtual vstd::MD5 GetMD5() = 0;
};

}// namespace toolhub::db
