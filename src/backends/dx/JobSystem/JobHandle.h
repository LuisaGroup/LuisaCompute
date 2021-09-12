#pragma once
#include <vstl/vstlconfig.h>
#include <vstl/vector.h>
#include <span>
class JobNode;
class JobBucket;
class LUISA_DLL JobHandle {
	friend class JobBucket;
	friend class JobNode;

private:
	JobBucket* bucket;
	size_t start;
	size_t end;

public:
	JobHandle(
		JobBucket* bucket,
		size_t start,
		size_t end)
		: start(start),
		  bucket(bucket),
		  end(end) {
	}
	JobHandle() : bucket(nullptr), start(-1), end(-1) {}
	operator bool() const noexcept {
		return start != -1;
	}
	bool operator!() const {
		return !operator bool();
	}
	size_t Count() const {
		return end + 1 - start;
	}
	void Reset() { start = -1; }
	void AddDependency(JobHandle const& handle);
	void AddDependency(std::initializer_list<JobHandle const> handle);
	void AddDependency(JobHandle const* handles, size_t handleCount);
};
