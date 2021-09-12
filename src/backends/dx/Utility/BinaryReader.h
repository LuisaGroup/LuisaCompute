#pragma once
#include <Common/Common.h>
class BinaryReader {
private:
	struct FileSystemData {
		FILE* globalIfs;
		std::mutex* readMtx;
		uint64 offset;
	};
	bool isAvaliable = true;
	union {
		FileSystemData packageData;
		FILE* ifs;
	};
	uint64 length;
	uint64 currentPos;

public:
	BinaryReader(vstd::string const& path);
	void Read(char* ptr, uint64 len);
	operator bool() const {
		return isAvaliable;
	}
	bool operator!() const {
		return !isAvaliable;
	}
	void SetPos(uint64 pos);
	uint64 GetPos() const {
		return currentPos;
	}
	uint64 GetLength() const {
		return length;
	}
	~BinaryReader();
	VSTL_OVERRIDE_OPERATOR_NEW
	VSTL_DELETE_COPY_CONSTRUCT(BinaryReader)
};