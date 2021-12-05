#pragma once
#include <vstl/Common.h>
class VENGINE_DLL_COMMON BinaryReader
{
private:
	struct FileSystemData
	{
		FILE* globalIfs;
		std::mutex* readMtx;
		uint64 offset;
	};
	bool isAvaliable = true;
	union
	{
		FileSystemData packageData;
		FILE* ifs;
	};
	uint64 length;
	uint64 currentPos;
public:
	BinaryReader(vstd::string const& path);
	void Read(char* ptr, uint64 len);
	vstd::vector<uint8_t> Read(bool addNullEnd = false);
	inline operator bool() const {
		return isAvaliable;
	}
	inline bool operator!() const {
		return !operator bool();
	}
	void SetPos(uint64 pos);
	inline uint64 GetPos() const
	{
		return currentPos;
	}
	inline uint64 GetLength() const
	{
		return length;
	}
	~BinaryReader();
//	DECLARE_VENGINE_OVERRIDE_OPERATOR_NEW
	KILL_COPY_CONSTRUCT(BinaryReader)
};