#pragma once
#include <vstl/common.h>
#include <EASTL/vector.h>
class LC_VSTL_API BinaryReader {
private:
    struct FileSystemData {
        FILE *globalIfs;
        std::mutex *readMtx;
        uint64 offset;
    };
    bool isAvaliable = true;
    union {
        FileSystemData packageData;
        FILE *ifs;
    };
    uint64 length;
    uint64 currentPos;

public:
    BinaryReader(vstd::string const &path);
    void Read(void *ptr, uint64 len);
    luisa::vector<uint8_t> Read(bool addNullEnd = false);
    vstd::string ReadToString();
    inline operator bool() const {
        return isAvaliable;
    }
    inline bool operator!() const {
        return !operator bool();
    }
    inline uint64 GetPos() const {
        return currentPos;
    }
    inline uint64 GetLength() const {
        return length;
    }
    ~BinaryReader();
    //	DECLARE_VENGINE_OVERRIDE_OPERATOR_NEW
    KILL_COPY_CONSTRUCT(BinaryReader)
};