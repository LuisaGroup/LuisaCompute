#pragma once
#include <vstl/common.h>
#include <EASTL/vector.h>
#ifdef _WIN32
#define VSTD_FSEEK _fseeki64_nolock
#define VSTD_FTELL _ftelli64_nolock
#define VSTD_FREAD _fread_nolock
#define VSTD_FWRITE _fwrite_nolock
#define VSTD_FCLOSE _fclose_nolock
#else
#define VSTD_FSEEK fseek
#define VSTD_FTELL ftell
#define VSTD_FREAD fread
#define VSTD_FWRITE fwrite
#define VSTD_FCLOSE fclose
#endif
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