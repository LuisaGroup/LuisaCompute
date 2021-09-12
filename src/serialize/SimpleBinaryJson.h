#pragma once
#pragma once
#include <util/VGuid.h>
#include <serialize/SimpleJsonValue.h>
namespace toolhub::db {

class SimpleBinaryJson final : public IJsonDatabase, public vstd::IOperatorNewBase {
protected:
public:
    vstd::StackObject<SimpleJsonValueDict> root;
    //SimpleJsonValueDict root: error
    SimpleBinaryJson();
    ~SimpleBinaryJson();
    vstd::Pool<SimpleJsonValueArray, VEngine_AllocType::VEngine, true> arrValuePool;
    vstd::Pool<SimpleJsonValueDict, VEngine_AllocType::VEngine, true> dictValuePool;
    std::vector<uint8_t> Serialize() override;
    bool Read(
        std::span<uint8_t const> data,
        bool clearLast) override;
    vstd::optional<ParsingException> Parse(
        std::string_view str,
        bool clearLast) override;
    std::string Print() override;
    IJsonDict *GetRootNode() override;
    UniquePtr<IJsonDict> CreateDict() override;
    UniquePtr<IJsonArray> CreateArray() override;
    SimpleJsonValueDict *CreateDict_Nake();
    SimpleJsonValueArray *CreateArray_Nake();
    void Dispose() override {
        delete this;
    }
    vstd::MD5 GetMD5() override;
    KILL_COPY_CONSTRUCT(SimpleBinaryJson)
    KILL_MOVE_CONSTRUCT(SimpleBinaryJson)
};
class ConcurrentBinaryJson final : public IJsonDatabase, public vstd::IOperatorNewBase {
protected:
public:
    vstd::StackObject<ConcurrentJsonValueDict> root;
    //ConcurrentJsonValueDict root: error
    ConcurrentBinaryJson();
    ~ConcurrentBinaryJson();
    vstd::Pool<ConcurrentJsonValueArray, VEngine_AllocType::VEngine, true> arrValuePool;
    vstd::Pool<ConcurrentJsonValueDict, VEngine_AllocType::VEngine, true> dictValuePool;
    luisa::spin_mutex arrPoolMtx;
    luisa::spin_mutex dictPoolMtx;
    std::vector<uint8_t> Serialize() override;
    bool Read(
        std::span<uint8_t const> data,
        bool clearLast) override;
    vstd::optional<ParsingException> Parse(
        std::string_view str,
        bool clearLast) override;
    std::string Print() override;
    IJsonDict *GetRootNode() override;
    UniquePtr<IJsonDict> CreateDict() override;
    UniquePtr<IJsonArray> CreateArray() override;
    ConcurrentJsonValueDict *CreateDict_Nake();
    ConcurrentJsonValueArray *CreateArray_Nake();
    vstd::MD5 GetMD5() override;
    void Dispose() override {
        delete this;
    }
    KILL_COPY_CONSTRUCT(ConcurrentBinaryJson)
    KILL_MOVE_CONSTRUCT(ConcurrentBinaryJson)
};
}// namespace toolhub::db