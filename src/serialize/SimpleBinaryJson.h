#pragma once

#include <vstl/VGuid.h>
#include <serialize/SimpleJsonValue.h>

namespace toolhub::db {

class SimpleBinaryJson final : public IJsonDatabase, public vstd::IOperatorNewBase {

private:
    SimpleJsonValueDict root;

public:
    SimpleBinaryJson();
    vstd::Pool<SimpleJsonValueArray, VEngine_AllocType::VEngine, true> arrValuePool;
    vstd::Pool<SimpleJsonValueDict, VEngine_AllocType::VEngine, true> dictValuePool;
    luisa::vector<uint8_t> Serialize() override;
    bool Read(
        std::span<uint8_t const> data,
        bool clearLast) override;
    std::optional<ParsingException> Parse(
        std::string_view str,
        bool clearLast) override;
    luisa::string Print() override;
    IJsonDict *GetRootNode() override;
    UniquePtr<IJsonDict> CreateDict() override;
    UniquePtr<IJsonArray> CreateArray() override;
    SimpleJsonValueDict *CreateDict_Nake();
    SimpleJsonValueArray *CreateArray_Nake();
    void Dispose() override {
        delete this;
    }
    vstd::MD5 GetMD5() override;
    VSTL_DELETE_COPY_CONSTRUCT(SimpleBinaryJson)
    VSTL_DELETE_MOVE_CONSTRUCT(SimpleBinaryJson)
#ifdef VENGINE_PYTHON_SUPPORT
    bool CompileFromPython(char const *code) override;
#endif
};

class ConcurrentBinaryJson final : public IJsonDatabase, public vstd::IOperatorNewBase {

private:
    ConcurrentJsonValueDict root;

public:
    ConcurrentBinaryJson();
    vstd::Pool<ConcurrentJsonValueArray, VEngine_AllocType::VEngine, true> arrValuePool;
    vstd::Pool<ConcurrentJsonValueDict, VEngine_AllocType::VEngine, true> dictValuePool;
    luisa::spin_mutex arrPoolMtx;
    luisa::spin_mutex dictPoolMtx;
    luisa::vector<uint8_t> Serialize() override;
    bool Read(
        std::span<uint8_t const> data,
        bool clearLast) override;
    std::optional<ParsingException> Parse(
        std::string_view str,
        bool clearLast) override;
    luisa::string Print() override;
    IJsonDict *GetRootNode() override;
    UniquePtr<IJsonDict> CreateDict() override;
    UniquePtr<IJsonArray> CreateArray() override;
    ConcurrentJsonValueDict *CreateDict_Nake();
    ConcurrentJsonValueArray *CreateArray_Nake();
    vstd::MD5 GetMD5() override;
    void Dispose() override {
        delete this;
    }
    VSTL_DELETE_COPY_CONSTRUCT(ConcurrentBinaryJson)
    VSTL_DELETE_MOVE_CONSTRUCT(ConcurrentBinaryJson)
};

}// namespace toolhub::db