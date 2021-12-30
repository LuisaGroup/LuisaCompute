#pragma once
#include <vstl/Common.h>
namespace toolhub::directx {
enum class ShaderType : uint8_t {
    Compute,
    RayTracing
};
class CompileResult;
class ByteBlob {
    friend class CompileResult;
    void *ptr;
    ByteBlob(void *ptr)
        : ptr(ptr) {}

public:
    ByteBlob(ByteBlob const &) = delete;
    ByteBlob(ByteBlob &&o) {
        ptr = o.ptr;
        o.ptr = nullptr;
    }
    ~ByteBlob();
    vstd::span<vbyte const> GetData() const;
};
class CharBlob {
    friend class CompileResult;
    void *ptr;
    CharBlob(void *ptr)
        : ptr(ptr) {}

public:
    CharBlob(CharBlob const &) = delete;
    CharBlob(CharBlob &&o) {
        ptr = o.ptr;
        o.ptr = nullptr;
    }
    ~CharBlob();
    vstd::string_view GetString() const;
};
class ShaderCompiler;
class CompileResult {
    friend class ShaderCompiler;
    void *ptr;
    bool isSuccess;
    CompileResult(void *ptr, bool isSuccess)
        : ptr(ptr), isSuccess(isSuccess) {}

public:
    vstd::variant<
        ByteBlob,
        CharBlob>
    GetResult();
    ~CompileResult();
};
class ShaderCompiler {
    void *compiler;

public:
    ShaderCompiler();
    ~ShaderCompiler();
    CompileResult Compile(
        vstd::string_view str,
        ShaderType shaderType,
        bool optimize,
        uint shaderModel = 63);
};
}// namespace toolhub::directx