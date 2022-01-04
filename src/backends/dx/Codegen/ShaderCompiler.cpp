#pragma vengine_package vengine_directx
#include <core/dynamic_module.h>
#include <Codegen/ShaderCompiler.h>
namespace toolhub::directx {
namespace detail {
static luisa::DynamicModule dynaModule(".", "VEngine_Compiler");
template<typename T>
struct GetFuncPtrType;
template<typename Ret, typename... Args>
struct GetFuncPtrType<Ret (*)(Args...)> {
    using Type = Ret(Args...);
};
template<typename T>
using GetFuncPtrType_t = typename GetFuncPtrType<T>::Type;
struct Compiler_Func {
    void *(*GetCompiler)();
    bool (*CompileCompute)(
        void *,
        char const *,
        size_t,
        uint,
        ShaderType,
        bool,
        void *&);
    void *(*GetCompileResultBlob)(
        void *);
    void *(*GetCompileResultError)(
        void *);
    void (*GetBlobData)(
        void *,
        void const *&,
        size_t &);
    void (*GetEncodingBlobData)(
        void *,
        char const *&,
        size_t &);
    void (*DeleteResult)(void *);
    void (*DeleteBlob)(void *);
    void (*DeleteEncodingBlob)(void *);
    void (*DeleteCompiler)(void *);
    Compiler_Func() {
        auto get = [&](auto &&func, char const *name) {
            func = dynaModule.function<GetFuncPtrType_t<std::remove_cvref_t<decltype(func)>>>(name);
        };
#define GET(x) get(x, #x)
        void (*SetMemoryFunc)(
            void *(*mallocFunc)(size_t),
            void (*freeFunc)(void *));
        GET(SetMemoryFunc);
        GET(GetCompiler);
        GET(CompileCompute);
        GET(GetCompileResultBlob);
        GET(GetCompileResultError);
        GET(GetBlobData);
        GET(GetEncodingBlobData);
        GET(DeleteResult);
        GET(DeleteBlob);
        GET(DeleteEncodingBlob);
        GET(DeleteCompiler);
        SetMemoryFunc(
            vengine_malloc,
            vengine_free);
    }
};
static Compiler_Func compFunc;
}// namespace detail

ShaderCompiler::ShaderCompiler() {
    using namespace detail;
    compiler = compFunc.GetCompiler();
}
ShaderCompiler::~ShaderCompiler() {
    using namespace detail;
    if (compiler)
        compFunc.DeleteCompiler(compiler);
}
CompileResult::~CompileResult() {
    using namespace detail;
    if (ptr)
        compFunc.DeleteResult(ptr);
}
CompileResult ShaderCompiler::Compile(
    vstd::string_view str,
    ShaderType shaderType,
    bool optimize,
    uint shaderModel) {
    using namespace detail;
    void *result = nullptr;
    bool v = compFunc.CompileCompute(
        compiler,
        str.data(),
        str.size(),
        shaderModel,
        shaderType,
        optimize,
        result);
    return {result, v};
}
vstd::variant<
    ByteBlob,
    CharBlob>
CompileResult::GetResult() {
    using namespace detail;
    if (isSuccess) {
        return ByteBlob(
            compFunc.GetCompileResultBlob(ptr));
    } else {
        return CharBlob(
            compFunc.GetCompileResultError(ptr));
    }
}
CharBlob::~CharBlob() {
    using namespace detail;
    if (ptr)
        compFunc.DeleteEncodingBlob(ptr);
}
vstd::string_view CharBlob::GetString() const {
    using namespace detail;
    char const *charPtr;
    size_t size;
    compFunc.GetEncodingBlobData(
        ptr,
        charPtr,
        size);
    return {
        charPtr,
        size};
}
vstd::span<vbyte const> ByteBlob::GetData() const {
    using namespace detail;
    void const *dataPtr;
    size_t size;
    compFunc.GetBlobData(
        ptr,
        dataPtr,
        size);
    return {
        reinterpret_cast<vbyte const *>(dataPtr),
        size};
}

ByteBlob::~ByteBlob() {
    using namespace detail;
    if (ptr)
        compFunc.DeleteBlob(ptr);
}
}// namespace toolhub::directx