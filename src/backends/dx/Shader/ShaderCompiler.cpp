#include <Shader/ShaderCompiler.h>
#include <core/dynamic_module.h>
#include <vstl/string_utility.h>
#ifndef NDEBUG
#include <vstl/binary_reader.h>
#endif
#include <core/logging.h>
namespace toolhub::directx {
DXByteBlob::DXByteBlob(
    ComPtr<IDxcBlob> &&b,
    ComPtr<IDxcResult> &&rr)
    : blob(std::move(b)),
      comRes(std::move(rr)) {}
std::byte *DXByteBlob::GetBufferPtr() const {
    return reinterpret_cast<std::byte *>(blob->GetBufferPointer());
}
size_t DXByteBlob::GetBufferSize() const {
    return blob->GetBufferSize();
}
static vstd::wstring GetSM(uint shaderModel) {
    vstd::string smStr;
    smStr << vstd::to_string(shaderModel / 10) << '_' << vstd::to_string(shaderModel % 10);
    vstd::wstring wstr;
    wstr.resize(smStr.size());
    for (auto i : vstd::range(smStr.size())) {
        wstr[i] = smStr[i];
    }
    return wstr;
}
IDxcCompiler3 *ShaderCompiler::Compiler() {
    std::lock_guard lck{moduleInstantiateMtx};
    if (module) return module->comp.Get();
    module.create(path);
    return module->comp.Get();
}
ShaderCompiler::~ShaderCompiler() {}
ShaderCompilerModule::ShaderCompilerModule(std::filesystem::path const &path)
    : dxil(luisa::DynamicModule::load(path, "dxil")),
      dxcCompiler(luisa::DynamicModule::load(path, "dxcompiler")) {
    if (!dxil) {
        LUISA_ERROR("dxil.dll not found.");
    }
    if (!dxcCompiler) {
        LUISA_ERROR("dxcompiler.dll not found.");
    }
    auto voidPtr = dxcCompiler.address("DxcCreateInstance"sv);
    HRESULT(__stdcall * DxcCreateInstance)
    (const IID &, const IID &, LPVOID *) =
        reinterpret_cast<HRESULT(__stdcall *)(const IID &, const IID &, LPVOID *)>(voidPtr);
    DxcCreateInstance(CLSID_DxcCompiler, IID_PPV_ARGS(comp.GetAddressOf()));
}
ShaderCompilerModule::~ShaderCompilerModule() {
    comp = nullptr;
}
ShaderCompiler::ShaderCompiler(std::filesystem::path const &path)
    : path(path) {}
CompileResult ShaderCompiler::Compile(
    vstd::string_view code,
    vstd::span<LPCWSTR> args) {
    DxcBuffer buffer{
        code.data(),
        code.size(),
        CP_ACP};
    ComPtr<IDxcResult> compileResult;

    ThrowIfFailed(Compiler()->Compile(
        &buffer,
        args.data(),
        args.size(),
        nullptr,
        IID_PPV_ARGS(compileResult.GetAddressOf())));
    HRESULT status;
    ThrowIfFailed(compileResult->GetStatus(&status));
    if (status == 0) {
        ComPtr<IDxcBlob> resultBlob;
        ThrowIfFailed(compileResult->GetResult(resultBlob.GetAddressOf()));
        return vstd::create_unique(new DXByteBlob(std::move(resultBlob), std::move(compileResult)));
    } else {
        ComPtr<IDxcBlobEncoding> errBuffer;
        ThrowIfFailed(compileResult->GetErrorBuffer(errBuffer.GetAddressOf()));
        auto errStr = vstd::string_view(
            reinterpret_cast<char const *>(errBuffer->GetBufferPointer()),
            errBuffer->GetBufferSize());
        return vstd::string(errStr);
    }
}
template<typename Vec>
static void AddCompileFlags(Vec &args) {
    vstd::push_back_all(
        args,
        {L"-Gfa",
         L"-all-resources-bound",
         L"-no-warnings",
         L"-HV 2021",
         L"-opt-enable",
         L"-funsafe-math-optimizations",
         L"-opt-enable",
         L"-fassociative-math",
         L"-opt-enable",
         L"-freciprocal-math"});
}
CompileResult ShaderCompiler::CompileCompute(
    vstd::string_view code,
    bool optimize,
    uint shaderModel) {
#ifndef NDEBUG
    if (shaderModel < 10) {
        LUISA_ERROR("Illegal shader model!");
    }
#endif
    vstd::fixed_vector<LPCWSTR, 32> args;
    vstd::wstring smStr;
    smStr << L"cs_" << GetSM(shaderModel);
    args.push_back(L"/T");
    args.push_back(smStr.c_str());
    AddCompileFlags(args);
    if (optimize) {
        args.push_back(L"-O3");
    }
    return Compile(code, args);
}
RasterBin ShaderCompiler::CompileRaster(
    vstd::string_view code,
    bool optimize,
    uint shaderModel) {
#ifndef NDEBUG
    if (shaderModel < 10) {
        LUISA_ERROR("Illegal shader model!");
    }
#endif
    vstd::fixed_vector<LPCWSTR, 32> args;
    AddCompileFlags(args);
    if (optimize) {
        args.push_back(L"-O3");
    }
    args.push_back(L"/T");
    auto size = args.size();
    vstd::wstring smStr;
    smStr << L"vs_" << GetSM(shaderModel);
    args.push_back(smStr.c_str());
    args.push_back(L"/DVS");
    RasterBin bin;
    bin.vertex = Compile(code, args);
    args.resize_uninitialized(size);
    smStr.clear();
    smStr << L"ps_" << GetSM(shaderModel);
    args.push_back(smStr.c_str());
    args.push_back(L"/DPS");
    bin.pixel = Compile(code, args);
    return bin;
}
#ifdef SHADER_COMPILER_TEST

CompileResult ShaderCompiler::CustomCompile(
    vstd::string_view code,
    vstd::span<vstd::string const> args) {
    vstd::vector<vstd::wstring> wstrArgs;
    vstd::push_back_func(
        wstrArgs,
        args.size(),
        [&](size_t i) {
            auto &&strv = args[i];
            vstd::wstring wstr;
            wstr.resize(strv.size());
            for (auto ite : vstd::range(strv.size())) {
                wstr[ite] = strv[ite];
            }
            return wstr;
        });
    vstd::vector<LPCWSTR> argPtr;
    vstd::push_back_func(
        argPtr,
        args.size(),
        [&](size_t i) {
            return wstrArgs[i].data();
        });

    return Compile(code, argPtr);
}
#endif
/*
CompileResult ShaderCompiler::CompileRayTracing(
    vstd::string_view code,
    bool optimize,
    uint shaderModel) {
    if (shaderModel < 10) {
        return "Illegal shader model!"_sv;
    }
    vstd::fixed_vector<LPCWSTR, 32> args;
    vstd::wstring smStr;
    smStr << L"lib_" << GetSM(shaderModel);
    args.push_back(L"/T");
    args.push_back(smStr.c_str());
    args.push_back_all(
        {L"-Qstrip_debug",
         L"-Qstrip_reflect",
         L"/enable_unbounded_descriptor_tables",
         L"-HV 2021"});
    if (optimize) {
        args.push_back(L"-O3");
    }
    return Compile(code, args);
}*/

}// namespace toolhub::directx