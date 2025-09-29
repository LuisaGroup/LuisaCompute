#include "shader_compiler.h"
#include <luisa/core/dynamic_module.h>
#include <luisa/vstl/string_utility.h>
#include <luisa/core/logging.h>
#include <luisa/vstl/spin_mutex.h>
#ifndef _WIN32
#define WINAPI
#endif
namespace lc::hlsl {
#ifndef LC_DXC_THROW_IF_FAILED
#define LC_DXC_THROW_IF_FAILED(x)                  \
    {                                              \
        HRESULT hr_ = (x);                         \
        LUISA_ASSERT(hr_ == S_OK, "bad HRESULT."); \
    }
#endif
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
IDxcCompiler3 *ShaderCompiler::compiler() {
    return compiler_module.comp;
}
IDxcUtils *ShaderCompiler::utils() {
    return compiler_module.utils;
}
IDxcLibrary *ShaderCompiler::library() {
    return compiler_module.library;
}
ShaderCompiler::~ShaderCompiler() {
}
ShaderCompilerModule::ShaderCompilerModule(std::filesystem::path const &path)
    : dxil(luisa::DynamicModule::load(path, "dxil")),
      dxcCompiler(luisa::DynamicModule::load(path, "dxcompiler")) {
    if (!dxil) {
        LUISA_ERROR("dxil.dll not found.");
    }
    if (!dxcCompiler) {
        LUISA_ERROR("dxcompiler.dll not found.");
    }
    auto voidPtr = dxcCompiler.address("DxcCreateInstance");
    HRESULT(WINAPI * DxcCreateInstance)
    (const IID &, const IID &, LPVOID *) =
        reinterpret_cast<HRESULT(WINAPI *)(const IID &, const IID &, LPVOID *)>(voidPtr);
    DxcCreateInstance(CLSID_DxcCompiler, IID_PPV_ARGS(&comp));
    DxcCreateInstance(CLSID_DxcLibrary, IID_PPV_ARGS(&library));
    DxcCreateInstance(CLSID_DxcUtils, IID_PPV_ARGS(&utils));
}
ShaderCompilerModule::~ShaderCompilerModule() {
    // TODO: directx-compiler may crash here
    utils->Release();
    library->Release();
    comp->Release();
}
ShaderCompiler::ShaderCompiler(std::filesystem::path const &path)
    : compiler_module(path) {
}
CompileResult ShaderCompiler::compile(
    vstd::string_view code,
    vstd::span<LPCWSTR> args) {
    DxcBuffer buffer{
        code.data(),
        code.size(),
        CP_ACP};
    ComUniquePtr<IDxcResult> compileResult;
    auto comp = compiler();
    if (comp) {
        IDxcResult *ptr{};
        auto compile_result = comp->Compile(
            &buffer,
            args.data(),
            args.size(),
            nullptr,
            IID_PPV_ARGS(&ptr));
        LC_DXC_THROW_IF_FAILED(compile_result);
        compileResult = ComUniquePtr<IDxcResult>{ptr};
    }
    HRESULT status;
    LC_DXC_THROW_IF_FAILED(compileResult->GetStatus(&status));
    if (status == 0) {
        ComUniquePtr<IDxcBlob> resultBlob;
        IDxcBlob *resultBlobPtr{};
        LC_DXC_THROW_IF_FAILED(compileResult->GetResult(&resultBlobPtr));
        resultBlob = ComUniquePtr<IDxcBlob>{resultBlobPtr};
        return resultBlob;
    } else {
        ComUniquePtr<IDxcBlobEncoding> errBuffer;
        IDxcBlobEncoding *errBufferPtr{};
        LC_DXC_THROW_IF_FAILED(compileResult->GetErrorBuffer(&errBufferPtr));
        errBuffer = ComUniquePtr<IDxcBlobEncoding>{errBufferPtr};
        auto errStr = vstd::string_view(
            reinterpret_cast<char const *>(errBuffer->GetBufferPointer()),
            errBuffer->GetBufferSize());
        return vstd::string(errStr);
    }
}
template<typename Vec>
static void AddCompileFlags(Vec &args, bool debug) {
    vstd::push_back_all(
        args,
        {DXC_ARG_ALL_RESOURCES_BOUND,
         L"-enable-16bit-types",
         DXC_ARG_PACK_MATRIX_ROW_MAJOR,
         DXC_ARG_AVOID_FLOW_CONTROL,
         L"-HV 2021"});
    if (debug) {
        args.emplace_back(DXC_ARG_DEBUG);
    } else {
        args.emplace_back(L"-no-warnings");
    }
}
template<typename Vec>
static void AddUnsafeMathFlags(Vec &args) {
    // unsafe opt may conflict with dxc
    // vstd::push_back_all(
    //     args,
    //     {L"-opt-enable",
    //      L"-funsafe-math-optimizations",
    //      L"-opt-enable",
    //      L"-fassociative-math",
    //      L"-opt-enable",
    //      L"-freciprocal-math"});
}
CompileResult ShaderCompiler::compile_compute(
    vstd::string_view code,
    bool optimize,
    uint shaderModel,
    bool enableUnsafeMath,
    bool spirv,
    bool debug) {
#ifndef NDEBUG
    if (shaderModel < 10) {
        LUISA_ERROR("Illegal shader model!");
    }
#endif
    vstd::fixed_vector<LPCWSTR, 32> args;
    vstd::wstring smStr;
    smStr << L"cs_" << GetSM(shaderModel);
    if (spirv) {
        args.emplace_back(L"-spirv");
        args.emplace_back(L"/DSPV");
        if (shaderModel > 60) {
            args.emplace_back(L"-fspv-target-env=vulkan1.1");
        } else if (shaderModel > 65) {
            args.emplace_back(L"-fspv-target-env=vulkan1.3");
        }
    }
    args.emplace_back(L"-T");
    args.emplace_back(smStr.c_str());
    AddCompileFlags(args, debug);
    if (enableUnsafeMath) {
        AddUnsafeMathFlags(args);
    }
    if (optimize) {
        args.emplace_back(DXC_ARG_OPTIMIZATION_LEVEL3);
    }
    return compile(code, args);
}
RasterBin ShaderCompiler::compile_raster(
    vstd::string_view code,
    bool optimize,
    uint shaderModel,
    bool enableUnsafeMath,
    bool spirv,
    bool debug) {
#ifndef NDEBUG
    if (shaderModel < 10) {
        LUISA_ERROR("Illegal shader model!");
    }
#endif
    vstd::fixed_vector<LPCWSTR, 32> args;
    AddCompileFlags(args, debug);
    if (spirv) {
        args.emplace_back(L"/DSPV");
        args.emplace_back(L"-spirv");
        if (shaderModel > 60) {
            args.emplace_back(L"-fspv-target-env=vulkan1.1");
        } else if (shaderModel > 65) {
            args.emplace_back(L"-fspv-target-env=vulkan1.3");
        }
    }
    if (enableUnsafeMath) {
        AddUnsafeMathFlags(args);
    }
    if (optimize) {
        args.emplace_back(DXC_ARG_OPTIMIZATION_LEVEL3);
    }
    args.emplace_back(L"-T");
    auto size = args.size();
    vstd::wstring smStr;
    smStr << L"vs_" << GetSM(shaderModel);
    args.emplace_back(smStr.c_str());
    args.emplace_back(L"/DVS");
    RasterBin bin;
    bin.vertex = compile(code, args);
    luisa::vector_resize(args, size);
    smStr.clear();
    smStr << L"ps_" << GetSM(shaderModel);
    args.emplace_back(smStr.c_str());
    args.emplace_back(L"/DPS");
    bin.pixel = compile(code, args);
    return bin;
}
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
    args.emplace_back(L"-T");
    args.emplace_back(smStr.c_str());
    args.push_back_all(
        {L"-Qstrip_debug",
         L"-Qstrip_reflect",
         L"/enable_unbounded_descriptor_tables",
         L"-HV 2021"});
    if (optimize) {
        args.emplace_back(DXC_ARG_OPTIMIZATION_LEVEL3);
    }
    return compile(code, args);
}*/
#undef LC_DXC_THROW_IF_FAILED
}// namespace lc::hlsl
