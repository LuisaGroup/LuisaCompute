#pragma vengine_package ispc_vsproject

#include "ispc_shader.h"
#include "ispc_codegen.h"
#include <vstl/file_system.h>
#include <vstl/MD5.h>
namespace lc::ispc {
////////////////////////// Windows use DLL
#ifdef LUISA_PLATFORM_WINDOWS
namespace detail {
static std::string_view FOLDER_NAME = "backends\\ispc_support\\";
static void GenerateDll(
    std::string_view code,
    luisa::string const &fileName,
    luisa::string const &libName) {
    //write text
    luisa::string textName = fileName + ".txt";
    {
        auto f = fopen(textName.c_str(), "wb");
        if (f) {
            auto disp = vstd::create_disposer([&] { fclose(f); });
            fwrite(code.data(), code.size(), 1, f);
        }
    }
    //compile
    luisa::string compileCmd(detail::FOLDER_NAME);
    compileCmd << "ispc.exe -O2 "
               << textName << " -o " << fileName << ".obj -woff";
    system(compileCmd.c_str());
    compileCmd.clear();
    compileCmd << detail::FOLDER_NAME
               << "link.exe /DLL /NOLOGO /OUT:"sv
               << libName
               << " /DYNAMICBASE \""
               << detail::FOLDER_NAME
               << "msvcrt.lib\" /NOENTRY /EXPORT:run /NODEFAULTLIB "sv
               << fileName
               << ".obj"sv;
    system(compileCmd.c_str());
    remove((fileName + ".obj").c_str());
    //remove((fileName + ".txt").c_str());
    remove((fileName + ".exp").c_str());
    remove((fileName + ".lib").c_str());
    //link
}
static luisa::string CompileCode(std::string_view code) {
    vstd::MD5 md5(code);
    luisa::string fileName;
    fileName << detail::FOLDER_NAME << md5.ToString();
    constexpr size_t MD5_BASE64_STRLEN = 22;
    fileName.resize(MD5_BASE64_STRLEN + detail::FOLDER_NAME.size());
    luisa::string dllName = fileName + ".dll";
    if (!vstd::FileSystem::IsFileExists(dllName)) {
        GenerateDll(code, fileName, dllName);
    }
    return fileName;
}
}// namespace detail
class WinDllExecutable : public vstd::IOperatorNewBase {
public:
    DynamicModule dllModule;
    using FuncType = void(
        uint, //blk_cX,
        uint, //blk_cY,
        uint, //blk_cZ,
        uint, // thd_idX,
        uint, //thd_idY,
        uint, // thd_idZ,
        uint, //dsp_cX
        uint, //dsp_cY
        uint, //dsp_cZ
        uint64// arg
    );
    vstd::funcPtr_t<FuncType> exportFunc;
    WinDllExecutable(Function func, std::string_view strv)
        : dllModule(strv) {
        exportFunc = dllModule.function<FuncType>("run");
    }
    void Execute(
        uint3 const &blockCount,
        uint3 const &blockSize,
        uint3 const &dispatchCount,
        void *args) const {
        exportFunc(
            blockCount.x,
            blockCount.y,
            blockCount.z,
            blockSize.x,
            blockSize.y,
            blockSize.z,
            dispatchCount.x,
            dispatchCount.y,
            dispatchCount.z,
            (uint64)args);
    }
};
static void *GetExecutable(Function func) {
    luisa::string binName;
    luisa::string result;
    CodegenUtility::PrintFunction(func, result, func.block_size());
    binName = detail::CompileCode(result);
    return new WinDllExecutable(func, binName);
}
Shader::~Shader() {
    delete reinterpret_cast<WinDllExecutable *>(executable);
}
#else
Shader::~Shader() {
}
////////////////////////// TODO: LLVM backend
#endif
Shader::Shader(
    Function func)
    : func(func),
      executable(GetExecutable(func)) {
    size_t sz = 0;
    for (auto &&i : func.arguments()) {
        varIdToArg.Emplace(i.uid(), sz);
        sz++;
    }
}
size_t Shader::GetArgIndex(uint varID) const {
    auto ite = varIdToArg.Find(varID);
    if (!ite) return std::numeric_limits<size_t>::max();
    return ite.Value();
}

ThreadTaskHandle Shader::dispatch(
    ThreadPool *tPool,
    uint3 sz,
    ArgVector vec,
    bool needSync) const {
    auto blockSize = func.block_size();
    auto blockCount = (sz + blockSize - uint3(1, 1, 1)) / blockSize;
    auto totalCount = blockCount.x * blockCount.y * blockCount.z;
    auto sharedCounter = vstd::MakeObjectPtr(vengine_new<std::atomic_uint, uint>(0), [](void *ptr) { vengine_free(ptr); });
    auto handle = tPool->GetParallelTask(
        [=, vec = std::move(vec), sharedCounter = std::move(sharedCounter)](size_t) {
            auto &&counter = *sharedCounter;
            for (auto i = counter.fetch_add(1u); i < totalCount; i = counter.fetch_add(1u)) {
                uint blockIdxZ = i / (blockCount.y * blockCount.x);
                i -= blockCount.y * blockCount.x * blockIdxZ;
                uint blockIdxY = i / blockCount.x;
                i -= blockIdxY * blockCount.x;
                uint blockIdxX = i;
#ifdef LUISA_PLATFORM_WINDOWS
                reinterpret_cast<WinDllExecutable *>(executable)->Execute(blockCount, make_uint3(blockIdxX, blockIdxY, blockIdxZ), sz, vec.data());
#else
//LLVM
#endif
            }
        },
        std::thread::hardware_concurrency(),
        needSync);
    return handle;
    //exportFunc(sz.x, sz.y, sz.z, (uint64)vec.data());
}
}// namespace lc::ispc
