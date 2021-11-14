#pragma vengine_package ispc_vsproject
#include <backends/ispc/runtime/ispc_compiler.h>
#ifdef _WIN32

namespace lc::ispc {
namespace detail {
static std::string_view FOLDER_NAME = "ispc_backend\\";
}
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
luisa::string Compiler::CompileCode(
    std::string_view code) const {
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
}// namespace lc::ispc
#endif
//TODO: other platforms