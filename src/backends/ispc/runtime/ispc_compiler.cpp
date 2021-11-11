#pragma vengine_package ispc_vsproject
#include <backends/ispc/runtime/ispc_compiler.h>
#ifdef _WIN32
namespace lc::ispc {
static void GenerateDll(
    std::string_view code,
    std::string const &fileName,
    std::string const &libName) {
    //write text
    std::string textName = fileName + ".txt";
    {
        auto f = fopen(textName.c_str(), "wb");
        if (f) {
            auto disp = vstd::create_disposer([&] { fclose(f); });
            fwrite(code.data(), code.size(), 1, f);
        }
    }
    //compile
    std::string compileCmd = "ispc.exe -O2 ";
    compileCmd << textName << " -o " << fileName << ".obj -woff";
    system(compileCmd.c_str());
    compileCmd.clear();
    compileCmd << "link.exe /DLL /NOLOGO /OUT:"sv
               << libName
               << " /DYNAMICBASE msvcrt.lib /NOENTRY /NODEFAULTLIB run_ispc.exp "sv
               << fileName
               << ".obj"sv;
    system(compileCmd.c_str());
    compileCmd.clear();
    compileCmd << "del " << fileName << ".obj"sv;
    system(compileCmd.c_str());
    compileCmd.clear();
    compileCmd << "del " << fileName << ".txt"sv;
    system(compileCmd.c_str());
    //link
}
std::string Compiler::CompileCode(
    std::string_view code) const {
    vstd::MD5 md5(code);
    std::string fileName;
    vstd::StringUtil::EncodeToBase64({(uint8_t const *)(&md5.ToBinary()), sizeof(vstd::MD5::MD5Data)}, fileName);
    constexpr size_t MD5_BASE64_STRLEN = 22;
    fileName.resize(MD5_BASE64_STRLEN);
    std::string dllName = fileName + ".dll";
    if (!vstd::FileSystem::IsFileExists(dllName)) {
        GenerateDll(code, fileName, dllName);
    }
    return fileName;
}
}// namespace lc::ispc
#endif
//TODO: other platforms