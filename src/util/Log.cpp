#pragma vengine_package vengine_dll
#include <util/Log.h>
#include <mutex>
#include <cstdio>
#include <string>
namespace vstd {

namespace LogGlobal {
static bool isInitialized = false;
static std::mutex mtx;
}// namespace LogGlobal
void VEngine_Log(char const *chunk) {
    using namespace LogGlobal;
    std::lock_guard<decltype(mtx)> lck(mtx);
    FILE *file = nullptr;
    if (!isInitialized) {
        isInitialized = true;
        file = fopen("LoggerFile.log", "w");
        if (file) {
            auto cc = "This is a log file from last run: \n";
            fwrite(cc, strlen(cc), 1, file);
        }
    } else {
        file = fopen("LoggerFile.log", "a+");
    }
    if (file) {
        fwrite(chunk, strlen(chunk), 1, file);
        fclose(file);
    }
}
void VEngine_Log(char const *const *chunks, size_t chunkCount) {
    using namespace LogGlobal;
    std::lock_guard<decltype(mtx)> lck(mtx);
    FILE *file = nullptr;
    if (!isInitialized) {
        isInitialized = true;
        file = fopen("LoggerFile.log", "w");
        if (file) {
            auto cc = "This is a log file from last run: \n";
            fwrite(cc, strlen(cc), 1, file);
        }
    } else {
        file = fopen("LoggerFile.log", "a+");
    }
    if (file) {
        for (size_t i = 0; i < chunkCount; ++i)
            fwrite(chunks[i], strlen(chunks[i]), 1, file);
        fclose(file);
    }
}
void VEngine_Log(std::initializer_list<char const *> initList) {
    VEngine_Log(initList.begin(), initList.size());
}
void VEngine_Log(std::type_info const &t) {
    VEngine_Log(
        {t.name(),
         " runtime error! Usually did mistake operation, like std::optional\n"});
}

void VEngine_Log_PureVirtual(Type tarType) {
    std::string d = "Try call pure virtual function in ";
    d += tarType.GetType().name();
    d += '\n';
    VEngine_Log(d.c_str());
    VSTL_ABORT();
}
}// namespace vstd