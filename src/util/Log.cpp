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

void vstl_log(char const *chunk) {
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

void vstl_log(char const *const *chunks, size_t chunkCount) {
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

void vstl_log(std::initializer_list<char const *> initList) {
    vstl_log(initList.begin(), initList.size());
}

void vstl_log(std::type_info const &t) {
    vstl_log(
        {t.name(),
         " runtime error! Usually did mistake operation, like std::optional\n"});
}

void vstl_log_error_pure_virtual(Type tarType) {
    std::string d = "Try call pure virtual function in ";
    d += tarType.GetType().name();
    d += '\n';
    vstl_log(d.c_str());
    VSTL_ABORT();
}

}// namespace vstd