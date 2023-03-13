
#include <vstl/log.h>
#include <mutex>
#include <cstdio>
#include <core/logging.h>
namespace LogGlobal {
static bool isInitialized = false;
static std::mutex mtx;
}// namespace LogGlobal
void VEngine_Log(std::string_view const &chunk) {
    LUISA_ERROR("{}", chunk);
}
void VEngine_Log(std::string_view const *chunk, size_t chunkCount) {
    vstd::string str;
    for (auto i : vstd::range(chunkCount)) {
        str << chunk[i];
    }
    LUISA_ERROR("{}", str);
}
void VEngine_Log(std::initializer_list<std::string_view> const &initList) {
    VEngine_Log(initList.begin(), initList.size());
}
void VEngine_Log(std::type_info const &t) {
    VEngine_Log(
        {t.name(),
         " runtime error! Usually did mistake operation, like vstd::optional\n"sv});
}

void VEngine_Log(char const *chunk) {
    VEngine_Log(std::string_view(chunk));
}