
#include <vstl/log.h>
#include <mutex>
#include <cstdio>
#include <core/logging.h>
namespace LogGlobal {
static bool isInitialized = false;
static std::mutex mtx;
}// namespace LogGlobal
void vengine_log(std::string_view const &chunk) {
    LUISA_ERROR("{}", chunk);
}
void vengine_log(std::string_view const *chunk, size_t chunkCount) {
    vstd::string str;
    for (auto i : vstd::range(chunkCount)) {
        str << chunk[i];
    }
    LUISA_ERROR("{}", str);
}
void vengine_log(std::initializer_list<std::string_view> const &initList) {
    vengine_log(initList.begin(), initList.size());
}
void vengine_log(std::type_info const &t) {
    vengine_log(
        {t.name(),
         " runtime error! Usually did mistake operation, like vstd::optional\n"sv});
}

void vengine_log(char const *chunk) {
    vengine_log(std::string_view(chunk));
}