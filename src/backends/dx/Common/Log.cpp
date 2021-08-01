#pragma vengine_package vengine_dll
#include <Common/vstring.h>
#include <mutex>
#include <cstdio>
namespace LogGlobal {
static bool isInitialized = false;
static std::mutex mtx;
}// namespace LogGlobal
void VEngine_Log(vstd::string_view const& chunk) {
	using namespace LogGlobal;
	std::lock_guard<decltype(mtx)> lck(mtx);
	FILE* file = nullptr;
	if (!isInitialized) {
		isInitialized = true;
		file = fopen("LoggerFile.log", "w"_sv);
		if (file) {
			vstd::string_view chunk = "This is a log file from last run: \n";
			fwrite(chunk.c_str(), chunk.size(), 1, file);
		}
	} else {
		file = fopen("LoggerFile.log", "a+"_sv);
	}
	if (file) {
		fwrite(chunk.c_str(), chunk.size(), 1, file);
		fclose(file);
	}
}
void VEngine_Log(vstd::string_view const* chunk, size_t chunkCount) {
	using namespace LogGlobal;
	std::lock_guard<decltype(mtx)> lck(mtx);
	FILE* file = nullptr;
	if (!isInitialized) {
		isInitialized = true;
		file = fopen("LoggerFile.log", "w"_sv);
		if (file) {
			vstd::string_view chunk = "This is a log file from last run: \n";
			fwrite(chunk.c_str(), chunk.size(), 1, file);
		}
	} else {
		file = fopen("LoggerFile.log", "a+"_sv);
	}
	if (file) {
		for (size_t i = 0; i < chunkCount; ++i)
			fwrite(chunk[i].c_str(), chunk[i].size(), 1, file);
		fclose(file);
	}
}
void VEngine_Log(std::initializer_list<vstd::string_view> const& initList) {
	VEngine_Log(initList.begin(), initList.size());
}
void VEngine_Log(std::type_info const& t) {
	VEngine_Log(
		{t.name(),
		 " runtime error! Usually did mistake operation, like vstd::optional\n"_sv});
}
void VEngine_Log(char const* chunk) {
	VEngine_Log(vstd::string_view(chunk));
}
