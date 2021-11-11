#pragma vengine_package vengine_dll
#include <vstl/file_system.h>

#ifdef _WIN32
#include <shlwapi.h>
#include <io.h>
#include <stdio.h>
#include <direct.h>
#include <fileapi.h>
namespace vstd {
bool FileSystem::IsFileExists(std::string const& path) {
	return PathFileExistsA(path.c_str());
}
void FileSystem::GetFiles(
	std::string const& mPath,
	vstd::function<bool(std::string&&)> const& func,
	bool recursionFolder) {
	auto innerFunc = [&](std::string const& path, auto&& innerFunc) -> bool {
		uint64 hFile = 0;
		struct _finddata_t fileinfo;
		if ((hFile = _findfirst((path + "/*").c_str(), &fileinfo)) != -1) {
			do {
				if ((fileinfo.attrib & _A_SUBDIR)) {
					if (recursionFolder && (strcmp(fileinfo.name, ".") != 0 && strcmp(fileinfo.name, "..") != 0)) {
						if (!innerFunc(path + ('/') + (fileinfo.name), innerFunc)) return false;
					}
				} else {
					if (!func(path + '/' + (fileinfo.name))) return false;
				}
			} while (_findnext(hFile, &fileinfo) == 0);
			_findclose(hFile);
		}
		return true;
	};
	innerFunc(mPath, innerFunc);
}
}// namespace vstl
#elif defined(__linux__) || defined(__unix__)
//TODO
#elif defined(__APPLE__)
//TODO
#endif
