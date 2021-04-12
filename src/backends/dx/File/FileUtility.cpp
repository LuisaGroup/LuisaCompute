#include <File/FileUtility.h>
#include <io.h>
#include <stdio.h>
#include <direct.h>
#include <Utility/StringUtility.h>
void FileUtility::GetFiles(vengine::string const& path, vengine::vector<vengine::string>& files, HashMap<vengine::string, bool> const& ignoreFolders) {
	uint64 hFile = 0;
	struct _finddata_t fileinfo;

	if ((hFile = _findfirst((path + "\\*").c_str(), &fileinfo)) != -1) {
		do {
			if ((fileinfo.attrib & _A_SUBDIR)) {
				if (ignoreFolders.Contains(fileinfo.name)) continue;
				if (strcmp(fileinfo.name, ".") != 0 && strcmp(fileinfo.name, "..") != 0)
					GetFiles(path + ("\\") + (fileinfo.name), files, ignoreFolders);
			} else {
				files.push_back(path + ("\\") + (fileinfo.name));
			}
		} while (_findnext(hFile, &fileinfo) == 0);
		_findclose(hFile);
	}
}
void GetFileFixedExtense_Global_Func(vengine::string const& path, vengine::vector<vengine::string>& files, HashMap<vengine::string, bool> const& extense, vengine::string& cache) {
	uint64 hFile = 0;
	struct _finddata_t fileinfo;

	if ((hFile = _findfirst((path + "\\*").c_str(), &fileinfo)) != -1) {
		do {
			if ((fileinfo.attrib & _A_SUBDIR)) {
				//if (ignoreFolders.Contains(fileinfo.name)) continue;
				if (strcmp(fileinfo.name, ".") != 0 && strcmp(fileinfo.name, "..") != 0) {
					vengine::string folderPath = path + ("\\") + (fileinfo.name);
					//GetFiles(folderPath, files, folders, ignoreFolders);
					GetFileFixedExtense_Global_Func(folderPath, files, extense, cache);
				}
			} else {
				char const* startPtr = fileinfo.name;
				char const* ptr;
				for (ptr = fileinfo.name; *ptr != 0; ++ptr) {
					if (*ptr == '.') {
						startPtr = ptr + 1;
					}
				}
				cache.clear();
				cache.push_back_all(startPtr, ptr - startPtr);
				auto ite = extense.Find(cache);
				if (ite && ite.Value()) {
					files.push_back(path + ("\\") + (fileinfo.name));
				}
			}
		} while (_findnext(hFile, &fileinfo) == 0);
		_findclose(hFile);
	}
}
void FileUtility::GetFilesFixedExtense(vengine::string const& originPath, vengine::vector<vengine::string>& files, HashMap<vengine::string, bool> const& extense) {
	vengine::string cache;
	GetFileFixedExtense_Global_Func(originPath, files, extense, cache);
}

vengine::string FileUtility::GetFileExtension(vengine::string const& filePath) {
	char* ptr = filePath.data() + filePath.size() - 1;
	for (; *ptr != '.' && ptr >= filePath.data(); --ptr) {}
	return vengine::string(ptr, filePath.data() + filePath.size());
}

void FileUtility::GetFiles(vengine::string const& path, vengine::vector<vengine::string>& files, vengine::vector<vengine::string>& folders, HashMap<vengine::string, bool> const& ignoreFolders) {
	uint64 hFile = 0;
	struct _finddata_t fileinfo;

	if ((hFile = _findfirst((path + "\\*").c_str(), &fileinfo)) != -1) {
		do {
			if ((fileinfo.attrib & _A_SUBDIR)) {
				if (ignoreFolders.Contains(fileinfo.name)) continue;
				if (strcmp(fileinfo.name, ".") != 0 && strcmp(fileinfo.name, "..") != 0) {
					vengine::string& folderPath = folders.emplace_back(path + ("\\") + (fileinfo.name));
					GetFiles(folderPath, files, folders, ignoreFolders);
				}
			} else {
				files.push_back(path + ("\\") + (fileinfo.name));
			}
		} while (_findnext(hFile, &fileinfo) == 0);
		_findclose(hFile);
	}
}

bool FileUtility::ReadCommandFile(vengine::string const& path, HashMap<vengine::string, Runnable<void(vengine::string const&)>>& rnb) {
	std::ifstream ifs(path.c_str());
	if (!ifs)
		return false;
	vengine::vector<vengine::string> lines;
	StringUtil::ReadLines(ifs, lines);

	//TODO
	//Add Command
	vengine::vector<vengine::string> commands;
	for (auto ite = lines.begin(); ite != lines.end(); ++ite) {
		StringUtil::Split(*ite, ' ', commands);
		if (commands.size() < 1) continue;
		auto hashIte = rnb.Find(commands[0]);
		if (hashIte) {
			if (commands.size() >= 2) {
				vengine::string str;
				for (uint a = 1; a < commands.size() - 1; ++a) {
					str += commands[a];
					str += ' ';
				}
				str += commands[commands.size() - 1];
				hashIte.Value()(str);
			} else {
				hashIte.Value()(vengine::string());
			}
		}
	}
	return true;
}

void FileUtility::GetTrivialFiles(vengine::string const& path, vengine::vector<vengine::string>& files) {
	uint64 hFile = 0;
	struct _finddata_t fileinfo;

	if ((hFile = _findfirst((path + "\\*").c_str(), &fileinfo)) != -1) {
		do {
			if (!(fileinfo.attrib & _A_SUBDIR)) {
				files.push_back(fileinfo.name);
			}
		} while (_findnext(hFile, &fileinfo) == 0);
		_findclose(hFile);
	}
}

vengine::string FileUtility::GetProgramPath() {
	vengine::string str;
	str.resize(4096);
	_getcwd(str.data(), str.size());
	str.resize(strlen(str.data()));
	return str;
}

void FileUtility::GetFolders(vengine::vector<vengine::string>& files) {
	auto str = GetProgramPath();
	str += "\\*";
	uint64 hFile = 0;
	//??????
	struct _finddata_t fileinfo;

	if ((hFile = _findfirst(str.c_str(), &fileinfo)) != -1) {
		do {
			if ((fileinfo.attrib & _A_SUBDIR)) {
				if (fileinfo.name[0] != '.')
					files.push_back(fileinfo.name);
			}
		} while (_findnext(hFile, &fileinfo) == 0);
		_findclose(hFile);
	}
}