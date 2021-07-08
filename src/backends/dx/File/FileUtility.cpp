#include <File/FileUtility.h>
#include <io.h>
#include <stdio.h>
#include <direct.h>
#include <Utility/StringUtility.h>
void FileUtility::GetFiles(vstd::string const& path, vstd::vector<vstd::string>& files, HashMap<vstd::string, bool> const& ignoreFolders) {
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
void GetFileFixedExtense_Global_Func(vstd::string const& path, vstd::vector<vstd::string>& files, HashMap<vstd::string, bool> const& extense, vstd::string& cache) {
	uint64 hFile = 0;
	struct _finddata_t fileinfo;

	if ((hFile = _findfirst((path + "\\*").c_str(), &fileinfo)) != -1) {
		do {
			if ((fileinfo.attrib & _A_SUBDIR)) {
				//if (ignoreFolders.Contains(fileinfo.name)) continue;
				if (strcmp(fileinfo.name, ".") != 0 && strcmp(fileinfo.name, "..") != 0) {
					vstd::string folderPath = path + ("\\") + (fileinfo.name);
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
void FileUtility::GetFilesFixedExtense(vstd::string const& originPath, vstd::vector<vstd::string>& files, HashMap<vstd::string, bool> const& extense) {
	vstd::string cache;
	GetFileFixedExtense_Global_Func(originPath, files, extense, cache);
}

vstd::string FileUtility::GetFileExtension(vstd::string const& filePath) {
	char* ptr = filePath.data() + filePath.size() - 1;
	for (; *ptr != '.' && ptr >= filePath.data(); --ptr) {}
	return vstd::string(ptr, filePath.data() + filePath.size());
}

void FileUtility::GetFiles(vstd::string const& path, vstd::vector<vstd::string>& files, vstd::vector<vstd::string>& folders, HashMap<vstd::string, bool> const& ignoreFolders) {
	uint64 hFile = 0;
	struct _finddata_t fileinfo;

	if ((hFile = _findfirst((path + "\\*").c_str(), &fileinfo)) != -1) {
		do {
			if ((fileinfo.attrib & _A_SUBDIR)) {
				if (ignoreFolders.Contains(fileinfo.name)) continue;
				if (strcmp(fileinfo.name, ".") != 0 && strcmp(fileinfo.name, "..") != 0) {
					vstd::string& folderPath = folders.emplace_back(path + ("\\") + (fileinfo.name));
					GetFiles(folderPath, files, folders, ignoreFolders);
				}
			} else {
				files.push_back(path + ("\\") + (fileinfo.name));
			}
		} while (_findnext(hFile, &fileinfo) == 0);
		_findclose(hFile);
	}
}

bool FileUtility::ReadCommandFile(vstd::string const& path, HashMap<vstd::string, Runnable<void(vstd::string const&)>>& rnb) {
	std::ifstream ifs(path.c_str());
	if (!ifs)
		return false;
	vstd::vector<vstd::string> lines;
	StringUtil::ReadLines(ifs, lines);

	//TODO
	//Add Command
	vstd::vector<vstd::string> commands;
	for (auto ite = lines.begin(); ite != lines.end(); ++ite) {
		StringUtil::Split(*ite, ' ', commands);
		if (commands.size() < 1) continue;
		auto hashIte = rnb.Find(commands[0]);
		if (hashIte) {
			if (commands.size() >= 2) {
				vstd::string str;
				for (uint a = 1; a < commands.size() - 1; ++a) {
					str += commands[a];
					str += ' ';
				}
				str += commands[commands.size() - 1];
				hashIte.Value()(str);
			} else {
				hashIte.Value()(vstd::string());
			}
		}
	}
	return true;
}

void FileUtility::GetTrivialFiles(vstd::string const& path, vstd::vector<vstd::string>& files) {
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

vstd::string FileUtility::GetProgramPath() {
	vstd::string str;
	str.resize(4096);
	_getcwd(str.data(), str.size());
	str.resize(strlen(str.data()));
	return str;
}

void FileUtility::GetFolders(vstd::vector<vstd::string>& files) {
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