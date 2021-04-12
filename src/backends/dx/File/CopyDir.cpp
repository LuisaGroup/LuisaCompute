#include <File/CopyDir.h>
#include <File/FileUtility.h>
#include <iostream>
#include <fstream>

#if defined(_WIN32)
#include <direct.h>
#include <io.h>
#include <shlobj.h>
#include <sys/stat.h>
#include <sys/types.h>
#else// Linux
#include <dirent.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <pwd.h>
#endif

CopyDir::CopyDir() {
}
CopyDir::~CopyDir() {
}
int64_t StringFind(vengine::string const& str, char tar, size_t startPos) {
	char* start = str.data() + startPos;
	char* end = str.data() + str.size();
	if (start > end) return -1;
	for (char* c = start; c != end; ++c) {
		if (*c == tar) {
			return c - start + startPos;
		}
	}
	return -1;
}
void CopyDir::copy(const vengine::string& srcDirPath, const vengine::string& desDirPath, HashMap<vengine::string, bool> const& ignoreFolders, HashMap<vengine::string, bool> const& avaliableExtension, JobBucket* bucket) {
	this->srcDirPath = srcDirPath;
	vengine::string srcDir;
	std::string str;
#ifdef _WIN32
	int n = 0;

	while (true) {
		int64_t v = StringFind(srcDirPath, '/', n);
		if (v < 0) break;
		n = v + 1;
	}
	if (n == 0) {
		std::cout << "src path error" << std::endl;
		return;
	}

	srcDir = vengine::string(srcDirPath.data() + n - 1, srcDirPath.data() + srcDirPath.size());//srcDirPath.substr(n - 1, srcDirPath.size());

#else// Linux
	int n = 0;
	while (srcDirPath.find('/', n) != vengine::string::npos) {
		n = srcDirPath.find('/', n) + 1;
	}
	if (n == 0) {
		std::cout << "src path error" << std::endl;
		return;
	}
	srcDir = srcDirPath.substr(n - 1, srcDirPath.size());

#endif
	this->desDirPath = desDirPath;

	if (!make_dir(this->desDirPath)) {
		return;
	}

	fileNameList.clear();
	if (!get_src_files_name(srcDirPath, desDirPath, "", fileNameList, ignoreFolders)) {
		return;
	}

	if (fileNameList.empty()) {
		std::cout << "src dir is empty" << std::endl;
		return;
	}

	do_copy(fileNameList, avaliableExtension, bucket);
}

bool CopyDir::make_dir(const vengine::string& pathName) {
#ifdef _WIN32
	::_mkdir(pathName.c_str());
#else// Linux
	if (::mkdir(pathName.c_str(), S_IRWXU | S_IRGRP | S_IXGRP) < 0) {
		std::cout << "create path error" << std::endl;
		return false;
	}
#endif

	return true;
}

bool CopyDir::get_src_files_name(vengine::string const& srcFolder,
								 vengine::string const& destFolder,
								 vengine::string const& root,
								 vengine::vector<vengine::string>& fileNameList,
								 HashMap<vengine::string, bool> const& ignoreFolders) {
#ifdef _WIN32
	_finddata_t file;
	uint64_t lf;
	vengine::string src = srcFolder + "/*";
	if ((lf = _findfirst(src.c_str(), &file)) == -1) {
		std::cout << this->srcDirPath << " not found" << std::endl;
		return false;
	} else {
		while (_findnext(lf, &file) == 0) {
			//Folder
			if ((file.attrib & _A_SUBDIR)) {
				if (!ignoreFolders.Contains(file.name) && strcmp(file.name, ".") != 0 && strcmp(file.name, "..") != 0) {
					vengine::string srcFolderPath = srcFolder + '/' + file.name;
					vengine::string destFolderPath = destFolder + '/' + file.name;
					vengine::string relative = root + '/' + file.name;
					_mkdir(destFolderPath.c_str());
					get_src_files_name(srcFolderPath, destFolderPath, relative, fileNameList, ignoreFolders);
				}
			} else
				fileNameList.push_back(root + '/' + file.name);
		}
	}

	_findclose(lf);
#else// Linux
	DIR* dir;
	struct dirent* ptr;

	if ((dir = opendir(this->srcDirPath.c_str())) == NULL) {
		std::cout << this->srcDirPath << " not found" << std::endl;
		return false;
	}

	while ((ptr = readdir(dir)) != NULL) {
		if ((ptr->d_name == ".") || (ptr->d_name == ".."))//current / parent
			continue;
		else if (ptr->d_type == 8)//file
			fileNameList.push_back(ptr->d_name);
		else if (ptr->d_type == 10)//link file
			continue;
		else if (ptr->d_type == 4)//dir
			fileNameList.push_back(ptr->d_name);
	}
	closedir(dir);

#endif

	return true;
}

void CopyDir::do_copy(const vengine::vector<vengine::string>& fileNameList, HashMap<vengine::string, bool> const& avaliableExtension, JobBucket* bucket) {
	success = true;
	for (int i = 0; i < fileNameList.size(); i++) {
		bucket->GetTask({}, [&, i]() -> void {
			vengine::string nowSrcFilePath, nowDesFilePath;
			vengine::string ext = FileUtility::GetFileExtension(fileNameList[i]);
			if (!avaliableExtension.Contains(ext)) return;
#ifdef _WIN32
			nowSrcFilePath = this->srcDirPath + fileNameList[i];
			nowDesFilePath = this->desDirPath + fileNameList[i];

#else
				nowSrcFilePath = this->srcDirPath + "/" + fileNameList.at(i);
				nowDesFilePath = this->desDirPath + "/" + fileNameList.at(i);

#endif
#ifdef UNICODE
			if (!CopyFile(vengine::wstring(nowSrcFilePath).c_str(), vengine::wstring(nowDesFilePath).c_str(), false))
#else
			if (!CopyFile(nowSrcFilePath.c_str(), nowDesFilePath.c_str(), false))

#endif
				success = false;
		});
	}
}
