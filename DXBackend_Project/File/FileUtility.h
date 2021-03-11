#pragma once
#include <Common/Common.h>
class FileUtility {
private:
	FileUtility() = delete;
	~FileUtility() = delete;

public:
	static bool ReadCommandFile(vengine::string const& path, HashMap<vengine::string, Runnable<void(vengine::string const&)>>& rnb);
	static void GetFiles(vengine::string const& path, vengine::vector<vengine::string>& files, HashMap<vengine::string, bool> const& ignoreFolders);
	static void GetFilesFixedExtense(vengine::string const& path, vengine::vector<vengine::string>& files, HashMap<vengine::string, bool> const& extense);
	static void GetFiles(vengine::string const& path, vengine::vector<vengine::string>& files, vengine::vector<vengine::string>& folders, HashMap<vengine::string, bool> const& ignoreFolders);
	static void GetFolders(vengine::vector<vengine::string>& files);
	static void GetTrivialFiles(vengine::string const& path, vengine::vector<vengine::string>& files);
	static vengine::string GetProgramPath();
	static vengine::string GetFileExtension(vengine::string const& filePath);
};