#pragma once


#include <Common/Common.h>
#include <JobSystem/JobInclude.h>
class CopyDir
{
public:
	CopyDir();
	~CopyDir();
	bool success = true;
	void copy(
		const vengine::string& srcDirPath,
		const vengine::string& desDirPath, 
		HashMap<vengine::string, bool> const& ignoreFolders,
		HashMap<vengine::string, bool> const& avaliableExtension,
		JobBucket* bucket);

private:
	bool make_dir(const vengine::string& pathName);
	//    bool mkdir (char const* pathname/*, mode_t mode*/);
	bool get_src_files_name(
		vengine::string const& srcFolder, vengine::string const& destFolder,
		vengine::string const& root, vengine::vector<vengine::string>& fileNameList, HashMap<vengine::string, bool> const& ignoreFolders);
	void do_copy(const vengine::vector<vengine::string>& fileNameList, HashMap<vengine::string, bool> const& avaliableExtension, JobBucket* bucket);

	vengine::string srcDirPath, desDirPath;
	vengine::vector<vengine::string> fileNameList;

};