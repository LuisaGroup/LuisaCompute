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
		const vstd::string& srcDirPath,
		const vstd::string& desDirPath, 
		HashMap<vstd::string, bool> const& ignoreFolders,
		HashMap<vstd::string, bool> const& avaliableExtension,
		JobBucket* bucket);

private:
	bool make_dir(const vstd::string& pathName);
	//    bool mkdir (char const* pathname/*, mode_t mode*/);
	bool get_src_files_name(
		vstd::string const& srcFolder, vstd::string const& destFolder,
		vstd::string const& root, vstd::vector<vstd::string>& fileNameList, HashMap<vstd::string, bool> const& ignoreFolders);
	void do_copy(const vstd::vector<vstd::string>& fileNameList, HashMap<vstd::string, bool> const& avaliableExtension, JobBucket* bucket);

	vstd::string srcDirPath, desDirPath;
	vstd::vector<vstd::string> fileNameList;

};