#pragma once
#include <File/MGUID.h>
#include <File/MGUID.h>
#include <Common/Common.h>
class FilePacker
{
public:
	static bool PackAllData(
		vstd::string const& packagePath,
		vstd::vector<vstd::string> const& paths,
		vstd::vector<std::pair<uint64, uint64>>& outputOffsets
	);
	static bool PackAllContents(
		vstd::string const& contentPath,
		vstd::vector<vstd::string> const& paths,
		vstd::vector<std::pair<uint64, uint64>> const& outputOffsets
	);
	FilePacker() = delete;
	VSTL_DELETE_COPY_CONSTRUCT(FilePacker)
};