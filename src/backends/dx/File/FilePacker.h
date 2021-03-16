#pragma once
#include "MGUID.h"
#include "MGUID.h"
#include <Common/Common.h>
class FilePacker
{
public:
	static bool PackAllData(
		vengine::string const& packagePath,
		vengine::vector<vengine::string> const& paths,
		vengine::vector<std::pair<uint64, uint64>>& outputOffsets
	);
	static bool PackAllContents(
		vengine::string const& contentPath,
		vengine::vector<vengine::string> const& paths,
		vengine::vector<std::pair<uint64, uint64>> const& outputOffsets
	);
	FilePacker() = delete;
	KILL_COPY_CONSTRUCT(FilePacker)
};