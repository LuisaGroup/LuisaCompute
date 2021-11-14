#pragma once
#include <vstl/Common.h>
#include <vstl/functional.h>
namespace vstd {

class FileSystem {
public:
	struct FileTime {
		uint64 creationTime;
		uint64 lastAccessTime;
		uint64 lastWriteTime;
	};
	static bool IsFileExists(string const& path);
	static void GetFiles(
		string const& path,
		function<bool(string&&)> const& callBack,
		bool recursionFolder);
};
}// namespace vstl