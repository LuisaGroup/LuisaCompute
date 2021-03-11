#include "FilePacker.h"
#include <fstream>

bool FilePacker::PackAllData(
	vengine::string const& packagePath,
	vengine::vector<vengine::string> const& paths,
	vengine::vector<std::pair<uint64, uint64>>& outputOffsets)
{
	std::ofstream package(packagePath.c_str(), std::ios::binary);
	if (!package)
	{
		std::cout << "Failed Output " << packagePath.c_str() << std::endl;
		return false;
	}
	uint64 offset = 0;
	outputOffsets.clear();
	outputOffsets.resize(paths.size());
	vengine::vector<char> c;
	for (uint64 i = 0; i < paths.size(); ++i)
	{
		auto&& ofst = outputOffsets[i];
		ofst.first = offset;
		{
			std::ifstream ifs(paths[i].c_str(), std::ios::binary);
			if (!ifs)
			{
				std::cout << "Failed Input " << paths[i].c_str() << std::endl;
				return false;
			}
			ifs.seekg(0, std::ios::end);
			size_t sz = ifs.tellg();
			ofst.second = sz;
			offset += sz;
			ifs.seekg(0, std::ios::beg);
			c.clear();
			c.resize(sz);
			ifs.read(c.data(), sz);
		}
		package.write(c.data(), c.size());
	}
	return true;
}

bool FilePacker::PackAllContents(
	vengine::string const& contentPath,
	vengine::vector<vengine::string> const& paths,
	vengine::vector<std::pair<uint64, uint64>> const& outputOffsets)
{
	if (paths.size() != outputOffsets.size()) return false;
	std::ofstream ofs(contentPath.data(), std::ios::binary);
	uint64 stringLen = 0;
	for (auto& i : paths)
	{
		stringLen += i.size();
	}
	//Input All String Length for allocation
	ofs.write((char const*)&stringLen, sizeof(stringLen));

	if (!ofs) return false;
	uint64 sz = paths.size();
	//Input File Count
	ofs.write((char const*)&sz, sizeof(sz));
	for (size_t i = 0; i < sz; ++i)
	{
		auto&& path = paths[i];
		uint pathLen = path.size();
		//Input String
		ofs.write((char const*)&pathLen, sizeof(pathLen));
		ofs.write(path.data(), pathLen);
		auto&& offset = outputOffsets[i];
		//Input Offset And Count
		ofs.write((char const*)&offset, sizeof(offset));
	}
	return true;
}
