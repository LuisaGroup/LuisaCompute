#include <File/Path.h>
#include <File/FileUtility.h>
#include <io.h>
Path::Path(vengine::string const& path)
{
	operator=(path);
}
Path::Path(Path const& absolutePath, vengine::string const& relativePath)
{
	Path curPath = absolutePath;
	vengine::vector<vengine::string> blocks;
	SeparatePath(relativePath, blocks);
	for (auto ite = blocks.begin(); ite != blocks.end(); ++ite)
	{
		if (*ite == ".." || *ite == ".")
		{
			curPath = curPath.GetParentLevelPath();
		}
		else
		{
			curPath = curPath.GetSubLevelPath(*ite);
		}
	}
	pathData = curPath.pathData;
}

bool Path::operator==(Path const& a) const
{
	return pathData == a.pathData;
}
bool Path::operator!=(Path const& a) const
{
	return !operator==(a);
}

bool Path::Exists() const
{
	return _access(pathData.c_str(), 0) != -1;
}

Path& Path::operator=(vengine::string const& path)
{
	if (path.empty())
	{
		pathData = GetProgramPath().GetPathStr();
		return *this;
	}
	char* start = path.data();
	char* end = start + path.size();
	//Not Absolute 
	if (*start < 'A' || *start > 'Z' || start[1] != ':')
	{
		static Path selfPath = GetProgramPath();
		Path curPath = selfPath;
		vengine::vector<vengine::string> blocks;
		SeparatePath(path, blocks);
		for (auto ite = blocks.begin(); ite != blocks.end(); ++ite)
		{
			if (*ite == ".." || *ite == ".")
			{
				curPath = curPath.GetParentLevelPath();
			}
			else
			{
				curPath = curPath.GetSubLevelPath(*ite);
			}
		}
		pathData = curPath.pathData;
	}
	//Absolute Pos
	else
	{
		pathData = path;
	}
	start = pathData.data();
	end = start + pathData.size();
	for (char* i = start; i != end; ++i)
	{
		if (*i == '\\')
			*i = '/';
	}
	return *this;
}

bool Path::IsSubPathOf(Path const& parentPath) const
{
	return parentPath.IsParentPathOf(*this);

}
bool Path::IsParentPathOf(Path const& subPath) const
{
	auto&& otherPath = subPath.GetPathStr();
	vengine::vector<vengine::string> parentSplit;
	vengine::vector<vengine::string> subSplit;
	SeparatePath(pathData, parentSplit);
	SeparatePath(otherPath, subSplit);
	if (parentSplit.size() >= subSplit.size()) return false;
	for (auto ite = parentSplit.begin(); ite != parentSplit.end(); ++ite)
	{
		if (*ite != subSplit[ite.GetIndex()]) return false;
	}
	return true;
}
vengine::string Path::TryGetSubPath(Path const& subPath) const
{
	auto&& otherPath = subPath.GetPathStr();
	vengine::vector<vengine::string> parentSplit;
	vengine::vector<vengine::string> subSplit;
	SeparatePath(pathData, parentSplit);
	SeparatePath(otherPath, subSplit);
	if (parentSplit.size() >= subSplit.size()) return vengine::string();
	auto ite = parentSplit.begin();
	for (; ite != parentSplit.end(); ++ite)
	{
		if (*ite != subSplit[ite.GetIndex()]) return vengine::string();
	}
	vengine::string str;
	for (uint i = ite.GetIndex(); i < subSplit.size() - 1; ++i)
	{
		str += subSplit[i] + '/';
	}
	str += subSplit[subSplit.size() - 1];
	return str;
}
Path& Path::operator=(Path const& v)
{
	auto&& path = v.GetPathStr();
	return operator=(path);
}
void Path::SeparatePath(vengine::string const& path, vengine::vector<vengine::string>& blocks)
{
	blocks.clear();
	char* start = path.data();
	char* end = start + path.size();
	char* i;
	for (i = start; i < end; ++i)
	{
		if (*i == '\\' || *i == '/')
		{
			blocks.emplace_back(start, i);
			start = i + 1;
		}
	}
	if (start < end)
	{
		blocks.emplace_back(start, end);
	}
}
Path Path::GetProgramPath()
{
	return Path(FileUtility::GetProgramPath());
}
Path Path::GetParentLevelPath() const
{
	char* start = pathData.data();
	char* end = start + pathData.size();
	for (char* i = end; i >= start; i--)
	{
		if (*i == '/' || *i == '\\')
		{
			return Path(vengine::string(start, i));
		}
	}
	return *this;
}
Path Path::GetSubLevelPath(vengine::string const& subName) const
{
	return Path(pathData + '/' + subName);
}
bool Path::IsFile() const
{
	char* start = pathData.data();
	char* end = start + pathData.size();
	for (char* i = end; i >= start; i--)
	{
		if (*i == '.')
			return true;
	}
	return false;

}
bool Path::IsDirectory() const
{
	return !IsFile();
}

vengine::string Path::GetExtension()
{
	char* start = pathData.data();
	char* end = start + pathData.size();
	for (char* i = end; i >= start; i--)
	{
		if (*i == '.')
			return vengine::string(i + 1, end);
	}
	return vengine::string();
}

void Path::TryCreateDirectory()
{
	vengine::vector<uint> slashIndex;
	slashIndex.reserve(20);
	for (uint i = 0; i < pathData.length(); ++i)
	{
		if (pathData[i] == '/' || pathData[i] == '\\')
		{
			slashIndex.push_back(i);
			pathData[i] = '\\';
		}
	}
	if (!IsFile())
	{
		slashIndex.push_back(pathData.length());
	}
	if (slashIndex.empty()) return;
	vengine::string command;
	command.reserve(slashIndex[slashIndex.size() - 1] + 3);
	uint startIndex = 0;
	for (uint i = 0; i < slashIndex.size(); ++i)
	{
		uint value = slashIndex[i];
		for (uint x = startIndex; x < value; ++x)
		{
			command += pathData[x];
		}
		if (_access(command.data(), 0) == -1)
		{
			std::system(("md " + command).data());
		}
		startIndex = slashIndex[i];
	}
	
}