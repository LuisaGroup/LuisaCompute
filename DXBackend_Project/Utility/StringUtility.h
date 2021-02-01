#pragma once
#include "../Common/vector.h"
#include <fstream>
#include "../Common/vstring.h"
#include "../Common/Memory.h"
#include "../Common/string_chunk.h"
typedef uint32_t uint;
class  StringUtil
{
private:
	StringUtil() = delete;
	KILL_COPY_CONSTRUCT(StringUtil)
public:
	static void IndicesOf(vengine::string const& str, vengine::string const& sign, vengine::vector<uint>& v);
	static void IndicesOf(vengine::string const& str, char, vengine::vector<uint>& v);
	static void CutToLine(vengine::string const& str, vengine::vector<vengine::string>& lines);
	static void CutToLine(const char* str, int64_t size, vengine::vector<vengine::string>& lines);
	static void ReadLines(std::ifstream& ifs, vengine::vector<vengine::string>& lines);
	static int32_t GetFirstIndexOf(vengine::string const& str, vengine::string const& sign);
	static int32_t GetFirstIndexOf(vengine::string const& str, char sign);
	static void Split(vengine::string const& str, vengine::string const& sign, vengine::vector<vengine::string>& v);
	static void Split(vengine::string const& str, char sign, vengine::vector<vengine::string>& v);
	static void GetDataFromAttribute(vengine::string const& str, vengine::string& result);
	static void GetDataFromBrackets(vengine::string const& str, vengine::string& result);
	static int32_t StringToInteger(vengine::string const& str);
	static double StringToFloat(vengine::string const& str);
	static int32_t StringToInteger(string_chunk str);
	static double StringToFloat(string_chunk str);
	static void ToLower(vengine::string& str);
	static void ToUpper(vengine::string& str);
};