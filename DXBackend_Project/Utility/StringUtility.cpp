#include "StringUtility.h"
void StringUtil::IndicesOf(vengine::string const& str, vengine::string const& sign, vengine::vector<uint>& v) {
	v.clear();
	if (str.empty()) return;
	int32_t count = str.length() - sign.length() + 1;
	v.reserve(10);
	for (int32_t i = 0; i < count; ++i) {
		bool success = true;
		for (int32_t j = 0; j < sign.length(); ++j) {
			if (sign[j] != str[i + j]) {
				success = false;
				break;
			}
		}
		if (success)
			v.push_back(i);
	}
}
void StringUtil::IndicesOf(vengine::string const& str, char sign, vengine::vector<uint>& v) {
	v.clear();
	int32_t count = str.length();
	v.reserve(10);
	for (int32_t i = 0; i < count; ++i) {
		if (sign == str[i]) {
			v.push_back(i);
		}
	}
}
void StringUtil::CutToLine(const char* str, int64_t size, vengine::vector<vengine::string>& lines) {
	lines.clear();
	lines.reserve(32);
	vengine::string buffer;
	buffer.reserve(32);
	for (size_t i = 0; i < size; ++i) {
		auto&& value = str[i];
		if (value == '\0') break;
		if (!(value == '\n' || value == '\r')) {
			buffer.push_back(value);
		} else {
			if (value == '\n' || (value == '\r' && i < size - 1 && str[i + 1] == '\n')) {
				if (!buffer.empty())
					lines.push_back(buffer);
				buffer.clear();
			}
		}
	}
	if (!buffer.empty())
		lines.push_back(buffer);
}
void StringUtil::CutToLine(vengine::string const& str, vengine::vector<vengine::string>& lines) {
	lines.clear();
	lines.reserve(32);
	vengine::string buffer;
	buffer.reserve(32);
	for (size_t i = 0; i < str.length(); ++i) {
		if (!(str[i] == '\n' || str[i] == '\r')) {
			buffer.push_back(str[i]);
		} else {
			if (str[i] == '\n' || (str[i] == '\r' && i < str.length() - 1 && str[i + 1] == '\n')) {
				if (!buffer.empty())
					lines.push_back(buffer);
				buffer.clear();
			}
		}
	}
	if (!buffer.empty())
		lines.push_back(buffer);
}
void StringUtil::ReadLines(std::ifstream& ifs, vengine::vector<vengine::string>& lines) {
	ifs.seekg(0, std::ios::end);
	int64_t size = ifs.tellg();
	ifs.seekg(0, std::ios::beg);
	vengine::vector<char> buffer(size);
	memset(buffer.data(), 0, size);
	ifs.read(buffer.data(), size);
	CutToLine(buffer.data(), size, lines);
}
int32_t StringUtil::GetFirstIndexOf(vengine::string const& str, char sign) {
	int32_t count = str.length();
	for (int32_t i = 0; i < count; ++i) {
		if (sign == str[i]) {
			return i;
		}
	}
	return -1;
}
int32_t StringUtil::GetFirstIndexOf(vengine::string const& str, vengine::string const& sign) {
	int32_t count = str.length() - sign.length() + 1;
	for (int32_t i = 0; i < count; ++i) {
		bool success = true;
		for (int32_t j = 0; j < sign.length(); ++j) {
			if (sign[j] != str[i + j]) {
				success = false;
				break;
			}
		}
		if (success)
			return i;
	}
	return -1;
}
void StringUtil::Split(vengine::string const& str, char sign, vengine::vector<vengine::string>& v) {
	vengine::vector<uint> indices;
	IndicesOf(str, sign, indices);
	v.clear();
	v.reserve(10);
	vengine::string s;
	s.reserve(str.size());
	int32_t startPos = 0;
	for (auto index = indices.begin(); index != indices.end(); ++index) {
		s.clear();
		for (int32_t i = startPos; i < *index; ++i) {
			s.push_back(str[i]);
		}
		startPos = *index + 1;
		if (!s.empty())
			v.push_back(s);
	}
	s.clear();
	for (int32_t i = startPos; i < str.length(); ++i) {
		s.push_back(str[i]);
	}
	if (!s.empty())
		v.push_back(s);
}
void StringUtil::Split(vengine::string const& str, vengine::string const& sign, vengine::vector<vengine::string>& v) {
	vengine::vector<uint> indices;
	IndicesOf(str, sign, indices);
	v.clear();
	v.reserve(10);
	vengine::string s;
	s.reserve(str.size());
	int32_t startPos = 0;
	for (auto index = indices.begin(); index != indices.end(); ++index) {
		s.clear();
		for (int32_t i = startPos; i < *index; ++i) {
			s.push_back(str[i]);
		}
		startPos = *index + sign.size();
		if (!s.empty())
			v.push_back(s);
	}
	s.clear();
	for (int32_t i = startPos; i < str.length(); ++i) {
		s.push_back(str[i]);
	}
	if (!s.empty())
		v.push_back(s);
}
void StringUtil::GetDataFromAttribute(vengine::string const& str, vengine::string& result) {
	int32_t firstIndex = GetFirstIndexOf(str, '[');
	result.clear();
	if (firstIndex < 0) return;
	result.reserve(5);
	for (int32_t i = firstIndex + 1; str[i] != ']' && i < str.length(); ++i) {
		result.push_back(str[i]);
	}
}
void StringUtil::GetDataFromBrackets(vengine::string const& str, vengine::string& result) {
	int32_t firstIndex = GetFirstIndexOf(str, '<');
	result.clear();
	if (firstIndex < 0) return;
	result.reserve(5);
	for (int32_t i = firstIndex + 1; str[i] != '>' && i < str.length(); ++i) {
		result.push_back(str[i]);
	}
}
int32_t StringUtil::StringToInteger(vengine::string const& str) {
	if (str.empty()) return 0;
	uint i;
	int32_t value = 0;
	int32_t rate;
	if (str[0] == '-') {
		rate = -1;
		i = 1;
	} else {
		rate = 1;
		i = 0;
	}
	for (; i < str.length(); ++i) {
		value *= 10;
		value += (int32_t)str[i] - 48;
	}
	return value * rate;
}
double StringUtil::StringToFloat(vengine::string const& str) {
	if (str.empty()) return 0;
	uint i;
	double value = 0;
	int32_t rate;
	if (str[0] == '-') {
		rate = -1;
		i = 1;
	} else {
		rate = 1;
		i = 0;
	}
	for (; i < str.length(); ++i) {
		auto c = str[i];
		if (c == '.') {
			i++;
			break;
		}
		value *= 10;
		value += (int32_t)c - 48;
	}
	double afterPointRate = 1;
	for (; i < str.length(); ++i) {
		afterPointRate *= 0.1;
		value += afterPointRate * ((int32_t)str[i] - 48);
	}
	return value * rate;
}
int32_t StringUtil::StringToInteger(string_chunk str) {
	if (str.size() == 0) return 0;
	uint i;
	int32_t value = 0;
	int32_t rate;
	if (str.begin()[0] == '-') {
		rate = -1;
		i = 1;
	} else {
		rate = 1;
		i = 0;
	}
	for (; i < str.size(); ++i) {
		value *= 10;
		value += (int32_t)str.begin()[i] - 48;
	}
	return value * rate;
}
double StringUtil::StringToFloat(string_chunk str) {
	if (str.size() == 0) return 0;
	uint i;
	double value = 0;
	int32_t rate;
	if (str.begin()[0] == '-') {
		rate = -1;
		i = 1;
	} else {
		rate = 1;
		i = 0;
	}
	for (; i < str.size(); ++i) {
		auto c = str.begin()[i];
		if (c == '.') {
			i++;
			break;
		}
		value *= 10;
		value += (int32_t)c - 48;
	}
	double afterPointRate = 1;
	for (; i < str.size(); ++i) {
		afterPointRate *= 0.1;
		value += afterPointRate * ((int32_t)str.begin()[i] - 48);
	}
	return value * rate;
}
inline constexpr void mtolower(char& c) {
	if ((c >= 'A') && (c <= 'Z'))
		c = c + ('a' - 'A');
}
inline constexpr void mtoupper(char& c) {
	if ((c >= 'a') && (c <= 'z'))
		c = c + ('A' - 'a');
}
void StringUtil::ToLower(vengine::string& str) {
	char* c = str.data();
	const uint size = str.length();
	for (uint i = 0; i < size; ++i) {
		mtolower(c[i]);
	}
}
void StringUtil::ToUpper(vengine::string& str) {
	char* c = str.data();
	const uint size = str.length();
	for (uint i = 0; i < size; ++i) {
		mtoupper(c[i]);
	}
}
