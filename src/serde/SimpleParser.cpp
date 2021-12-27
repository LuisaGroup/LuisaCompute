#pragma vengine_package vengine_database

#include <serde/config.h>
#include <vstl/config.h>
#include <serde/SimpleBinaryJson.h>
#include <serde/ParserException.h>
#include <serde/DatabaseInclude.h>
#include <vstl/BinaryReader.h>
#include <serde/Parser/StateRecorder.h>
namespace toolhub::db {
namespace parser {
enum class OutsideState : uint8_t {
    None,
    BeginDict,
    EndDict,
    BeginArray,
    EndArray,
    Continue,
    Cut,
    Comment,
    Number,
    String,
    Characters,
    Guid,
    Keyword,
    Path
};
struct StateRecorders {
    StateRecorder<OutsideState> outsideStates;
    StateRecorder<bool> keywordStates;
    StateRecorder<bool> guidCheck;
    StateRecorder<bool> numCheck;
    StateRecorder<bool> spaces;
    static void InitOutSideState(OutsideState *ptr) {
        ptr['{'] = OutsideState::BeginDict;
        ptr['}'] = OutsideState::EndDict;
        ptr['['] = OutsideState::BeginArray;
        ptr[']'] = OutsideState::EndArray;
        ptr[','] = OutsideState::Continue;
        ptr['<'] = OutsideState::Path;
        for (auto &&i : vstd::ptr_range(ptr + 48, ptr + 58)) {
            i = OutsideState::Number;
        }
        ptr['-'] = OutsideState::Number;
        ptr['$'] = OutsideState::Guid;
        ptr['\"'] = OutsideState::String;
        ptr['\''] = OutsideState::Characters;
        ptr[':'] = OutsideState::Cut;
        ptr['/'] = OutsideState::Comment;
        ptr['_'] = OutsideState::Keyword;
        for (auto &&i : vstd::ptr_range(ptr + 'a', ptr + 'z' + 1)) {
            i = OutsideState::Keyword;
        }
        for (auto &&i : vstd::ptr_range(ptr + 'A', ptr + 'Z' + 1)) {
            i = OutsideState::Keyword;
        }
    }
    using BoolArrayType = typename StateRecorder<bool>::ArrayType;
    static void InitKeywordState(BoolArrayType &ptr) {
        for (auto i : vstd::range(48, 58)) {
            ptr[i] = true;
        }
        for (auto i : vstd::range('a', 'z' + 1)) {
            ptr[i] = true;
        }
        for (auto i : vstd::range('A', 'Z' + 1)) {
            ptr[i] = true;
        }
        ptr['_'] = true;
    }
    StateRecorders()
        : outsideStates(InitOutSideState),
          keywordStates(InitKeywordState),
          guidCheck(
              [&](BoolArrayType &ptr) {
                  std::string_view sv =
                      "ABCDEFGHIJKLMNOP"
                      "QRSTUVWXYZabcdef"
                      "ghijklmnopqrstuv"
                      "wxyz0123456789+/="sv;
                  for (auto i : sv) {
                      ptr[i] = true;
                  }
              }),
          numCheck(
              [&](BoolArrayType &ptr) {
                  std::string_view sv = "0123456789.eE-+"sv;
                  for (auto i : sv) {
                      ptr[i] = true;
                  }
              }),
          spaces(
              [&](BoolArrayType &ptr) {
                  ptr[' '] = true;
                  ptr['\t'] = true;
                  ptr['\r'] = true;
                  ptr['\n'] = true;
              })

    {
    }
};

static vstd::spin_mutex recorderIsInited;
static vstd::optional<StateRecorders> recorders;
class SimpleJsonParser {

public:
    struct Field {
        vstd::variant<
            vstd::vector<char>,
            vstd::vector<std::pair<
                vstd::variant<
                    vstd::vector<char>,
                    int64,
                    vstd::Guid>,
                Field *>>,
            vstd::vector<Field *>,
            int64,
            double,
            vstd::Guid,
            bool,
            std::nullptr_t>
            data;
        template<typename... Args>
        Field(Args &&...a)
            : data(std::forward<Args>(a)...) {
        }
    };

private:
    vstd::vector<SimpleJsonParser> *subParsers;
    vstd::Pool<Field, false> *fieldPool;

    vstd::HashMap<std::string_view, Field *> keywords;
    std::string_view keywordName;
    vstd::vector<Field *> fieldStack;
    enum class StreamState : uint8_t {
        None,
        SearchKey,
        SearchElement,
        SearchKeyValueCut,
        SearchContinue,
        SearchKeyword
    };
    StreamState streamer = StreamState::None;
    Field *lastField = nullptr;
    Field *rootField = nullptr;
    static char const *GetNextChar(char const *ptr, char const *end) {
        auto &&s = recorders->spaces;
        while (ptr != end) {
            if (!s[*ptr]) return ptr;
            ptr++;
        }
        return ptr;
    }
    bool BeginDict(char const *&ptr, char const *const end, vstd::string &errorStr) {
        switch (streamer) {
            case StreamState::SearchElement:
            case StreamState::SearchKeyword:
                break;
            case StreamState::None: {
                if (lastField) {
                    errorStr = "Illegal character '{'";
                    return false;
                }
            } break;
            default:
                errorStr = "Illegal character '{'";
                return false;
        }

        lastField = fieldStack.emplace_back(fieldPool->New());
        lastField->data.update(1, [&](void *ptr) {
            new (ptr) vstd::vector<std::pair<std::string_view, Field *>>();
        });
        streamer = StreamState::SearchKey;
        ++ptr;
        return true;
    }
    bool SetLastFieldState(vstd::string &errorStr) {
        if (fieldStack.empty()) {
            streamer = StreamState::None;
            //set keyword
            if (keywordName.data() != nullptr) {
                auto iteValue = keywords.TryEmplace(
                    keywordName,
                    lastField);
                if (!iteValue.second) {
                    errorStr << "keyword " << keywordName << " conflict!";
                    return false;
                }
                keywordName = std::string_view(nullptr, 0);
            } else {
                rootField = lastField;
            }
            lastField = nullptr;
        } else {
            auto curField = lastField;
            lastField = *(fieldStack.end() - 1);
            if (lastField->data.GetType() == 1) {
                streamer = StreamState::SearchContinue;
                (lastField->data.get<1>().end() - 1)->second = curField;
            } else {
                streamer = StreamState::SearchContinue;
                lastField->data.get<2>().emplace_back(curField);
            }
        }
        return true;
    }
    bool EndDict(char const *&ptr, char const *const end, vstd::string &errorStr) {
        if (fieldStack.empty() || (*(fieldStack.end() - 1))->data.GetType() == 0) {
            errorStr = ("Illegal character '}'");
            return false;
        }
        fieldStack.erase_last();
        ++ptr;
        return SetLastFieldState(errorStr);
    }
    bool BeginComment(char const *&ptr, char const *end, vstd::string &errorStr) {
        ++ptr;
        switch (*ptr) {
            default:
                errorStr = "Illegal character '/'";
                return false;
            case '/':
                while (*ptr != '\n' && ptr != end) {
                    ptr++;
                }
                break;
            case '*':
                ptr++;
                end--;
                while (ptr < end && *ptr != '*' && ptr[1] != '/') {
                    ptr++;
                }
                ptr += 2;
                break;
        }
        return true;
    }
    bool BeginArray(char const *&ptr, char const *const end, vstd::string &errorStr) {
        switch (streamer) {
            case StreamState::SearchElement:
            case StreamState::SearchKeyword:
                break;
            case StreamState::None: {
                if (lastField) {
                    errorStr = "Illegal character '{'";
                    return false;
                }
            } break;
            default:
                errorStr = ("Illegal character '['");
                return false;
        }
        lastField = fieldStack.emplace_back(fieldPool->New());
        lastField->data.update(2, [&](void *ptr) {
            new (ptr) vstd::vector<Field *>();
        });
        streamer = StreamState::SearchElement;
        ++ptr;
        return true;
    }
    bool EndArray(char const *&ptr, char const *const end, vstd::string &errorStr) {
        if (fieldStack.empty() || (*(fieldStack.end() - 1))->data.GetType() == 1) {
            errorStr = ("Illegal character ']'");
            return false;
        }
        fieldStack.erase_last();
        ++ptr;
        return SetLastFieldState(errorStr);
    }
    bool Continue(char const *&ptr, char const *const end, vstd::string &errorStr) {
        if (streamer != StreamState::SearchContinue) {
            errorStr = ("Illegal character ','");
            return false;
        }
        ptr++;
        switch (lastField->data.GetType()) {
            case 1: {
                streamer = StreamState::SearchKey;
            } break;
            case 2: {
                streamer = StreamState::SearchElement;
            } break;
            default:
                errorStr = ("Compile Internal Error");
                return false;
        }
        return true;
    }
    template<typename T>
    bool SetStringField(T &&str, vstd::string &errorStr) {
        switch (streamer) {
            case StreamState::SearchKeyword: {

                auto iteValue = keywords.TryEmplace(
                    keywordName,
                    vstd::MakeLazyEval([&]() {
                        return fieldPool->New(std::forward<T>(str));
                    }));
                if (!iteValue.second) {
                    errorStr << "keyword " << keywordName << " conflict!";
                    return false;
                }
                streamer = StreamState::None;
                keywordName = std::string_view(nullptr, 0);
            } break;
            case StreamState::SearchKey: {
                if constexpr (std::is_same_v<double, std::remove_cvref_t<T>>) {
                    errorStr = ("float number cannot be a key");
                    return false;
                }
                lastField->data.get<1>().emplace_back(std::forward<T>(str), nullptr);
                streamer = StreamState::SearchKeyValueCut;
            } break;
            case StreamState::SearchElement:
                if (lastField->data.GetType() == 1) {
                    Field *field = fieldPool->New(std::forward<T>(str));
                    (lastField->data.get<1>().end() - 1)->second = field;
                    streamer = StreamState::SearchContinue;
                } else {
                    Field *field = fieldPool->New(std::forward<T>(str));
                    lastField->data.get<2>().emplace_back(field);
                    streamer = StreamState::SearchContinue;
                }
                break;
            default:
                errorStr = ("Illegal value field");
                return false;
        }
        return true;
    }
    template<typename A, typename B, bool value>
    struct TypeSelector;
    template<typename A, typename B>
    struct TypeSelector<A, B, false> {
        using Type = A;
    };
    template<typename A, typename B>
    struct TypeSelector<A, B, true> {
        using Type = B;
    };
    bool Number(char const *&ptr, char const *const end, vstd::string &errorStr) {
        char const *start = ptr;
        auto error = [&]() {
            errorStr = ("Incorrect numeric format!");
            return false;
        };
        do {
            ++ptr;
            if (ptr == end) {
                errorStr = ("Error end");
                return false;
            }
        } while (recorders->numCheck[*ptr]);
        auto numStr = std::string_view(start, ptr - start);
        char const *pin = start;

        int64 integerPart = 0;
        vstd::optional<double> floatPart;
        vstd::optional<int64> ratePart;
        float sign = 1;
        auto GetInt =
            [&](int64 &v,
                auto &&processFloat,
                auto &&processRate) -> bool {
            bool isNegative;
            if (numStr[0] == '-') {
                isNegative = true;
                pin++;
            } else if (numStr[0] == '+') {
                isNegative = false;
                pin++;
            } else {
                isNegative = false;
            }
            while (pin != ptr) {
                if (*pin >= '0' && *pin <= '9') {
                    v *= 10;
                    v += (*pin) - 48;
                    pin++;
                } else if (*pin == '.') {
                    pin++;
                    if (!processFloat()) return false;
                    break;
                } else if (*pin == 'e') {
                    pin++;
                    if (!processRate()) return false;
                    break;
                } else {
                    return error();
                }
            }
            if (isNegative) {
                v *= -1;
                sign = -1;
            }
            return true;
        };
        auto GetFloat = [&](double &v, auto &&processRate) -> bool {
            double rate = 1;
            while (pin != ptr) {
                if (*pin >= '0' && *pin <= '9') {
                    rate *= 0.1;
                    v += ((*pin) - 48) * rate;
                    pin++;
                } else if (*pin == 'e') {
                    pin++;
                    return processRate();
                } else {
                    return error();
                }
            }
            return true;
        };

        auto GetRate = [&]() -> bool {
            ratePart.New();
            return GetInt(*ratePart, error, error);
        };
        auto GetNumFloatPart = [&]() -> bool {
            floatPart.New();
            return GetFloat(*floatPart, GetRate);
        };
        if (!GetInt.operator()(
                integerPart,
                GetNumFloatPart,
                GetRate)) return false;
        if (ratePart || floatPart) {
            double value = (integerPart + (floatPart ? *floatPart : 0) * sign) * (ratePart ? pow(10, *ratePart) : 1);
            return SetStringField(value, errorStr);
        } else {
            return SetStringField(integerPart, errorStr);
        }
    }
    bool Guid(char const *&ptr, char const *const end, vstd::string &errorStr) {
        ptr++;
        char const *start = ptr;
        while (recorders->guidCheck[*ptr]) {
            ++ptr;
            if (ptr == end) {
                errorStr = ("Error end!");
                return false;
            }
        }
        auto guidStr = std::string_view(start, ptr - start);
        auto guid = vstd::Guid::TryParseGuid(guidStr);
        if (!guid) {
            errorStr = ("Incorrect guid format!");
            return false;
        }
        return SetStringField(*guid, errorStr);
    }
    template<char beginEnd>
    bool String(char const *&ptr, char const *const end, vstd::string &errorStr) {
        ptr++;
        char const *start = ptr;
        bool isSlash = false;
        vstd::vector<char> chars;
        chars.reserve(32);
        auto func = [&]() {
            while (true) {
                if (isSlash) {
                    isSlash = false;
                    switch (*ptr) {
                        case '\\':
                            chars.push_back('\\');
                            break;
                        case 't':
                            chars.push_back('\t');
                            break;
                        case 'r':
                            chars.push_back('\r');
                            break;
                        case 'n':
                            chars.push_back('\n');
                            break;
                        case '\'':
                            chars.push_back('\'');
                            break;
                        case '\"':
                            chars.push_back('\"');
                            break;
                    }
                } else {
                    switch (*ptr) {
                        case '\\':

                            isSlash = true;
                            break;
                        case beginEnd:
                            return true;
                        default:
                            chars.push_back(*ptr);
                            break;
                    }
                }
                ++ptr;
                if (ptr == end) {
                    errorStr = ("Error end!");
                    return false;
                }
            }
            return true;
        };
        if (!func()) return false;
        ++ptr;
        return SetStringField(std::move(chars), errorStr);
    }
    bool Cut(char const *&ptr, char const *const end, vstd::string &errorStr) {
        if (streamer != StreamState::SearchKeyValueCut) {
            errorStr = ("Illegal character ':'");
            return false;
        }
        ptr++;
        streamer = StreamState::SearchElement;
        return true;
    }
    bool BeginPath(char const *&ptr, char const *const end, vstd::string &errorStr);

    bool M_Parse(std::string_view str, vstd::string &errorStr) {
        char const *ptr = GetNextChar(str.data(), str.data() + str.size());
        while (true) {
            auto ed = str.data() + str.size();
            ptr = GetNextChar(ptr, ed);
            if (ptr == ed) return true;
            switch (recorders->outsideStates[*ptr]) {
                case OutsideState::BeginDict:
                    if (!BeginDict(ptr, ed, errorStr)) return false;
                    break;
                case OutsideState::EndDict:
                    if (!EndDict(ptr, ed, errorStr)) return false;
                    break;
                case OutsideState::BeginArray:
                    if (!BeginArray(ptr, ed, errorStr)) return false;
                    break;
                case OutsideState::EndArray:
                    if (!EndArray(ptr, ed, errorStr)) return false;
                    break;
                case OutsideState::Continue:
                    if (!Continue(ptr, ed, errorStr)) return false;
                    break;
                case OutsideState::Number:
                    if (!Number(ptr, ed, errorStr)) return false;
                    break;
                case OutsideState::Guid:
                    if (!Guid(ptr, ed, errorStr)) return false;
                    break;
                case OutsideState::String:
                    if (!String<'\"'>(ptr, ed, errorStr)) return false;
                    break;
                case OutsideState::Characters:
                    if (!String<'\''>(ptr, ed, errorStr)) return false;
                    break;
                case OutsideState::Cut:
                    if (!Cut(ptr, ed, errorStr)) return false;
                    break;
                case OutsideState::Keyword:
                    if (!BeginKeyword(ptr, ed, errorStr)) return false;
                    break;
                case OutsideState::Comment:
                    if (!BeginComment(ptr, ed, errorStr)) return false;
                    break;
                case OutsideState::Path:
                    if (!BeginPath(ptr, ed, errorStr)) return false;
                    break;
                default: {
                    errorStr = (vstd::string("Illegal character '") + *ptr + "'");
                    return false;
                }
            }
        }
        return true;
    }
    WriteJsonVariant PrintField(Field *field) {
        switch (field->data.GetType()) {
            case 0: {
                auto &&v = field->data.get<0>();
                return WriteJsonVariant(std::string_view(v.data(), v.size()));
            }
            case 1: {
                auto &&v = field->data.get<1>();
                return WriteJsonVariant(PrintDict(v));
            }
            case 2: {
                auto &&v = field->data.get<2>();
                return WriteJsonVariant(PrintArray(v));
            }
            case 3: {
                auto &&v = field->data.get<3>();
                return WriteJsonVariant(v);
            }
            case 4: {
                auto &&v = field->data.get<4>();
                return WriteJsonVariant(v);
            }
            case 5: {
                auto &&v = field->data.get<5>();
                return WriteJsonVariant(v);
            }
            case 6: {
                auto &&v = field->data.get<6>();
                return WriteJsonVariant(v);
            }
            default:
                return WriteJsonVariant();
        }
    }
    IJsonDatabase *db;
    void SetDict(
        IJsonDict *dict,
        vstd::vector<std::pair<
            vstd::variant<
                vstd::vector<char>,
                int64,
                vstd::Guid>,
            Field *>> const &v) {
        dict->Reserve(v.size());
        for (auto &&i : v) {
            auto key = [&]() {
                switch (i.first.GetType()) {
                    case 0: {
                        auto &&v = i.first.get<0>();
                        return Key(std::string_view(v.begin(), v.size()));
                    }
                    case 1: {
                        auto &&v = i.first.get<1>();
                        return Key(v);
                    }
                    case 2: {
                        auto &&v = i.first.get<2>();
                        return Key(v);
                    }
                    default:
                        return Key();
                }
            }();

            auto value = PrintField(i.second);
            dict->Set(std::move(key), std::move(value));
        }
    }
    vstd::unique_ptr<IJsonDict> PrintDict(
        vstd::vector<std::pair<
            vstd::variant<
                vstd::vector<char>,
                int64,
                vstd::Guid>,
            Field *>> const &v) {
        auto dict = db->CreateDict();
        SetDict(dict.get(), v);
        return dict;
    }
    void SetArray(
        IJsonArray *arr,
        vstd::vector<Field *> const &v) {
        arr->Reserve(v.size());
        for (auto &&i : v) {
            arr->Add(PrintField(i));
        }
    }
    vstd::unique_ptr<IJsonArray> PrintArray(vstd::vector<Field *> const &v) {
        auto arr = db->CreateArray();
        SetArray(arr.get(), v);
        return arr;
    }
    template<bool escapeFirst>
    std::string_view GetKeyword(char const *&ptr, char const *const end) {
        char const *start = ptr;
        if constexpr (!escapeFirst) {
            if (recorders->outsideStates[*ptr] != OutsideState::Keyword) {
                //				return std::string_view(start, start);
                return {};
            }
        }
        ptr++;
        while (ptr != end) {
            if (!recorders->keywordStates[*ptr]) {
                return std::string_view(start, ptr - start);
            }
            ptr++;
        }
        return std::string_view(start, end - start);
    }
    bool BeginKeyword(char const *&ptr, char const *const end, vstd::string &errorStr) {
        auto var = GetKeyword<true>(ptr, end);
        switch (streamer) {
            case StreamState::None: {
                if (var != "var"sv) {
                    errorStr << "Illegal word: " << var;
                    return false;
                }
                ptr = GetNextChar(ptr, end);
                keywordName = GetKeyword<false>(ptr, end);
                if (keywordName.size() == 0) {
                    errorStr << "Need variable name!";
                    return false;
                }
                ptr = GetNextChar(ptr, end);
                if (*ptr != '=') {
                    errorStr << "Require '=' after variable name!";
                    return false;
                }
                ptr++;
                streamer = StreamState::SearchKeyword;
            } break;
            case StreamState::SearchKey:
            case StreamState::SearchElement: {
                auto ite = keywords.Find(var);
                if (!ite) {
                    errorStr << "Illegal keyword: " << var;
                    return false;
                }
                auto &&value = *ite.Value();
                switch (value.data.GetType()) {
                    case 0://string
                        if (!SetStringField(value.data.get<0>(), errorStr)) return false;
                        break;
                    case 1://dict
                    {
                        if (lastField->data.GetType() == 1) {
                            streamer = StreamState::SearchContinue;
                            (lastField->data.get<1>().end() - 1)->second = fieldPool->New(value.data.get<1>());
                        } else {
                            streamer = StreamState::SearchContinue;
                            lastField->data.get<2>().emplace_back(fieldPool->New(value.data.get<1>()));
                        }
                    } break;
                    case 2://array
                        if (lastField->data.GetType() == 1) {
                            streamer = StreamState::SearchContinue;
                            (lastField->data.get<1>().end() - 1)->second = fieldPool->New(value.data.get<2>());
                        } else {
                            streamer = StreamState::SearchContinue;
                            lastField->data.get<2>().emplace_back(fieldPool->New(value.data.get<2>()));
                        }
                        break;
                    case 3://int
                        if (!SetStringField(value.data.get<3>(), errorStr)) return false;
                        break;
                    case 4://double
                        if (!SetStringField(value.data.get<4>(), errorStr)) return false;
                        break;
                    case 5://guid
                        if (!SetStringField(value.data.get<5>(), errorStr)) return false;
                        break;
                    case 6:
                        if (!SetStringField(value.data.get<6>(), errorStr)) return false;
                        break;
                    case 7:
                        if (!SetStringField(value.data.get<7>(), errorStr)) return false;
                        break;
                }
                //SetStringField<
            } break;
            default:
                errorStr << "unexpected word: " << var;
                return false;
        }
        return true;
    }

public:
    bool Parse(
        std::string_view str,
        IJsonDatabase *db,
        IJsonDict *dict, vstd::string &errorStr) {
        if (!M_Parse(str, errorStr)) return false;
        if (rootField == nullptr) {
            return true;
        }
        this->db = db;
        if (rootField->data.GetType() != 1) {
            errorStr = ("Try parsing a non-dict text to dict");
            return false;
        }
        SetDict(
            dict,
            rootField->data.get<1>());
        return true;
    }
    bool Parse(
        std::string_view str,
        IJsonDatabase *db,
        IJsonArray *arr, vstd::string &errorStr) {
        if (!M_Parse(str, errorStr)) return false;
        if (rootField == nullptr) {
            return true;
        }
        this->db = db;
        if (rootField->data.GetType() != 2) {
            errorStr = ("Try parsing a non-array text to array");
            return false;
        }
        SetArray(
            arr,
            rootField->data.get<2>());
        return true;
    }
    SimpleJsonParser(
        decltype(fieldPool) poolPtr,
        vstd::vector<SimpleJsonParser> *subParserPtr) {
        {
            std::lock_guard lck(recorderIsInited);
            recorders.New();
        }
        fieldPool = poolPtr;
        subParsers = subParserPtr;
        keywords.Emplace("null", fieldPool->New(nullptr));
        keywords.Emplace("true", fieldPool->New(true));
        keywords.Emplace("false", fieldPool->New(false));
    }
    ~SimpleJsonParser() {
    }
    SimpleJsonParser(SimpleJsonParser const &) = delete;
    SimpleJsonParser(SimpleJsonParser &&) = default;
};

bool SimpleJsonParser::BeginPath(char const *&ptr, char const *const end, vstd::string &errorStr) {
    SimpleJsonParser *subParser;
    ++ptr;
    auto start = ptr;
    while (ptr != end && *ptr != '>') {
        ++ptr;
    }
    std::string_view pathName = std::string_view(start, ptr - start);
    ++ptr;
    auto GenerateSub = [&]() -> bool {
        BinaryReader red = vstd::string(pathName);
        if (!red) {
            errorStr << "Error file path " << pathName;
            return false;
        }
        auto fileVec = red.Read();
        subParser = &subParsers->emplace_back(SimpleJsonParser(fieldPool, subParsers));
        if (!subParser->M_Parse(std::string_view(reinterpret_cast<char const *>(fileVec.data()), fileVec.size()), errorStr)) {
            return false;
        }
        if (!subParser->rootField) {
            errorStr << "Illegal file " << pathName;
            return false;
        }
        return true;
    };
    switch (streamer) {
        case StreamState::SearchKeyword: {
            if (!GenerateSub()) return false;
            auto iteValue = keywords.TryEmplace(
                keywordName,
                subParser->rootField);
            if (!iteValue.second) {
                errorStr << "keyword " << keywordName << " conflict!";
                return false;
            }
            streamer = StreamState::None;
            keywordName = std::string_view(nullptr, 0);
        } break;
        case StreamState::SearchElement:
            if (!GenerateSub()) return false;
            if (lastField) {
                if (lastField->data.GetType() == 1) {
                    streamer = StreamState::SearchContinue;
                    (lastField->data.get<1>().end() - 1)->second = subParser->rootField;
                } else {
                    streamer = StreamState::SearchContinue;
                    lastField->data.get<2>().emplace_back(subParser->rootField);
                }
            } else {
                rootField = subParser->rootField;
                lastField = nullptr;
                streamer = StreamState::None;
            }
            break;
        case StreamState::None:
            if (!GenerateSub()) return false;
            rootField = subParser->rootField;
            break;
        default:
            errorStr << "Illegal include " << pathName;
            return false;
    }
    return true;
}
}// namespace parser
template<typename T>
vstd::optional<ParsingException> RunParse(
    T *ptr,
    IJsonDatabase *db,
    std::string_view str, bool clearLast) {
    if (clearLast) {
        ptr->Reset();
    }
    using namespace parser;
    vstd::vector<SimpleJsonParser> subParsers;
    vstd::Pool<SimpleJsonParser::Field, false> fieldPool(32, false);
    SimpleJsonParser parser(&fieldPool, &subParsers);
    vstd::string msg;
    if (!parser.Parse(str, db, ptr, msg)) {
        return ParsingException(std::move(msg));
    }
    return vstd::optional<ParsingException>();
}
vstd::optional<ParsingException> SimpleJsonValueDict::Parse(
    std::string_view str, bool clearLast) {
    using namespace parser;
    return RunParse<SimpleJsonValueDict>(
        this, db, str, clearLast);
}
vstd::optional<ParsingException> SimpleJsonValueArray::Parse(
    std::string_view str, bool clearLast) {
    using namespace parser;
    return RunParse<SimpleJsonValueArray>(
        this, db, str, clearLast);
}

}// namespace toolhub::db