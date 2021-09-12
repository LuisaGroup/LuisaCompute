#pragma vengine_package vengine_database

#include <cmath>

#include <serialize/DatabaseInclude.h>
#include <serialize/SimpleBinaryJson.h>
#include <serialize/SimpleParser.h>

namespace toolhub::db {

namespace parser {

template<typename T>
struct StateRecorder {

    T states[std::numeric_limits<char>::max() + 1];

    template<typename Mutex, typename Func>
    StateRecorder(
        Mutex &globalMutex,
        Func &&initFunc) {
        std::lock_guard<Mutex> lck(globalMutex);
        if constexpr (std::is_trivially_constructible_v<T>)
            memset(states, 0, sizeof(states));
        initFunc(states);
    }

    template<typename Func>
    explicit StateRecorder(Func &&initFunc) {
        if constexpr (std::is_trivially_constructible_v<T>)
            memset(states, 0, sizeof(states));
        initFunc(states);
    }

    [[nodiscard]] T const &Get(char p) const { return states[p]; }
    T const &operator[](char p) const { return states[p]; }
};

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
    Keyword
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

    static void InitKeywordState(bool *ptr) {
        for (auto &&i : vstd::ptr_range(ptr + 48, ptr + 58)) {
            i = true;
        }
        for (auto &&i : vstd::ptr_range(ptr + 'a', ptr + 'z' + 1)) {
            i = true;
        }
        for (auto &&i : vstd::ptr_range(ptr + 'A', ptr + 'Z' + 1)) {
            i = true;
        }
        ptr['_'] = true;
    }

    StateRecorders()
        : outsideStates(InitOutSideState),
          keywordStates(InitKeywordState),
          guidCheck(
              [&](bool *ptr) {
                  std::string_view sv = "0123456789abcedfABCDEF";
                  for (auto i : sv) {
                      ptr[i] = true;
                  }
              }),
          numCheck(
              [&](bool *ptr) {
                  std::string_view sv = "0123456789.eE-+";
                  for (auto i : sv) {
                      ptr[i] = true;
                  }
              }),
          spaces(
              [&](bool *ptr) {
                  ptr[' '] = true;
                  ptr['\t'] = true;
                  ptr['\r'] = true;
                  ptr['\n'] = true;
              }) {}
};

static luisa::spin_mutex recorderIsInited;
static vstd::optional<StateRecorders> recorders;

class SimpleJsonParser : public vstd::IOperatorNewBase {

    static char const *GetNextChar(char const *ptr, char const *end) {
        auto &&s = recorders->spaces;
        while (ptr != end) {
            if (!s[*ptr])
                return ptr;
            ptr++;
        }
        return ptr;
    }

public:
    struct Field {
        vstd::variant<
            std::vector<char>,
            std::vector<std::pair<
                vstd::variant<
                    std::vector<char>,
                    int64,
                    vstd::Guid>,
                Field *>>,
            std::vector<Field *>,
            int64,
            double,
            vstd::Guid>
            data;

        template<typename... Args>
        Field(Args &&...a)
            : data(std::forward<Args>(a)...) {
        }
    };

private:
    std::vector<SimpleJsonParser> *subParsers;
    vstd::Pool<Field, VEngine_AllocType::VEngine, false> *fieldPool;

    vstd::HashMap<std::string_view, Field *> keywords;
    std::string_view keywordName;
    std::vector<Field *> fieldStack;

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

    bool BeginDict(char const *&ptr, char const *const end, std::string &errorStr) {
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
            new (ptr) std::vector<std::pair<std::string_view, Field *>>();
        });
        streamer = StreamState::SearchKey;
        ++ptr;
        return true;
    }

    bool SetLastFieldState(std::string &errorStr) {
        if (fieldStack.empty()) {
            streamer = StreamState::None;
            //set keyword
            if (keywordName.data() != nullptr) {
                auto iteValue = keywords.TryEmplace(
                    keywordName,
                    lastField);
                if (!iteValue.second) {
                    errorStr += "keyword ";
                    errorStr += keywordName;
                    errorStr += " conflict!";
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

    bool EndDict(char const *&ptr, char const *const end, std::string &errorStr) {
        if (fieldStack.empty() || (*(fieldStack.end() - 1))->data.GetType() == 0) {
            errorStr = ("Illegal character '}'");
            return false;
        }
        if (!fieldStack.empty()) {
            fieldStack.erase(fieldStack.end() - 1);
        }
        ++ptr;
        return SetLastFieldState(errorStr);
    }

    bool BeginComment(char const *&ptr, char const *end, std::string &errorStr) {
        static_cast<void>(this);
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

    bool BeginArray(char const *&ptr, char const *const end, std::string &errorStr) {
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
            new (ptr) std::vector<Field *>();
        });
        streamer = StreamState::SearchElement;
        ++ptr;
        return true;
    }

    bool EndArray(char const *&ptr, char const *const end, std::string &errorStr) {
        if (fieldStack.empty() || (*(fieldStack.end() - 1))->data.GetType() == 1) {
            errorStr = ("Illegal character ']'");
            return false;
        }
        if (!fieldStack.empty()) {
            fieldStack.erase(fieldStack.end() - 1);
        }
        ++ptr;
        return SetLastFieldState(errorStr);
    }

    bool Continue(char const *&ptr, char const *const end, std::string &errorStr) {
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
    bool SetStringField(T &&str, std::string &errorStr) {
        switch (streamer) {
            case StreamState::SearchKeyword: {

                auto iteValue = keywords.TryEmplace(
                    keywordName,
                    vstd::MakeLazyEval([&]() {
                        return fieldPool->New(std::forward<T>(str));
                    }));
                if (!iteValue.second) {
                    errorStr += "keyword ";
                    errorStr += keywordName;
                    errorStr += " conflict!";
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

    bool Number(char const *&ptr, char const *const end, std::string &errorStr) {
        char const *start = ptr;
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
                    if (!processFloat())
                        return false;
                    break;
                } else if (*pin == 'e') {
                    pin++;
                    if (!processRate())
                        return false;
                    break;
                }
            }
            if (isNegative)
                v *= -1;
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
                }
            }
            return true;
        };
        auto error = [&]() {
            errorStr = ("Incorrect numeric format!");
            return false;
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
                GetRate))
            return false;
        if (ratePart || floatPart) {
            double value = (static_cast<double>(integerPart) + (floatPart ? *floatPart : 0)) * (ratePart ? pow(10, *ratePart) : 1);
            return SetStringField(value, errorStr);
        } else {
            return SetStringField(integerPart, errorStr);
        }
    }

    bool Guid(char const *&ptr, char const *const end, std::string &errorStr) {
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
        if (guidStr.size() != 32) {
            errorStr = ("Incorrect guid format!");
            return false;
        }
        return SetStringField(vstd::Guid(guidStr), errorStr);
    }

    template<char beginEnd>
    bool String(char const *&ptr, char const *const end, std::string &errorStr) {
        ptr++;
        char const *start = ptr;
        bool isSlash = false;
        std::vector<char> chars;
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
        if (!func())
            return false;
        ++ptr;
        return SetStringField(std::move(chars), errorStr);
    }

    bool Cut(char const *&ptr, char const *const end, std::string &errorStr) {
        if (streamer != StreamState::SearchKeyValueCut) {
            errorStr = ("Illegal character ':'");
            return false;
        }
        ptr++;
        streamer = StreamState::SearchElement;
        return true;
    }

    bool M_Parse(std::string_view str, std::string &errorStr) {
        auto endPtr = str.data() + str.size();
        char const *ptr = GetNextChar(str.data(), endPtr);
        while (true) {
            ptr = GetNextChar(ptr, endPtr);
            if (ptr == endPtr)
                return true;
            switch (recorders->outsideStates[*ptr]) {
                case OutsideState::BeginDict:
                    if (!BeginDict(ptr, endPtr, errorStr))
                        return false;
                    break;
                case OutsideState::EndDict:
                    if (!EndDict(ptr, endPtr, errorStr))
                        return false;
                    break;
                case OutsideState::BeginArray:
                    if (!BeginArray(ptr, endPtr, errorStr))
                        return false;
                    break;
                case OutsideState::EndArray:
                    if (!EndArray(ptr, endPtr, errorStr))
                        return false;
                    break;
                case OutsideState::Continue:
                    if (!Continue(ptr, endPtr, errorStr))
                        return false;
                    break;
                case OutsideState::Number:
                    if (!Number(ptr, endPtr, errorStr))
                        return false;
                    break;
                case OutsideState::Guid:
                    if (!Guid(ptr, endPtr, errorStr))
                        return false;
                    break;
                case OutsideState::String:
                    if (!String<'\"'>(ptr, endPtr, errorStr))
                        return false;
                    break;
                case OutsideState::Characters:
                    if (!String<'\''>(ptr, endPtr, errorStr))
                        return false;
                    break;
                case OutsideState::Cut:
                    if (!Cut(ptr, endPtr, errorStr))
                        return false;
                    break;
                case OutsideState::Keyword:
                    if (!BeginKeyword(ptr, endPtr, errorStr))
                        return false;
                    break;
                case OutsideState::Comment:
                    if (!BeginComment(ptr, endPtr, errorStr))
                        return false;
                    break;
                default: {
                    errorStr = (std::string("Illegal character '") + *ptr + "'");
                    return false;
                }
            }
        }
        return true;
    }

    WriteJsonVariant PrintField(Field *field) {
        return field->data.visit_with_default(
            WriteJsonVariant(),
            [&](std::vector<char> const &v) {
                return WriteJsonVariant(std::string_view(v.data(), v.size()));
            },
            [&](auto &&v) {
                return WriteJsonVariant(PrintDict(v));
            },
            [&](auto &&v) {
                return WriteJsonVariant(PrintArray(v));
            },
            [&](int64 const &v) {
                return WriteJsonVariant(v);
            },
            [&](double const &v) {
                return WriteJsonVariant(v);
            },
            [&](vstd::Guid const &v) {
                return WriteJsonVariant(v);
            });
    }

    IJsonDatabase *db{};

    void SetDict(
        IJsonDict *dict,
        std::vector<std::pair<
            vstd::variant<
                std::vector<char>,
                int64,
                vstd::Guid>,
            Field *>> const &v) {
        dict->Reserve(v.size());
        for (auto &&i : v) {
            auto key = i.first.visit_with_default(
                Key(),
                [&](std::vector<char> const &v) {
                    return Key(std::string_view(v.data(), v.size()));
                },
                [&](int64 const &v) {
                    return Key(v);
                },
                [&](vstd::Guid const &v) {
                    return Key(v);
                });
            auto value = PrintField(i.second);
            dict->Set(key, std::move(value));
        }
    }

    UniquePtr<IJsonDict> PrintDict(
        std::vector<std::pair<
            vstd::variant<
                std::vector<char>,
                int64,
                vstd::Guid>,
            Field *>> const &v) {
        auto dict = db->CreateDict();
        SetDict(dict.get(), v);
        return dict;
    }

    void SetArray(
        IJsonArray *arr,
        std::vector<Field *> const &v) {
        arr->Reserve(v.size());
        for (auto &&i : v) {
            arr->Add(PrintField(i));
        }
    }

    UniquePtr<IJsonArray> PrintArray(std::vector<Field *> const &v) {
        auto arr = db->CreateArray();
        SetArray(arr.get(), v);
        return arr;
    }

    template<bool escapeFirst>
    std::string_view GetKeyword(char const *&ptr, char const *const end) {
        char const *start = ptr;
        if constexpr (!escapeFirst) {
            if (recorders->outsideStates[*ptr] != OutsideState::Keyword) {
                return {};
            }
        }
        ptr++;
        while (ptr != end) {
            if (!recorders->keywordStates[*ptr]) {
                return {start, static_cast<size_t>(ptr - start)};
            }
            ptr++;
        }
        return {start, static_cast<size_t>(end - start)};
    }

    bool BeginKeyword(char const *&ptr, char const *const end, std::string &errorStr) {
        auto var = GetKeyword<true>(ptr, end);
        switch (streamer) {
            case StreamState::None: {
                if (var != "var") {
                    errorStr += "Illegal word: ";
                    errorStr += var;
                    return false;
                }
                ptr = GetNextChar(ptr, end);
                keywordName = GetKeyword<false>(ptr, end);
                if (keywordName.empty()) {
                    errorStr += "Need variable name!";
                    return false;
                }
                ptr = GetNextChar(ptr, end);
                if (*ptr != '=') {
                    errorStr += "Require '=' after variable name!";
                    return false;
                }
                ptr++;
                streamer = StreamState::SearchKeyword;
                break;
            }
            case StreamState::SearchKey:
            case StreamState::SearchElement: {
                auto ite = keywords.Find(var);
                if (!ite) {
                    errorStr += "Illegal keyword: ";
                    errorStr += var;
                    return false;
                }
                auto &&value = *ite.Value();
                switch (value.data.GetType()) {
                    case 0://string
                        if (!SetStringField(value.data.get<0>(), errorStr))
                            return false;
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
                        if (!SetStringField(value.data.get<3>(), errorStr))
                            return false;
                        break;
                    case 4://double
                        if (!SetStringField(value.data.get<4>(), errorStr))
                            return false;
                        break;
                    case 5://guid
                        if (!SetStringField(value.data.get<5>(), errorStr))
                            return false;
                        break;
                }
                break;
            }
            default:
                errorStr += "unexpected word: ";
                errorStr += var;
                return false;
        }
        return true;
    }

public:
    bool Parse(
        std::string_view str,
        IJsonDatabase *database,
        IJsonDict *dict, std::string &errorStr) {
        if (!M_Parse(str, errorStr))
            return false;
        if (rootField == nullptr) {
            return true;
        }
        this->db = database;
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
        IJsonDatabase *database,
        IJsonArray *arr, std::string &errorStr) {
        if (!M_Parse(str, errorStr))
            return false;
        if (rootField == nullptr) {
            return true;
        }
        this->db = database;
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
        std::vector<SimpleJsonParser> *subParserPtr) {
        {
            std::lock_guard lck(recorderIsInited);
            recorders.New();
        }
        fieldPool = poolPtr;
        subParsers = subParserPtr;
    }
    ~SimpleJsonParser() override = default;
    SimpleJsonParser(SimpleJsonParser const &) = delete;
    SimpleJsonParser(SimpleJsonParser &&) = default;
};

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
    std::vector<SimpleJsonParser> subParsers;
    vstd::Pool<SimpleJsonParser::Field, VEngine_AllocType::VEngine, false> fieldPool(32, false);
    SimpleJsonParser parser(&fieldPool, &subParsers);
    std::string msg;
    if (!parser.Parse(str, db, ptr, msg)) {
        return ParsingException(std::move(msg));
    }
    return {};
}

vstd::optional<ParsingException> SimpleBinaryJson::Parse(
    std::string_view str, bool clearLast) {
    return RunParse<SimpleJsonValueDict>(
        static_cast<SimpleJsonValueDict *>(GetRootNode()), this, str, clearLast);
}

vstd::optional<ParsingException> SimpleJsonValueDict::Parse(
    std::string_view str, bool clearLast) {
    using namespace parser;
    return RunParse<IJsonDict>(
        this, db, str, clearLast);
}

vstd::optional<ParsingException> SimpleJsonValueArray::Parse(
    std::string_view str, bool clearLast) {
    using namespace parser;
    return RunParse<IJsonArray>(
        this, db, str, clearLast);
}

vstd::optional<ParsingException> ConcurrentBinaryJson::Parse(
    std::string_view str, bool clearLast) {
    using namespace parser;
    return RunParse<ConcurrentJsonValueDict>(
        static_cast<ConcurrentJsonValueDict *>(GetRootNode()), this, str, clearLast);
}

vstd::optional<ParsingException> ConcurrentJsonValueDict::Parse(
    std::string_view str, bool clearLast) {
    using namespace parser;
    return RunParse<IJsonDict>(
        this, db, str, clearLast);
}

vstd::optional<ParsingException> ConcurrentJsonValueArray::Parse(
    std::string_view str, bool clearLast) {
    using namespace parser;
    return RunParse<IJsonArray>(
        this, db, str, clearLast);
}

}// namespace toolhub::db
