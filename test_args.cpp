#include <windows.h>
#include <shellapi.h>
#include <iostream>
#include <vector>
#include <string>

int main() {
    int n = 0;
    LPWSTR* args = CommandLineToArgvW(GetCommandLineW(), &n);
    std::cout << "argc=" << n << std::endl;
    for (int i = 0; i < n; ++i) {
        int len = WideCharToMultiByte(CP_UTF8, 0, args[i], -1, nullptr, 0, nullptr, nullptr);
        std::string s;
        if (len > 0) {
            s.resize(len - 1);
            WideCharToMultiByte(CP_UTF8, 0, args[i], -1, s.data(), len, nullptr, nullptr);
        }
        std::cout << "argv[" << i << "]=" << s << std::endl;
    }
    LocalFree(args);
    return 0;
}
