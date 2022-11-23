//
// Created by ChenXin on 2022/11/20.
//

#include <core/thread_pool.h>
#include <core/logging.h>

#include <Windows.h>

using namespace luisa;

int main() {
    ThreadPool::global().async([]() {
        LUISA_INFO("Start thread 1");
        ThreadPool::global().async([]() {
            LUISA_INFO("Start thread 2a");
            ThreadPool::global().async([]() {
                LUISA_INFO("Start thread 3");
                Sleep(1000);
                LUISA_INFO("End thread 3");
            });
            Sleep(1000);
            LUISA_INFO("End thread 2a");
        });
        Sleep(2000);
        ThreadPool::global().async([]() {
            LUISA_INFO("Start thread 2b");
            Sleep(1000);
            LUISA_INFO("End thread 2b");
        });
        LUISA_INFO("End thread 1");
    });
    ThreadPool::global().synchronize();
    Sleep(1000);
    return 0;
}