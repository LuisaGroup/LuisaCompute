//
// Created by ChenXin on 2022/11/20.
//

#include <thread>
#include <chrono>
#include <core/thread_pool.h>
#include <core/logging.h>

using namespace luisa;

int main() {

    static constexpr auto sleep = [](auto n) noexcept {
        using namespace std::chrono_literals;
        std::this_thread::sleep_for(n * 1s);
    };

    ThreadPool::global().async([]() {
        LUISA_INFO("Start thread 1");
        ThreadPool::global().async([]() {
            LUISA_INFO("Start thread 2a");
            ThreadPool::global().async([]() {
                LUISA_INFO("Start thread 3");
                sleep(1u);
                LUISA_INFO("End thread 3");
            });
            sleep(1u);
            LUISA_INFO("End thread 2a");
        });
        sleep(2u);
        ThreadPool::global().async([]() {
            LUISA_INFO("Start thread 2b");
            sleep(1u);
            LUISA_INFO("End thread 2b");
        });
        LUISA_INFO("End thread 1");
    });
    ThreadPool::global().synchronize();
    sleep(1u);
}