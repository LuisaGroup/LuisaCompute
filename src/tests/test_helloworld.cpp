#include <luisa/core/fiber.h>
#include <luisa/core/logging.h>
using namespace luisa;
using namespace luisa::fiber;

int main(int argc, char *argv[]) {
    scheduler sc;
    auto ft = async([&](){
        std::this_thread::sleep_for(std::chrono::seconds(1));
        return 666;
    });
    LUISA_INFO("{}", ft.wait());
}
