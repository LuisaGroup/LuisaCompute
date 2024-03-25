#include <luisa/core/fiber.h>
namespace luisa::fiber {
scheduler::scheduler() noexcept
    : internal(internal_t::Config::allCores()) {
    internal.bind();
}
scheduler::scheduler(uint32_t thread_count) noexcept
    : internal(internal_t::Config().setWorkerThreadCount(static_cast<int>(thread_count))) {
    internal.bind();
}
scheduler::~scheduler() noexcept {
    internal.unbind();
}
counter::counter(uint32_t init_count) noexcept : internal(init_count) {}
void counter::wait() const noexcept { internal.wait(); }
void counter::add(const uint32_t x) const noexcept { internal.add(x); }
bool counter::decrement() const noexcept { return internal.done(); }
counter::counter(internal_t &&other) noexcept : internal(std::move(other)) {}
event::event() noexcept : internal(marl::Event::Mode::Manual) {}

void event::wait() const noexcept {
    internal.wait();
}
void event::signal() const noexcept { internal.signal(); }
bool event::test() const noexcept { return internal.test(); }
void event::clear() const noexcept { internal.clear(); }
event::event(internal_t &&other) noexcept : internal(std::move(other)) {}
void schedule(luisa::function<void()> &&func) {
    marl::schedule(std::move(func));
}
uint32_t worker_thread_count() {
    return marl::Scheduler::get()->config().workerThread.count;
}
namespace detail {
typeless_future::typeless_future(size_t mem_size) noexcept
    : internal{mem_size} {
}

void *typeless_future::wait() const noexcept {
    return internal.wait();
}
void typeless_future::signal(eastl::move_only_function<void(void*)> const& new_ctor, void (*new_dtor)(void *)) const noexcept {
    internal.signal(new_ctor, new_dtor);
}
bool typeless_future::test() const noexcept {
    return internal.test();
}
void typeless_future::clear() const noexcept {
    internal.clear();
}
}// namespace detail
}// namespace luisa::fiber