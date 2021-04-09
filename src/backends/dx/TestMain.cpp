#include <iostream>
#include <optional>
#include <Common/DLL.h>
#include <vector>
#include <render_graph/rg_system.h>
#include <render_graph/rg_node.h>
#include <render_graph/rg_executor.h>
#include <Common/vstring.h>
namespace luisa::compute {
static uint64_t GLOBAL_SIGNAL = 1;

class FuckingExecutor final : public RGExecutor {
public:
	vengine::string_view name;
	uint64_t signal() override {
		std::cout
			<< "signal "
			<< GLOBAL_SIGNAL
			<< " by "
			<< name
			<< '\n';
			
		return GLOBAL_SIGNAL++;
	}
	void gpu_wait(uint64_t signal, RGExecutor* signal_source) override {
		std::cout
			<< name
			<< " sync to "
			<< signal
			<< " by signal "
			<< static_cast<FuckingExecutor*>(signal_source)->name
			<< '\n';
	}
	void cpu_sync(uint64_t signal) override {
		std::cout
			<< "cpu sync to "
			<< signal
			<< '\n';
	}
	void execute(std::function<void()>&& func) override {
		func();
	}
};
}// namespace luisa::compute

int main() {
	using namespace luisa::compute;
	std::function<void()> copy0([]() {
		std::cout << "Working FIRST" << std::endl;
	});
	std::function<void()> copy1([]() {
		std::cout << "Working SECOND" << std::endl;
	});
	std::function<void()> copy2([]() {
		std::cout << "Working THIRD" << std::endl;
	});

	std::function<void()> compute0([]() {
		std::cout << "StillWorking FIRST" << std::endl;
	});
	std::function<void()> compute1([]() {
		std::cout << "StillWorking SECOND" << std::endl;
	});
	std::function<void()> compute2([]() {
		std::cout << "StillWorking THIRD" << std::endl;
	});
	FuckingExecutor copyExecutor;
	copyExecutor.name = "copy"_sv;
	FuckingExecutor computeExecutor;
	computeExecutor.name = "compute"_sv;
	constexpr uint COPY_QUEUE = 0;
	constexpr uint COMPUTE_QUEUE = 1;
	RGSystem sys;
	RGNode node0(std::move(copy0), COMPUTE_QUEUE);
	RGNode node1(std::move(copy1), COMPUTE_QUEUE);
	RGNode node2(std::move(copy2), COMPUTE_QUEUE);

	RGNode cnode0(std::move(compute0), COMPUTE_QUEUE);
	RGNode cnode1(std::move(compute1), COMPUTE_QUEUE);
	RGNode cnode2(std::move(compute2), COMPUTE_QUEUE);

	cnode0.add_depend_node(&node0);

	node0.push_to_system(&sys);
	node1.push_to_system(&sys);
	node2.push_to_system(&sys);
	cnode0.push_to_system(&sys);
	cnode1.push_to_system(&sys);
	cnode2.push_to_system(&sys);
	RGExecutor* exes[] = {
		&copyExecutor,
		&computeExecutor};
	sys.execute(std::span<RGExecutor*>(exes));
}