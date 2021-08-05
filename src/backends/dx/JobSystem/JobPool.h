#include <util/vector.h>
#include <Common/spin_mutex.h>

template<typename T>
class JobPool {
private:
	ArrayList<T*> allocatedPool;
	ArrayList<T*> list[2];
	luisa::spin_mutex mtx;
	bool switcher = false;
	uint32_t capacity;
	void ReserveList(ArrayList<T*>& vec) {
		T* t = new T[capacity];
		allocatedPool.push_back(t);
		vec.resize(capacity);
		for (uint32_t i = 0; i < capacity; ++i) {
			vec[i] = t + i;
		}
	}

public:
	JobPool(uint32_t capacity) : capacity(capacity) {
		allocatedPool.reserve(10);
		list[0].reserve(capacity * 2);
		list[1].reserve(capacity * 2);
		ReserveList(list[0]);
		ReserveList(list[1]);
	}

	void UpdateSwitcher() {
		switcher = !switcher;
	}

	T* New() {
		ArrayList<T*>& lst = list[switcher];
		if (lst.empty()) ReserveList(lst);
		T* value = lst.erase_last();
		value->Reset();
		return value;
	}

	void Delete(T* value) {
		ArrayList<T*>& lst = list[!switcher];
		value->Dispose();
		std::lock_guard<luisa::spin_mutex> lck(mtx);
		lst.push_back(value);
	}

	~JobPool() {
		for (auto ite = allocatedPool.begin(); ite != allocatedPool.end(); ++ite) {
			delete[] * ite;
		}
	}
};
