

from time import perf_counter

def test(n):
    t1 = perf_counter()
    s = sum([[2,3] for _ in range(n)], [])
    # s = []
    # for _ in range(n):
    #     s += [2,3]
    print(f"time({n}):", perf_counter()-t1)

test(2**15)
test(2**16)
test(2**17)
test(2**18)
test(2**19)
test(2**20)
test(2**21)
test(2**22)