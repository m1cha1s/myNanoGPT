from time import perf_counter

start = perf_counter()
for i in range(1_000_000_000):
    pass
t = perf_counter()-start

print(f"{t/1_000_000_000}s/it")