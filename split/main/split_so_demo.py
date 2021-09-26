import ctypes
import time

so = ctypes.CDLL('split.so')
start = time.time()
result = so.Run_greedy("httpResponse")
end = time.time()
print(f'斐波那契数列第40项：{result}，耗时：{end - start}')
