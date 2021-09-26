import ctypes
from ctypes import c_char_p
import time

so = ctypes.CDLL("split.so")
so.Run_greedy.restype = ctypes.c_char_p
start = time.time()
result = so.Run_greedy(c_char_p("httpResponse".encode("utf-8")))
end = time.time()
print(f'greedy split：{result}，耗时：{end - start}')

so.Run_conserv.restype = ctypes.c_char_p
start = time.time()
result = so.Run_conserv(c_char_p("httpResponse".encode("utf-8")))
end = time.time()
print(f'consev split：{result}，耗时：{end - start}')

so.Run_gentest.restype = ctypes.c_char_p
start = time.time()
result = so.Run_gentest(c_char_p("httpResponse".encode("utf-8")))
end = time.time()
print(f'gentest split：{result}，耗时：{end - start}')