import ctypes
from ctypes import c_char_p
import time

so = ctypes.CDLL("split.so")
so.Run_greedy.restype = ctypes.c_char_p
start = time.time()
result = so.Run_greedy(c_char_p("httpResponse".encode("utf-8")))
end = time.time()
print(f'greedy split：{result}，time spent：{end - start}')

so.Run_greedy_prefix.restype = ctypes.c_char_p
start = time.time()
result = so.Run_greedy_prefix(c_char_p("httpResponse".encode("utf-8")))
end = time.time()
print(f'greedy prefix split：{result}，time spent：{end - start}')

so.Run_greedy_suffix.restype = ctypes.c_char_p
start = time.time()
result = so.Run_greedy_suffix(c_char_p("httpResponse".encode("utf-8")))
end = time.time()
print(f'greedy suffix split：{result}，time spent：{end - start}')

so.Run_conserv.restype = ctypes.c_char_p
start = time.time()
result = so.Run_conserv(c_char_p("httpResponse".encode("utf-8")))
end = time.time()
print(f'consev split：{result}，time spent：{end - start}')
