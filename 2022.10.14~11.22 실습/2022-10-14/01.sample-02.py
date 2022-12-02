#내 pc에 ip address를 확인하는 코드
import socket

in_addr = socket.gethostbyname(socket.gethostname()) 

print(in_addr)

# 2^8 = 256
# 8x4 = 32bit => IPv4 | 8x6 = 48bit => IPv6
# 2^32 = 4G 