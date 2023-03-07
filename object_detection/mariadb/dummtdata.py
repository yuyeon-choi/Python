import random
import datetime

# strftime(%Y-%m-%d %H:%M:%S)
for i in range(150):
    m = random.randint(1, 12)
    d = random.randint(1, 30)

    H = random.randint(00, 23)
    M = random.randint(00, 59)
    S = random.randint(00, 59)
    print('2023-'+ str(m) +'-' + str(d), str(H)+ ':'+ str(M)+ ':'+ str(S))