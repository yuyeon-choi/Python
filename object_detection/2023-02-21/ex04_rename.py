import os

# 주어진 디렉토리에 있는 항목들의 이름을 담고 있는 리스트를 반환합니다.
# 리스트는 임의의 순서대로 나열됩니다.
file_path = 'C:/Users/yuyeon/Downloads/military UAV 02-21 15-10까지 작업물'
file_names = os.listdir(file_path)
file_names

i = 1336
for name in file_names:
    src = os.path.join(file_path, name)
    dst = 'military drone_' + str(i) + '.png'
    dst = os.path.join(file_path, dst)
    os.rename(src, dst)
    i += 1