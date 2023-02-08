# 전처리 이후 이미지와 라벨의 데이터 수가 다를 시
# 다른 데이터를 찾은 후 삭제해주는 작업

import os, glob
a,b = [], []
for mode in ['train', 'val']:
    path1 = f"./dataset/{mode}/images"
    image_paths = glob.glob(os.path.join(path1, "*.jpg"))
    for image_path in image_paths :
        image_path = os.path.basename(image_path).split('\\')[-1].replace('.jpg', '')
        a.append(image_path)
    a = sorted(a)

    path2 = f"D:/dataset/{mode}/labels"
    label_paths = glob.glob(os.path.join(path2, "*.text"))
    for label_path in label_paths :
        label_path = os.path.basename(label_path).split('\\')[-1].replace('.text', '')
        b.append(label_path)
    b = sorted(b)
    diffence_list = list(set(a) - set(b))

    for item in diffence_list : # 기존 34463 | 변경 후 34334
        os.remove(f'./datasets/{mode}/images/{item}.jpg')
