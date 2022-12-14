import os
import json
import cv2

# json path
json_path = "./cv2_ex/2.Data_Augmented_etc/2022-12-13/image01_T/annotations/instances_default_box.json"

# json 파일 읽기

with open(json_path, "r") as f:
    coco_info = json.load(f)

# print(coco_info)

assert len(coco_info) > 0, "파일 읽기 실패"

# 카테고리 정보 수집
categories = dict()
for category in coco_info['categories']:
    print(category)
    categories[category["id"]] = category["name"]

# print("categories info >> ", categories)

# annotation 정보 수집
''' 
annotation :
본래 주석이란 뜻으로, 주석과는 그 역할이 다르지만 주석처럼 코드에 달아 클래스에 특별한 기능을 주입할 수 있다.
'''
ann_info = dict()
for annotation in coco_info['annotations']:
    image_id = annotation["image_id"]
    bbox = annotation["bbox"]
    category_id = annotation["category_id"]
    # print(
    #     f"image_id : {image_id}, category_id : {category_id} , bbox : {bbox}")
    ''' 결과 
    image_id : 1, category_id : 1 , bbox : [0.0, 29.22, 251.91, 191.06]
    .
    .
    .
    '''
    if image_id not in ann_info:
        ann_info[image_id] = {
            "boxes": [bbox], "categories": [category_id]
        }
    else:
        ann_info[image_id]["boxes"].append(bbox)
        ann_info[image_id]["categories"].append(categories[category_id])

# print("ann_info >> ", ann_info)

for image_info in coco_info['images']:
    # print(image_info)
    filename = image_info['file_name']
    width = image_info['width']
    height = image_info['height']
    img_id = image_info['id']
    print(filename, width, height, img_id)

    # 이미지 가져오기 위한 처리
    file_path = os.path.join("./cv2_ex/2.Data_Augmented_etc/2022-12-13/image01_T/images", filename)
    img = cv2.imread(file_path)
    try:
        annotation = ann_info[img_id]
    except KeyError:
        continue
    
    
    # box category
    for bbox, category in zip(annotation['boxes'], annotation['categories']):
        x1, y1, w, h = bbox
        print(filename, x1, y1, w, h)

        rec_img = cv2.rectangle(img, (int(x1), int(y1)),
                                (int(x1+w), int(y1+h)), (225, 0, 255), 2)
    
    # cv2.imwrite(f"./cv2_ex/2.Data_Augmented_etc/2022-12-13/image01_T/save_image/{filename}", rec_img)   # 결과 이미지 저장
    # cv2.imshow("test", rec_img)
    # cv2.waitKey(0)