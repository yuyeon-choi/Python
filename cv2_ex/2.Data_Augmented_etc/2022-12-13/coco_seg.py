import os
import json
import cv2

json_path = "./cv2_ex/2.Data_Augmented_etc/2022-12-13/image01_T/annotations/instances_default_seg.json"

with open(json_path, "r") as f:
    coco_info = json.load(f)

print(coco_info)
# 파일 읽기 실패
assert len(coco_info) > 0, "파일 읽기 실패"

# 카테고리 수집
categories = dict()
for category in coco_info['categories']:
    categories[category["id"]] = category["name"]
    
    # annotation 정보
ann_info = dict()
for annotation in coco_info['annotations']:
    image_id = annotation["image_id"]
    bbox = annotation["bbox"]
    category_id = annotation["category_id"]
    segmentation = annotation["segmentation"]

    # print(image_id, category_id, bbox, segmentation)

    if image_id not in ann_info:
        ann_info[image_id] = {
            "boxes": [bbox], "segmentation": [segmentation],
            "categories": [category_id]
        }
    else:
        ann_info[image_id]["boxes"].append(bbox)
        ann_info[image_id]["segmentation"].append(segmentation)
        ann_info[image_id]["categories"].append(categories[category_id])    # 주의! <리스트를 하나의 객체로 추가>

for image_info in coco_info["images"]:
    filename = image_info['file_name']
    width = image_info['width']
    height = image_info['height']
    img_id = image_info['id']

    file_path = os.path.join("./cv2_ex/2.Data_Augmented_etc/2022-12-13/image01_T/images", filename)
    img = cv2.imread(file_path)

    try:
        annotation = ann_info[img_id]
    except KeyError:
        continue

    for bbox, segmentation, category in zip(annotation['boxes'],
                                            annotation['segmentation'], annotation['categories']):
        x1, y1, w, h = bbox
        import numpy as np
        for seg in segmentation:
            # print(seg)
            poly = np.array(seg, np.int32).reshape((int(len(seg)/2), 2))    # reshape((int(len(seg)/2) : 차원수 줄이기
            # print(poly)
            poly_img = cv2.polylines(img, [poly], True, (255, 0, 0), 2)
    cv2.imshow("test", poly_img)
    cv2.waitKey(0)

    cv2.imwrite(f"./cv2_ex/2.Data_Augmented_etc/2022-12-13/image01_T/save_image_seg/{filename}", poly_img)   # 결과 이미지 저장
    cv2.imshow("test", poly_img)
    cv2.waitKey(0)
'''
+) 
for category in coco_info['categories']:
    categories[category["id"]] = category["name"]
    pass <- ?

pass는 실행할 것이 아무 것도 없다는 것을 의미한다. 
따라서 아무런 동작을 하지 않고 다음 코드를 실행한다.
'''