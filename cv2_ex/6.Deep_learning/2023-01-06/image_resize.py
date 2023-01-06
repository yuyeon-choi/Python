from PIL import Image # pip install Pillow
import os

def expand2square(pil_img, background_color) :
    width, height = pil_img.size
    # print(width, height)
    if width == height :
        return pil_img
    elif width > height :
        result = Image.new(pil_img.mode, (width, width), background_color)
        # image add(추가할 이미지, 붙일 위치(가로, 세로))
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else :
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


def image_file(image_folder_path) :
    all_root = []
    for (path, dir, files)  in os.walk(image_folder_path) :
        for filename in files :
            # image.png --> (splitext(ㅁㅁㅁㅁ)[-1]) --> .png
            ext = os.path.splitext(filename)[-1]
            # print(ext)
            # print(filename)
            if ext == ".png" :
                root = os.path.join(path, filename)
                # ./cvat_annotations/annotations.xml | root 안에 저장될 내용
                all_root.append(root)
            else :
                print("no image file...")
                continue
    return all_root

img_path_list = image_file("./dataset/val/r/")
print(img_path_list)

for img_path in img_path_list :
    print(img_path)
    # image_name_temp = img_path.split("/")
    image_name_temp = os.path.basename(img_path)
    # image_name_temp2 = os.path.abspath(img_path) # abspath : 절대 경로 뽑아줌
    # print(image_name_temp2)
    image_name = image_name_temp.replace(".png", "")
    print(image_name_temp)

    img = Image.open(img_path)
    img_new = expand2square(img,(0,0,0)).resize((224,224))
    os.makedirs("./resize/val/r", exist_ok=True)
    img_new.save(f"./resize/val/r/{image_name}.png", quality=100)