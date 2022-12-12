import PIL
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time
from selenium.webdriver.common.by import By
import base64
from PIL import Image
from io import BytesIO
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import urllib.request
from selenium.webdriver.chrome.options import Options

def selenium_scroll_option(driver):
    SCROLL_PAUSE_SEC = 3
    # 스크롤 높이 가져옴
    last_height = driver.execute_script("return document.body.scrollHeight")
    while True:
        # 끝까지 스크롤 다운
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        # 3초 대기
        time.sleep(SCROLL_PAUSE_SEC)
        # 스크롤 다운 후 스크롤 높이 다시 가져옴
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            break
        last_height = new_height

def main():
    option = Options()
    option.add_argument(
        'User_Agent : Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36')
    os.makedirs('./search_data', exist_ok=True)
    search_list = input('검색어 입력(,로 구분 입력) : ')
    N = input('수집할 데이터 개수 : ')
    search_list = [i.strip() for i in search_list.split(',')]
    # 브라우저 로딩 시간 고려한 시간
    # time sleep과는 다르게 로딩 다되면 바로 실행
    driver = webdriver.Chrome(options=option)
    driver.implicitly_wait(10)
    for search in search_list:
        img_dir = f'./search_data/{search}'
        os.makedirs(img_dir, exist_ok=True)
        driver.get('https://www.google.com')
        elem = driver.find_element(By.NAME, 'q')
        elem.clear()
        elem.send_keys(search)
        elem.send_keys(Keys.RETURN)
        assert "No results found." not in driver.page_source

        # 이미지 메뉴 누르기
        driver.find_element(By.XPATH, '/html/body/div[7]/div/div[4]/div/div[1]/div/div[1]/div/div[2]/a').click()
        selenium_scroll_option(driver)

        driver.find_element(By.XPATH, '//*[@id="islmp"]/div/div/div/div/div[1]/div[2]/div[2]/input').click()  # 이미지 / 지도 /등등 바꿀수 있음 
        selenium_scroll_option(driver)
        img_srcs = driver.find_elements(By.CLASS_NAME, 'rg_i')

        url_list = []
        last = 0
        for idx, img_src in enumerate(img_srcs):
            base64_image = img_src.get_attribute('src')
            try:
                if base64_image:
                    if 'base64' in base64_image:
                        img = Image.open(BytesIO(base64.b64decode(base64_image.split(',')[-1])))
                        img.save(os.path.join(img_dir, str(idx)+'.png'))
                        last = idx + 1
                    else:
                        url_list.append(base64_image)

            except PIL.UnidentifiedImageError:
                print(base64_image)
            except AttributeError:
                print(base64_image)
                continue
            if int(N) == idx:
                break
        [urllib.request.urlretrieve(url, os.path.join(img_dir, str(last+i)+'.png')) for i, url in enumerate(set(url_list))]
        print(len(set(url_list)))
    driver.close()

if __name__ =='__main__':
    main()

    # rose flower, sunflower, Cherry Blossom, gypsophila, Tulip