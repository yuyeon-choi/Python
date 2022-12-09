from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time
from selenium.webdriver.common.by import By
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

Chrome_web = webdriver.Chrome()
Chrome_web.implicitly_wait(10)
Chrome_web.get('http://www.google.com')

elem = Chrome_web.find_element(By.NAME, 'q')
elem.clear()
search = 'banana'
elem.send_keys(search)
elem.send_keys(Keys.RETURN)
# elem.clear()

# 이미지 메뉴 고르기
Chrome_web.find_element(By.XPATH, '/html/body/div[7]/div/div[4]/div/div[1]/div/div[1]/div/div[2]/a').click()


elem = Chrome_web.find_element(By.XPATH, '//*[@id="islmp"]/div/div/div/div/div[1]/div[2]/div[2]/input')
elem.send_keys(Keys.RETURN)
input()

# selenium_scroll_option(driver)


# driver.find_element(By.XPATH, '//*[@id="islmp"]/div/div/div/div/div[1]/div[2]/div[2]/input').click()
# selenium_scroll_option(driver)
# img_srcs = driver.find_elements(By.CLASS_NAME, 'rg_i')

# for idx, img_src in enumerate(img_srcs):
#     base64_image = img_src.get_attribute('src')
#     try:
#         if base64_image:
#             if 'base64' in base64_image:
#                 img = Image.open(BytesIO())