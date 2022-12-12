'''
CVAT GITHUB
https://github.com/opencv/cvat

IMAGE 처리 프로젝트할 때 참고!!
https://github.com/pytorch/examples/blob/main/imagenet/main.py

파이토치 튜토리얼 한번은 꼭보기!
https://tutorials.pytorch.kr/recipes/recipes_index.html
'''

''' [크롤링 용어 정리]
- 셀레늄(selenium): 본래는 원격으로 웹브라우저를 컨트롤하려는 목표로 만들어 졌으나 크롤링에도 쓰이는 도구
- HTML: 웹페이지 상의 데이터들을 글자형식의 코드로 표현한 형태의 언어
- HTML 태그: 위의 HTML 코드 중에서 여러 성격의 데이터들을 책갈피로 찾을 수 있게 표시한 값
- 엘리먼트 : HTML 태그를 이용해 찾은 데이터
'''

from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time
import os
import urllib.request



# =============================================================================
# 폴더 구성
# =============================================================================
def create_folder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print("error : Creating directory ... " + directory)



# =============================================================================
# 키워드 입력 , chromedriver 실행
# =============================================================================
options = webdriver.ChromeOptions()
# Bluetooth: bluetooth_adapter_winrt.cc:1074 Getting Default Adapter failed. 오류 해결 코드
options.add_experimental_option('excludeSwitches', ['enable-logging']) 
# # 함수 실행 완료 후 강제종료되는데 이 코드를 써도 해결되지않음 -> while(True): pass 사용
# options.add_experimental_option("detach", True)

keywords = "사과"
chromedriver_path = "./cv2_ex/2.Data_Augmented_etc/2022-12-12/chromedriver.exe"
driver = webdriver.Chrome(chromedriver_path, options=options)
driver.implicitly_wait(3)



# =============================================================================
# 키워드 입력 selenium 실행
# =============================================================================
driver.get("https://www.google.co.kr/imghp?h1=ko")

# input -> /html/body/div[1]/div[3]/form/div[1]/div[1]/div[1]/div/div[2]/input
# button -> /html/body/div[1]/div[3]/form/div[1]/div[1]/div[1]/button

keyword = driver.find_element_by_xpath(
    '/html/body/div[1]/div[3]/form/div[1]/div[1]/div[1]/div/div[2]/input')
keyword.send_keys(keywords)
# keyword.send_keys(Keys.RETURN)

driver.find_element_by_xpath(
    '/html/body/div[1]/div[3]/form/div[1]/div[1]/div[1]/button').click()

# elem = driver.find_element_by_name("q")
# elem.send_keys(keyword)
# elem.send_keys(Keys.RETURN)

# (F12 개발자 도구)-------------------------------------------------------------
# <input class = "gLFyf" jsaction = "paste:puy29d;"
# maxlength = "2048" name = "q" type = "text" aria-autocomplete = "both"
# aria-haspopup = "false" autocapitalize = "off" autocomplete = "off"
# autocorrect = "off" autofocus = "" role = "combobox"
# spellcheck = "false" title = "검색" value = ""
# aria-label = "검색" data-ved = "0ahUKEwjK5OHq__L7AhVeQPUHHZemCioQ39UDCAM" >
# -----------------------------------------------------------------------------



# =============================================================================
#  스크롤 
# =============================================================================
print(keywords + '스크롤 중 .......')
elem = driver.find_element_by_tag_name('body')
for i in range(60):
    elem.send_keys(Keys.PAGE_DOWN)
    time.sleep(0.2) # 너무 빠르게 스크롤 내리면 봇이 차단당함! 꼭 써줘야함.

try:
    # //*[@id="islmp"]/div/div/div/div[2]/div[1]/div[2]/div[2]/input
    driver.find_element_by_xpath(
        '//*[@id="islmp"]/div/div/div/div[2]/div[1]/div[2]/div[2]/input').click()
    for i in range(60):
        elem.send_keys(Keys.PAGE_DOWN)
        time.sleep(0.2)
except:
    pass
'''
PAGE_DOWN키를 통해 검색결과가 더 많이 나올 수 있도록 스크롤해주며, 지속적으로 스크롤하다보면 
아래 '결과 더보기' 버튼이 생성된다.
해당 버튼이 나오면 클릭 후 계속 검색 진행
'''



# =============================================================================
# 이미지 개수
# =============================================================================
links = []
images = driver.find_elements_by_css_selector("img.rg_i.Q4LuWd")

for image in images:
    if image.get_attribute('src') != None:
        links.append(image.get_attribute('src'))

# <image src = "base64" # 이미지 파일을 업로드할때 대부분 base64로 인코딩하여 사용

print(keywords + "찾은 이미지 개수 : ", len(links))
time.sleep(2)

''' 결과
사과스크롤 중 .......
사과찾은 이미지 개수 :  446
'''



# =============================================================================
# 데이터 다운로드 
# =============================================================================
create_folder('./cv2_ex/2.Data_Augmented_etc/2022-12-12/'+keywords+'_img_download')
for index, i in enumerate(links):
    url = i
    start = time.time()
    urllib.request.urlretrieve(
        url, "./cv2_ex/2.Data_Augmented_etc/2022-12-12/" + keywords + "_img_download/" + keywords + "_" + str(index) + ".jpg")
    print(str(index) + "/" + str(len(links)) + " " + keywords +
          " 다운로드 시간 ------ : ", str(time.time() - start)[:5] + '초')

print(keywords + "다운로드 완료 !!")

# -----------------------------------------------------------------------------

while(True):
    pass