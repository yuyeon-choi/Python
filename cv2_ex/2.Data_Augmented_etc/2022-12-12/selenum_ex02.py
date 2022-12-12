from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from multiprocessing import Pool
import time
import os
import urllib.request
import pandas as pd


def create_folder(directory):
    # 폴더 생성
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print("error : Creating directory ... " + directory)

# 검색 키워드 호출
key = pd.read_csv("./cv2_ex/2.Data_Augmented_etc/2022-12-12/keyword.txt", encoding='utf-8', names=['keyword'])
keyword = []
[keyword.append(key['keyword'][x]) for x in range(len(key))]
print(keyword)