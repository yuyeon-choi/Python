## 실시간 데이터 업데이트

import pandas as pd
import pymysql
import csv

# 기본 setting
conn = pymysql.connect(
    user='bbongs',
    passwd='Qweasd123$',
    host='219.251.99.114',
    db='drone_project',
    charset='utf8'
)
cursor = conn.cursor()
conn.commit()

