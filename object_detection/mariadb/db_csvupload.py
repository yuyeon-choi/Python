import pandas as pd
import pymysql
import csv
# from mysql.connector import (connection)

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

# sql = "SELECT * FROM test" 
# with conn:
#     with conn.cursor() as cur:
#         cur.execute(sql)
#         result = cur.fetchall()
#         for data in result:
#             print(data)


# # 5. 데이터 csv, txt 파일 업로드
# # CSV 파일 경로...
with open('C:/Users/yuyeon/Desktop/dummy_dataset/dummy3.csv', 'r', encoding='utf8') as f:       # 경로변경
    csvReader = csv.reader(f)
    next(csvReader)     # 첫줄에 row값 있을때

#     # 컬럼 매핑
    for row in csvReader:
        id = (row[0])
        Latitude = (row[1])
        Longitude = (row[2])
        StartTime = (row[3])
        EndTime = (row[4])
        detection = (row[5])
        # print(id)
        # print(Latitude)
        # print(Longitude)
        # print(StartTime)
        # print(EndTime)
        # print(detection)
        sql = """INSERT INTO dummy3 (id, Latitude, Longitude, StartTime, EndTime, detection) VALUES (%s, %s, %s, %s, %s, %s)"""     # 테이블명 바꾸기
        cursor.execute(sql, (id, Latitude, Longitude, StartTime, EndTime, detection))

#     # DB의 변화 저장
    conn.commit()
    f.close()
    conn.close()