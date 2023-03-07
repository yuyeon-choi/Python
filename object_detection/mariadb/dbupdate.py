# 테이블 정보
detect ='''
    CREATE TABLE `detect` (
	`id` INT(11) NULL DEFAULT NULL,
	`cctv` VARCHAR(50) NULL DEFAULT NULL COLLATE 'utf8_general_ci',
	`Latitude` DECIMAL(20,5) NULL DEFAULT NULL,
	`Longitude` DECIMAL(20,5) NULL DEFAULT NULL,
	`StartTime` DATETIME NULL DEFAULT NULL,
	`EndTime` DATETIME NULL DEFAULT NULL,
	`detection` VARCHAR(50) NULL DEFAULT NULL COLLATE 'utf8_general_ci'
    )
    COLLATE='utf8_general_ci'
    ENGINE=InnoDB
;
    '''

import pandas as pd
import pymysql
import csv
import keyboard

# from mysql.connector import (connection)

# 기본 setting
conn = pymysql.connect(
    user='bbongs',
    passwd='Qweasd123$',
    host='219.251.99.114',
    db='drone_project',
    charset='utf8'
)

detect_info = []
# [1, cctv1, 38.38274, 128.33009, 2023-01-03 01:40:00, 2023-01-03 01:43:00, drone] list 형 데이터로 부탁드립니당
cursor = conn.cursor()  
answer = ""
while(True) :
    detect_info.append()        # 출력결과 리스트에 추가
    
    sql="""
    INSERT INTO detect (id, cctv, Latitude, Longitude, StartTime, EndTime, detection) VALUES (%s, %s, %s, %s, %s, %s);
    """   
    cursor.execute(sql, detect_info)

    conn.commit()

    if keyboard.read_key() == "q":
        conn.close()
        break
