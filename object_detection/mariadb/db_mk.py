'''
[mariasql info]
    host = "219.251.99.114"
    user = "bbongs"
    pw = "Qweasd123$"
    db = "team"
    port = 3306
=====================================================================================
Database: team
    - table1 : webcam
        row : label | num | detected_label | threshold | flag
        output data : 0 : 760x1280 7 airplanes, 1: 760x1280 2 balloons, 74.0ms 
    - table2 : detect_info
        row : time | gps 
    - table3 : customer_info
        row : id | name | email_address
'''
import pandas as pd
from mysql.connector import (connection)

# ## sql 모음
sql_0 = "DROP DATABASE drone_project;"
sql_1 = "CREATE DATABASE drone_project;"
######## 테이블 명 잘보고 바꾸기!!!#####################
detect ='''
    CREATE TABLE `detect` (
	`id` int NOT NULL AUTO_INCREMENT,
	`cctv` VARCHAR(50) NULL DEFAULT NULL COLLATE 'utf8_general_ci',
	`Latitude` DECIMAL(20,5) NULL DEFAULT NULL,
	`Longitude` DECIMAL(20,5) NULL DEFAULT NULL,
	`StartTime` DATETIME(2) NULL DEFAULT NULL,
	`EndTime` DATETIME(2) NULL DEFAULT NULL,
	`detection` VARCHAR(50) NULL DEFAULT NULL COLLATE 'utf8_general_ci',

    PRIMARY KEY (id)
)
COLLATE='utf8_general_ci'
ENGINE=InnoDB
;
'''
# sql_3 ='''
# CREATE TABLE webcam (
#     id INT(10) UNSIGNED NOT NULL,
#     label VARCHAR(15) NOT NULL COLLATE 'utf8mb4_general_ci',
#     num INT(10) UNSIGNED ZEROFILL NOT NULL,
#     threshold FLOAT UNSIGNED NOT NULL,
#     time DATETIME NOT NULL,
#     latitude VARCHAR(25) NOT NULL COLLATE 'utf8mb4_general_ci',
#     longitude VARCHAR(25) NOT NULL DEFAULT '' COLLATE 'utf8mb4_general_ci',
#     send_email TINYINT(3) UNSIGNED ZEROFILL NOT NULL)
# COLLATE='utf8mb4_general_ci'
# ENGINE=InnoDB;
# '''

# 기본 setting
conn = connection.MySQLConnection(
    # user     = "root",
    # password = "admin",
    # host     = "192.168.208.1",
    user     = "bbongs",
    password = "Qweasd123$",
    host     = "219.251.99.114"
)
cur = conn.cursor()

# # 0. Databse 삭제 (있으면 삭제)
# try:
#     cur.execute(sql_0)
# except Exception as e:
#     print(e)

# # 1. Database 생성
# cur.execute()
# conn.close()

# 2. Table 생성
conn = connection.MySQLConnection(
    user     = "bbongs",
    password = "Qweasd123$",
    host     = "219.251.99.114",
    database = "drone_project"
)
cur = conn.cursor()
cur.execute(detect)
conn.close()

# 3. data 추가 
# csv 파일읽어오기
# df_customer = pd.read_csv('./2023.02/02.14.d93_team_study/customer.csv', )
# for index, data in enumerate(df_customer):
#     print(data)

# sql_insert ='''
# INSERT TABLE customer ()
# '''

# cur.execute(sql_insert)

