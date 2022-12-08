# 구두점 삭제
# 구두점 글자의 딕셔너리를 만들어 translate() 적용
import unicodedata
import sys
text_data=["HI!!!!!!!!! I. love. This. Song...!!!!",
           "1222222%% Agree?! #AA",
           "Reight!@!@"]    
punctuation = dict.fromkeys(i for i in range(sys.maxunicode)
                            if unicodedata.category(chr(i)).startswith('P'))

test = [string.translate(punctuation) for string in text_data]
print(test)

"""
결과
['HI I love This Song', '1222222 Agree AA', 'Reight']
"""