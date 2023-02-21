import sys
from PyQt5.QtWidgets import *
from PyQt5 import uic

#UI파일 연결
#단, UI파일은 Python 코드 파일과 같은 디렉토리에 위치해야한다.
form_class = uic.loadUiType("test_pyqt.ui")[0]

#메인 윈도우 클래스
class WindowClass(QMainWindow, form_class) :
    #초기화 메서드
    def __init__(self) :
        super().__init__()
        self.setupUi(self)
        #pushButton (시작버튼)을 클릭하면 아래 fuctionStart 메서드와 연결 됨.
        self.pushButton.clicked.connect(self.functionStart) 

    # 시작버튼을 눌렀을 때 실행되는 메서드
    def functionStart(self):
      self.progressBar.setRange(0, 19) #progressbar 초기 설정(100을 0~19, 20단계로 나눔)
      for i in range(0, 20):
            print("출력 : ",str(i))
            self.progressBar.setValue(i) #progress bar 진행률 올리기
            self.textBrowser.append("출력 : "+str(i)) #text browser 문자열 추가하기

#코드 실행시 GUI 창을 띄우는 부분
#__name__ == "__main__" : 모듈로 활용되는게 아니라 해당 .py파일에서 직접 실행되는 경우에만 코드 실행
if __name__ == "__main__" :
    app = QApplication(sys.argv)
    myWindow = WindowClass()
    myWindow.show()
    app.exec_()