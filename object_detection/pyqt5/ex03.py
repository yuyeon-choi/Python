# 폰트 다이얼로그 (QFontDialog)는 폰트를 선택할 수 있게 해주는 다이얼로그입니다.
## Ex 6-3. QFontDialog.

import sys
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout
, QPushButton, QSizePolicy, QLabel, QFontDialog)


class MyApp(QWidget):

    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        btn = QPushButton('Dialog', self)
        btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        btn.move(20, 20)
        btn.clicked.connect(self.showDialog)

        vbox = QVBoxLayout()
        vbox.addWidget(btn)

        self.lbl = QLabel('The quick brown fox jumps over the lazy dog', self)
        self.lbl.move(130, 20)

        vbox.addWidget(self.lbl)
        self.setLayout(vbox)

        self.setWindowTitle('Font Dialog')
        self.setGeometry(300, 300, 250, 180)
        self.show()

    def showDialog(self):
        font, ok = QFontDialog.getFont()
        '''
        폰트 다이얼로그를 띄우고, getFont() 메서드를 사용해서 선택한 폰트와 불 값을 반환받습니다.
        앞의 예제와 마찬가지로 'OK' 버튼을 클릭하면 True를, 'Cancel' 버튼을 클릭하면 False를 반환합니다.
        '''
        if ok:
           self.lbl.setFont(font)
        '''
        setFont() 메서드를 사용해서 선택한 폰트를 라벨의 폰트로 설정해줍니다.
        '''

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MyApp()
    sys.exit(app.exec_())