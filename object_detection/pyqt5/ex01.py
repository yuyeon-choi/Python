## Ex 6-1. QInputDialog.

import sys
from PyQt5.QtWidgets import (QApplication, QWidget, QPushButton, QLineEdit, QInputDialog)


class MyApp(QWidget):

    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.btn = QPushButton('Dialog', self)
        self.btn.move(30, 30)
        self.btn.clicked.connect(self.showDialog)

        self.le = QLineEdit(self)
        self.le.move(120, 35)

        self.setWindowTitle('Input dialog')
        self.setGeometry(300, 300, 300, 200)
        self.show()

    def showDialog(self):
        text, ok = QInputDialog.getText(self, 'Input Dialog', 'Enter your name:')
        '''
        이 코드를 통해 입력 대화창이 나타납니다.
        두 번째 매개변수는 대화창의 타이틀, 세 번째 매개변수는 대화창 안에 보여질 메세지를 의미합니다. 입력 대화창은 입력한 텍스트와 불 (Boolean) 값을 반환합니다.
        텍스트를 입력한 후 'OK' 버튼을 누르면 불 값은 True, 'Cancel' 버튼을 누르면 불 값은 False가 됩니다.
        '''
        if ok:
            self.le.setText(str(text))
        '''
        입력한 값을 setText() 메서드를 통해 줄 편집 위젯에 표시되도록 합니다.
        '''


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MyApp()
    sys.exit(app.exec_())
