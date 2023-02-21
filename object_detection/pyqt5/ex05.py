'''
QMessageBox 클래스는 사용자에게 정보를 제공하거나 질문과 대답을 할 수 있는 대화창을 제공합니다.
'''

## Ex 6-5. QMessageBox.

import sys
from PyQt5.QtWidgets import QApplication, QWidget, QMessageBox


class MyApp(QWidget):

    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('QMessageBox')
        self.setGeometry(300, 300, 300, 200)
        self.show()

    def closeEvent(self, event):
        reply = QMessageBox.question(self, 'Message', 'Are you sure to quit?',
                                    QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        '''
        두번째 매개변수는 타이틀바에 나타날 문자열 ('Message'), 세번째 매개변수는 대화창에 나타날 문자열 ('Are you sure to quit?')을 입력합니다.
        네 번째에는 대화창에 보여질 버튼의 종류를 입력하고, 마지막으로 디폴트로 선택될 버튼을 설정해줍니다.
        QMessageBox.No로 설정할 경우, 메세지 박스가 나타났을 때 'No' 버튼이 선택되어 있습니다.
        반환값은 reply 변수에 저장됩니다.
        '''
    

        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()
        '''
        'Yes' 버튼을 클릭했을 경우, 이벤트를 받아들이고 위젯을 종료합니다.
        'No' 버튼을 클릭하면, 이벤트를 무시하고 위젯을 종료하지 않습니다.
        '''


if __name__ == '__main__':
   app = QApplication(sys.argv)
   ex = MyApp()
   sys.exit(app.exec_())

'''
QWidget을 종료할 때, QCloseEvent가 생성되어 위젯에 전달됩니다.
위젯의 동작을 수정하기 위해 closeEvent() 이벤트 핸들러를 수정해야합니다.
'''