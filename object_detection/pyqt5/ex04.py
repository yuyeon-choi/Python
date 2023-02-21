# QFileDialog는 사용자가 파일 또는 경로를 선택할 수 있도록 하는 다이얼로그입니다.
## Ex 6-4. QFileDialog.

import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QTextEdit, QAction, QFileDialog
from PyQt5.QtGui import QIcon


class MyApp(QMainWindow):

    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.textEdit = QTextEdit()
        self.setCentralWidget(self.textEdit)
        self.statusBar()

        openFile = QAction(QIcon('open.png'), 'Open', self)
        openFile.setShortcut('Ctrl+O')
        openFile.setStatusTip('Open New File')
        openFile.triggered.connect(self.showDialog)

        menubar = self.menuBar()
        menubar.setNativeMenuBar(False)
        fileMenu = menubar.addMenu('&File')
        fileMenu.addAction(openFile)

        self.setWindowTitle('File Dialog')
        self.setGeometry(300, 300, 300, 200)
        self.show()

    def showDialog(self):
        fname = QFileDialog.getOpenFileName(self, 'Open file', './')
        '''
        QFileDialog를 띄우고, getOpenFileName() 메서드를 사용해서 파일을 선택합니다.
        세 번째 매개변수를 통해 기본 경로를 설정할 수 있습니다. 또한 기본적으로 모든 파일( * )을 열도록 되어있습니다.
        '''
        if fname[0]:
            f = open(fname[0], 'r')

            with f:
                data = f.read()
                self.textEdit.setText(data)
        '''
        선택한 파일을 읽어서, setText() 메서드를 통해 텍스트 편집 위젯에 불러옵니다.
        '''

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MyApp()
    sys.exit(app.exec_())