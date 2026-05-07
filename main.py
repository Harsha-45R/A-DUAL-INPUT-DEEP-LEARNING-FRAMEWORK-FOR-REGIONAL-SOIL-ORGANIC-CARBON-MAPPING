import sys
from PyQt5.QtWidgets import QApplication
from gui import SpectrometerApp
if __name__ == '__main__':
    print("app starting")

    app = QApplication(sys.argv)
    print("QApplication created")

    ex = SpectrometerApp()
    print("Window created")

    ex.show()
    print("Window shown")

    sys.exit(app.exec_())