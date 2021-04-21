"""

The main file that starts the SVM maker software. Only initializes and
starts the GUI

"""


#metaData
__author__              = "Scott Howes"
__credits__             = "Scott Howes"
__email__               = "showes@unbc.ca"
__python_version__      = "3.9.4"


#imports
import sys
from svm_gui import *
from PyQt5.QtWidgets import QApplication


#the main function
def main():
    app = QApplication(sys.argv)
    gui = svm_gui()
    gui.show()
    sys.exit(app.exec_())


#rinngin main()
if __name__ == "__main__":
    main()
