"""

The main file that starts the gender recognition software. Only initializes and
starts the GUI

"""


#metaData
__author__              = "Scott Howes"
__credits__             = "Scott Howes"
__email__               = "showes@unbc.ca"
__python_version__      = "3.9.4"


#imports
import sys
from gr_gui import *
from PyQt5.QtWidgets import QApplication
from HaarCascadeSGD import haarcascade
from RGB2GrayTransformer import RGB2GrayTransformer
from HogTransformer import HogTransformer


#the main function
def main():
    #joshthing = haarcascade('C:/Users/Josh/PycharmProjects/CPSC371Project/biden.jpg')
    app = QApplication(sys.argv)
    gui = gr_gui()
    gui.show()
    sys.exit(app.exec_())


#rinngin main()
if __name__ == "__main__":
    main()
