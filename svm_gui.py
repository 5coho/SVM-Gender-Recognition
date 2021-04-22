"""

The Gui file Gender Recognition
loads the gender_recognition_gui.ui file and adds functionality.

"""


__author__          = "Scott Howes"
__credits__         = "Scott Howes"
__email__           = "showes@unbc.ca"
__python_version__  = "3.9.4"


#imports
import cv2
import sys
import time
from datetime import date
from joblib import *
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication
from PyQt5.QtWidgets import QWidget
from PyQt5.QtWidgets import QLabel
from PyQt5.QtWidgets import QFileDialog
from PyQt5.uic import loadUi
import numpy as np


#The GUI class
class svm_gui(QWidget):


    #the constructor
    def __init__(self):
        super(svm_gui, self).__init__()
        loadUi("gui/svm_maker_gui.ui", self)

        #initializing svm to None
        self.svm = None

        #folder path to the training data
        self.folderPath = ""

        #making the buttons unclickable unclickable
        self.bttn_save.setEnabled(False)
        self.bttn_createSVM.setEnabled(False)

        #loading connection for buttons
        self._load_connects()


    #load the connects for buttons functionality
    def _load_connects(self):
        self.bttn_browse.clicked.connect(self.bttn_browse_clicked)
        self.bttn_createSVM.clicked.connect(self.bttn_createSVM_clicked)
        self.bttn_clear.clicked.connect(self.bttn_clear_clicked)
        self.bttn_save.clicked.connect(self.bttn_save_clicked)
        self.bttn_saveOutput.clicked.connect(self.bttn_saveOutput_clicked)


    #Connection for the browse button tho get the training data file path
    @pyqtSlot()
    def bttn_browse_clicked(self):

        #opening a QFileDialog to get folder path
        self.folderPath = str(QFileDialog.getExistingDirectory(self, "Select training Data folder"))

        #setting lineEdit_folderPath to self.folderPath
        self.lineEdit_folderPath.setText(self.folderPath)

        #outputting to console text edit box
        self.textEdit_output.append(f"<b>Training Data Location: </b>" + self.folderPath)
        self.textEdit_output.append("")

        #making createSVM button clickable
        self.bttn_createSVM.setEnabled(True)


    #function for when the Create SVM button is clicked. This is where code for
    #svm creation takes place. sets self.svm to the created svm
    @pyqtSlot()
    def bttn_createSVM_clicked(self):

        #outputting parameters
        self.textEdit_output.append("<b>parameters Used:</b>")
        self.textEdit_output.append(f"<b>C:</b> {self.lineEdit_c.text()}")
        self.textEdit_output.append(f"<b>Kernel:</b> {self.comboBox_kernel.currentText()}")
        self.textEdit_output.append(f"<b>Degree:</b> {self.lineEdit_degree.text()}")
        self.textEdit_output.append(f"<b>Gamma:</b> {self.comboBox_gamma.currentText()}")
        self.textEdit_output.append(f"<b>Coef0:</b> {self.lineEdit_coef0.text()}")
        self.textEdit_output.append(f"<b>Shrinking:</b> {self.comboBox_shrinking.currentText()}")
        self.textEdit_output.append(f"<b>Probability:</b> {self.comboBox_prob.currentText()}")
        self.textEdit_output.append(f"<b>Tol:</b> {self.lineEdit_tol.text()}")
        self.textEdit_output.append(f"<b>Cache Size:</b> {self.lineEdit_cache.text()}")
        self.textEdit_output.append(f"<b>Max Iterations:</b> {self.lineEdit_maxIter.text()}")
        self.textEdit_output.append(f"<b>Decision Function Shape:</b> {self.comboBox_dfs.currentText()}")
        self.textEdit_output.append(f"<b>Break Ties:</b> {self.comboBox_breakTies.currentText()}")
        self.textEdit_output.append("")

        #casting parameters to usable data types and assigning variables
        c_val = float(self.lineEdit_c.text())
        kernel_val = str(self.comboBox_kernel.currentText())
        degree_val = int(self.lineEdit_degree.text())
        gamma_val = str(self.comboBox_gamma.currentText())
        coef0_val = float(self.lineEdit_coef0.text())
        shrinking_val = bool(self.comboBox_shrinking.currentText())
        probability_val = bool(self.comboBox_prob.currentText())
        tol_val = float(self.lineEdit_tol.text())
        cache_size_val = float(self.lineEdit_cache.text())
        max_iter_val = int(self.lineEdit_maxIter.text())
        decision_function_shape_val = str(self.comboBox_dfs.currentText())
        break_ties_val = bool(self.comboBox_breakTies.currentText())

        #making save button clickable AFTER SVM is created
        self.bttn_save.setEnabled(True)


    #button to save an svm as a joblib. Only become clickable after an svm
    #has been created
    @pyqtSlot()
    def bttn_save_clicked(self):
        #path, _ = QFileDialog.getOpenFileName(None, "Load Protein", "stufffffffffffffffff", Joblib Files (*.joblib)")
        self.textEdit_output.append("<b>Save Button clicked</b>")


    #function for when the clear output button is clicked
    @pyqtSlot()
    def bttn_clear_clicked(self):
            self.textEdit_output.clear()


    #saves the output as a text file
    @pyqtSlot()
    def bttn_saveOutput_clicked(self):

        #generating a default filename
        outDate = date.today().strftime("%b-%d-%Y")
        filename = f"svm_output_{outDate}_{self.lcd_accuracy.value()}"

        path, _ = QFileDialog.getSaveFileName(None, "Save Output", filename, "*.txt")
        if path:
            file = open(path, "w+")
            contents = self.textEdit_output.toPlainText()
            file.write(contents)
            file.close()
