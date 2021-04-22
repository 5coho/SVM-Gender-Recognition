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
import os
from datetime import date
from joblib import dump
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication
from PyQt5.QtWidgets import QWidget
from PyQt5.QtWidgets import QLabel
from PyQt5.QtWidgets import QFileDialog
from PyQt5.uic import loadUi
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import numpy


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

        #folder names in folderPath.
        #these will be used as labels for svm_maker_gui
        self.folders = []

        #sizes for resizing the training images
        self.h_resize = 130
        self.w_resize = 70

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
    # if path needs to be added
    @pyqtSlot()
    def bttn_browse_clicked(self):

        #opening a QFileDialog to get folder path
        self.folderPath = str(QFileDialog.getExistingDirectory(self, "Select training Data folder"))

        #setting lineEdit_folderPath to self.folderPath
        self.lineEdit_folderPath.setText(self.folderPath)

        #getting label names from folder names in self.folderPath
        self.folders = os.listdir(self.folderPath)

        #outputting to console text edit box
        self.textEdit_output.append(f"<b>Training Data Location: </b>" + self.folderPath)
        self.textEdit_output.append("")
        self.textEdit_output.append(f"<b>Labels For Training: </b>")

        for folder in self.folders:
            self.textEdit_output.append(folder)
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

        #lists for training data
        X = []
        y = []

        self.textEdit_output.append("<b>Creating X, y...</b>")

        #gathering images, creating X, y and image processing
        for folder in self.folders:
            label = folder
            for file in os.listdir(os.path.join(self.folderPath, folder)):

                #making image GrayScale
                image = cv2.imread(os.path.join(os.path.join(self.folderPath, folder), file), cv2.IMREAD_GRAYSCALE)
                image = cv2.resize(image, (self.h_resize, self.w_resize))

                X.append(image)
                y.append(label)

        self.textEdit_output.append("DONE")
        self.textEdit_output.append("")
        self.textEdit_output.append("<b>Converting to numpy array...</b>")

        #converting X and y to numpy arrays
        X = numpy.array(X)
        y = numpy.array(y)

        self.textEdit_output.append("DONE")
        self.textEdit_output.append("")

        self.textEdit_output.append("<b>Reshaping Numpy Arrays...</b>")

        #rehaping the arrays
        Xori = X.reshape(X.shape[0], X.shape[1]*X.shape[2])
        yori = y.reshape(y.shape[0])

        self.textEdit_output.append("DONE")
        self.textEdit_output.append("")
        self.textEdit_output.append("<b>Shuffling X, y...</b>")

        #shuffling Xori an yori
        X, y = shuffle(Xori, yori)

        self.textEdit_output.append("DONE")
        self.textEdit_output.append("")
        self.textEdit_output.append("<b>Spitting Training data 80-20...</b>")

        #splitting
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        self.textEdit_output.append("DONE")
        self.textEdit_output.append("")
        self.textEdit_output.append("<b>Creating SVM...</b>")

        #creating the svm
        self.svm = SVC(C=c_val, kernel=kernel_val, degree=degree_val, gamma=gamma_val, coef0=coef0_val, shrinking=shrinking_val, probability=probability_val, tol=tol_val, cache_size=cache_size_val, class_weight=None, verbose=False, max_iter=max_iter_val, decision_function_shape=decision_function_shape_val, break_ties=break_ties_val, random_state=None)


        self.textEdit_output.append("DONE")
        self.textEdit_output.append("")
        self.textEdit_output.append("<b>Training SVM (This may take awhile)...</b>")

        #training the svm
        start = time.time()
        self.svm.fit(X_train, y_train)
        end = time.time()

        self.textEdit_output.append("DONE")
        self.textEdit_output.append("")
        self.textEdit_output.append(f"<b>Time Taken:</b> {round(end-start,2)}")

        #getting training accuracy
        training_accuracy = self.svm.score(X_test, y_test)

        self.textEdit_output.append(f"<b>Training Accuracy:</b> {training_accuracy}")
        self.textEdit_output.append("")

        #setting the lcd_accuracy to training_accuracy
        self.lcd_accuracy.display(training_accuracy)

        #making save button clickable AFTER SVM is created
        self.bttn_save.setEnabled(True)


    #button to save an svm as a joblib. Only become clickable after an svm
    #has been created
    @pyqtSlot()
    def bttn_save_clicked(self):
        #generating a default filename
        creationDate = date.today().strftime("%b-%d-%Y")
        filename = f"svm_{creationDate}_{self.h_resize}x{self.w_resize}_{round(self.lcd_accuracy.value(), 4)}"

        #getting save path
        path, name = QFileDialog.getSaveFileName(None, "Save SVM", filename, "*.joblib")

        if path:

            #dumping object using joblib
            dump(self.svm, path)

            #outputting
            self.textEdit_output.append(f"<b>Save Location:</b> {path}")
            self.textEdit_output.append("")


    #function for when the clear output button is clicked
    @pyqtSlot()
    def bttn_clear_clicked(self):
            self.textEdit_output.clear()


    #saves the output as a text file
    @pyqtSlot()
    def bttn_saveOutput_clicked(self):

        #generating a default filename
        outDate = date.today().strftime("%b-%d-%Y")
        filename = f"svm_output_{outDate}_{round(self.lcd_accuracy.value(), 4)}"

        path, _ = QFileDialog.getSaveFileName(None, "Save Output", filename, "*.txt")
        if path:
            file = open(path, "w+")
            contents = self.textEdit_output.toPlainText()
            file.write(contents)
            file.close()
