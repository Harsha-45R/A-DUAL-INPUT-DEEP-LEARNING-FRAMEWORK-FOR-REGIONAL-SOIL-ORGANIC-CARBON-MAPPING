import sys
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFileDialog, QMessageBox
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from spectrometer import Spectrometer
from styles import button_style, disabled_button_style, title_style, success_message_style

class TrainModelThread(QThread):
    training_finished = pyqtSignal(bool)

    def __init__(self, spectrometer):
        super().__init__()
        self.spectrometer = spectrometer

    def run(self):
        try:
            self.spectrometer.train_model()
            self.training_finished.emit(True)
        except Exception as e:
            print(f"Training failed: {e}")
            self.training_finished.emit(False)

class SpectrometerApp(QWidget):
    def __init__(self):
        super().__init__()
        self.spectrometer = Spectrometer()
        print("fxfnc")
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Soil Organic Carbon Prediction')
        self.setGeometry(100, 100, 800, 600)
        self.setStyleSheet("background-color: white;")

        main_layout = QVBoxLayout()
        button_layout = QVBoxLayout()
        graph_layout = QVBoxLayout()

        self.title_label = QLabel("Soil Organic Carbon Prediction", self)
        self.title_label.setStyleSheet(title_style)
        main_layout.addWidget(self.title_label, alignment=Qt.AlignCenter)

        self.init_button = QPushButton('Initialize Spectrometer', self)
        self.init_button.clicked.connect(self.initialize_spectrometer)
        self.init_button.setStyleSheet(button_style)
        button_layout.addWidget(self.init_button)

        self.upload_sample_button = QPushButton('Upload Sample Spectrum', self)
        self.upload_sample_button.clicked.connect(self.upload_sample_spectrum)
        self.upload_sample_button.setEnabled(False)
        self.upload_sample_button.setStyleSheet(disabled_button_style)
        button_layout.addWidget(self.upload_sample_button)

        self.train_model_button = QPushButton('Train Model', self)
        self.train_model_button.clicked.connect(self.train_model)
        self.train_model_button.setEnabled(False)
        self.train_model_button.setStyleSheet(disabled_button_style)
        button_layout.addWidget(self.train_model_button)

        self.predict_soc_button = QPushButton('Predict SoC', self)
        self.predict_soc_button.clicked.connect(self.predict_soc)
        self.predict_soc_button.setEnabled(False)
        self.predict_soc_button.setStyleSheet(disabled_button_style)
        button_layout.addWidget(self.predict_soc_button)

        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        graph_layout.addWidget(self.canvas)

        self.message_label = QLabel("", self)
        self.message_label.setStyleSheet(success_message_style)
        self.message_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.message_label, alignment=Qt.AlignCenter)

        layout = QHBoxLayout()
        layout.addLayout(button_layout)
        layout.addLayout(graph_layout)

        main_layout.addLayout(layout)
        self.setLayout(main_layout)

    def initialize_spectrometer(self):
        self.spectrometer.initialize()
        self.upload_sample_button.setEnabled(True)
        self.train_model_button.setEnabled(True)
        self.upload_sample_button.setStyleSheet(button_style)
        self.train_model_button.setStyleSheet(button_style)
        self.show_success_message("Spectrometer initialized.")

    def upload_sample_spectrum(self):
        file_path, _ = QFileDialog.getOpenFileName(self, 'Open Sample Spectrum File', '', 'CSV Files (*.csv)')
        if file_path:
            self.spectrometer.upload_sample_spectrum(file_path)
            self.show_success_message("Sample spectrum uploaded.")

    def train_model(self):
        self.train_model_thread = TrainModelThread(self.spectrometer)
        self.train_model_thread.training_finished.connect(self.on_training_complete)
        self.train_model_thread.start()
        self.show_success_message("Training started...")

    def on_training_complete(self, success):
        if success:
            self.predict_soc_button.setEnabled(True)
            self.predict_soc_button.setStyleSheet(button_style)
            self.show_success_message("Model trained successfully.")
        else:
            self.show_error_message("Error during training.")

    def predict_soc(self):
        soc_value = self.spectrometer.predict_soc()
        if soc_value is not None:
            self.show_success_message(f"Predicted SoC: {soc_value:.4f}")

    def show_success_message(self, message):
        self.message_label.setText(message)
        QTimer.singleShot(3000, self.clear_message)

    def clear_message(self):
        self.message_label.setText("")
