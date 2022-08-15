import os
from PyQt5 import QtCore, QtGui, QtWidgets

from lib.search_engine import full_text_search, get_image_details


class ImageDetailsWindow(QtWidgets.QMainWindow):
    def __init__(self, img_dir, file_name, text, keywords, pred_label, origin, query):
        super().__init__()
        self.img_dir = img_dir
        self.query = query
        self.text = QtWidgets.QPlainTextEdit(text)
        self.text.setReadOnly(True)
        self.keywords = QtWidgets.QPlainTextEdit(keywords)
        self.keywords.setReadOnly(True)
        self.keywords.setMaximumHeight(100)
        self.pred_label = QtWidgets.QPlainTextEdit(pred_label)
        self.pred_label.setReadOnly(True)
        self.pred_label.setMaximumHeight(27)
        self.origin = QtWidgets.QPlainTextEdit(origin) 
        self.origin.setMaximumHeight(27)
        self.origin.setReadOnly(True)
        self.file_name = file_name
        
        self.label = QtWidgets.QLabel()
        self.img = QtGui.QPixmap(f"{self.img_dir}/{file_name}")
        if self.img.width() > 400:
            self.img = self.img.scaledToWidth(400)
        if self.img.height() > 650:
            self.img = self.img.scaledToHeight(650)
        self.label.setPixmap(self.img)

        self.setWindowTitle(f'News Finder - {self.query.text()} - {file_name}')
        self.setFixedSize(QtCore.QSize(800, 700))

        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        grid_layout = QtWidgets.QGridLayout(central_widget)
        grid_layout.addWidget(self.label, 0, 0, 8, 1)
        grid_layout.addWidget(QtWidgets.QLabel('<b>Keywords</b>'), 0, 1)
        grid_layout.addWidget(self.keywords, 1, 1)
        grid_layout.addWidget(QtWidgets.QLabel('<b>Label</b>'), 2, 1)
        grid_layout.addWidget(self.pred_label, 3, 1)
        grid_layout.addWidget(QtWidgets.QLabel('<b>OCR text</b>'), 4, 1)
        grid_layout.addWidget(self.text, 5, 1)
        grid_layout.addWidget(QtWidgets.QLabel('<b>Origin file</b>'), 6, 1)
        grid_layout.addWidget(self.origin, 7, 1)

        qr = self.frameGeometry()
        cp = QtWidgets.QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())
        self.move(self.pos().x()+100, self.pos().y()-25)

        self.show()


class WelcomeWindow(QtWidgets.QMainWindow):
    def __init__(self, main_dir):
        super().__init__()
        self.main_dir = main_dir
        self.label = QtWidgets.QLabel()
        try:
            logo_path = f"{self.main_dir}/gui_graphics/WelcomeScreen.png"
            self.logo = QtGui.QPixmap(logo_path)
        except:
            raise FileNotFoundError(
                f"File '{logo_path}' not found, code will be forced to quit..."
        )
        self.label.setPixmap(self.logo)
        
        self.setWindowTitle('News Finder')
        self.setFixedSize(QtCore.QSize(826, 480)) 
        
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        grid_layout = QtWidgets.QGridLayout(central_widget)
        grid_layout.addWidget(self.label, 0, 0)

        qr = self.frameGeometry()
        cp = QtWidgets.QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())
        
        self.show()


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, img_dir, ix, parent=None):
        super().__init__(parent)
        self.img_dir = img_dir
        self.ix = ix
        self.main_dir = '/'.join(self.img_dir.split('/')[:-2])
        
        self.label = QtWidgets.QLabel()
        try:
            logo_path = f"{self.main_dir}/gui_graphics/Logo.png"
            self.logo = QtGui.QPixmap(logo_path)
        except:
            raise FileNotFoundError(
                f"File '{logo_path}' not found, code will be forced to quit..."
        )
        self.label.setPixmap(self.logo)
        self.label.resize(self.logo.width(), self.logo.height())
        
        self.search_btn = QtWidgets.QPushButton(
            self.tr("Search"), clicked=self.on_search_btn_clicked
        )
        self.search_btn.setFixedSize(100, 32)
        self.clear_btn = QtWidgets.QPushButton(
            self.tr("Clear results"), clicked=self.on_clear_btn_clicked
        )
        self.clear_btn.setFixedSize(100, 32)
        
        self.query = QtWidgets.QLineEdit(placeholderText='Type any phrase')
        
        self.pixmap_lw = QtWidgets.QListWidget(
            viewMode=QtWidgets.QListView.IconMode,
            iconSize=250 * QtCore.QSize(1, 1),
            movement=QtWidgets.QListView.Static,
            resizeMode=QtWidgets.QListView.Adjust,
        )
        self.pixmap_lw.itemClicked.connect(self.on_item_clicked)
        
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        self.grid_layout = QtWidgets.QGridLayout(central_widget)
        self.grid_layout.addWidget(self.label, 0, 0, 1, 3)
        self.grid_layout.addWidget(self.query, 1, 0)
        self.grid_layout.addWidget(self.search_btn, 1, 1)
        self.grid_layout.addWidget(self.clear_btn, 1, 2)
        self.grid_layout.addWidget(self.pixmap_lw, 2, 0, 1, 3)

        self.timer_loading = QtCore.QTimer(interval=50, timeout=self.load_image)
        self.filenames_iterator = None

        self.setFixedSize(QtCore.QSize(923, 700)) 
        self.setWindowTitle('News Finder')
        
        qr = self.frameGeometry()
        cp = QtWidgets.QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())
        self.show()

    @QtCore.pyqtSlot()
    def on_item_clicked(self):
        clickedItem = self.pixmap_lw.currentItem().text()
        img_details_df = get_image_details(main_dir=self.main_dir, image_name=clickedItem)
        self.w_in = ImageDetailsWindow(
            query=self.query,
            img_dir=self.img_dir, 
            file_name=clickedItem,
            text=img_details_df['OCR_RAW_TEXT'][0].replace(',', ', ').replace('.', '. '), 
            keywords=img_details_df['KEYWORDS'][0].replace(',', ', '), 
            pred_label=img_details_df['PRED_LABEL'][0], 
            origin=img_details_df['ORIGIN_FILE_NAME'][0]
        )

    @QtCore.pyqtSlot()
    def on_clear_btn_clicked(self):
        self.setWindowTitle(f'News Finder')
        self.query = QtWidgets.QLineEdit(placeholderText='Type any phrase')
        self.grid_layout.addWidget(self.query, 1, 0)
        self.pixmap_lw.clear()

    @QtCore.pyqtSlot()
    def on_search_btn_clicked(self):
        self.setWindowTitle(f'News Finder - {self.query.text()}')
        self.file_names = full_text_search(ix=self.ix, query=self.query.text())
        self.start_loading()

    @QtCore.pyqtSlot()
    def start_loading(self):
        if self.timer_loading.isActive():
            self.timer_loading.stop()
        self.filenames_iterator = self.load_images()
        self.pixmap_lw.clear()
        self.timer_loading.start()

    @QtCore.pyqtSlot()
    def load_image(self):
        try:
            filename = next(self.filenames_iterator)
        except StopIteration:
            self.timer_loading.stop()
        else:
            name = os.path.basename(filename)
            it = QtWidgets.QListWidgetItem(name)
            it.setIcon(QtGui.QIcon(filename))
            self.pixmap_lw.addItem(it)

    def load_images(self):
        it = QtCore.QDirIterator(
            self.img_dir,
            self.file_names,
            QtCore.QDir.Files
        )
        while it.hasNext():
            filename = it.next()
            yield filename

