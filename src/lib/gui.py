import os
import sys
from PyQt5 import QtCore, QtGui, QtWidgets

from lib.search_engine import prepare_data, remove_temp_dir, full_text_search


class WelcomeWindow(QtWidgets.QMainWindow):
    def __init__(self, main_dir):
        super().__init__()
        self.main_dir = main_dir
        self.label = QtWidgets.QLabel()
        self.logo = QtGui.QPixmap(f"{self.main_dir}/gui_graphics/WelcomeScreen.png")
        self.label.setPixmap(self.logo)
        
        self.setWindowTitle('News Finder')
        self.setMinimumSize(QtCore.QSize(800, 451)) 
        
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        grid_layout = QtWidgets.QGridLayout(central_widget)
        grid_layout.addWidget(self.label, 0, 0)
        
        self.show()


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, img_dir, ix, parent=None):
        super().__init__(parent)
        self.img_dir = img_dir
        self.ix = ix
        
        self.label = QtWidgets.QLabel()
        self.logo = QtGui.QPixmap(f"{'/'.join(self.img_dir.split('/')[:-2])}/gui_graphics/Logo.png")
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
        
        self.query = QtWidgets.QLineEdit('Type any phrase')
        
        self.pixmap_lw = QtWidgets.QListWidget(
            viewMode=QtWidgets.QListView.IconMode,
            iconSize=250 * QtCore.QSize(1, 1),
            movement=QtWidgets.QListView.Static,
            resizeMode=QtWidgets.QListView.Adjust,
        )
        
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        grid_layout = QtWidgets.QGridLayout(central_widget)
        grid_layout.addWidget(self.label, 0, 0, 1, 3)
        grid_layout.addWidget(self.query, 1, 0)
        grid_layout.addWidget(self.search_btn, 1, 1)
        grid_layout.addWidget(self.clear_btn, 1, 2)
        grid_layout.addWidget(self.pixmap_lw, 2, 0, 1, 3)

        self.timer_loading = QtCore.QTimer(interval=50, timeout=self.load_image)
        self.filenames_iterator = None

        self.resize(900, 800)
        self.setWindowTitle('News Finder')

        qr = self.frameGeometry()
        cp = QtWidgets.QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())
        self.show()

    @QtCore.pyqtSlot()
    def on_clear_btn_clicked(self):
        self.pixmap_lw.clear()

    @QtCore.pyqtSlot()
    def on_search_btn_clicked(self):
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


if __name__ == "__main__":
    main_dir = '/Users/alexdrozdz/Desktop/Studia/00. Seminarium magisterskie/Master_degree'
    app = QtWidgets.QApplication(sys.argv)
    w = WelcomeWindow(main_dir=main_dir)
    QtWidgets.qApp.processEvents()
    ix = prepare_data(main_dir)
    QtCore.QCoreApplication.instance().quit
    img_dir = main_dir + '/cropped_visual_content/'
    w = MainWindow(img_dir=img_dir, ix=ix)
    w.show()
    status = app.exec_()
    remove_temp_dir(main_dir)
    sys.exit(status)
