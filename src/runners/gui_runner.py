import os
import sys

import click
from constants import General
from PyQt5 import QtCore, QtWidgets

from lib.gui import MainWindow, WelcomeWindow
from lib.logs import Log
from lib.search_engine import prepare_data, remove_temp_dir


@click.command(context_settings=dict(help_option_names=["-h", "--help"]))
@click.option(
    "--main_dir",
    default=General.MAIN_DIR,
    type=str,
    help="Working directory path.",
    show_default=True,
)
def gui_runner(main_dir):
    # initialize logger
    logger = Log("gui_runner", main_dir)
    logger.log_start()

    # check provided path
    assert os.path.exists(main_dir) == True
    img_dir = main_dir + "/cropped_visual_content/"
    assert os.path.exists(img_dir) == True
    tmp_dir = main_dir + "/.tmp/"
    assert os.path.exists(tmp_dir) == False
    grph_dir = main_dir + "/gui_graphics/"
    assert os.path.exists(grph_dir) == True
    db_dir = main_dir + "/ocr_database/"
    assert os.path.exists(db_dir) == True

    # app instance
    app = QtWidgets.QApplication(sys.argv)

    # welcoming window
    _ = WelcomeWindow(main_dir=main_dir)
    QtWidgets.qApp.processEvents()

    # prepare search engine input
    ix = prepare_data(main_dir)
    QtCore.QCoreApplication.instance().quit

    # app main window
    _ = MainWindow(img_dir=img_dir, ix=ix)

    # terminate after closing main window
    status = app.exec_()
    logger.log_end()
    remove_temp_dir(main_dir)
    sys.exit(status)


if __name__ == "__main__":
    gui_runner()
