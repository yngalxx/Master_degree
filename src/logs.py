import contextlib
import logging
import os
import pathlib
import time
from logging.handlers import WatchedFileHandler


class Log:
    def __init__(self, script_name: str):
        self.script_name = script_name
        self.log_path = f'{"/".join(str(pathlib.Path(__file__).parent.resolve()).split("/")[:-1])}/logs/'
        self.start_time = time.time()
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)
        log_file = f"{self.log_path}{self.script_name.lower()}.log"
        self.log_file = log_file
        handler = WatchedFileHandler(os.environ.get("LOGFILE", log_file))
        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        self.handler = handler
        log = logging.getLogger()
        log.setLevel(os.environ.get("LOGLEVEL", "INFO"))
        log.addHandler(handler)
        handler_to_console = logging.StreamHandler()
        handler_to_console.setFormatter(formatter)
        self.handler_to_console = handler_to_console
        log.addHandler(handler_to_console)
        self.log = log

    def log_start(self):
        with contextlib.suppress(IndexError):
            with open(self.log_file, "r") as f:
                lines = f.read().splitlines()
                last_line = lines[-1]

            if (last_line != "") and (
                os.stat(self.log_file).st_size == 0
            ) == False:
                with open(self.log_file, "a") as file:
                    file.write(f"\n############################\n\n")

        logging.info(self.script_name.upper() + " - STARTED!")

    def log_end(self):
        # time
        end_time = round(time.time() - self.start_time, 2)
        logging.info(
            f"{self.script_name.upper()} - COMPLETED! RUNTIME: {end_time} sec."
        )

        self.log.removeHandler(self.handler_to_console)
        self.log.removeHandler(self.handler)
        logging.shutdown()
