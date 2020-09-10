

import os

class LogFile:
    def __init__(self):
        self.filename = None

    def _create_log(self, log_path, filename):
        self.filename = filename
        self.log_path = log_path
        self.log_full_path = os.path.join(self.log_path, filename)
        #os.system("touch " + log_full_path)

        return self

    def write(self, values):
        f = open(self.log_full_path, "a+")
        f.write(values)
        f.close()

