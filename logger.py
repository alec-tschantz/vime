import matplotlib.pyplot as plt
import os

class Logger(object):

    def __init__(self):
        self.current_log = ""
        self.file_dir = ""
        self.values = dict()

    def log(self, string):
        self.current_log += string+"\n"

    def set_log_file(self, file_dir):
        self.file_dir = 'logs/' + file_dir + '/'
        if not os.path.exists(self.file_dir):
            os.makedirs(self.file_dir)

    def log_value(self, key, value):
        if key in self.values:
            self.values[key].append(value)
        else:
            self.values[key] = []
            self.values[key].append(value)

    def save_values(self):
        for key, val in self.values.items():
            plt.plot(val)
            plt.title(key)
            plt.savefig(self.file_dir+key)
            plt.close()

    def print_log(self):
        print(self.current_log)
        self.current_log = ""


def log(string):
    _log.log(string)

def log_value(key, value):
    _log.log_value(key, value)

def set_log_file(file_dir):
    _log.set_log_file(file_dir)

def save_values():
    _log.save_values()

def print_log():
    _log.print_log()


_log = Logger()
