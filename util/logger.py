# A simple torch style logger
# (C) Wei YANG 2017
from __future__ import absolute_import

import os
import sys
import numpy as np

__all__ = ['Logger', 'LoggerMonitor', 'savefig']

def savefig(fname, dpi=None):
    import matplotlib.pyplot as plt
    dpi = 150 if dpi == None else dpi
    plt.savefig(fname, dpi=dpi)
    
def plot_overlap(logger, names=None):
    import matplotlib.pyplot as plt
    names = logger.names if names == None else names
    numbers = logger.numbers
    for _, name in enumerate(names):
        x = np.arange(len(numbers[name]))
        y = [float(i) for i in numbers[name]]
        plt.plot(x, np.asarray(y))
        plt.show()
    return [logger.title + '(' + name + ')' for name in names]


class Logger(object):
    '''Save training process to log file with simple plot function.'''
    def __init__(self, fpath, title=None, resume=False): 
        self.file = None
        self.resume = resume
        self.title = '' if title == None else title
        if fpath is not None:
            if resume: 
                self.file = open(fpath, 'r') 
                name = self.file.readline()
                self.names = name.rstrip().split('\t')
                self.numbers = {}
                for _, name in enumerate(self.names):
                    self.numbers[name] = []

                for numbers in self.file:
                    numbers = numbers.rstrip().split('\t')
                    for i in range(0, len(numbers)):
                        self.numbers[self.names[i]].append(numbers[i])
                self.file.close()
                self.file = open(fpath, 'a')  
            else:
                self.file = open(fpath, 'w')

    def set_names(self, names):
        if self.resume: 
            pass
        # initialize numbers as empty list
        self.numbers = {}
        self.names = names
        for _, name in enumerate(self.names):
            self.file.write(name)
            self.file.write('\t')
            self.numbers[name] = []
        self.file.write('\n')
        self.file.flush()

    def append(self, numbers):
        assert len(self.names) == len(numbers), 'Numbers do not match names'
        for index, num in enumerate(numbers):
            self.file.write("{0:.10f}".format(num))
            self.file.write('\t')
            self.numbers[self.names[index]].append(num)
        self.file.write('\n')
        self.file.flush()

    def plot(self, names=None):
        import matplotlib.pyplot as plt
        names = self.names if names == None else names
        numbers = self.numbers
        for _, name in enumerate(names):
            x = np.arange(len(numbers[name]))
            plt.plot(x, np.asarray(numbers[name]))
        plt.legend([self.title + '(' + name + ')' for name in names])
        plt.grid(True)

    def close(self):
        if self.file is not None:
            self.file.close()

class LoggerMonitor(object):
    '''Load and visualize multiple logs.'''
    def __init__ (self, paths):
        '''paths is a distionary with {name:filepath} pair'''
        self.loggers = []
        for title, path in paths.items():
            logger = Logger(path, title=title, resume=True)
            self.loggers.append(logger)

    def plot(self):
        import matplotlib.pyplot as plt
        f, axarr = plt.subplots(2,2)
        lines = []
        titles = []
        for logger in self.loggers:
            numbers = logger.numbers
            x = np.arange(len(numbers['Train Acc']))
            y = [float(i) for i in numbers['Train Acc']]
            axarr[0, 0].plot(x, y , label=logger.title)
            axarr[0, 0].set_title('Train Acc')
            x = np.arange(len(numbers['Train Loss']))
            y = [float(i) for i in numbers['Train Loss']]
            axarr[0, 1].plot(x, y,label=logger.title)
            axarr[0, 1].set_title('Train Loss')
            x = np.arange(len(numbers['Val Acc']))
            y = [float(i) for i in numbers['Val Acc']]
            axarr[1, 0].plot(x, y,label=logger.title)
            axarr[1, 0].set_title('Val Acc')
            x = np.arange(len(numbers['Val Loss']))
            y = [float(i) for i in numbers['Val Loss']]
            line, = axarr[1, 1].plot(x, y,label=logger.title)
            axarr[1, 1].set_title('Val Loss')
            lines.append((line))
            titles.append(logger.title)
        plt.figlegend(lines, titles, loc = 'upper center')
        plt.show()

if __name__ == '__main__':
    # # Example
    # logger = Logger('test.txt')
    # logger.set_names(['Train loss', 'Valid loss','Test loss'])

    # length = 100
    # t = np.arange(length)
    # train_loss = np.exp(-t / 10.0) + np.random.rand(length) * 0.1
    # valid_loss = np.exp(-t / 10.0) + np.random.rand(length) * 0.1
    # test_loss = np.exp(-t / 10.0) + np.random.rand(length) * 0.1

    # for i in range(0, length):
    #     logger.append([train_loss[i], valid_loss[i], test_loss[i]])
    # logger.plot()

    # Example: logger monitor
    paths = {
    'pointnet2':'/mnt/gypsum/home/zhanxu/Proj/joint3d/checkpoint/checkpoint_pointnet2_v3/log.txt',
    'pointnet2_2':'/mnt/gypsum/home/zhanxu/Proj/joint3d/checkpoint/checkpoint_pointnet2_v2/log.txt',
    'pointnet2_1': '/mnt/gypsum/home/zhanxu/Proj/joint3d/checkpoint/checkpoint_pointnet2/log.txt',
    }

    #field = ['Train Acc', 'Train Loss','Val Acc','Val Loss']
    field = ['Train Loss']

    monitor = LoggerMonitor(paths)
    monitor.plot()
    savefig('test.eps')