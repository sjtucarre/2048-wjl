import sys
import csv, os
import torch
import numpy as np
import torchvision.transforms as transforms
from time import sleep

PATH = '/data/nextcloud/dbc2017/files/jupyter/v1/game2048/gru_model_final.pkl'


class Agent:
    '''Agent Base.'''

    def __init__(self, game, display=None):
        self.game = game
        self.display = display

    def play(self, max_iter=np.inf, verbose=False):
        n_iter = 0
        while (n_iter < max_iter) and (not self.game.end):
            direction = self.step()
            self.game.move(direction)
            n_iter += 1
            if verbose:
                print("Iter: {}".format(n_iter))
                print("======Direction: {}======".format(
                    ["left", "down", "right", "up"][direction]))
                if self.display is not None:
                    self.display.display(self.game)

    def step(self):
        direction = int(input("0: left, 1: down, 2: right, 3: up = ")) % 4
        return direction

class MyAgent(Agent):

    def __init__(self, game, display=None):
        if game.size != 4:
            raise ValueError(
                "`%s` can only work with game of `size` 4." % self.__class__.__name__)
        super().__init__(game, display)

        self.model = torch.load(PATH)
        self.model.eval()

    def step(self):


        board = np.where(self.game.board == 0, 1, self.game.board)
        board = np.log2(board)

        board = board.reshape((4, 4))
        # sleep(3600)
        board = board[:, :, np.newaxis]
        board = board / 11.0
        trans = transforms.Compose([transforms.ToTensor()])
        board = trans(board)
        board = torch.unsqueeze(board, dim=0)
        board = board.type(torch.float)
        out = self.model(board)
        direction = torch.max(out, 1)[1]
        return int(direction)

class MyGRUAgent(Agent):
    
    def __init__(self, game, display=None):
        if game.size != 4:
            raise ValueError(
                "`%s` can only work with game of `size` 4." % self.__class__.__name__)
        super().__init__(game, display)

        self.model = torch.load(PATH,map_location='cpu')
        #self.model = torch.load(PATH,map_location=lambda storage, loc: storage.cuda(0))
        #self.model = torch.load(PATH)
        self.model = torch.load(PATH,map_location=torch.device('cpu'))
        self.model.cuda()
        self.model.eval()

    def step(self):

        board = np.where(self.game.board == 0, 1, self.game.board)
        # print(board)
        #将board矩阵的元素用log变到1-11之间的整数
        board = np.log2(board)
        #对矩阵进行转置，以便将列的信息考虑在哪
        board1=board.T
        board2=np.vstack((board,board1))
        board = board2[:, :, np.newaxis]
        #将矩阵元素进一步压缩到0-1之间的float
        board = board/ 11.0

        trans = transforms.Compose([transforms.ToTensor()])
        board = trans(board)

        board = board.type(torch.float)
        device1 = torch.device("cuda")
        #将输入数据load到GPU上
        board=board.to(device1)
        out = self.model(board)

        direction = torch.max(out, 1)[1]
        # sleep(3600)
        return int(direction)




