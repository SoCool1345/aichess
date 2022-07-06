"""自我对弈收集数据"""

from multiprocessing import Process, Manager, Pool, freeze_support, current_process, Pipe
from config import CONFIG



import random
from collections import deque
import copy
import os
import pickle
import time
import my_redis, redis
import zip_array
from game import Board, Game, move_action2move_id, move_id2move_action, flip_map
from multi_batch_mcts import MCTSPlayer



# if CONFIG['use_frame'] == 'paddle':
#     from paddle_net import PolicyValueNet
# elif CONFIG['use_frame'] == 'pytorch':
#     from pytorch_net import PolicyValueNet
# else:
#     print('暂不支持您选择的框架')


# 定义整个对弈收集数据流程
class CollectPipeline(Process):


    def __init__(self,pipes):
        Process.__init__(self)
        # 象棋逻辑和棋盘
        self.board = Board()
        self.game = Game(self.board)
        # 对弈参数
        self.temp = 1  # 温度
        self.n_playout = CONFIG['play_out']  # 每次移动的模拟次数
        self.c_puct = CONFIG['c_puct']  # u的权重
        self.iters = 0
        self.pipes = pipes
        self.n_processes = CONFIG['n_processes']
        self.n_threads = CONFIG['n_threads']


    def get_equi_data(self, play_data):
        """左右对称变换，扩充数据集一倍，加速一倍训练速度"""
        extend_data = []
        # 棋盘状态shape is [9, 10, 9], 走子概率，赢家
        for state, mcts_prob, winner in play_data:
            # 原始数据
            extend_data.append(zip_array.zip_state_mcts_prob((state, mcts_prob, winner)))
            # 水平翻转后的数据
            state_flip = state.transpose([1, 2, 0])
            state = state.transpose([1, 2, 0])
            for i in range(10):
                for j in range(9):
                    state_flip[i][j] = state[i][8 - j]
            state_flip = state_flip.transpose([2, 0, 1])
            mcts_prob_flip = copy.deepcopy(mcts_prob)
            for i in range(len(mcts_prob_flip)):
                mcts_prob_flip[i] = mcts_prob[move_action2move_id[flip_map(move_id2move_action[i])]]
            extend_data.append(zip_array.zip_state_mcts_prob((state_flip, mcts_prob_flip, winner)))
        return extend_data

    def collect_selfplay_data(self,pipe ,n_games=1):
        # 收集自我对弈的数据
        for i in range(n_games):
            mcts_player = MCTSPlayer(pipe,
                                     c_puct=self.c_puct,
                                     n_playout=self.n_playout,
                                     is_selfplay=1)


            winner, play_data = self.game.start_self_play(mcts_player, temp=self.temp, is_shown=False)
            play_data = list(play_data)[:]
            self.episode_len = len(play_data)
            # 增加数据
            play_data = self.get_equi_data(play_data)
            # 存储数据
            self.push_play_data(play_data)
        return self.iters

    # 存储数据
    def push_play_data(self, play_data):
        redis_cli = my_redis.get_redis_cli()
        while True:
            try:

                for d in play_data:
                    redis_cli.rpush('train_data_buffer', pickle.dumps(d))
                redis_cli.incr('iters')
                self.iters = redis_cli.get('iters')
                print("存储完成")
                break
            except:
                print("存储失败")
                time.sleep(1)
        redis_cli.close()



    def run(self):
        """开始收集数据"""
        try:
            print(str(current_process().pid)+"开始收集数据")
            while True:
                iters = self.collect_selfplay_data(self.pipes[0])
                print('batch i: {}, episode_len: {}'.format(
                    iters, self.episode_len))
        except KeyboardInterrupt:
            print('\n\rquit')

    def close(self) -> None:
        self.pipe.close()
        super().close()


if __name__ == '__main__':
    freeze_support()
    # pool = Pool(processes=CONFIG['n_processes'])
    # m = Manager()
    n_processes = CONFIG['n_processes']
    n_threads = CONFIG['n_threads']
    pipes = [] #[n_processes,n_threads]
    for _ in range(n_processes):
        pipes.append([(Pipe(duplex=False),Pipe(duplex=False)) for _ in range(n_threads)])

    if CONFIG['use_frame'] == 'paddle':
        for i in range(CONFIG['n_processes']):
            collecting_pipeline = CollectPipeline(pipes[i])
            collecting_pipeline.run()
    elif CONFIG['use_frame'] == 'pytorch':
        for i in range(CONFIG['n_processes']):
            collecting_pipeline = CollectPipeline(pipes[i])
            collecting_pipeline.start()
    else:
        print('暂不支持您选择的框架')
        print('训练结束')
    model = __import__("model_process", globals(), locals(), ['ModelProcess'])
    model_process = model.ModelProcess(pipes)
    model_process.start()
    model_process.join()



