"""自我对弈收集数据"""
import queue
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy
import torch.multiprocessing as mp

from config import CONFIG

import copy
import pickle
import time
import my_redis
import zip_array
from game import Board, Game, move_action2move_id, move_id2move_action, flip_map
from batch_mcts import MCTSPlayer

if CONFIG['use_frame'] == 'pytorch':
    from pytorch_net import PolicyValueNet
else:
    print('暂不支持您选择的框架')

# 定义整个对弈收集数据流程
from utils.timefn import timefn


class CollectPipeline(mp.Process):

    def __init__(self,model):
        mp.Process.__init__(self)

        # 对弈参数
        self.temp = 1  # 温度
        self.n_playout = CONFIG['play_out']  # 每次移动的模拟次数
        self.c_puct = CONFIG['c_puct']  # u的权重
        self.iters = 0
        # self.queues = queues
        self.n_processes = CONFIG['n_processes']
        self.n_threads = CONFIG['n_threads']
        self.model = model
    def create(self):
        self.redis_cli = my_redis.get_redis_cli()
        self.pool = ThreadPoolExecutor(max_workers=self.n_threads)

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

    # @profile
    def collect_selfplay_data(self, queue, n_games=1):
        # 收集自我对弈的数据
        for i in range(n_games):
            mcts_player = MCTSPlayer(self.model.policy_value_fn,
                                     c_puct=self.c_puct,
                                     n_playout=self.n_playout,
                                     is_selfplay=1)
            # 象棋逻辑和棋盘
            board = Board()
            game = Game(board)
            winner, play_data = game.start_self_play(mcts_player, temp=self.temp, is_shown=False)
            play_data = list(play_data)[:]
            self.episode_len = len(play_data)
            # 增加数据
            play_data = self.get_equi_data(play_data)
            # 存储数据
            self.push_play_data(play_data)
        return self.iters

    # 存储数据
    def push_play_data(self, play_data):

        while True:
            try:

                for d in play_data:
                    self.redis_cli.rpush('train_data_buffer', pickle.dumps(d))
                self.redis_cli.incr('iters')
                self.iters = self.redis_cli.get('iters')
                print('存储完成 batch i: {}, episode_len: {}'.format(
                    self.iters, self.episode_len))
                break
            except:
                print("存储失败")
                time.sleep(1)

    def run(self):
        self.create()
        """开始收集数据"""
        try:
            for i in range(self.n_threads):
                self.pool.submit(self.thread_run, self.queues[i])
            # self.thread_run(self.queues[0])
        except KeyboardInterrupt:
            print('\n\rquit')
        except Exception as e:
            print(e)

    def thread_run(self, queue):
        print(str(mp.current_process().pid) + threading.current_thread().getName() + "开始收集数据")
        while True:
            self.collect_selfplay_data(queue)

    def close(self) -> None:
        super().close()
        self.redis_cli.close()
        self.pool.shutdown()

    def update_model_state(self):
        try:

            version = self.redis_cli.get('update_model_version')
            if self.update_model_version != version:
                self.policy_value_net.update_state(self.model_path)  # 从本体处加载最新模型
                self.update_model_version = version
                print('已更新模型参数,version : {}'.format(version))

        except:
            pass


def load_model():
    if CONFIG['use_frame'] == 'paddle':
        model_path = CONFIG['paddle_model_path']
    elif CONFIG['use_frame'] == 'pytorch':
        model_path = CONFIG['pytorch_model_path']
    else:
        print('暂不支持所选框架')
    try:
        policy_value_net = PolicyValueNet(model_file=model_path)
        print('已加载最新模型')
    except:
        policy_value_net = PolicyValueNet()
        print('已加载初始模型')
    return policy_value_net


if __name__ == '__main__':
    mp.freeze_support()
    # mp.set_start_method('fork')
    # manager = multiprocessing.managers.SyncManager()
    # manager.start()

    n_processes = CONFIG['n_processes']
    n_threads = CONFIG['n_threads']
    # l  = manager.list(asyncio.Queue() for _ in range(n_processes))

    # async_queues = [asyncio.Queue() for _ in range(n_processes)]
    # pool = mp.Pool(processes=n_processes)
    # q = []
    model = load_model()
    # model.policy_value_net.to('cpu')
    # model.share_memory()
    # model.policy_value_net.to('cuda')
    if CONFIG['use_frame'] == 'pytorch':
        for i in range(CONFIG['n_processes']):
            # queues = [(mp.Queue(), mp.Queue()) for _ in range(n_threads)]
            # q.append(queues)
            collecting_pipeline = CollectPipeline( model)
            collecting_pipeline.start()

    else:
        print('暂不支持您选择的框架')
        print('训练结束')
    # q = numpy.array(q)
    # q = numpy.transpose(q, axes=(1, 0, 2))
    # model = __import__("multi_main.model_process", globals(), locals(), ['ModelProcess'])
    # model_process = model.ModelProcess(q)
    # model_process.start()
    # model_process.join()
    collecting_pipeline.join()

