"""自我对弈收集数据"""
import torch.multiprocessing as mp
import asyncio
from collections import deque
import copy
import pickle
import time
import my_redis
import zip_array
from async_main.async_game import Board, Game, move_action2move_id, move_id2move_action, flip_map
from async_main.async_batch_mcts import MCTSPlayer
from config import CONFIG

if CONFIG['use_frame'] == 'pytorch':
    from async_main.async_pytorch_net import PolicyValueNet
else:
    print('暂不支持您选择的框架')


# 定义整个对弈收集数据流程
class CollectPipeline:

    def __init__(self, init_model=None):

        # 对弈参数
        self.temp = 1  # 温度
        self.n_playout = CONFIG['play_out']  # 每次移动的模拟次数
        self.c_puct = CONFIG['c_puct']  # u的权重
        self.buffer_size = CONFIG['buffer_size']  # 经验池大小
        self.data_buffer = deque(maxlen=self.buffer_size)
        self.iters = 0
        self.update_model_version = 0
        self.redis_cli = my_redis.get_redis_cli()
        self.loop = asyncio.get_event_loop()
        self.queue = asyncio.Queue()
        self.lock = asyncio.Lock()

    # 从主体加载模型
    def load_model(self):
        if CONFIG['use_frame'] == 'paddle':
            self.model_path = CONFIG['paddle_model_path']
        elif CONFIG['use_frame'] == 'pytorch':
            self.model_path = CONFIG['pytorch_model_path']
        else:
            print('暂不支持所选框架')
        try:
            self.policy_value_net = PolicyValueNet(model_file=self.model_path)
            print('已加载最新模型')
        except:
            self.policy_value_net = PolicyValueNet()
            print('已加载初始模型')

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

        # 管理队列数据，一旦队列中有数据，就统一传给神经网络，获得预测结果

    async def prediction_worker(self):

        while True:
            board_list = []
            future_list = []
            index_list = [0]
            for _ in range(8):
                features, future = await self.queue.get()
                board_list.extend(features)
                future_list.append(future)
                index_list.append(len(board_list))
            async with self.lock:
                action_prob_list, leaf_value_list = await self.policy_value_net.policy_value_fn(board_list)
            for i in range(len(future_list)):
                future_list[i].set_result((action_prob_list[index_list[i]:index_list[i + 1]],
                                          leaf_value_list[index_list[i]:index_list[i + 1]]))



    async def collect_selfplay_data(self, n_games=1):
        # 收集自我对弈的数据
        for i in range(n_games):
            version = self.redis_cli.get('update_model_version')
            if self.update_model_version != version:
                async with self.lock:
                    self.policy_value_net.update_state(self.model_path)  # 从本体处加载最新模型
                self.update_model_version = version
                print('已更新模型参数')

            mcts_player = MCTSPlayer(self.queue,
                                     c_puct=self.c_puct,
                                     n_playout=self.n_playout,
                                     is_selfplay=1)
            # 象棋逻辑和棋盘
            board = Board()
            game = Game(board)
            winner, play_data = await game.start_self_play(mcts_player, temp=self.temp, is_shown=False)
            play_data = list(play_data)[:]
            self.episode_len = len(play_data)
            # 增加数据
            play_data = self.get_equi_data(play_data)
            while True:
                try:

                    for d in play_data:
                        self.redis_cli.rpush('train_data_buffer', pickle.dumps(d))
                    self.redis_cli.incr('iters')
                    self.iters = self.redis_cli.get('iters')
                    print("存储完成")
                    print('batch i: {}, episode_len: {}'.format(
                        self.iters, self.episode_len))
                    break
                except:
                    print("存储失败")
                    time.sleep(1)
        return self.iters

    def run(self):
        """开始收集数据"""
        try:
            self.load_model()
            while True:
                coroutine_list = []
                coroutine_list.append(self.prediction_worker())
                for _ in range(16):
                    coroutine_list.append(self.collect_selfplay_data())
                self.loop.run_until_complete(asyncio.gather(*coroutine_list))

        except KeyboardInterrupt:
            print('\n\rquit')


if __name__ == '__main__':
    if CONFIG['use_frame'] == 'pytorch':
        collecting_pipeline = CollectPipeline(init_model='current_policy.pkl')
        collecting_pipeline.run()
    else:
        print('暂不支持您选择的框架')
        print('训练结束')
