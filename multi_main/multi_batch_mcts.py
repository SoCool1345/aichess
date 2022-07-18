"""蒙特卡洛树搜索"""

import numpy as np
import copy
from config import CONFIG
import torch

from utils.timefn import timefn


def softmax(x):
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs


# 定义叶子节点
class TreeNode(object):
    """
    mcts树中的节点，树的子节点字典中，键为动作，值为TreeNode。记录当前节点选择的动作，以及选择该动作后会跳转到的下一个子节点。
    每个节点跟踪其自身的Q，先验概率P及其访问次数调整的u
    """
    __slots__ = ('_parent', '_children','_n_visits', '_Q', '_u', 'W', '_P', 'is_expending')

    def __init__(self, parent, prior_p):
        """
        :param parent: 当前节点的父节点
        :param prior_p:  当前节点被选择的先验概率
        :param in_state: 当前节点的状态
        """
        self._parent = parent
        self._children = {} # 从动作到TreeNode的映射
        self._n_visits = 0  # 当前当前节点的访问次数
        self._Q = torch.tensor(0.,dtype=torch.float16)         # 当前节点对应动作的平均动作价值
        self._u = torch.tensor(0.,dtype=torch.float16)         # 当前节点的置信上限         # PUCT算法
        self.W = torch.tensor(0.,dtype=torch.float16)
        self._P = prior_p
        self.is_expending = False

    def add_virtual_value(self, value):
        """
        计算节点的虚拟损失
        """
        self.W -= value
        self._n_visits += value
        self._Q = self.W / self._n_visits if self._n_visits > 0 else 0



    def expand(self, action_priors):    # 这里把不合法的动作概率全部设置为0
        """通过创建新子节点来展开树"""
        for action, prob in action_priors:
            if action not in self._children:
                self._children[action] =  TreeNode(self, prob)
        self.is_expending = False

    def select(self, c_puct):
        """
        在子节点中选择能够提供最大的Q+U的节点
        return: (action, next_node)的二元组
        """
        return max(self._children.items(),
                   key=lambda act_node: act_node[1].get_value(c_puct))

    def get_value(self, c_puct):
        """
        计算并返回此节点的值，它是节点评估Q和此节点的先验的组合
        c_puct: 控制相对影响（0， inf）
        """
        self._u = (c_puct * self._P *
                   np.sqrt(self._parent._n_visits) / (1 + self._n_visits))
        return self._Q + self._u

    def update(self, leaf_value):
        """
        从叶节点评估中更新节点值
        leaf_value: 这个子节点的评估值来自当前玩家的视角
        """
        # 统计访问次数
        self._n_visits += 1
        self.W += leaf_value
        # 更新Q值，取决于所有访问次数的平均树，使用增量式更新方式
        self._Q = self.W / self._n_visits

    # 使用递归的方法对所有节点（当前节点对应的支线）进行一次更新
    def update_recursive(self, leaf_value):
        """就像调用update()一样，但是对所有直系节点进行更新"""
        # 如果它不是根节点，则应首先更新此节点的父节点
        if self._parent:
            self._parent.update_recursive(-leaf_value)
        self.update(leaf_value)

    def is_leaf(self):
        """检查是否是叶节点，即没有被扩展的节点"""
        return self._children == {}

    def is_root(self):
        return self._parent is None

    @property
    def P(self):
        return self._P


# 蒙特卡洛搜索树
class MCTS(object):

    def __init__(self, queue,policy_value_fn, c_puct=5, n_playout=2000):
        """policy_value_fn: 接收board的盘面状态，返回落子概率和盘面评估得分"""
        self._root = TreeNode(None, 1.0)
        self.queue = queue # (input,output)
        self._c_puct = c_puct
        self._n_playout = n_playout

        self.virtual_loss = 3
        self.search_batch_size = 16

        self._policy = policy_value_fn


    # @timefn
    def _playout(self, board):
        """
        进行一次搜索，根据叶节点的评估值进行反向更新树节点的参数
        注意：state已就地修改，因此必须提供副本
        """
        board_list = []
        node_list = []

        for _ in range(self.search_batch_size):
            node = self._root
            _board = copy.deepcopy(board)
            while True:
                if node.is_leaf() or node.is_expending:
                    break
                # 贪心算法选择下一步行动
                action, node = node.select(self._c_puct)
                # if action not in _board.availables:
                #     print("action:"+str(action))
                _board.do_move(action.item())
            if not node.is_expending:
                node.is_expending = True
                node.add_virtual_value(self.virtual_loss)
                board_list.append(_board)
                node_list.append(node)
            else:
                break
            if len(board_list) == 0:
                return

        # legal_positions = [torch.as_tensor(b.availables).share_memory_() for b in board_list]
        # current_state_list = [torch.as_tensor(b.current_state().astype('float16')).share_memory_() for b in board_list]

        # 使用网络评估叶子节点，网络输出（动作，概率）元组p的列表以及当前玩家视角的得分[-1, 1]
        # input, output = self.queue
        # output.put((current_state_list,legal_positions))
        # action_prob_list, leaf_value_list = input.get()
        # print(multiprocessing.current_process().pid,str(threading.currentThread().getName()))
        action_prob_list, leaf_value_list = self._policy(board_list)

        for node, action_probs, leaf_value, _board in zip(node_list, action_prob_list, leaf_value_list, board_list):
            node.add_virtual_value(-self.virtual_loss)
            node.is_expending = False
            # 查看游戏是否结束
            end, winner = _board.game_end()
            if not end:
                node.expand(action_probs)
            else:
                # 对于结束状态，将叶子节点的值换成1或-1
                if winner == -1:  # Tie
                    leaf_value = 0.0
                else:
                    leaf_value = (
                        1.0 if winner == _board.get_current_player_id() else -1.0
                    )
            # 在本次遍历中更新节点的值和访问次数
            # 必须添加符号，因为两个玩家共用一个搜索树
            node.update_recursive(-leaf_value)

        # for node, action_probs,leaf_value,_board in zip(node_list, action_prob_list,leaf_value_list,board_list):
        #     node.add_virtual_value(-self.virtual_loss)
        #     node.is_expending = False
        #     # 查看游戏是否结束
        #     end, winner = _board.game_end()
        #     if not end:
        #         node.expand(action_probs)
        #     else:
        #         # 对于结束状态，将叶子节点的值换成1或-1
        #         if winner == -1:  # Tie
        #             leaf_value = 0.0
        #         else:
        #             leaf_value = (
        #                 1.0 if winner == _board.get_current_player_id() else -1.0
        #             )
        #         leaf_value = torch.as_tensor([leaf_value], dtype=torch.float16)
        #         # 在本次遍历中更新节点的值和访问次数
        #         # 必须添加符号，因为两个玩家共用一个搜索树
        #     node.update_recursive(-leaf_value[0])





    def get_move_probs(self, board, temp=1e-3):
        """
        按顺序运行所有搜索并返回可用的动作及其相应的概率
        state:当前游戏的状态
        temp:介于（0， 1]之间的温度参数
        """
        for n in range(self._n_playout//self.search_batch_size):
            self._playout(board)

        # 跟据根节点处的访问计数来计算移动概率
        act_visits= [(act, node._n_visits)
                     for act, node in self._root._children.items()]
        acts, visits = zip(*act_visits)
        act_probs = softmax(1.0 / temp * np.log(np.array(visits) + 1e-10))
        return acts, act_probs

    def update_with_move(self, last_move):
        """
        在当前的树上向前一步，保持我们已经直到的关于子树的一切
        """
        if last_move in self._root._children:
            self._root = self._root._children[last_move]
            del self._root._parent
            self._root._parent = None
        else:
            self._root = TreeNode(None, 1.0)

    def __str__(self):
        return 'MCTS'


# 基于MCTS的AI玩家
class MCTSPlayer(object):

    def __init__(self, queue,policy_value_fn, c_puct=5, n_playout=2000, is_selfplay=0):
        self.mcts = MCTS(queue,policy_value_fn, c_puct, n_playout)
        self._is_selfplay = is_selfplay
        self.agent = "AI"

    def set_player_ind(self, p):
        self.player = p

    # 重置搜索树
    def reset_player(self):
        self.mcts.update_with_move(-1)

    def __str__(self):
        return 'MCTS {}'.format(self.player)

    # 得到行动
    def get_action(self, board, temp=1e-3, return_prob=0):
        # 像alphaGo_Zero论文一样使用MCTS算法返回的pi向量
        move_probs = np.zeros(2086)

        acts, probs = self.mcts.get_move_probs(board, temp)
        move_probs[list(acts)] = probs
        if self._is_selfplay:
            # 添加Dirichlet Noise进行探索（自我对弈需要）
            move = np.random.choice(
                acts,
                p=0.75*probs + 0.25*np.random.dirichlet(CONFIG['dirichlet'] * np.ones(len(probs)))
            )
            # 更新根节点并重用搜索树
            self.mcts.update_with_move(move)
        else:
            # 使用默认的temp=1e-3，它几乎相当于选择具有最高概率的移动
            move = np.random.choice(acts, p=probs)
            # 重置根节点
            self.mcts.update_with_move(-1)
        if return_prob:
            return move, move_probs
        else:
            return move


