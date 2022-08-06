import os
import random

import elo

from batch_mcts import MCTSPlayer
from game import Board
from pytorch_net import PolicyValueNet

net1 = PolicyValueNet()
net2 = PolicyValueNet()
player1 = MCTSPlayer(net1.policy_value_fn,
                     c_puct=5,
                     n_playout=150,
                     is_selfplay=0)
player2 = MCTSPlayer(net2.policy_value_fn,
                     c_puct=5,
                     n_playout=150,
                     is_selfplay=0)
start_player = 1
board=Board()

p1, p2 = 1, 2
player1.set_player_ind(1)
player2.set_player_ind(2)
players = {p1: player1, p2: player2}
board.init_board(start_player)
p1, p2 = 1, 2
player1.set_player_ind(1)
player2.set_player_ind(2)
players = {p1: player1, p2: player2}


def self_play_elo(model_files,elos,n_games=100):
    """
    Self-play function.
    """
    if elos is None:
        elos = [elo.Rating(1000) for _ in range(len(model_files))]
    for i in range(n_games):
        first = random.randint(0,len(model_files)-1)
        back = random.randint(0,len(model_files)-1)
        if first == back:
            back = (back+1)%len(model_files)
        net1.update_state(model_files[first])
        net2.update_state(model_files[back])
        winner = start_self_play(players)
        if winner == -1:
            elos[first],elos[back] = elo.rate_1vs1(elos[first],elos[back], drawn=True)
        elif winner == 1:
            elos[first],elos[back] = elo.rate_1vs1(elos[first],elos[back], drawn=False)
        else:
            elos[back], elos[first] = elo.rate_1vs1(elos[back], elos[first], drawn=False)
        print('先手',first,'后手',back,'赢家',winner)
        print('当前elo:',elos)
    return elos
def start_self_play(players):
    board.init_board(start_player)
    while True:
        current_player = board.get_current_player_id()  # 红子对应的玩家id
        player_in_turn = players[current_player]  # 决定当前玩家的代理
        move = player_in_turn.get_action(board)  # 当前玩家代理拿到动作
        board.do_move(move)
        end, winner = board.game_end()
        if end:
            return winner


if __name__ == '__main__':
    paths = []
    for path in os.listdir('../models'):
        if path.find('pkl') >= 0 or path.find('model') >= 0:
            paths.append(os.path.join('../models', path))
    print(paths)
    elos = self_play_elo(paths,None,100)
    print(paths,elos)