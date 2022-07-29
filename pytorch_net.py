"""策略价值网络"""

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from config import CONFIG
from torch.cuda.amp import autocast


# 搭建残差块
from utils.timefn import timefn


class ResBlock(nn.Module):

    def __init__(self, num_filters=256):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=num_filters, out_channels=num_filters, kernel_size=(3, 3), stride=(1, 1),
                               padding=1)
        self.conv1_bn = nn.BatchNorm2d(num_filters, )
        self.conv1_act = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=num_filters, out_channels=num_filters, kernel_size=(3, 3), stride=(1, 1),
                               padding=1)
        self.conv2_bn = nn.BatchNorm2d(num_filters, )
        self.conv2_act = nn.ReLU()

    def forward(self, x):
        y = self.conv1(x)
        y = self.conv1_bn(y)
        y = self.conv1_act(y)
        y = self.conv2(y)
        y = self.conv2_bn(y)
        y = x + y
        return self.conv2_act(y)


# 搭建骨干网络，输入：N, 9, 10, 9 --> N, C, H, W
class Net(nn.Module):

    def __init__(self, num_channels=256, num_res_blocks=7):
        super().__init__()
        # 全局特征
        # self.global_conv = nn.Conv2d(in_channels=9, out_channels=512, kernel_size=(10, 9))
        # self.global_bn = nn.BatchNorm1d(512)
        # 初始化特征
        self.conv_block = nn.Conv2d(in_channels=9, out_channels=num_channels, kernel_size=(3, 3), stride=(1, 1),
                                    padding=1)
        self.conv_block_bn = nn.BatchNorm2d(256)
        self.conv_block_act = nn.ReLU()
        # 残差块抽取特征
        self.res_blocks = nn.ModuleList([ResBlock(num_filters=num_channels) for _ in range(num_res_blocks)])
        # 策略头
        # self.global_policy_fc = nn.Linear(512, 2086)
        self.policy_conv = nn.Conv2d(in_channels=num_channels, out_channels=16, kernel_size=(1, 1), stride=(1, 1))
        self.policy_bn = nn.BatchNorm2d(16)
        self.policy_act = nn.ReLU()
        self.policy_fc = nn.Linear(16 * 9 * 10, 2086)
        # 价值头
        # self.global_value_fc = nn.Linear(512, 256)
        self.value_conv = nn.Conv2d(in_channels=num_channels, out_channels=8, kernel_size=(1, 1), stride=(1, 1))
        self.value_bn = nn.BatchNorm2d(8)
        self.value_act1 = nn.ReLU()
        self.value_fc1 = nn.Linear(8 * 9 * 10, 256)
        self.value_act2 = nn.ReLU()
        self.value_fc2 = nn.Linear(256, 1)

    # 定义前向传播
    # @timefn
    def forward(self, x):
        # 公共头
        # global_x = self.global_conv(x)
        # global_x = torch.reshape(global_x, [-1, 512])
        # global_x = self.global_bn(global_x)
        x = self.conv_block(x)
        x = self.conv_block_bn(x)
        x = self.conv_block_act(x)
        for layer in self.res_blocks:
            x = layer(x)
        # 策略头
        policy = self.policy_conv(x)
        policy = self.policy_bn(policy)
        policy = self.policy_act(policy)
        policy = torch.reshape(policy, [-1, 16 * 10 * 9])
        policy = self.policy_fc(policy)
        # global_policy = self.policy_act(self.global_policy_fc(global_x))
        # policy = F.log_softmax(policy + global_policy)
        policy = F.log_softmax(policy)
        # 价值头
        value = self.value_conv(x)
        value = self.value_bn(value)
        value = self.value_act1(value)
        value = torch.reshape(value, [-1, 8 * 10 * 9])
        # global_value = self.value_act1(self.global_value_fc(global_x))
        value = self.value_fc1(value)
        value = self.value_act1(value)
        # value = self.value_fc2(value + global_value)
        value = self.value_fc2(value)
        value = F.tanh(value)

        return policy, value


# 策略值网络，用来进行模型的训练
class PolicyValueNet:

    def __init__(self, model_file=None, use_gpu=True, device='cuda'):
        self.use_gpu = use_gpu
        self.l2_const = 2e-3  # l2 正则化
        self.device = device
        torch.backends.cudnn.benchmark = True
        self.policy_value_net = Net(num_res_blocks=16).to(self.device)
        self.optimizer = torch.optim.Adam(params=self.policy_value_net.parameters(), lr=1e-3, betas=(0.9, 0.999),
                                          eps=1e-8, weight_decay=self.l2_const)
        self.update_state(model_file)

        # 更新模型参数

    def share_memory(self):
        self.policy_value_net.share_memory()

    def update_state(self, model_file=None):
        if model_file:
            self.policy_value_net.load_state_dict(torch.load(model_file),strict=False)  # 加载模型参数

    # 输入一个批次的状态，输出一个批次的动作概率和状态价值
    def policy_value(self, state_batch):
        self.policy_value_net.eval()
        state_batch = torch.as_tensor(state_batch,device=self.device)
        with autocast():  # 半精度fp16
            log_act_probs, value = self.policy_value_net(state_batch)
        log_act_probs, value = log_act_probs.cpu(), value.cpu()
        act_probs = np.exp(log_act_probs.detach().numpy())
        return act_probs, value.detach().numpy()

    # 输入棋盘，返回每个合法动作的（动作，概率）元组列表，以及棋盘状态的分数
    # @timefn
    def policy_value_fn(self, board_list):
        self.policy_value_net.eval()
        # 获取合法动作列表
        try:
            legal_positions = [board.availables for board in board_list]
        except:
            print(board_list)
        current_state_list = [torch.as_tensor(board.current_state().astype('float16'),device=self.device) for board in board_list]
        if len(current_state_list) > 1:
            current_state = torch.stack(current_state_list)
        elif len(current_state_list) == 1:
            current_state = current_state_list[0].reshape(-1, 9, 10, 9)
        else:
            return
        # current_state = torch.as_tensor(current_state)
        # 使用神经网络进行预测
        with autocast():  # 半精度fp16
            log_act_probs, value = self.policy_value_net(current_state)
        log_act_probs, value = log_act_probs.cpu(), value.cpu()
        act_prob_list = []
        for i in range(len(log_act_probs)):
            act_probs = np.exp(log_act_probs[i].numpy().astype('float16').flatten()) if CONFIG['use_frame'] == 'paddle' else np.exp(
                log_act_probs[i].detach().numpy().astype('float16').flatten())
            # 只取出合法动作
            act_probs = zip(legal_positions[i], act_probs[legal_positions[i]])
            act_prob_list.append(act_probs)
        # 返回动作概率，以及状态价值
        return np.asarray(act_prob_list), value.detach().numpy()

    # 保存模型
    def save_model(self, model_file):
        torch.save(self.policy_value_net.state_dict(), model_file)

    # 执行一步训练
    def train_step(self, state_batch, mcts_probs, winner_batch, lr=0.002):
        self.policy_value_net.train()
        # 包装变量
        state_batch = torch.as_tensor(state_batch,device=self.device)
        mcts_probs = torch.as_tensor(mcts_probs,device=self.device)
        winner_batch = torch.as_tensor(winner_batch,device=self.device)
        # 清零梯度
        self.optimizer.zero_grad()
        # 设置学习率
        for params in self.optimizer.param_groups:
            # 遍历Optimizer中的每一组参数，将该组参数的学习率 * 0.9
            params['lr'] = lr
        # 前向运算
        log_act_probs, value = self.policy_value_net(state_batch)
        value = torch.reshape(value, shape=[-1])
        # 价值损失
        value_loss = F.mse_loss(input=value, target=winner_batch)
        # 策略损失
        policy_loss = -torch.mean(torch.sum(mcts_probs * log_act_probs, dim=1))  # 希望两个向量方向越一致越好
        # 总的损失，注意l2惩罚已经包含在优化器内部
        loss = value_loss + policy_loss
        # 反向传播及优化
        loss.backward()
        self.optimizer.step()
        # 计算策略的熵，仅用于评估模型
        with torch.no_grad():
            entropy = -torch.mean(
                torch.sum(torch.exp(log_act_probs) * log_act_probs, dim=1)
            )
        return loss.detach().cpu().numpy(), entropy.detach().cpu().numpy()


if __name__ == '__main__':
    net = Net().to('cuda')
    test_data = torch.ones([8, 9, 10, 9]).to('cuda')
    x_act, x_val = net(test_data)
    print(x_act.shape)  # 8, 2086
    print(x_val.shape)  # 8, 1
