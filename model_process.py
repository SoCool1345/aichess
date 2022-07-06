import multiprocessing
import time
from multiprocessing import Process

import my_redis
from config import CONFIG

if CONFIG['use_frame'] == 'paddle':
    from paddle_net import PolicyValueNet
elif CONFIG['use_frame'] == 'pytorch':
    from pytorch_net import PolicyValueNet
else:
    print('暂不支持您选择的框架')

#模型推理独立进程
class ModelProcess(Process):


    def __init__(self,pipes):
        Process.__init__(self)
        self.update_model_version = 0
        self.pipe_list = pipes #(output,input)
        self.n_processes = CONFIG['n_processes']
        self.n_threads = CONFIG['n_threads']

    def get_pipe(self):
        # 创建双管道(out,in)
        input = multiprocessing.Pipe(False)
        output = multiprocessing.Pipe(False)
        self.pipes.append((output,input))
        return (output,input)


    def run(self) -> None:
        """
        通过管道接收训练数据
        """
        self.load_model()
        while True:
            # time.sleep(1e-3)
            board_list = []
            index_list = [0]
            for pipes in self.pipe_list:
                for _,input in pipes:
                    conn1 ,conn2  = input
                    if conn1.poll(1e9):
                        boards = conn1.recv()
                    else:
                        boards = None
                    if boards is not None and len(boards) > 0:
                        board_list.extend(boards)
                    index_list.append(len(board_list))

            if len(board_list) > 0:
                action_prob_list, leaf_value_list = self.policy_value_net.policy_value_fn(board_list)
                index = 0
                for pipes in self.pipe_list:
                    for i,(output,_) in enumerate(pipes):
                        if index_list[index] == index_list[index+1]:
                            continue
                        conn1, conn2 = output
                        conn2.send((action_prob_list[index_list[index]:index_list[index+1]],leaf_value_list[index_list[index]:index_list[index+1]]))
                        index += 1



    def update_model_state(self):
        redis_cli = my_redis.get_redis_cli()
        version = redis_cli.get('update_model_version')
        if self.update_model_version != version:
            self.policy_value_net.update_state(self.model_path)  # 从本体处加载最新模型
            self.update_model_version = version
            print('已更新模型参数')
        redis_cli.close()

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

    def close(self):
        for pipe in self.pipes:
            pipe.close()

# if __name__ == '__main__':
#     process = ModelProcess()
#     process.start()
#     process.join()