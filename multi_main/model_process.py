from torch.multiprocessing import Process

import my_redis
from config import CONFIG

if CONFIG['use_frame'] == 'pytorch':
    from multi_main.multi_pytorch_net import PolicyValueNet
else:
    print('暂不支持您选择的框架')

#模型推理独立进程
class ModelProcess(Process):

    def __init__(self,queues):
        Process.__init__(self)
        self.update_model_version = 0
        self.queues = queues #(output,input)
        self.n_processes = CONFIG['n_processes']
        self.n_threads = CONFIG['n_threads']


    def run(self) -> None:
        self.redis_cli = my_redis.get_redis_cli()
        """
        通过管道接收训练数据
        """
        self.load_model()
        try:
            while True:

                # time.sleep(1e-3)
                index_list = [0]
                current_state_list, legal_position_list = [], []
                for i in range(self.n_threads):

                    for _,input in self.queues[i]:
                        try:
                            state,legal_positions = input.get_nowait()
                        except:
                            state = None

                        if state is not None and len(state) > 0:
                            current_state_list.extend(state)
                            legal_position_list.extend(legal_positions)
                        index_list.append(len(current_state_list))
                    # if len(current_state_list) > 0:
                    #     self.update_model_state()
                    #     action_prob_list, leaf_value_list = self.policy_value_net.policy_value_fn(current_state_list,
                    #                                                                               legal_position_list)
                    #     index = -1
                    #     for i, (output, _) in enumerate(self.queues[i]):
                    #         index += 1
                    #         if index_list[index] == index_list[index + 1]:
                    #             continue
                    #         output.put((action_prob_list[index_list[index]:index_list[index + 1]],
                    #                     leaf_value_list[index_list[index]:index_list[index + 1]]))

                if len(current_state_list) > 0:
                    self.update_model_state()
                    action_prob_list, leaf_value_list = self.policy_value_net.policy_value_fn(current_state_list,legal_position_list)
                    index = -1
                    for i in range(self.n_threads):
                        for i,(output,_) in enumerate(self.queues[i]):
                            index += 1
                            if index_list[index] == index_list[index+1]:
                                continue
                            output.put((action_prob_list[index_list[index]:index_list[index+1]],leaf_value_list[index_list[index]:index_list[index+1]]))

        except Exception as e:
            print(e)



    def update_model_state(self):
        try:

            version = self.redis_cli.get('update_model_version')
            if self.update_model_version != version:
                self.policy_value_net.update_state(self.model_path)  # 从本体处加载最新模型
                self.update_model_version = version
                print('已更新模型参数,version : {}'.format(version))

        except:
            pass

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
        self.redis_cli.close()

# if __name__ == '__main__':
#     process = ModelProcess()
#     process.start()
#     process.join()