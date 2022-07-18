import json
import pickle
from config import CONFIG
import aioredis as redis



def get_redis_cli():
    r = redis.StrictRedis(host=CONFIG['redis_host'], port=CONFIG['redis_port'], db=CONFIG['redis_db'])
    return r
def get_list_range(redis_cli,name,l,r=-1):
    assert isinstance(redis_cli,redis.Redis)
    list = redis_cli.lrange(name,l,r)
    return [pickle.loads(d) for d in list]

# if __name__ == '__main__':
#     r = get_redis_cli()
    # with open(CONFIG['train_data_buffer_path'], 'rb') as data_dict:
    #     data_file = pickle.load(data_dict)
    #     data_buffer = data_file['data_buffer']
    # for d in data_buffer:
    #     r.rpush('train_data_buffer',pickle.dumps(d))
    # r.rpush('test',pickle.dumps(([8,2],[2,4],5)))
    # p = get_list_range(r,'test',0,-1)
    # print(p)
    # if r.llen('train_data_buffer') > 100000:
    #     for i in range(7000):
    #         r.lpop('train_data_buffer')

    # policy_value_net = PolicyValueNet()
    # s = pickle.dumps(policy_value_net)
    # print(policy_value_net)
    # t = pickle.loads(s)
    # print(t)
    # print(r.get('a'))
    # r.set('b',json.dumps({'a':1,'b':2}))
