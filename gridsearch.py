from finetune import api
from config import Params
from tools import *
import os

data_dict = [
    {'data': r'./processed_signal/HKU956/772_12s_step_2s.pkl',
        'spliter': r'./processed_signal/HKU956/772_12s_step_2s_spliter5.pkl'},
    {'data': r'./processed_signal/HKU956/516_8s_step_2s.pkl',
        'spliter': r'./processed_signal/HKU956/516_8s_step_2s_spliter5.pkl'},
    {'data': r'./processed_signal/HKU956/260_4s_step_2s.pkl',
        'spliter': r'./processed_signal/HKU956/260_4s_step_2s_spliter5.pkl'},
]

targets = ['valence_label', 'arousal_label']

all_results = []

for e in data_dict:
    data_name = os.path.split(e['data'])[-1].replace('.pkl')
    data = e['data']
    spliter = e['spliter']
    for targ in targets:
        args = Params(target=targ, debug=False)
        args.data = data
        args.spliter = spliter
        args.save_path = args.save_path + '_' + data_name
        results = api(args=args)
        all_results.append([args.save_path, results])
        print('\n\n')


for res in all_results:
    print(res[0], res[1])
    # print(data.shape)
