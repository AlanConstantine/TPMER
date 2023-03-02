from finetune import api
from config import Params
from tools import *
import os

data_dict = [
    {'data': r'./processed_signal/HKU956/1540_24s_step_2s.pkl',
        'spliter': r'./processed_signal/HKU956/1540_24s_step_2s_spliter5.pkl'},
    {'data': r'./processed_signal/HKU956/772_12s_step_2s.pkl',
        'spliter': r'./processed_signal/HKU956/772_12s_step_2s_spliter5.pkl'},
    {'data': r'./processed_signal/HKU956/516_8s_step_2s.pkl',
        'spliter': r'./processed_signal/HKU956/516_8s_step_2s_spliter5.pkl'},
    {'data': r'./processed_signal/HKU956/260_4s_step_2s.pkl',
        'spliter': r'./processed_signal/HKU956/260_4s_step_2s_spliter5.pkl'},
    {'data': r'./processed_signal/HKU956/772_12s_step_6s.pkl',
        'spliter': r'./processed_signal/HKU956/772_12s_step_6s_spliter5.pkl'},
]

targets = ['valence_label', 'arousal_label']

all_results = []

for e in data_dict:
    data = e['data']
    spliter = e['spliter']
    for targ in targets:
        args = Params(target=targ, debug=False, data=data, spliter=spliter, use_cuda=True, batch_size=128)
        results = api(args=args)
        all_results.append([args.save_path, results])
        print('\n\n\n')


for res in all_results:
    print(res[0], res[1])
    # print(data.shape)
