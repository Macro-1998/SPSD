"""
CUDA_VISIBLE_DEVICES=1 python -m train --config /data/wjc/clean_mmt/configs_pub/ablation/cgsd/t1v1_lambd30_512to1024_sd0.json
CUDA_VISIBLE_DEVICES=1 python -m train --config /data/wjc/clean_mmt/configs_pub/ablation/cgsd/t02v02_lambd30_512to1024_sd0.json
CUDA_VISIBLE_DEVICES=1 python -m train --config /data/wjc/clean_mmt/configs_pub/ablation/cgsd/t04v04_lambd30_512to1024_sd0.json
CUDA_VISIBLE_DEVICES=1 python -m train --config /data/wjc/clean_mmt/configs_pub/ablation/cgsd/t06v06_lambd30_512to1024_sd0.json
CUDA_VISIBLE_DEVICES=1 python -m train --config /data/wjc/clean_mmt/configs_pub/ablation/cgsd/t08v08_lambd30_512to1024_sd0.json
"""

import glob
import os
import math
import argparse
import torch
import torch.multiprocessing as mp


def main():
    parser = argparse.ArgumentParser(description="PyTorch Template")
    parser.add_argument("--config_dir", type=str, help="config files dir")

    args = parser.parse_args()
    config_dir = args.config_dir
    #config_dir = 'configs_pub/ablation/cgsd_t10v10_lambda'
    config_files = sorted(glob.glob(config_dir + '/*.json'))
    
    n_gpu = torch.cuda.device_count()
    mp.spawn(worker, nprocs=n_gpu, args=(n_gpu, config_files,))  


def worker(gpu, n_gpu, config_files):
    n_config = math.ceil(len(config_files) / n_gpu)
    start = gpu * n_config
    end = start + n_config if gpu != n_gpu-1 else len(config_files)
    configs_this = config_files[start:end]
    print("gpu {}:".format(gpu))
    cmds = ["python -m train --config {}".format(config_file) for config_file in configs_this]
    print('\t' + '\n\t'.join(cmds))
    for _cmd in cmds:
        os.system(_cmd)
        #print(_cmd)


if __name__ == '__main__':
    main()

"""
#skip_files = ['configs_pub/ablation/cgsd_lambd/t04v02_lambd100_512to1024_sd0.json']
skip_files = []
for config_file in config_files:
    _exp_path = config_file.replace('configs_pub', 'exps').rstrip('.json')
    if os.path.exists(_exp_path):
        skip_files.append(config_file)
    if config_file in skip_files:
        print('Skip', config_file)
        continue
    _cmd = "python -m train --config {}".format(config_file)
    print(_cmd)
    os.system(_cmd)
"""