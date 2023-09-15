import os
import sys
import time

import gpustat
import subprocess

cmd = 'python train.py'


def gpu_info():
    GPU_ID = subprocess.getoutput(
        'nvidia-smi --query-gpu=memory.free --format=csv,nounits,noheader | nl -v 0 | sort -nrk 2 | cut -f 1| head -n 1 | xargs')
    # GPU_ID = '0'
    command = 'nvidia-smi -i {}| grep %'.format(GPU_ID)
    gpu_status = os.popen(command).read().split('|')
    gpu_memory = int(gpu_status[2].split('/')[0].split('M')[0].strip())
    gpu_power = int(gpu_status[1].split('   ')[-1].split('/')[0].split('W')[0].strip())
    return gpu_power, gpu_memory, GPU_ID


def narrow_setup(interval=1):
    i=0
    while True:
        gpu_power, gpu_memory, ID = gpu_info()
        # print('GPU_ID: {}'.format(ID))

        if gpu_memory < 8000 or gpu_power < 150:
            print('\n' + cmd)
            os.system(cmd)

        i = i%5
        symbol = 'monitoring: ' + '>' * i + ' ' * (10 - i - 1) + '|' + 'GPU ID:{}'.format(ID)
        gpu_power_str = 'gpu power:%d W |' % gpu_power
        gpu_memory_str = 'gpu memory:%d MiB |' % gpu_memory
        sys.stdout.write('\r' + gpu_memory_str + ' ' + gpu_power_str + ' ' + symbol)
        sys.stdout.flush()
        i += 1

        time.sleep(interval)


if __name__ == '__main__':
    os.system('gpustat')
    narrow_setup()