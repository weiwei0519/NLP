# coding=UTF-8
# Python实现progress进度条的工具类

'''
@File: progress_bar
@Author: WeiWei
@Time: 2022/11/6
@Email: weiwei_519@outlook.com
@Software: PyCharm
'''

import sys
import time
from time import sleep
from tqdm import tqdm
from progress.bar import IncrementalBar


# 普通进度条
def progress_bar():
    for i in range(1, 101):
        print("\r", end="")
        print("Download progress: {}%: ".format(i), "▋" * (i // 2), end="")
        sys.stdout.flush()
        time.sleep(0.05)


progress_bar()

# 带时间进度条
scale = 50
print("执行开始，祈祷不报错".center(scale // 2, "-"))
start = time.perf_counter()
for i in range(scale + 1):
    a = "*" * i
    b = "." * (scale - i)
    c = (i / scale) * 100
    dur = time.perf_counter() - start
    print("\r{:^3.0f}%[{}->{}]{:.2f}s".format(c, a, b, dur), end="")
    time.sleep(0.1)
print("\n" + "执行结束，万幸".center(scale // 2, "-"))

# tpdm进度条，一个专门生成进度条的工具包，可以使用pip在终端进行下载，当然还能切换进度条风格
for i in tqdm(range(1, 500)):
    # 模拟你的任务
    sleep(0.01)
sleep(0.5)

# progress bar风格进度条
mylist = [1, 2, 3, 4, 5, 6, 7, 8]
bar = IncrementalBar('Countdown', max=len(mylist))
for item in mylist:
    bar.next()
    time.sleep(1)
    bar.finish()
