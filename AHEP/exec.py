#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/1/22 3:46 PM
# @Author  : Yugang.ji
# @Site    : 
# @File    : exec.py
# @Software: PyCharm
import os

seed = [1]
methods = ["HEP", "Batch_Fast_Typeless_IS", "Batch_Fast_Typeless_SNIS", "Batch_RS_Typeless_IS", "Batch_RS_Typeless_SNIS",
           "Batch_Fast_Type_IS", "Batch_Fast_Type_SNIS", "Batch_RS_Type_IS", "Batch_RS_Type_SNIS"]

samples = [4096]

for s in seed:
    for n in samples:
        for method in methods:
            strs = "python main_Aminer.py --model {} --alpha 0.4 --num_epochs 1 --seed {} --num_sample {}".format(method, s, n)
            # strs_test = 'python main_Aminer.py --model {} --alpha 0.4 --seed {} --num_epochs 1 --num_sample {} --opt test --model_name model.ckpt-180'.format(
            #     method, s, n)
            os.system(strs)
            # os.system(strs_test)
