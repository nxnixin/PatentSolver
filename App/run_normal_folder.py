#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @File  : run_normal_folder.py
# @Author: nixin
# @Date  : 2021/11/11
from App.bin import constants
from App.bin.InputHandler import InputHandler
from App.bin.PatentHandler import PatentHandler
from App.bin.CorpusProcessor import CorpusProcessor
import time

start_time = time.time()

input_folder = constants.DATA_INPUT + 'US_patents'
files_extension = "*." + 'txt'

iInput = InputHandler(input_folder, files_extension)
input_data = iInput.get_input()

pretreat_data = PatentHandler(input_data)
clean_patent_data = pretreat_data.pretreat_data()


process_data = CorpusProcessor(clean_patent_data,input_folder, files_extension)
processed_data = process_data.process_corpus()

print("Process is finished within %s seconds" % round(time.time() - start_time,2))