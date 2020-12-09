# -*- coding: utf-8 -*-
# Author: mzk
# Version: 0.1
# This module handles audio pre processing: cut the activity part of cough
# time: 2020.1.6

import numpy as np
import sys

# sys.path.append("./tools")
sys.path.append("D:/work/牧原数字/智能化部门/vad")
from tools.audioUtility import selectWindow
from tools.audioFeature import audioFrameEnergy, audioFrameRMS, audioFrameZeroCrossingRate
from tools.audioPreProcess import segmentVoiceByWord, segmentVoiceByZero, segmentVoiceByZero_start_end
from tools.audioFileInput import openAudioFile
from tools.utils import read_textgrids, generate_textgrid_file
from pydub import AudioSegment
import os
import time

if __name__ == "__main__":
    # samplingRate, data = openAudioFile("fugou_outside.wav", 0) # _2020_01_03_23_24_45_00079
    samplingRate, data = openAudioFile("20200110175035_20.wav", 0)
    data_count = samplingRate * 90

    # print("data length = ", data_count, "time length = ", (data_count / 16000), "s", data[:10])
    label_path = "D:/work/牧原数字/智能化部门/vad/vad/20200110175035_20.TextGrid"
    label = read_textgrids(label_path)
    i = 23
    # data = data[samplingRate * i * 10:samplingRate * 10 * (i + 1)]
    time_s = time.clock()
    print("time:", time_s, len(data))
    # numWord, timeInfo = segmentVoiceByZero(samplingRate, data[:], label, 20, 10)
    numWord, timeInfo = segmentVoiceByZero_start_end(samplingRate, data[:], end=1600000, label=label, frameLength=20, hopLength=10)
    # labels = ['-'] * numWord
    # generate_textgrid_file(timeInfo / samplingRate, labels, "20200110175035_20.wav")
    # print(timeInfo)
    print("end cost:", time.clock() - time_s)
