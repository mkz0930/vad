# -*- coding: utf-8 -*-
# Version: 0.2
# 该模块主要做vad切割，核心函数是segmentVoiceByZero.

import librosa
import numpy as np
import matplotlib.pyplot as plt
import time

import sys
import os

matrix_size = 500

def smooth_filter(data):
    '''
    filter audio data by smooth
    Parameters
    ----------
    data: numpy array of float32
        audio PCM data

    Returns
    ----------
    smooth_data: numpy array of float32
        audio PCM data
    '''

    return np.append((data[:-1] + data[1:]) / 2, data[-1])


def audioFrameZeroCrossingRate_m(segment, threshold=0.01):
    x = segment
    x[x >  threshold] = 1
    x[x < - threshold] = -1
    x = x.astype(np.int)
    y = x[:-1] * x[1:]
    z = y[y == -1]
    return - np.sum(z)/ len(segment)


def audioFrameZeroCrossingRate_matrix(data, windowSize=320, length=matrix_size, threshold=0.01):
    '''
    将每帧数据叠加成矩阵，用numpy矩阵计算每帧的过零率
    Parameters
    ----------
    data: 加过窗的每帧数据，nFrame*windowSize
    length:默认100， 即每次计算100*320的矩阵，一秒钟大小的数据

    Returns
    ----------
    rms_lst:每帧的过零率
    '''
    n = len(data) // length
    zs_lst = []
    for i in range(n):
        data_matrix = data[i*length:(i+1)*length]
        data_matrix[data_matrix > threshold] = 1
        data_matrix[data_matrix < - threshold] = -1
        data_matrix = data_matrix.astype(np.int)
        data_matrix = data_matrix[:, :-1]*data_matrix[:, 1:]
        data_matrix[data_matrix > -1] = 0
        data_matrix = -np.sum(data_matrix, axis=1) / windowSize
        zs_lst.append(data_matrix)
    zs_lst = np.array(zs_lst).flatten()

    if len(data) - n*length > 0:
        data_matrix = data[n*length:]
        data_matrix[data_matrix > threshold] = 1
        data_matrix[data_matrix < - threshold] = -1
        data_matrix = data_matrix.astype(np.int)
        data_matrix = data_matrix[:, :-1]*data_matrix[:, 1:]
        data_matrix[data_matrix > -1] = 0
        data_matrix = -np.sum(data_matrix, axis=1) / windowSize
        zs_lst = np.append(zs_lst, data_matrix)
    zs_lst = smooth_filter(zs_lst)

    return zs_lst


def audioFrameRMS_m(segment):
    segment = segment*segment
    return np.sqrt(np.sum(segment)/len(segment))


def audioFrameRMS_matrix(data, windowSize=320, length=matrix_size):
    '''
    将每帧数据叠加成矩阵，用numpy矩阵计算每帧的能量特征
    Parameters
    ----------
    data: 加过窗的每帧数据，nFrame*windowSize
    length:默认100， 即每次计算100*320的矩阵，一秒钟大小的数据

    Returns
    ----------
    rms_lst:每帧的能量值
    '''
    n = len(data) // length
    rms_lst = []
    for i in range(n):
        data_matrix = data[i*length:(i+1)*length]
        data_matrix = data_matrix*data_matrix
        data_sum = np.sum(data_matrix, axis=1) / windowSize
        data_sqart = np.sqrt(data_sum)
        rms_lst.append(data_sqart)
    rms_lst = np.array(rms_lst).flatten()

    if len(data) - n*length > 0:
        data_matrix = data[n*length:]
        data_matrix = data_matrix*data_matrix
        data_sum = np.sum(data_matrix, axis=1) / windowSize
        data_sqart = np.sqrt(data_sum)
        rms_lst = np.append(rms_lst, data_sqart)

    rms_lst = smooth_filter(rms_lst)
    return rms_lst


def selectWindow(windowSize):
    ''' window selection '''
    # Window 3 - Hann Window
    windowHann = 0.5 * (1 - np.cos(2 * np.pi / (windowSize - 1) * np.arange(windowSize)))
    windowName = 'Hanning'
    return windowHann, windowName


def data_to_frame(data, windowHann, hopSize=160, windowSize=320):
    data_matrix = []
    nFrame = data.size // hopSize - 1  # 舍弃最后不满一帧的数据
    for i in range(nFrame):
        frameStart = i * hopSize
        frame_x = data[frameStart:(frameStart + windowSize)]
        # Multiply frame data with window
        frame_x = frame_x * windowHann
        data_matrix.append(frame_x)

    return np.array(data_matrix)


def plot_piece(data, runningFrameFeature1, runningFrameFeature2, segmentPosition,
               start=0, label=None, hopSize=160, samplingRate=16000):
    '''
    将切割出的片段画出来，分别在原始数据，能量，过零率上显示。.

    Parameters
    ----------
    data: numpy array of float32
        audio PCM data
    runningFrameFeature1: 能量
    runningFrameFeature2: 过零率
    segStart：每个片段的开头位置
    segEnd：每个片段的结尾位置
    nFrame：总帧数

    Returns
    ----------
    '''
    nFrame = data.size // hopSize - 1 # 舍弃最后不满一帧的数据
    # Plot audio waveform
    segStart = np.array(segmentPosition[:,0]) / samplingRate
    segEnd = np.array(segmentPosition[:,1]) / samplingRate

    fig1 = plt.figure(1, figsize=(18, 9))
    time = np.arange(0, data.size) * 1.0 / samplingRate
    time_x = np.arange(nFrame) * hopSize * 1.0 / samplingRate

    numWord = len(segStart)
    plt.subplot(311)
    plt.plot(time, data, 'b', label='Waveform')
    plt.legend()
    label_max = np.max(data)
    for i in range(numWord):
        plt.plot([segStart[i], segStart[i]], [-label_max, label_max], 'r')
    for i in range(numWord):
        plt.plot([segEnd[i], segEnd[i]], [-label_max*0.5, label_max*0.5], 'r')
    # Plot energy in RMS
    plt.subplot(312)
    plt.plot(time_x, runningFrameFeature1, 'g', label='RMS')
    plt.legend()
    label_max = np.max(runningFrameFeature1)
    for i in range(numWord):
        plt.plot([segStart[i], segStart[i]], [0, label_max], 'r')
    for i in range(numWord):
        plt.plot([segEnd[i], segEnd[i]], [0, label_max*0.5], 'r')
    energy_mean = np.mean(runningFrameFeature1) * 0.6
    plt.plot([0, data.size / samplingRate], [energy_mean, energy_mean], "y")
    # Plot Zero-crossing rate
    plt.subplot(313)
    plt.plot(time_x, runningFrameFeature2, 'y', label='Zero-Crossing')
    plt.xlabel('Time (sec)')
    plt.legend()
    label_max = np.max(runningFrameFeature2)
    # plt.plot([0, nFrame*hopLength/1000], [ambientZeroCrossingRateLevel, ambientZeroCrossingRateLevel], 'r')
    for i in range(numWord):
        plt.plot([segStart[i], segStart[i]], [0, label_max], 'r')
    for i in range(numWord):
        plt.plot([segEnd[i], segEnd[i]], [0, label_max * 0.5], 'r')
    energy_mean = np.mean(runningFrameFeature2) * 2
    plt.plot([0, data.size / samplingRate], [energy_mean, energy_mean], "b")
    plt.plot([0, data.size / samplingRate], [energy_mean * 0.1, energy_mean * 0.1], "g")
    # label
    if label is not None:
        n = len(label)
        for i in range(n):
            if label[i].start - start / samplingRate > segEnd[numWord - 1]:
                break
            if label[i].start >= start / samplingRate:
                plt.plot([label[i].start - start / samplingRate, label[i].start - start / samplingRate], [0, 0.15], 'g')
        for i in range(n):
            if label[i].start - start / samplingRate > segEnd[numWord - 1]:
                break
            if label[i].start >= start / samplingRate:
                plt.plot([label[i].end - start / samplingRate, label[i].end - start / samplingRate], [0, 0.1], 'g')

    # fig1.canvas.manager.window.move(-1900, 850)  # 调整窗口在屏幕上弹出的位置
    plt.show()
    # plt.close(fig1)


def running_feature(data, nFrame, window, hopSize=160, windowSize=320):
    '''
       计算每帧的特征。

       Parameters
       ----------
       data: numpy array of float32
           audio PCM data

       window：汉明窗
       segEnd：每个片段的结尾位置
       nFrame：总帧数

       Returns
       ----------
       runningFrameFeature2: 过零率
       max_zs_peak_mean：过零率波峰的平均值
       '''
    # mean value of data
    mean_value = np.mean(np.abs(data))
    runningFrameFeature1 = []  # Short-time RMS
    runningFrameFeature2 = []  # Short-time zero-crossing rate

    for i in range(nFrame):
        frameStart = i * hopSize
        frame_x = data[frameStart:(frameStart + windowSize)]
        # Multiply frame data with window
        frame_x = frame_x * window
        # Compute frame feature 1 energy
        frameRMS = audioFrameRMS_m(frame_x)
        runningFrameFeature1.append(frameRMS)
        # runningFrameFeature1[i] = (runningFrameFeature1[i - 1] + runningFrameFeature1[i]) / 2
        # Compute frame feature 2 zero-crossing rate
        frameZeroCrossingRate = audioFrameZeroCrossingRate_m(frame_x, mean_value)

        runningFrameFeature2.append(frameZeroCrossingRate)
        # if i > 2:
        #     runningFrameFeature1[i] = (runningFrameFeature1[i - 2] + runningFrameFeature1[i - 1] + runningFrameFeature1[
        #         i]) / 3

    # 平滑特征值
    runningFrameFeature1 = smooth_filter(np.array(runningFrameFeature1))
    runningFrameFeature2 = smooth_filter(np.array(runningFrameFeature2))
    return runningFrameFeature1, runningFrameFeature2


def double_gate_zs(runningFrameFeature1, runningFrameFeature2, Zs, ZL, ML, thresholdLength, min_frame=0.05,
                   top_limit=0.8, hopSize=160, sr=16000):
    '''
          使用双门限法对特征进行切割。

          Parameters
          ----------
          runningFrameFeature2: 每帧过零率
          runningFrameFeature1: 每帧能量值
          Zs：过零率上限
          ZL：过零率下限
          ML：能量阈值
          thresholdLength：片段长度限制
          min_frame：过零率扩展后，舍弃小于0.05秒的片段
          top_limit：限制开头拓展的位置，不超过最大长度的0.8倍

          Returns
          ----------
          segStart: 每个片段的开头
          segEnd: 每个片段的结尾
          '''
    min_frame = min_frame * (sr / hopSize) # 每秒有100帧
    segStart = []
    segEnd = []
    # Step 1: Zs leveling
    isLookForStart = True
    for i in range(1, len(runningFrameFeature2)):
        if isLookForStart:
            if (runningFrameFeature2[i] >= Zs) & (runningFrameFeature2[i - 1] < Zs):
                # 片段融合，下一个开头到上一个开头小于限制长度的一半，则舍弃上一个结尾，重新找
                if len(segStart) > 0 and i - segStart[-1] < thresholdLength * 0.5:
                    segEnd = segEnd[:-1]
                    isLookForStart = False
                else:
                    segStart.append(i)
                    isLookForStart = False
        else:
            if i - segStart[-1] <= thresholdLength:
                if (runningFrameFeature2[i] < Zs):
                    segEnd.append(i)
                    isLookForStart = True
                elif i - segStart[-1] == thresholdLength:
                    segEnd.append(i)
                    isLookForStart = True
    if isLookForStart == False:
        segEnd.append(i)

    # Step 2: ZL leveling
    # Adjust end boundary
    numWord = len(segStart)
    for i in range(numWord):
        index = segEnd[i]
        if i == (numWord - 1):
            search = len(runningFrameFeature2)
        else:
            search = segStart[i + 1]
        while index < search:
            if (runningFrameFeature2[int(index)] < ZL):
                segEnd[i] = index
                break
            elif index - segStart[i] == thresholdLength:
                segEnd[i] = index
                break
            else:
                index += 1
    # Adjust start boundary
    for i in range(numWord):
        index = segStart[i]
        if i == 0:
            search = 0
        else:
            search = segEnd[i - 1]
        while index > search:
            if (runningFrameFeature2[int(index)]  < ZL):
                segStart[i] = index
                break
            elif segEnd[i] - index >= thresholdLength*top_limit:
                segStart[i] = index
                break
            else:
                index -= 1
    
    # 舍弃只有几帧的小片段
    segStart = np.array(segStart)
    segEnd = np.array(segEnd)
    segLengthMask = (segEnd - segStart) > min_frame
    segStart = segStart[segLengthMask]
    segEnd = segEnd[segLengthMask]
    
    # Step 3: ML leveling
    # Adjust end boundary
    numWord = len(segStart)
    for i in range(numWord):
        index = segEnd[i]
        if i == (numWord - 1):
            search = len(runningFrameFeature2)
        else:
            search = segStart[i+1]
        while index < search:
            if runningFrameFeature1[int(index)] < ML:
                segEnd[i] = index
                break
            elif index - segStart[i] == thresholdLength:
                segEnd[i] = index
                break
            else:
                index += 1
    # Adjust start boundary
    for i in range(numWord):
        index = segStart[i]
        if i == 0:
            search = 0
        else:
            search = segEnd[i - 1]
        while index > search:
            if (runningFrameFeature1[int(index)] < ML):
                segStart[i] = index
                break
            elif segEnd[i] - index >= thresholdLength*top_limit:
                segStart[i] = index
                break
            else:
                index -= 1

    return segStart, segEnd


def piece_choice(segStart, segEnd, thresholdLength_min, hopSize=160):
    '''
          对片段进行筛选，保证在一定长度内。

          Parameters
          ----------
          segStart：每个片段的开头位置
          segEnd：每个片段的结尾位置
          thresholdLength_min；片段长度限制

          Returns
          ----------
          segmentPosition: 每个片段的原始位置

          '''
    segStart = np.array(segStart)
    segEnd = np.array(segEnd)
    assert segStart.shape[0] == segEnd.shape[0]
    segLengthMask = (segEnd - segStart) > thresholdLength_min

    segStartMerge = segStart[segLengthMask]
    segEndMerge = segEnd[segLengthMask]

    numWord = len(segStartMerge)
    segmentPosition = np.vstack([segStartMerge, segEndMerge]).T * hopSize
    return numWord, segmentPosition


def segmentVoiceByZero(samplingRate, data, frameLength=20, hopLength=10, label=None, verbose=False):
    '''
    Segment audio data by zero crossing rate,data from start to end.
    Ref
    https://blog.csdn.net/rocketeerLi/article/details/83307435

    Parameters
    ----------
    samplingRate: float32
    data: numpy array of float32
        audio PCM data
    frameLength: 每帧长度
    hopLength: 滑窗大小
    start：可以选择从data的那个位置开始计算
    end：data结束的位置
    label：text文件中的标签值
    Returns
    ----------
    numWord: int
        number of words in data
    segmentPosition: numpy array of size (numWord, 2)
        position information for each segmented word
    '''
    # 0.Frame configuration
    frameSize = int(frameLength / 1000 * samplingRate)
    hopSize = int(hopLength / 1000 * samplingRate)

    # 1.adjustable parameter
    thresholdEnergyGain = 0.6  # threshold value of Energy
    thresholdZSGain = 2.0  # 过零率最大值的倍数，作为过零率阈值
    thresholdZSGainLow = 0.1 # 过零率阈值的倍数，作为延展阈值
    thresholdLength = samplingRate * 0.4 // hopSize  # limit the max length of vad pieces
    thresholdLength_min = samplingRate * 0.12 // hopSize  # limit the min length of vad pieces

    # 2.数据预处理
    data = smooth_filter(data)
    mean_value = np.mean(np.abs(data))
    # 加汉明窗
    windowSize = frameSize
    window, windowName = selectWindow(windowSize)
    # Total number of frames
    data_matrix = data_to_frame(data, window)

    # 3.特征计算
    runningFrameFeature1 = audioFrameRMS_matrix(data_matrix)
    runningFrameFeature2 = audioFrameZeroCrossingRate_matrix(data_matrix, threshold=mean_value)
    ambientZeroCrossingRateLevel = np.mean(runningFrameFeature2)
    Zs = ambientZeroCrossingRateLevel * thresholdZSGain  # 过零率阈值上限
    ZL = Zs * thresholdZSGainLow  # 过零率阈值下限
    ambientRMSLevel = np.mean(runningFrameFeature1)
    ML = ambientRMSLevel * thresholdEnergyGain # 能量阈值下限

    # 4.计算片段起始
    segStart, segEnd = double_gate_zs(runningFrameFeature1, runningFrameFeature2, Zs, ZL, ML, thresholdLength)

    # 5.筛选片段
    numWord, segmentPosition = piece_choice(segStart, segEnd, thresholdLength_min, hopSize=hopSize)
    if verbose:
        print('---------------------------------------------------------')
        print('Voice Word Segmentation')
        print('|\tnumWord\t\t=\t{0}'.format(numWord))
        for i in range(numWord):
            print('|\t#{0}\t{1}'.format(i, segmentPosition[i] / samplingRate),
                  "\t len: ", (segmentPosition[i][1] - segmentPosition[i][0]) / samplingRate, "s")
        print('---------------------------------------------------------')

        # 6.画图
        plot_piece(data, runningFrameFeature1, runningFrameFeature2, segmentPosition,
                   label=label, hopSize=hopSize, samplingRate=samplingRate)

    return numWord, segmentPosition


if __name__ == "__main__":
    print("ok")
    t0 = time.time()
    path = "C:/Users/44379/Desktop/b041/0002572461213034484205d7ff31_01_000a_2020_08_03_07_28_09_00048.wav"
    data, samplingRate = librosa.load(path, sr=16000, res_type="kaiser_fast")
    print("test start!  read wav cost time:", time.time()-t0)
    t1 = time.time()
    numWord, segmentPosition = segmentVoiceByZero(samplingRate, data, verbose=1)
    print("end seg time:", time.time()-t1)
    
    
