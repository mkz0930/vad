
import sys
import os
sys.path.append("D:/work/牧原数字")
from data.cough_detection.denoise.audio_denoising import perform_spectral_subtraction
from 智能化部门.vad.vad.project_vad import segmentVoiceByZero
import librosa
import time
from praatio import tgio
import numpy as np
import tensorflow as tf


LENGTH = 9600

def data_scale(data):
    """scaling data, prepare data for vad.

    Args:
        data: pcm data

    Returns:
        data : data.
    """
    data = np.float64(data / (2 ** (8 * 2)))
    # print("/2*** data : ", data[:10])
    boudnData = np.max([abs(np.max(data)), abs(np.min(data))])
    # print("boudnData:", boudnData)
    gain = 1 / boudnData
    # print("gain: ", gain)
    data = data * gain
    # print("gain control data: ", data[:10])
    return data


def vad_cnn_tset(wav_path):
    """vad+cnn system test.

    Args:
        data: pcm data

    Returns:
        cough_position : position of cough.
    """
    source_data, samplingRate = librosa.load(wav_path, sr=16000)
    # 1.denoise
    denosie_data = perform_spectral_subtraction(source_data, samplingRate)
    
    # 2.vad
    vad_data = data_scale(denosie_data)
    data = vad_data
    # numWord, timeInfo = segmentVoiceByZero_start_end(samplingRate, data, label=label)
    numWord, timeInfo = segmentVoiceByZero(samplingRate, data, verbose=1)
    print("vad piece:", numWord)
    
    # 3.classify model load
    # model_path = "./cnn_b_test0518_b.h5"
    # model = tf.keras.models.load_model(model_path)
    # model.summary()
    
    # 4.test
    vad_pieces = []
    cough_count = 0
    cough_position = []
    cough_p = []
    labels_result = []
    source_data_n = len(source_data)
    for i in range(numWord):
        labels_result.append(0)
        # start = int(timeInfo[i][0])
        # end = int(timeInfo[i][1])
        # n = end - start
        # cough_position.append([start, end])
        #
        # # 将pcm数据前后扩充到LENGTH大小，再进行预测
        # if n > LENGTH:
        #     pad_n = (n - LENGTH) // 2
        #     x = source_data[(start + pad_n):(start + LENGTH + pad_n)]
        # else:
        #     pad_n = (LENGTH - n) // 2
        #     start_p = max(start - pad_n, 0)
        #     end_p = min(start + LENGTH - pad_n, source_data_n)
        #     x = source_data[start_p:end_p]
        #     if (end_p - start_p) < LENGTH:
        #         x = np.pad(x, (0, LENGTH - end_p + start_p), "constant", constant_values=(0, 0))
        #
        # spec = librosa.feature.melspectrogram(x, sr=16000, n_mels=32, n_fft=512, hop_length=160)
        # spec_db = librosa.power_to_db(spec, ref=np.max)
        # spec_db = spec_db - np.mean(spec_db)
        # x = spec_db.T[:-1].reshape(60, 32, 1)
        # vad_pieces.append(x)
        
    
    # result = model.predict(np.array(vad_pieces))
    # result[result > 0.4] = 1
    # result = result.astype(np.int)
    #
    # labels_result = result.flatten()

    # print("result:", result, "\nlabels_result:", labels_result)
    
    print("cough count: ", cough_count)
    print("position: ", cough_p)
    timeInfo = timeInfo / samplingRate
    return cough_p, labels_result, timeInfo


def generate_textgrid_file(times, label, wav_filepath):
    """Generate a textgrid file according to the given onset and offset times as well as corresponding labels. The location and the filename of that textgrid file would be identical with wav file.

    Args:
        times: float32 ndarray, (n_clips, 2); Stores the onset and offset time of each clip. Unit: seconds
        labels: string list; The length of which should be equal to n_clips.
        wav_filepath: string; The path of the target WAV FILE.
    Returns:
        None
    """
    textgrid_filepath = f'{os.path.splitext(wav_filepath)[0]}_pred.TextGrid'
    print("textgrid_filepath: ", textgrid_filepath)
    
    tg0 = tgio.openTextgrid(f'{os.path.splitext(wav_filepath)[0]}.TextGrid')
    real_tier = tg0.tierDict[tg0.tierNameList[0]]
    # vad2 = tg0.tierDict[tg0.tierNameList[1]]
    # vad3 = tg0.tierDict[tg0.tierNameList[2]]
    # vad4 = tg0.tierDict[tg0.tierNameList[3]]
    
    
    tg = tgio.Textgrid()
    # real_tier = tgio.IntervalTier('vad-0', [], 0, pairedWav=wav_filepath)
    pred_tier = tgio.IntervalTier('vad-4', [], 0, pairedWav=wav_filepath)
    
    # for idx, time in enumerate(times):
    #     new_entry = tgio.Interval(time[0], time[1], str(0))
    #     real_tier.insertEntry(new_entry, warnFlag=False, collisionCode='replace')
    for idx, time in enumerate(times):
        new_entry = tgio.Interval(time[0], time[1], str(6))
        pred_tier.insertEntry(new_entry, warnFlag=False, collisionCode='replace')

    tg.addTier(real_tier)
    # tg.addTier(vad2)
    # tg.addTier(vad3)
    # tg.addTier(vad4)
    tg.addTier(pred_tier)
    # tg.addTier(cough_tier)
    tg.save(textgrid_filepath)
    
    
if __name__ == "__main__":
    # print("ok")
    t0 = time.time()
    wav_name = "dev04_2020_05_23_04_56_49.wav"
    data, samplingRate = librosa.load(wav_name, sr=16000)
    # segmentVoiceByZero(samplingRate, data, verbose=1)
    cough_p, labels_result, timeInfo = vad_cnn_tset(wav_name)
    generate_textgrid_file(timeInfo, labels_result, wav_name)

    print("cost time: ", (time.time() - t0) / 60, "min")