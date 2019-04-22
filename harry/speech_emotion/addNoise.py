import glob
import os
from scipy.io import wavfile
import pyaudio
import wave
import numpy as np
import tqdm
from tqdm import trange

WAV_PATH = "E:/Test/Session1-4/wav/"
OUT_File = "E:/Test/Session1-4/noises/"
p = pyaudio.PyAudio()
CHANNELS = 1
FORMAT = pyaudio.paInt16
RATE = 16000

DICT = {
    "doing_the_dishes.wav": 'n01',
    "dude_miaowing.wav": 'n02',
    "exercise_bike.wav": 'n03',
    "office_noise.wav": 'n04',
    "pink_noise.wav": 'n05',
    "running_tap.wav": 'n06',
    "white_noise.wav": 'n07',
}


def mix(f1, f2, out):
    with wave.open(f1, 'rb') as f:
        params1 = f.getparams()
        nchannels1, sampwidth1, framerate1, nframes1, comptype1, compname1 = params1[:6]
        # print(nchannels1, sampwidth1, framerate1, nframes1, comptype1, compname1)
        f1_str_data = f.readframes(nframes1)
        f1_wave_data = np.fromstring(f1_str_data, dtype=np.int16)
    with wave.open(f1, 'rb') as f:
        params2 = f.getparams()
        nchannels2, sampwidth2, framerate2, nframes2, comptype2, compname2 = params2[:6]
        # print(nchannels2, sampwidth2, framerate2, nframes2, comptype2, compname2)
        f2_str_data = f.readframes(nframes2)
        f2_wave_data = np.fromstring(f2_str_data, dtype=np.int16)
    if nframes1 < nframes2:
        length = nframes1
    else:
        length = nframes2

    new_wave_data = f1_wave_data[:length] + f2_wave_data[:length]
    new_wave = new_wave_data.tostring()
    record(new_wave, out)


def record(re_frames, WAVE_OUTPUT_FILENAME):
    # print("开始录音")
    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(re_frames)
    wf.close()
    # print("关闭录音")


if __name__ == '__main__':
    Wav = glob.glob(WAV_PATH + '/*.wav')
    Noise = glob.glob("D:/Program Files/PyCharm/SER/speech_emotion/noises/*.wav")
    # for noise in Noise:
    #     data=wavfile.read(noise)
    #     wavfile.write("E:/Test/noises/"+os.path.basename(noise),data[0],data[1]//2)

    for noise in Noise:
        numstr = DICT[os.path.basename(noise)]
        for i in trange(len(Wav)):
            wav = Wav[i]
            out = OUT_File + '/' + numstr + '/' + os.path.basename(wav)[:-4] + '-' + numstr + '.wav'
            mix(wav, noise, out)
