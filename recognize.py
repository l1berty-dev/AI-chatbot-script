import queue
import sounddevice as sd
import vosk
import json
import threading

q = queue.Queue()
model = vosk.Model('model-small')
device = sd.default.device = 2, 4
samplerate = int(sd.query_devices(device[0], 'input')['default_samplerate'])
triggers = {'аксель', 'дубина'}
triggers_search = {'найди', 'узнай', 'поищи', 'кто'}


def callback(indata, frames, time, status):
    q.put(bytes(indata))


def get_data():
    with sd.RawInputStream(samplerate=samplerate, blocksize=16000, device=device, dtype='int16',
                           channels=1, callback=callback):
        rec = vosk.KaldiRecognizer(model, samplerate)
        while True:
            data = q.get()
            if rec.AcceptWaveform(data):
                data = json.loads(rec.Result())['text']
                # print(data)
                if data == '':
                    continue
                return data


def get_data_time():
    with sd.RawInputStream(samplerate=samplerate, blocksize=16000, device=device, dtype='int16',
                           channels=1, callback=callback):
        rec = vosk.KaldiRecognizer(model, samplerate)
        while True:
            data = q.get()
            if rec.AcceptWaveform(data):
                data = json.loads(rec.Result())['text']
                # print(data)
                return data

        # else:
        #     print(rec.PartialResult())


# if __name__ == '__main__':
#     thread = threading.Thread(target=get_data_time)
#     thread.daemon = True
#     thread.start()
#     thread.join(5)
