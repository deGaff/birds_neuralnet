import joblib
import wave
from scipy import signal
import numpy as np

# прорежает семплы
# thin_factor коэф. обрезания относительно среднего значения
def thinsamples(samples, thin_factor):
    thinned_samples = []  # создаем пустой список
    average = np.sum(np.absolute(samples)) / len(samples)  # абсолютное среднее значение
    for i in samples:
        if np.absolute(i) > average * thin_factor:
            thinned_samples.append(i)
    thinned_samples = np.asarray(thinned_samples) # превращает в массив нампая
    return thinned_samples

# определяет индекс по частоте, кол-ву семплов и фреймрейту
# если индекс получился дробным, то округляет его
def indOfFreq(freq, lenght, framerate):
    return round((freq/framerate) * lenght)

types = {
    1: np.int8,
    2: np.int16,
    4: np.int32
}
RFC_model = joblib.load("neuron.pkl")

for i in range(1,7):

    namefile = f"Demonstration\({i}).wav"
    w = wave.open(namefile, 'r')
    (nchannels, sampwidth, framerate, nframes, comptype, compname) = w.getparams() #получение параметров .wav файла
    frames = w.readframes(nframes) # считывание кадров файла
    samples = np.frombuffer(frames, dtype=types[sampwidth]) # получение семплов файла
    fft_len = 1024 # количество семплов в одном преобразовании фурье 
    thinfactor = 1 # коэффициент, на который умножается среднее значение семплов на всём протяжении при использовании алгоритма отсечения
    thinned_samples = thinsamples(samples, thinfactor) # вызов функции отсечения 
    fft = np.zeros(fft_len) # создание массива выбранной длины, заполненного нулями
    win = signal.gaussian(fft_len, std=15) # получение щначений окна
    for i in range(len(thinned_samples) // fft_len): # цикл по количеству промежутков выбранной длины в массиве семлов (после использования функции отсечения)
        temp = thinned_samples[i * fft_len:(i + 1) * fft_len] # получение промежутка выбранной длины
        fft += np.absolute(np.fft.fft(temp * win)) # сумма всех оконных преобразований фурье
    fft = fft / (len(thinned_samples) // fft_len) # получение среднего значения (деление полученной суммы на кол-во)
    fft = fft / fft.max() # нормировка в (0,1)
    a = indOfFreq(100, 1024, framerate) # получение индекса верхней границы частоты
    b = indOfFreq(6000, 1024, framerate) # получение индекса нижней границы частоты
    fft = fft[a:b] # получение нужного нам интервала 

    RFC_predictions = RFC_model.predict(fft.reshape(1, -1)) # получение прогноза нейронной сети 

    print(RFC_predictions) # вывод прогноза нейронной сети


# 1 - Nightingale
# 2 - Tit
# 3 - Duck
# 4 - Duck
# 5 - Nightingale
# 6 - Tit
