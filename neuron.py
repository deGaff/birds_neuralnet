import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import joblib

# Загрузка заранее обработанных записей птиц
# Процесс обработки приведен в файле для демонстрации результата работы
fft_train = np.load("Education/train_fft_new.npy")
names_train = np.load("Education/name_of_birds.npy")
fft_test = np.load("Education/test_fft_new.npy")
names_test = np.load("Education/test_names.npy")

# Создать классификатор c эмперически найденными параметрами
RFC_model = RandomForestClassifier(max_depth=5, n_estimators=198, criterion='entropy', max_features=3)
# Обучить классификатор на заранее приготовленных и обработанных записях
RFC_model.fit(fft_train, names_train.ravel())

# Получить предсказания классификатора, используя
RFC_predictions = RFC_model.predict(fft_test)

# Сохранить обученную нейронную сеть для последующего использования
joblib.dump(RFC_model, 'neuron.pkl')

# Результаты тестов
# Сначала идет матрица, по диагонали правильные прогнозы, если в строке не на основной диагонали не нули,
# значит была ошибка прогнозирования, далее приводится более подробная статистика, затем итоговая точность
# модели на данном наборе данных
# Пример: 78  0  0
#          4 73  4
#          0  1 80
# Значит была ошибка в прогнозах соловья (4 раза сказал утка, 4 - синица), в синице один раз сказал, что соловей
result = confusion_matrix(names_test, RFC_predictions)
print("Confusion Matrix:")
print(result)
result1 = classification_report(names_test, RFC_predictions)
print("Classification Report:",)
print(result1)
result2 = accuracy_score(names_test, RFC_predictions)
print("Accuracy:", result2)
