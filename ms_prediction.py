import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np

# 1. Загрузка данных
df = pd.read_excel('C:/Users/Гусь/Downloads/ForNetWork.xlsx')

# 2. Проверка на пропуски в данных
print("Проверка на пропущенные значения в данных:")
print(df.isnull().sum())

# 3. Обработка пропусков: заменим NaN на средние значения для каждого столбца
df.fillna(df.mean(), inplace=True)

# 4. Разделение данных на признаки и целевую переменную (ГПС)
X = df.drop(columns=['гпс'])  # Входные признаки
y = df['гпс']  # Целевая переменная

# Проверим, есть ли в целевой переменной значения, отличные от 0 и 1
print(f"Уникальные значения целевой переменной (ГПС): {y.unique()}")

# Преобразуем целевую переменную в 0 и 1, если необходимо
y = y.apply(lambda x: 1 if x == 1 else 0)

# 5. Предобработка данных (нормализация в диапазоне [0, 1])
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Проверим диапазон значений после нормализации
print(f"Минимум в нормализованных данных: {np.min(X_scaled)}, максимум: {np.max(X_scaled)}")

# Разделение данных на обучающую и тестовую выборку
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Преобразуем данные в тензоры PyTorch
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)


# 6. Создание более сложной модели нейронной сети с Dropout
class NeuralNet(nn.Module):
    def __init__(self, input_size):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)  # Увеличенное количество нейронов
        self.dropout1 = nn.Dropout(0.3)  # Dropout для предотвращения переобучения
        self.fc2 = nn.Linear(128, 64)  # Второй слой
        self.dropout2 = nn.Dropout(0.3)  # Dropout
        self.fc3 = nn.Linear(64, 32)  # Третий слой
        self.fc4 = nn.Linear(32, 1)  # Выходной слой
        self.relu = nn.ReLU()  # Функция активации
        self.sigmoid = nn.Sigmoid()  # Для классификации

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout1(x)  # Применяем Dropout
        x = self.relu(self.fc2(x))
        x = self.dropout2(x)  # Применяем Dropout
        x = self.sigmoid(self.fc3(x))
        x = self.sigmoid(self.fc4(x))
        return x


# Инициализация модели
input_size = X_train.shape[1]
model = NeuralNet(input_size)

# 7. Функция потерь и оптимизатор
criterion = nn.BCELoss()  # Бинарная кросс-энтропия
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 8. Обучение модели
num_epochs = 1500  # Увеличим количество эпох для более качественного обучения

for epoch in range(num_epochs):
    # Прямой проход
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)

    # Обратный проход и оптимизация
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 20 == 0:  # Изменение частоты вывода
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# 9. Оценка модели на тестовой выборке
with torch.no_grad():
    y_pred = model(X_test_tensor)
    y_pred_label = (y_pred >= 0.5).float()  # Преобразуем в бинарные метки
    accuracy = (y_pred_label == y_test_tensor).sum().item() / y_test_tensor.size(0)
    print(f'Accuracy: {accuracy * 100:.2f}%')


# Функция для предсказания на основе данных пациента
def predict_patient_data(model, scaler, patient_data):
    """
    Функция для предсказания ГПС на основе данных пациента.

    Аргументы:
    - model: обученная нейронная сеть (PyTorch).
    - scaler: объект MinMaxScaler для нормализации данных.
    - patient_data: список значений характеристик пациента в том же порядке, как в обучающей выборке.

    Возвращает:
    - Предсказание: 0 (нет ГПС) или 1 (ГПС возникнет).
    """

    # Преобразуем данные пациента в формат numpy и нормализуем
    patient_data_np = np.array(patient_data).reshape(1, -1)
    patient_data_scaled = scaler.transform(patient_data_np)

    # Преобразуем в тензор
    patient_tensor = torch.tensor(patient_data_scaled, dtype=torch.float32)

    # Прогнозирование
    model.eval()  # Переводим модель в режим предсказания
    with torch.no_grad():
        prediction = model(patient_tensor)
        prediction_label = (prediction >= 0.5).float()  # Прогноз метки 0 или 1

    # Возвращаем результат
    return int(prediction_label.item())


# Пример использования функции
patient_data_example = [
    1,       # Тип пациента
    0,       # Пол
    2,       # Препарат до
    35,      # Возраст
    5,       # Давность заболевания
    1,       # Активность до
    2,       # EDSS на
    0,       # Обострения до
    1,       # Обострения на
    0,       # Инъектор
    10,      # Время
    0,       # Руки
    1,       # Ноги
    0,       # Ягодицы
    0,       # Живот
    70,      # Вес
    170,     # Рост
    0,       # ИДыхП (инфекции дыхательных путей)
    0,       # Ангина, фарингит
    0,       # Ларингит, трахеит
    0,       # Пневмония
    0,       # ИМочП (инфекции мочеполовой системы)
    0,       # Уретрит
    0,       # Цистит
    0,       # Пиелонефрит
    0,       # Герпес
    0,       # Афты
    0,       # Изменение веса
    0,       # Голод
    0,       # Жажда
    0,       # Факт
    0,       # ОК
    0,       # Температура
    0,       # Тонзиллит
    0,       # Пиелонефрит (повторяющийся параметр, если есть)
    0,       # Другое
    0,       # Пыль
    0,       # Пыльца
    0,       # Цитрус
    0,       # Лекарства
    0,       # Признак 41
    0,       # Признак 42
    0,       # Признак 43
    0,       # Признак 44
    0,       # Признак 45
    0,       # Признак 46
    0,       # Признак 47
    0,       # Признак 48
    0,       # Признак 49
    0
]
patient = [
    1,       # Тип пациента
    1,       # Пол
    2,       # Препарат до
    43,      # Возраст
    228,       # Давность заболевания
    0,       # Активность до
    3,       # EDSS на
    4,       # Обострения до
    0,       # Обострения на
    0,       # Инъектор
    20,      # Время
    1,       # Руки
    0,       # Ноги
    0,       # Ягодицы
    0,       # Живот
    68,      # Вес
    172,     # Рост
    0,       # ИДыхП (инфекции дыхательных путей)
    0,       # Ангина, фарингит
    0,       # Ларингит, трахеит
    0,       # Пневмония
    0,       # ИМочП (инфекции мочеполовой системы)
    0,       # Уретрит
    0,       # Цистит
    0,       # Пиелонефрит
    2,       # Герпес
    0,       # Афты
    1,       # Изменение веса
    0,       # Голод
    0,       # Жажда
    2,       # Факт
    2,       # ОК
    3,       # Температура
    0,       # Тонзиллит
    1,       # Пиелонефрит (повторяющийся параметр, если есть)
    0,       # Другое
    0,       # Пыль
    0,       # Пыльца
    0,       # Цитрус
    1,       # Лекарства
    0,       # другое
    0,       # дерматит
    0,       #  ринит
    0,       # астма
    22,       # ИМТ
    12,       # Инсомния
    29,       # Тревожность
    5,       # Эйфория
    29,       # Астения
    41        # Депрессия
]
# Вызов функции для предсказания
predicted_gps = predict_patient_data(model, scaler, patient)
print(f'Предсказание для пациента: {predicted_gps} (1 - ГПС возникнет, 0 - ГПС не будет)')