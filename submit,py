import pandas as pd
from sklearn.preprocessing import LabelEncoder
from timezonefinder import TimezoneFinder
import pytz
from tqdm.auto import tqdm
import numpy as np

test_events = pd.read_csv('S:\\DataSpellProjects\\rurube_dem\\train_events.csv')
train_targets = pd.read_csv('S:\\DataSpellProjects\\rurube_dem\\train_targets.csv')
video_info = pd.read_csv('S:\\DataSpellProjects\\rurube_dem\\video_info_v2.csv')

test = pd.merge(left=test_events, right=train_targets, left_on="viewer_uid", right_on="viewer_uid")

import pandas as pd
from sklearn.model_selection import train_test_split

# Предположим, у тебя есть полный набор данных train_events
# Полный набор данных
X_full = test.drop('age_class', axis=1)  # Замените 'target_column' на вашу целевую переменную
y_full = test['age_class']

# Выбираем 100k строк с тем же балансом классов
X_sampled, _, y_sampled, _ = train_test_split(
    X_full, y_full,
    train_size=100000,  # Берем 100k строк
    stratify=y_full,  # Стратификация по классам
    random_state=42
)

# Объединяем обратно для получения итогового датасета
train_events_sampled = X_sampled.copy()
train_events_sampled['age_class'] = y_sampled  # Добавляем обратно целевую переменную

test = train_events_sampled

test = test.merge(right=video_info, left_on="rutube_video_id", right_on="rutube_video_id")

test_le = test.drop(columns=["author_id", "title"])
columns = test.drop(
    ["event_timestamp", "rutube_video_id", "viewer_uid", "title", "author_id", "total_watchtime",
     "duration"], axis=1).columns

for column in columns:
    le = LabelEncoder()
    test_le[column] = le.fit_transform(test[column])
test_le.head()

### Достаем основные сессионные фичи
test_le['event_timestamp'] = pd.to_datetime(test_le['event_timestamp'], errors='coerce')

# Создаем экземпляр TimezoneFinder
tf = TimezoneFinder()

# Пример координат для основных городов
region_coordinates = {
    'Moscow': (55.7558, 37.6173),
    'Vologda Oblast': (59.2187, 39.8918),
    'St.-Petersburg': (59.9343, 30.3351),
    'Tatarstan Republic': (55.7963, 49.1064),
    'Sverdlovsk Oblast': (56.8389, 60.6057),
    'Rostov': (47.2357, 39.7015),
    'Krasnodar Krai': (45.0393, 38.9872),
    'Krasnodarskiy': (45.0393, 38.9872),
    'Kursk Oblast': (51.7304, 36.1926),
    'Kaluga Oblast': (54.5060, 36.2516),
    'Khanty-Mansia': (61.0025, 69.0182),
    'Udmurtiya Republic': (56.8526, 53.2048),
    'Samara Oblast': (53.1959, 50.1002),
    'Bashkortostan Republic': (54.7348, 55.9579),
    'Komi': (61.6684, 50.8357),
    'Penza': (53.1959, 45.0183),
    'Moscow Oblast': (55.7558, 37.6173),
    'Stavropol Kray': (45.0448, 41.969),
    'Tambov Oblast': (52.7213, 41.4523),
    'Irkutsk Oblast': (52.2869, 104.305),
    "Leningradskaya Oblast'": (59.8287, 30.3347),
    'Dagestan': (42.2804, 47.5156),
    'Chelyabinsk': (55.1644, 61.4368),
    'Tyumen Oblast': (57.1530, 65.5343),
    'Perm Krai': (58.0104, 56.2294),
    'Yamalo-Nenets': (65.5343, 72.5167),
    'Ulyanovsk': (54.3142, 48.4031),
    'Saratov Oblast': (51.5331, 46.0342),
    'Altay Kray': (53.3481, 83.7798),
    'Kuzbass': (55.3547, 86.0876),
    'Voronezh Oblast': (51.6720, 39.1843),
    'Orenburg Oblast': (51.7682, 55.0969),
    'Vladimir Oblast': (56.1291, 40.4073),
    'Vladimir': (56.1291, 40.4073),
    'Novosibirsk Oblast': (55.0084, 82.9357),
    'Amur Oblast': (50.2907, 127.5272),
    'Khabarovsk': (48.4827, 135.0838),
    'Volgograd Oblast': (48.7080, 44.5133),
    'Novgorod Oblast': (58.5215, 31.2755),
    'Sverdlovsk': (56.8389, 60.6057),
    'Smolensk Oblast': (54.7867, 32.0504),
    'Chuvashia': (55.4746, 47.1073),
    'Nizhny Novgorod Oblast': (56.3269, 44.0059),
    'Pskov Oblast': (57.8194, 28.3310),
    'Omsk Oblast': (54.9893, 73.3682),
    'Primorye': (43.1737, 132.0064),
    'Astrakhan Oblast': (46.3476, 48.0336),
    'Krasnoyarskiy': (56.0153, 92.8932),
    'Karelia': (61.7849, 34.3469),
    'Belgorod Oblast': (50.5977, 36.5850),
    'Krasnoyarsk Krai': (56.0153, 92.8932),
    'Yaroslavl Oblast': (57.6216, 39.8978),
    'Tver Oblast': (56.8585, 35.9176),
    'Kirov Oblast': (58.6035, 49.6668),
    'Kurgan Oblast': (55.4507, 65.3411),
    'Kaliningrad Oblast': (54.7104, 20.4522),
    'Kostroma Oblast': (57.7676, 40.9267),
    'Kamchatka': (53.0370, 158.6559),
    'Tomsk Oblast': (56.4846, 84.9483),
    'Bryansk Oblast': (53.2521, 34.3717),
    'Tula Oblast': (54.1931, 37.6177),
    'Ivanovo': (57.0004, 40.9739),
    'Ivanovo Oblast': (57.0004, 40.9739),
    'Lipetsk Oblast': (52.6100, 39.5946),
    'Ryazan Oblast': (54.6250, 39.7359),
    'North Ossetia–Alania': (42.7924, 44.6216),
    'Murmansk': (68.9585, 33.0827),
    'Kabardino-Balkariya Republic': (43.4947, 43.6159),
    'Arkhangelskaya': (64.5399, 40.5182),
    'Penza Oblast': (53.1959, 45.0183),
    'Altai': (52.5205, 85.1602),
    'Kaliningrad': (54.7104, 20.4522),
    'Zabaykalskiy (Transbaikal) Kray': (52.0340, 113.4990),
    'Oryol oblast': (52.9685, 36.0697),
    'Tula': (54.1931, 37.6177),
    'Jaroslavl': (57.6216, 39.8978),
    'Kemerovo Oblast': (55.3547, 86.0876),
    'Khakasiya Republic': (53.7224, 91.4437),
    'Chechnya': (43.3179, 45.6989),
    'Kalmykiya Republic': (46.3083, 44.2702),
    'Sakha': (62.0280, 129.7326),
    'Omsk': (54.9893, 73.3682),
    'Sakhalin Oblast': (47.5172, 142.7970),
    'Karachayevo-Cherkesiya Republic': (43.7365, 41.7368),
    'Buryatiya Republic': (51.8335, 107.5846),
    'Smolenskaya Oblast’': (54.7867, 32.0504),
    'Sebastopol City': (44.6166, 33.5254),
    'Mariy-El Republic': (56.6344, 47.8990),
    'Voronezj': (51.6720, 39.1843),
    'Kursk': (51.7304, 36.1926),
    'Adygeya Republic': (44.6096, 40.1008),
    'Perm': (58.0104, 56.2294),
    'Primorskiy (Maritime) Kray': (43.1737, 132.0064),
    'Saratovskaya Oblast': (51.5331, 46.0342),
    'Mordoviya Republic': (54.1808, 45.1869),
    'Crimea': (44.9521, 34.1024),
    'Ingushetiya Republic': (43.1688, 44.8131),
    'Chukotka': (66.0000, 169.4902),
    'North Ossetia': (42.7924, 44.6216),
    'Tambov': (52.7213, 41.4523),
    'Kaluga': (54.5060, 36.2516),
    'Jewish Autonomous Oblast': (48.5382, 132.7369),
    'Orel Oblast': (52.9685, 36.0697),
    'Tver’ Oblast': (56.8585, 35.9176),
    'Tyumen’ Oblast': (57.1530, 65.5343),
    'Stavropol’ Kray': (45.0448, 41.969),
    'Magadan Oblast': (59.5682, 150.8085),
    'Tyva Republic': (51.7191, 94.4378),
    'Transbaikal Territory': (52.0340, 113.4990),
    'Nenets': (67.6380, 44.1212),
    'Smolensk': (54.7867, 32.0504),
    'Stavropol Krai': (45.0448, 41.969),
    'Vologda': (59.2187, 39.8918),
    'Astrakhan': (46.3476, 48.0336),
    'Kirov': (58.6035, 49.6668),
    'Arkhangelsk Oblast': (64.5399, 40.5182)
}


# Функция для получения часового пояса по региону
def get_timezone(region):
    if region in region_coordinates:
        lat, lon = region_coordinates[region]
        timezone_str = tf.timezone_at(lat=lat, lng=lon)
        if timezone_str:
            return pytz.timezone(timezone_str)
    # Если регион не найден, возвращаем московское время как дефолтное
    return pytz.timezone('Europe/Moscow')


# Коррекция времени для каждого пользователя
def correct_time(row):
    region = row['region']
    local_tz = get_timezone(region)
    moscow_tz = pytz.timezone('Europe/Moscow')

    # Проверяем, имеет ли временная метка tzinfo, чтобы не локализовать ее повторно
    timestamp_moscow = row['event_timestamp']
    if timestamp_moscow.tzinfo is None:
        # Локализуем, если временная метка наивная (без tzinfo)
        timestamp_moscow = moscow_tz.localize(timestamp_moscow)

    return timestamp_moscow.astimezone(local_tz)


tqdm.pandas()
# Применение функции для корректировки времени
test_le['local_time'] = test_le.progress_apply(correct_time, axis=1)

test_le['local_time'] = test_le['local_time'].astype(str)

# Удаляем информацию о временной зоне (+HH:MM, -HH:MM или Z) в конце строки
test_le['local_time'] = test_le['local_time'].str.replace(
    r'(Z|[\+\-]\d{2}:?\d{2})$', '', regex=True
)
test_le['local_time'] = pd.to_datetime(test_le['local_time'], errors='coerce')

# Шаг 1: Сортировка данных по 'viewer_uid' и 'local_time'
test_le = test_le.sort_values(['viewer_uid', 'local_time']).reset_index(drop=True)
test_le.head()

# Шаг 2: Извлечение временных признаков
test_le['hour'] = test_le['local_time'].dt.hour
test_le['day_of_week'] = test_le['local_time'].dt.dayofweek

# Создаём столбец 'end_time' = 'local_time' + 'total_watchtime'
test_le['end_time'] = test_le['local_time'] + pd.to_timedelta(test_le['total_watchtime'], unit='s')

# Шаг 3: Вычисление разницы во времени между последовательными действиями
# Вычисление разницы во времени между окончанием предыдущего видео и началом текущего
test_le['prev_end_time'] = test_le.groupby('viewer_uid')['end_time'].shift(1)
# Рассчитываем разницу: текущий 'local_time' - предыдущий 'end_time'
test_le['time_diff'] = test_le['local_time'] - test_le['prev_end_time']

# Шаг 4: Определение новой сессии ('new_session') с учётом отрицательных 'time_diff'
session_threshold = pd.Timedelta('20 minutes')
test_le['new_session'] = (
        (test_le['time_diff'] > session_threshold) |
        (test_le['time_diff'] < pd.Timedelta(0))
).astype(int)

test_le = test_le.drop("time_diff", axis=1)

# Шаг 5: Присвоение уникального идентификатора для каждой сессии ('session_id')
test_le['session_id'] = test_le.groupby('viewer_uid')['new_session'].cumsum()

# Шаг 6: Группировка по 'viewer_uid' и 'session_id' для создания сессионных признаков
session_features = test_le.groupby(['viewer_uid', 'session_id']).agg(
    total_watchtime_per_session=('total_watchtime', 'sum'),  # Суммарное время просмотра в сессии
    video_count_per_session=('local_time', 'count'),  # Количество видео в сессии
    session_start=('local_time', 'min'),  # Время начала сессии
    session_end=('local_time', 'max')  # Время окончания сессии
).reset_index()

# Вычисление длительности сессии в секундах
session_features['session_duration'] = (
            session_features['session_end'] - session_features['session_start']).dt.total_seconds()

# Удаление временных столбцов
session_features = session_features.drop(['session_start', 'session_end'], axis=1)

# Шаг 7: Объединение по 'viewer_uid' и 'session_id'
test_le = pd.merge(test_le, session_features, on=['viewer_uid', 'session_id'], how='left')

# Шаг 8: Группировка по 'viewer_uid' для создания признаков активности
activity_features = test_le.groupby('viewer_uid').agg(
    most_active_hour=('hour', lambda x: x.mode().iloc[0] if not x.mode().empty else x.min()),
    # Час наибольшей активности
    most_active_day=('day_of_week', lambda x: x.mode().iloc[0] if not x.mode().empty else x.min()),
    # День недели наибольшей активности
    avg_session_duration=('session_duration', 'mean'),  # Средняя длительность сессии
    avg_video_count_per_session=('video_count_per_session', 'mean'),  # Среднее количество видео за сессию
    avg_watchtime_per_session=('total_watchtime_per_session', 'mean')  # Среднее время просмотра за сессию
).reset_index()

# Объединение activity_features с train_small_2 по 'viewer_uid'
test_le = pd.merge(test_le, activity_features, on='viewer_uid', how='left')

# Группировка по 'viewer_uid' для создания дополнительных признаков на основе 'total_watchtime'
additional_watchtime_features = test_le.groupby('viewer_uid').agg(
    total_watchtime_total=('total_watchtime', 'sum'),  # Общее время просмотра
    avg_watchtime_per_video=('total_watchtime', 'mean'),  # Среднее время просмотра одного видео
    max_watchtime_per_video=('total_watchtime', 'max'),  # Максимальное время просмотра одного видео
    min_watchtime_per_video=('total_watchtime', 'min'),  # Минимальное время просмотра одного видео
    std_watchtime_per_video=('total_watchtime', 'std')  # Стандартное отклонение времени просмотра
).reset_index()

# Обработка NaN значений в 'std_watchtime_per_video' (например, для пользователей с одним видео)
additional_watchtime_features['std_watchtime_per_video'] = additional_watchtime_features[
    'std_watchtime_per_video'].fillna(0)

test_le = pd.merge(test_le, additional_watchtime_features, on='viewer_uid', how='left')

test_done = test_le.drop(
    columns=["event_timestamp", "viewer_uid", "rutube_video_id", "session_duration",
             "video_count_per_session", "total_watchtime_per_session", "session_id", "new_session", "day_of_week",
             "local_time", "end_time", "prev_end_time", "age", "sex", "age_class"])

y_sex = test_le['sex']
y_age_class = test_le['age_class']

from joblib import load

y_sex_pred_test = []
y_age_pred_test = []

for i in tqdm(range(3)):
    xgb_sex_loaded = load(f'1xgb_sex_model_fold_{i}.pkl')  # Загружаем модель для пола
    y_sex_pred_test.append(xgb_sex_loaded.predict_proba(test_done))

for i in tqdm(range(3)):
    # Загрузка модели для возрастных категорий
    xgb_age_class_loaded = load(f'1xgb_age_model_fold_{i}.pkl')  # Загружаем модель для возрастных категорий
    y_age_pred_test.append(xgb_age_class_loaded.predict_proba(test_done))

y_sex_pred_test_avg = np.mean(y_sex_pred_test, axis = 0)
y_age_pred_test_avg = np.mean(y_age_pred_test, axis = 0)

y_sex_pred_test_classes = np.argmax(y_sex_pred_test_avg, axis=1)
y_age_pred_test_classes = np.argmax(y_age_pred_test_avg, axis=1)  # Берем класс с максимальной вероятностью
#%%
from sklearn.metrics import accuracy_score, f1_score
# Accuracy и F1 для пола
sex_accuracy_test = accuracy_score(y_sex, y_sex_pred_test_classes)
sex_f1_test = f1_score(y_sex, y_sex_pred_test_classes, average='weighted')

# Accuracy и F1 для возрастных категорий
age_accuracy_test = accuracy_score(y_age_class, y_age_pred_test_classes)
age_f1_test = f1_score(y_age_class, y_age_pred_test_classes, average='weighted')

final_score_test = 0.7 * age_f1_test + 0.3 * sex_accuracy_test

# Печать результатов
print(f'F1 для пола: {sex_f1_test}')
print(f'F1 для возрастных категорий: {age_f1_test}')
print(f'Accuracy для пола: {sex_accuracy_test}')
print(f'Итоговая метрика: {final_score_test}')
