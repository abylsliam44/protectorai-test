# Face Detection Project

Проект для детекции лиц с использованием WIDER FACE датасета и метода Haar Cascade (OpenCV).

## Описание

Этот проект реализует систему детекции лиц с использованием классического метода Haar Cascade из OpenCV. Проект включает полную оценку качества детекции с использованием ground truth данных, вычисление метрик Precision, Recall, F1-score и визуализацию результатов.

## Структура проекта

```
face-detection/
├── face_detection.ipynb          # Основной Jupyter Notebook
├── complete_evaluation.py        # Полная оценка с Ground Truth
├── visualize_results.py          # Визуализация результатов
├── config_manager.py             # Менеджер конфигурации
├── test_configs.py               # Тестирование конфигураций
├── config.json                   # Файл конфигурации
├── data/
│   ├── images/                   # Изображения WIDER FACE (500 JPG)
│   └── annos.json                # Ground Truth аннотации
├── results/                      # Результаты и визуализации
├── requirements.txt              # Python зависимости
├── Dockerfile                    # Docker контейнер
└── README.md                     # Этот файл
```

## Установка и запуск

### Локальная установка

1. **Клонируйте репозиторий:**
```bash
git clone <repository-url>
cd face-detection
```

2. **Создайте виртуальное окружение:**
```bash
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# или
venv\Scripts\activate     # Windows
```

3. **Установите зависимости:**
```bash
pip install -r requirements.txt
```

4. **Запустите полную оценку:**
```bash
python complete_evaluation.py
```

5. **Создайте визуализации:**
```bash
python visualize_results.py
```

6. **Откройте Jupyter Notebook:**
```bash
jupyter notebook face_detection.ipynb
```

7. **Тестируйте различные конфигурации:**
```bash
python test_configs.py
```

### Docker

1. **Соберите образ:**
```bash
docker build -t face-detection .
```

2. **Запустите контейнер:**
```bash
docker run -p 8888:8888 face-detection
```

3. **Откройте браузер:**
```
http://localhost:8888
```

## Использование

### Основные скрипты

- **`complete_evaluation.py`** - Полная оценка детекции с ground truth данными
- **`visualize_results.py`** - Создание визуализаций результатов
- **`config_manager.py`** - Управление конфигурацией параметров детектора
- **`test_configs.py`** - Тестирование различных конфигураций
- **`face_detection.ipynb`** - Интерактивный Jupyter Notebook

### Пример использования

```python
import cv2
import numpy as np
from complete_evaluation import detect_faces, load_image

# Загружаем изображение
image = load_image("data/images/example.jpg")

# Детектируем лица
faces = detect_faces(image)

# Выводим результат
print(f"Найдено {len(faces)} лиц")
```

## Результаты

### Производительность (200 изображений)
- **Скорость**: 39.6 FPS
- **Время на изображение**: 0.057 сек
- **Обработано изображений**: 200
- **Общее время**: 11.49 сек

### Качество детекции (IoU ≥ 0.5)
- **Precision**: 0.640 (64.0%)
- **Recall**: 0.140 (14.0%)
- **F1-score**: 0.230 (23.0%)

### Детализация
- **True Positives**: 291
- **False Positives**: 164
- **False Negatives**: 1,789
- **Ground Truth лиц**: 2,080
- **Найдено лиц**: 455

## Детальный анализ результатов

### Производительность по категориям лиц
| GT лиц | Изображений | Precision | Recall | F1-score | FPS |
|--------|-------------|-----------|--------|----------|-----|
| 0      | 7           | 0.000     | 0.000  | 0.000    | 19.5|
| 1      | 67          | 0.382     | 0.582  | 0.438    | 18.8|
| 2      | 29          | 0.462     | 0.379  | 0.390    | 20.1|
| 3      | 11          | 0.579     | 0.424  | 0.474    | 21.0|
| 4      | 12          | 0.513     | 0.396  | 0.405    | 17.1|
| 5+     | 74          | 0.350     | 0.180  | 0.240    | 19.8|

### Распределение производительности
- **Высокая производительность** (F1 > 0.5): 66 изображений (33.0%)
- **Средняя производительность** (0.2 ≤ F1 ≤ 0.5): 33 изображения (16.5%)
- **Низкая производительность** (F1 < 0.2): 101 изображение (50.5%)

### Ключевые выводы
- **Лучшие результаты**: 1-3 лица на изображении (F1: 0.39-0.47)
- **Худшие результаты**: 5+ лиц на изображении (F1: 0.24)
- **Основная проблема**: С увеличением количества лиц качество падает
- **Пропущено лиц**: 94.3% на сложных изображениях

### Сильные стороны
- Высокая скорость работы (39.6 FPS)
- Хорошая точность на простых изображениях (64.0% Precision)
- Стабильная работа на изображениях с 1-3 лицами

### Проблемы
- Низкий Recall (14.0%) - пропускает много лиц
- Плохая работа на сложных изображениях (5+ лиц)
- Зависимость от количества лиц на изображении

### Рекомендации для улучшения
- **Для простых изображений** (1-3 лица): настройка параметров Haar Cascade
- **Для сложных изображений** (5+ лиц): использование современных deep learning методов (RetinaFace, SCRFD)
- **Общие улучшения**: 
  - Предобработка изображений
  - Multi-scale детекция
  - Ensemble методы

## Технические детали

### Алгоритм
- **Метод**: Haar Cascade (OpenCV)
- **Классификатор**: haarcascade_frontalface_default.xml
- **Параметры**: scale_factor=1.1, min_neighbors=5, min_size=(30,30)

### Метрики
- **IoU**: Intersection over Union для сопоставления детекций
- **Precision**: TP / (TP + FP)
- **Recall**: TP / (TP + FN)
- **F1-score**: 2 * (Precision * Recall) / (Precision + Recall)

### Датасет
- **WIDER FACE**: 500 изображений с аннотациями
- **Формат аннотаций**: JSON с bounding boxes
- **Фильтрация**: Только валидные аннотации (invalid=0)

## Требования

- Python 3.7+
- OpenCV 4.5+
- NumPy 1.21+
- Matplotlib 3.5+
- Jupyter Notebook
- Pillow 10.0+
- scikit-learn 1.0+

## Автор

Abylay Slamzhanov (@abylsliam44 Telegram)

## Поддержка

Для вопросов и предложений создайте issue в репозитории.