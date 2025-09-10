# API Документация

**GitHub:** https://github.com/abylsliam44/protectorai-test  
**Автор:** Abylay Slamzhanov

## Основные модули

### `src.complete_evaluation`
Основной модуль для оценки детекции лиц.

**Функции:**
- `load_images(data_dir)` - загрузка изображений
- `load_annotations(annos_file)` - загрузка аннотаций
- `detect_faces(image, config_manager)` - детекция лиц
- `evaluate_with_ground_truth()` - полная оценка

### `src.utils.config_manager`
Управление конфигурацией параметров.

**Класс ConfigManager:**
- `get_detector_params()` - параметры детектора
- `get_evaluation_params()` - параметры оценки
- `update_config(section, key, value)` - обновление конфигурации

### `src.visualization.visualize_results`
Создание визуализаций результатов.

**Функции:**
- `visualize_sample_results()` - визуализация примеров
- `create_metrics_visualization()` - графики метрик

## Примеры использования

### Базовое использование
```python
from src.complete_evaluation import detect_faces, load_image
from src.utils.config_manager import ConfigManager

# Загружаем изображение
image = load_image("data/images/example.jpg")

# Детектируем лица
faces = detect_faces(image)
print(f"Найдено {len(faces)} лиц")
```

### Работа с конфигурацией
```python
from src.utils.config_manager import ConfigManager

# Создаем менеджер конфигурации
config = ConfigManager()

# Обновляем параметры
config.update_config("detector", "scale_factor", 1.05)
config.update_config("detector", "min_neighbors", 3)

# Получаем параметры
params = config.get_detector_params()
```
