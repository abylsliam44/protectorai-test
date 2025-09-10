# Установка и настройка

## Требования
- Python 3.7+
- OpenCV 4.5+
- NumPy 1.21+
- Matplotlib 3.5+

## Установка

### Локальная установка
```bash
# Клонируйте репозиторий
git clone https://github.com/abylsliam44/protectorai-test.git
cd protectorai-test

# Создайте виртуальное окружение
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# или
venv\Scripts\activate     # Windows

# Установите зависимости
pip install -r requirements.txt
```

### Docker
```bash
docker build -t face-detection .
docker run -p 8888:8888 face-detection
```

## Запуск

### Основной скрипт
```bash
python main.py
```

### Отдельные модули
```bash
# Оценка
python src/complete_evaluation.py

# Визуализация
python src/visualization/visualize_results.py

# Тестирование конфигураций
python examples/test_configs.py

# Jupyter Notebook
jupyter notebook examples/face_detection.ipynb
```
