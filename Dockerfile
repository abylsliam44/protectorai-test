# Face Detection Project
# Author: Abylay Slamzhanov
# GitHub: https://github.com/abylsliam44/protectorai-test
FROM python:3.9-slim

# Устанавливаем системные зависимости
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем requirements и устанавливаем зависимости
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем код
COPY . .

# Создаем папку для результатов
RUN mkdir -p results

# Открываем порт для Jupyter
EXPOSE 8888

# Запускаем Jupyter Notebook
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
