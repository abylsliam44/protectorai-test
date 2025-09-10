#!/usr/bin/env python3
"""
Менеджер конфигурации для настройки параметров детектора
"""

import json
import os

class ConfigManager:
    """Класс для управления конфигурацией проекта"""
    
    def __init__(self, config_file="config.json"):
        self.config_file = config_file
        self.config = self.load_config()
    
    def load_config(self):
        """Загружает конфигурацию из файла"""
        if os.path.exists(self.config_file):
            with open(self.config_file, 'r') as f:
                return json.load(f)
        else:
            return self.get_default_config()
    
    def get_default_config(self):
        """Возвращает конфигурацию по умолчанию"""
        return {
            "detector": {
                "scale_factor": 1.1,
                "min_neighbors": 5,
                "min_size": [30, 30],
                "max_size": [300, 300]
            },
            "evaluation": {
                "iou_threshold": 0.5,
                "max_images": 200,
                "save_visualizations": True
            },
            "visualization": {
                "box_color": [0, 255, 0],
                "box_thickness": 2,
                "text_size": 0.7,
                "save_dpi": 150
            }
        }
    
    def get_detector_params(self):
        """Возвращает параметры детектора"""
        return self.config["detector"]
    
    def get_evaluation_params(self):
        """Возвращает параметры оценки"""
        return self.config["evaluation"]
    
    def get_visualization_params(self):
        """Возвращает параметры визуализации"""
        return self.config["visualization"]
    
    def update_config(self, section, key, value):
        """Обновляет значение в конфигурации"""
        if section in self.config and key in self.config[section]:
            self.config[section][key] = value
            self.save_config()
        else:
            raise ValueError(f"Ключ {key} не найден в секции {section}")
    
    def save_config(self):
        """Сохраняет конфигурацию в файл"""
        with open(self.config_file, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def reset_to_default(self):
        """Сбрасывает конфигурацию к значениям по умолчанию"""
        self.config = self.get_default_config()
        self.save_config()
