#!/usr/bin/env python3
"""
Тестирование различных конфигураций детектора
"""

from config_manager import ConfigManager
from complete_evaluation import load_images, load_annotations, evaluate_with_ground_truth
import time

def test_configuration(config_name, detector_params):
    """Тестирует конкретную конфигурацию"""
    print(f"\nТестируем конфигурацию: {config_name}")
    print("-" * 40)
    
    # Создаем конфигурацию
    config_manager = ConfigManager()
    for key, value in detector_params.items():
        config_manager.update_config("detector", key, value)
    
    # Загружаем данные
    image_paths = load_images()
    annotations = load_annotations()
    
    # Тестируем на 50 изображениях для быстроты
    config_manager.update_config("evaluation", "max_images", 50)
    
    start_time = time.time()
    results = evaluate_with_ground_truth(image_paths, annotations, config_manager)
    test_time = time.time() - start_time
    
    print(f"Результаты {config_name}:")
    print(f"  Precision: {results['overall_precision']:.3f}")
    print(f"  Recall: {results['overall_recall']:.3f}")
    print(f"  F1-score: {results['overall_f1']:.3f}")
    print(f"  FPS: {results['avg_fps']:.1f}")
    print(f"  Время теста: {test_time:.1f} сек")
    
    return results

def main():
    """Основная функция тестирования"""
    print("ТЕСТИРОВАНИЕ РАЗЛИЧНЫХ КОНФИГУРАЦИЙ ДЕТЕКТОРА")
    print("=" * 50)
    
    # Конфигурации для тестирования
    configs = {
        "Стандартная": {
            "scale_factor": 1.1,
            "min_neighbors": 5,
            "min_size": [30, 30],
            "max_size": [300, 300]
        },
        "Более чувствительная": {
            "scale_factor": 1.05,
            "min_neighbors": 3,
            "min_size": [20, 20],
            "max_size": [400, 400]
        },
        "Менее чувствительная": {
            "scale_factor": 1.2,
            "min_neighbors": 8,
            "min_size": [50, 50],
            "max_size": [200, 200]
        },
        "Для маленьких лиц": {
            "scale_factor": 1.05,
            "min_neighbors": 4,
            "min_size": [15, 15],
            "max_size": [500, 500]
        }
    }
    
    results = {}
    
    for config_name, params in configs.items():
        results[config_name] = test_configuration(config_name, params)
    
    # Сравниваем результаты
    print(f"\nСРАВНЕНИЕ РЕЗУЛЬТАТОВ:")
    print("=" * 50)
    print(f"{'Конфигурация':<20} {'Precision':<10} {'Recall':<10} {'F1':<10} {'FPS':<10}")
    print("-" * 60)
    
    for config_name, result in results.items():
        print(f"{config_name:<20} {result['overall_precision']:<10.3f} {result['overall_recall']:<10.3f} {result['overall_f1']:<10.3f} {result['avg_fps']:<10.1f}")
    
    # Находим лучшую конфигурацию
    best_f1 = max(results.items(), key=lambda x: x[1]['overall_f1'])
    best_fps = max(results.items(), key=lambda x: x[1]['avg_fps'])
    
    print(f"\nЛучшая F1-score: {best_f1[0]} ({best_f1[1]['overall_f1']:.3f})")
    print(f"Лучшая FPS: {best_fps[0]} ({best_fps[1]['avg_fps']:.1f})")

if __name__ == "__main__":
    main()
