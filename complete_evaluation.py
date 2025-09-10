#!/usr/bin/env python3
"""
Полная оценка детекции лиц с использованием ground truth данных
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import glob
import os
import json
from pathlib import Path
from config_manager import ConfigManager

# Создаем папку для результатов
os.makedirs('results', exist_ok=True)

def load_images(data_dir="data/images"):
    """Загружаем все jpg файлы из папки"""
    images = glob.glob(f"{data_dir}/*.jpg")
    print(f"Найдено {len(images)} изображений")
    return images

def load_image(path):
    """Загружаем одно изображение"""
    return cv2.imread(path)

def load_annotations(annos_file="data/annos.json"):
    """Загружаем ground truth аннотации"""
    with open(annos_file, 'r') as f:
        data = json.load(f)
    
    # Преобразуем в словарь для быстрого поиска
    annotations = {}
    for item in data:
        img_path = item['img_path'].replace('\\', '/')  # Исправляем пути для Unix
        img_name = os.path.basename(img_path)
        annotations[img_name] = item['annotations']
    
    print(f"Загружено {len(annotations)} аннотаций")
    return annotations

def detect_faces(image, config_manager=None):
    """Детектируем лица на изображении с использованием конфигурации"""
    if config_manager is None:
        config_manager = ConfigManager()
    
    detector_params = config_manager.get_detector_params()
    scale_factor = detector_params['scale_factor']
    min_neighbors = detector_params['min_neighbors']
    min_size = tuple(detector_params['min_size'])
    max_size = tuple(detector_params['max_size'])
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scale_factor, min_neighbors, minSize=min_size, maxSize=max_size)
    return faces

def calculate_iou(box1, box2):
    """Вычисляем IoU между двумя прямоугольниками"""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    # Координаты пересечения
    x_left = max(x1, x2)
    y_top = max(y1, y2)
    x_right = min(x1 + w1, x2 + w2)
    y_bottom = min(y1 + h1, y2 + h2)
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    intersection = (x_right - x_left) * (y_bottom - y_top)
    area1 = w1 * h1
    area2 = w2 * h2
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0

def match_detections_to_ground_truth(pred_boxes, gt_boxes, iou_threshold=0.5):
    """Сопоставляем предсказания с ground truth по IoU"""
    if len(pred_boxes) == 0 and len(gt_boxes) == 0:
        return [], [], [], []
    
    if len(pred_boxes) == 0:
        return [], [], [], list(range(len(gt_boxes)))
    
    if len(gt_boxes) == 0:
        return [], [], list(range(len(pred_boxes))), []
    
    # Вычисляем IoU для всех пар
    iou_matrix = np.zeros((len(pred_boxes), len(gt_boxes)))
    for i, pred_box in enumerate(pred_boxes):
        for j, gt_box in enumerate(gt_boxes):
            iou_matrix[i, j] = calculate_iou(pred_box, gt_box)
    
    # Жадное сопоставление
    matched_pred = []
    matched_gt = []
    unmatched_pred = list(range(len(pred_boxes)))
    unmatched_gt = list(range(len(gt_boxes)))
    
    # Сортируем по убыванию IoU
    sorted_indices = np.unravel_index(np.argsort(iou_matrix.ravel())[::-1], iou_matrix.shape)
    
    for pred_idx, gt_idx in zip(sorted_indices[0], sorted_indices[1]):
        if pred_idx in unmatched_pred and gt_idx in unmatched_gt:
            if iou_matrix[pred_idx, gt_idx] >= iou_threshold:
                matched_pred.append(pred_idx)
                matched_gt.append(gt_idx)
                unmatched_pred.remove(pred_idx)
                unmatched_gt.remove(gt_idx)
    
    return matched_pred, matched_gt, unmatched_pred, unmatched_gt

def calculate_metrics(pred_boxes, gt_boxes, iou_threshold=0.5):
    """Вычисляем метрики качества"""
    matched_pred, matched_gt, unmatched_pred, unmatched_gt = match_detections_to_ground_truth(
        pred_boxes, gt_boxes, iou_threshold
    )
    
    tp = len(matched_pred)  # True Positives
    fp = len(unmatched_pred)  # False Positives
    fn = len(unmatched_gt)  # False Negatives
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'matched_pred': matched_pred,
        'matched_gt': matched_gt,
        'unmatched_pred': unmatched_pred,
        'unmatched_gt': unmatched_gt
    }

def draw_boxes(image, boxes, color=(0, 255, 0), thickness=2, labels=None):
    """Рисуем bounding boxes на изображении"""
    result = image.copy()
    for i, (x, y, w, h) in enumerate(boxes):
        cv2.rectangle(result, (x, y), (x + w, y + h), color, thickness)
        if labels and i < len(labels):
            cv2.putText(result, str(labels[i]), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return result

def evaluate_with_ground_truth(image_paths, annotations, config_manager=None):
    """Полная оценка с ground truth данными"""
    if config_manager is None:
        config_manager = ConfigManager()
    
    eval_params = config_manager.get_evaluation_params()
    max_images = eval_params['max_images']
    iou_threshold = eval_params['iou_threshold']
    
    if max_images:
        image_paths = image_paths[:max_images]
    
    total_faces = 0
    total_gt_faces = 0
    total_time = 0
    all_metrics = []
    detailed_results = []
    
    print(f"Обрабатываем {len(image_paths)} изображений с ground truth...")
    
    for i, img_path in enumerate(image_paths):
        if i % 20 == 0:
            print(f"Обработано: {i}/{len(image_paths)} ({i/len(image_paths)*100:.1f}%)")
        
        # Загружаем изображение
        image = load_image(img_path)
        if image is None:
            continue
        
        img_name = os.path.basename(img_path)
        
        # Получаем ground truth аннотации
        gt_data = annotations.get(img_name, {})
        gt_boxes = gt_data.get('bbox', [])
        
        # Фильтруем валидные аннотации (invalid = 0)
        valid_gt_boxes = []
        if 'invalid' in gt_data:
            for j, box in enumerate(gt_boxes):
                if j < len(gt_data['invalid']) and gt_data['invalid'][j] == 0:
                    valid_gt_boxes.append(box)
        else:
            valid_gt_boxes = gt_boxes
        
        # Детектируем лица
        start_time = time.time()
        pred_boxes = detect_faces(image, config_manager)
        detection_time = time.time() - start_time
        
        # Вычисляем метрики
        metrics = calculate_metrics(pred_boxes, valid_gt_boxes, iou_threshold)
        
        total_faces += len(pred_boxes)
        total_gt_faces += len(valid_gt_boxes)
        total_time += detection_time
        
        all_metrics.append(metrics)
        
        detailed_results.append({
            'image': img_name,
            'faces_detected': len(pred_boxes),
            'gt_faces': len(valid_gt_boxes),
            'detection_time': detection_time,
            'fps': 1/detection_time if detection_time > 0 else 0,
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1': metrics['f1'],
            'tp': metrics['tp'],
            'fp': metrics['fp'],
            'fn': metrics['fn']
        })
    
    # Вычисляем общие метрики
    total_tp = sum(m['tp'] for m in all_metrics)
    total_fp = sum(m['fp'] for m in all_metrics)
    total_fn = sum(m['fn'] for m in all_metrics)
    
    overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    overall_f1 = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0.0
    
    avg_fps = total_faces / total_time if total_time > 0 else 0
    
    results = {
        'total_images': len(image_paths),
        'total_faces_detected': total_faces,
        'total_gt_faces': total_gt_faces,
        'total_time': total_time,
        'avg_fps': avg_fps,
        'avg_time_per_image': total_time / len(image_paths),
        'overall_precision': overall_precision,
        'overall_recall': overall_recall,
        'overall_f1': overall_f1,
        'total_tp': total_tp,
        'total_fp': total_fp,
        'total_fn': total_fn,
        'iou_threshold': iou_threshold,
        'detailed_results': detailed_results
    }
    
    return results

def main():
    """Основная функция"""
    print("Запускаем полную оценку с ground truth данными...")
    
    # Загружаем данные
    image_paths = load_images()
    if not image_paths:
        print("Ошибка: Изображения не найдены в data/images/")
        return
    
    annotations = load_annotations()
    if not annotations:
        print("Ошибка: Аннотации не найдены в data/annos.json")
        return
    
    # Загружаем конфигурацию
    config_manager = ConfigManager()
    eval_params = config_manager.get_evaluation_params()
    max_images = eval_params['max_images']
    
    print(f"Оцениваем на {min(max_images, len(image_paths))} изображениях...")
    results = evaluate_with_ground_truth(image_paths, annotations, config_manager)
    
    # Выводим результаты
    print(f"\nИТОГОВЫЕ РЕЗУЛЬТАТЫ:")
    print(f"Обработано изображений: {results['total_images']}")
    print(f"Найдено лиц: {results['total_faces_detected']}")
    print(f"Ground truth лиц: {results['total_gt_faces']}")
    print(f"Общее время: {results['total_time']:.2f} сек")
    print(f"Средний FPS: {results['avg_fps']:.1f}")
    print(f"Время на изображение: {results['avg_time_per_image']:.3f} сек")
    
    print(f"\nМЕТРИКИ КАЧЕСТВА (IoU ≥ {results['iou_threshold']}):")
    print(f"Precision: {results['overall_precision']:.3f}")
    print(f"Recall: {results['overall_recall']:.3f}")
    print(f"F1-score: {results['overall_f1']:.3f}")
    print(f"True Positives: {results['total_tp']}")
    print(f"False Positives: {results['total_fp']}")
    print(f"False Negatives: {results['total_fn']}")
    
    # Сохраняем результаты
    with open("results/complete_evaluation.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Создаем отчет
    report = f"""# Полный отчет по детекции лиц

## Метод: Haar Cascade (OpenCV)

## Результаты на {results['total_images']} изображениях:

### Производительность:
- **Скорость**: {results['avg_fps']:.1f} FPS
- **Время на изображение**: {results['avg_time_per_image']:.3f} сек
- **Найдено лиц**: {results['total_faces_detected']}
- **Ground truth лиц**: {results['total_gt_faces']}

### Качество (IoU ≥ {results['iou_threshold']}):
- **Precision**: {results['overall_precision']:.3f}
- **Recall**: {results['overall_recall']:.3f}
- **F1-score**: {results['overall_f1']:.3f}

### Детализация:
- **True Positives**: {results['total_tp']}
- **False Positives**: {results['total_fp']}
- **False Negatives**: {results['total_fn']}

## Заключение:
Haar Cascade показывает {'отличную' if results['overall_f1'] > 0.7 else 'хорошую' if results['overall_f1'] > 0.5 else 'удовлетворительную'} производительность.
{'Рекомендуется для production использования.' if results['overall_f1'] > 0.7 else 'Требует улучшения для production.' if results['overall_f1'] < 0.5 else 'Подходит для базовых задач.'}

Результаты сохранены в results/complete_evaluation.json
"""
    
    with open("results/complete_report.md", "w") as f:
        f.write(report)
    
    print(f"\nРезультаты сохранены:")
    print(f"   results/complete_evaluation.json")
    print(f"   results/complete_report.md")
    
    return results

if __name__ == "__main__":
    main()
