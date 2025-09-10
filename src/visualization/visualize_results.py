#!/usr/bin/env python3
"""
Визуализация результатов детекции с ground truth данными
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from pathlib import Path

def load_annotations(annos_file="data/annos.json"):
    """Загружаем ground truth аннотации"""
    with open(annos_file, 'r') as f:
        data = json.load(f)
    
    annotations = {}
    for item in data:
        img_path = item['img_path'].replace('\\', '/')
        img_name = os.path.basename(img_path)
        annotations[img_name] = item['annotations']
    
    return annotations

def detect_faces(image, scale_factor=1.1, min_neighbors=5, min_size=(30, 30)):
    """Детектируем лица на изображении"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scale_factor, min_neighbors, minSize=min_size)
    return faces

def draw_comparison(image, pred_boxes, gt_boxes, title="Face Detection Comparison"):
    """Рисуем сравнение предсказаний и ground truth"""
    # Создаем копии изображения
    pred_img = image.copy()
    gt_img = image.copy()
    combined_img = image.copy()
    
    # Рисуем предсказания (зеленые)
    for (x, y, w, h) in pred_boxes:
        cv2.rectangle(pred_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(pred_img, "Pred", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    # Рисуем ground truth (красные)
    for (x, y, w, h) in gt_boxes:
        cv2.rectangle(gt_img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(gt_img, "GT", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    
    # Рисуем комбинированное изображение
    for (x, y, w, h) in pred_boxes:
        cv2.rectangle(combined_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(combined_img, "Pred", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    for (x, y, w, h) in gt_boxes:
        cv2.rectangle(combined_img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(combined_img, "GT", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    
    # Создаем фигуру с тремя изображениями
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Оригинальное изображение
    axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    # Предсказания
    axes[1].imshow(cv2.cvtColor(pred_img, cv2.COLOR_BGR2RGB))
    axes[1].set_title(f"Predictions ({len(pred_boxes)} faces)")
    axes[1].axis('off')
    
    # Ground Truth
    axes[2].imshow(cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB))
    axes[2].set_title(f"Ground Truth ({len(gt_boxes)} faces)")
    axes[2].axis('off')
    
    plt.tight_layout()
    return fig

def visualize_sample_results():
    """Визуализируем результаты на нескольких примерах"""
    print("Создаем визуализации результатов...")
    
    # Загружаем данные
    annotations = load_annotations()
    
    # Выбираем несколько изображений для визуализации
    sample_images = [
        "2_Demonstration_Demonstration_Or_Protest_2_604.jpg",
        "5_Car_Accident_Accident_5_948.jpg", 
        "11_Meeting_Meeting_11_Meeting_Meeting_11_663.jpg",
        "39_Ice_Skating_Ice_Skating_39_992.jpg",
        "32_Worker_Laborer_Worker_Laborer_32_860.jpg"
    ]
    
    for i, img_name in enumerate(sample_images):
        img_path = f"data/images/{img_name}"
        
        if not os.path.exists(img_path):
            print(f"Предупреждение: Изображение {img_name} не найдено")
            continue
        
        # Загружаем изображение
        image = cv2.imread(img_path)
        if image is None:
            continue
        
        # Получаем ground truth
        gt_data = annotations.get(img_name, {})
        gt_boxes = gt_data.get('bbox', [])
        
        # Фильтруем валидные аннотации
        valid_gt_boxes = []
        if 'invalid' in gt_data:
            for j, box in enumerate(gt_boxes):
                if j < len(gt_data['invalid']) and gt_data['invalid'][j] == 0:
                    valid_gt_boxes.append(box)
        else:
            valid_gt_boxes = gt_boxes
        
        # Детектируем лица
        pred_boxes = detect_faces(image)
        
        # Создаем визуализацию
        fig = draw_comparison(image, pred_boxes, valid_gt_boxes, f"Sample {i+1}: {img_name}")
        
        # Сохраняем
        output_path = f"results/comparison_{i+1}.png"
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        print(f"Сохранено: {output_path}")
        print(f"   Предсказано: {len(pred_boxes)} лиц, Ground Truth: {len(valid_gt_boxes)} лиц")

def create_metrics_visualization():
    """Создаем визуализацию метрик"""
    print("Создаем визуализацию метрик...")
    
    # Загружаем результаты
    with open("results/complete_evaluation.json", "r") as f:
        results = json.load(f)
    
    detailed_results = results['detailed_results']
    
    # Извлекаем метрики
    precisions = [r['precision'] for r in detailed_results]
    recalls = [r['recall'] for r in detailed_results]
    f1_scores = [r['f1'] for r in detailed_results]
    fps_values = [r['fps'] for r in detailed_results]
    
    # Создаем графики
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Precision по изображениям
    axes[0, 0].plot(precisions, 'b-', alpha=0.7)
    axes[0, 0].axhline(y=np.mean(precisions), color='r', linestyle='--', label=f'Mean: {np.mean(precisions):.3f}')
    axes[0, 0].set_title('Precision по изображениям')
    axes[0, 0].set_xlabel('Изображение')
    axes[0, 0].set_ylabel('Precision')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Recall по изображениям
    axes[0, 1].plot(recalls, 'g-', alpha=0.7)
    axes[0, 1].axhline(y=np.mean(recalls), color='r', linestyle='--', label=f'Mean: {np.mean(recalls):.3f}')
    axes[0, 1].set_title('Recall по изображениям')
    axes[0, 1].set_xlabel('Изображение')
    axes[0, 1].set_ylabel('Recall')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # F1-score по изображениям
    axes[1, 0].plot(f1_scores, 'm-', alpha=0.7)
    axes[1, 0].axhline(y=np.mean(f1_scores), color='r', linestyle='--', label=f'Mean: {np.mean(f1_scores):.3f}')
    axes[1, 0].set_title('F1-score по изображениям')
    axes[1, 0].set_xlabel('Изображение')
    axes[1, 0].set_ylabel('F1-score')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # FPS по изображениям
    axes[1, 1].plot(fps_values, 'c-', alpha=0.7)
    axes[1, 1].axhline(y=np.mean(fps_values), color='r', linestyle='--', label=f'Mean: {np.mean(fps_values):.1f}')
    axes[1, 1].set_title('FPS по изображениям')
    axes[1, 1].set_xlabel('Изображение')
    axes[1, 1].set_ylabel('FPS')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("results/metrics_visualization.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print("Сохранено: results/metrics_visualization.png")

def main():
    """Основная функция"""
    print("Создаем визуализации результатов детекции лиц...")
    
    # Создаем папку для результатов
    os.makedirs('results', exist_ok=True)
    
    # Визуализируем примеры
    visualize_sample_results()
    
    # Создаем графики метрик
    create_metrics_visualization()
    
    print("\nВсе визуализации созданы!")

if __name__ == "__main__":
    main()
