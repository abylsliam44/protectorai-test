#!/usr/bin/env python3
"""
Face Detection Project - Main Entry Point
"""

import sys
import os

# Добавляем src в путь
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.complete_evaluation import main as run_evaluation
from src.visualization.visualize_results import main as run_visualization

def main():
    """Главная функция"""
    print("Face Detection Project")
    print("=" * 30)
    print("1. Run evaluation")
    print("2. Create visualizations")
    print("3. Run both")
    
    choice = input("\nВыберите опцию (1-3): ").strip()
    
    if choice == "1":
        print("\nЗапускаем оценку...")
        run_evaluation()
    elif choice == "2":
        print("\nСоздаем визуализации...")
        run_visualization()
    elif choice == "3":
        print("\nЗапускаем полный пайплайн...")
        run_evaluation()
        run_visualization()
    else:
        print("Неверный выбор!")

if __name__ == "__main__":
    main()
