import json
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from scipy.ndimage import gaussian_filter
from datetime import datetime
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict
from collections import defaultdict
import openai

def create_heatmap(coordinates: list, screen_size: dict) -> np.ndarray:
    """Создает тепловую карту на основе координат кликов"""
    if not coordinates:
        return np.zeros((100, 100))
        
    x = np.array([point['x'] for point in coordinates])
    y = np.array([point['y'] for point in coordinates])
    
    # Создаем сетку для тепловой карты
    xx, yy = np.mgrid[0:screen_size['width']:100j, 0:screen_size['height']:100j]
    
    # Если точек меньше 3, используем простой метод
    if len(coordinates) < 3:
        heatmap = np.zeros((100, 100))
        for point in coordinates:
            x_idx = int(point['x'] * 100 / screen_size['width'])
            y_idx = int(point['y'] * 100 / screen_size['height'])
            if 0 <= x_idx < 100 and 0 <= y_idx < 100:
                heatmap[y_idx, x_idx] += 1
        # Добавляем небольшое размытие
        heatmap = gaussian_filter(heatmap, sigma=1)
        # Нормализуем значения
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()
        return heatmap
    
    try:
        # Транспонируем данные для gaussian_kde
        values = np.vstack([x, y])
        positions = np.vstack([xx.ravel(), yy.ravel()])
        
        # Создаем ядро плотности
        kernel = gaussian_kde(values)
        # Вычисляем плотность
        f = np.reshape(kernel(positions).T, xx.shape)
        # Нормализуем значения
        f = (f - f.min()) / (f.max() - f.min())
        return f
    except Exception as e:
        print(f"Ошибка при создании тепловой карты: {e}")
        # Создаем простую тепловую карту
        heatmap = np.zeros((100, 100))
        for point in coordinates:
            x_idx = int(point['x'] * 100 / screen_size['width'])
            y_idx = int(point['y'] * 100 / screen_size['height'])
            if 0 <= x_idx < 100 and 0 <= y_idx < 100:
                heatmap[y_idx, x_idx] += 1
        # Добавляем небольшое размытие
        heatmap = gaussian_filter(heatmap, sigma=1)
        # Нормализуем значения
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()
        return heatmap

class OpenAIAnalyzer:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.client = openai.OpenAI(api_key=api_key)
        print("Инициализация OpenAI API...")
        
    def get_response(self, prompt: str) -> str:
        """Получение ответа от ChatGPT"""
        try:
            print("Подготовка промпта для ChatGPT...")
            
            # Подготавливаем промпт
            system_prompt = """Ты опытный UX-аналитик. Проанализируй данные о взаимодействии пользователей с интерфейсом и дай рекомендации.

Данные содержат информацию о:
1. Количестве кликов на каждом экране
2. Времени, проведенном на экране
3. Количестве посещений
4. Навигационных паттернах (откуда пришли, куда ушли)
5. Координатах кликов

Пожалуйста, проанализируй эти данные и дай рекомендации в следующем формате:

ОБЩИЙ АНАЛИЗ:
- Опиши основные паттерны использования
- Выдели ключевые метрики
- Отметь необычные паттерны

ПРОБЛЕМНЫЕ МЕСТА:
- Укажи экраны с низкой эффективностью
- Отметь места, где пользователи тратят много времени
- Выдели проблемные навигационные паттерны

РЕКОМЕНДАЦИИ:
- Дай конкретные рекомендации по улучшению
- Предложи изменения в навигации
- Укажи, где можно оптимизировать интерфейс

Используй простой язык и конкретные примеры. Избегай общих фраз."""
            
            print("Отправка запроса к ChatGPT...")
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Данные для анализа:\n{prompt}"}
                ],
                temperature=0.7,
                max_tokens=1000
            )
            
            print("Ответ получен от ChatGPT")
            return response.choices[0].message.content
                
        except Exception as e:
            error_msg = f"Ошибка при анализе: {str(e)}"
            print(error_msg)
            return error_msg

def load_all_heatmaps(data_folder: str) -> List[Dict]:
    """Загружает все файлы heatmap и сортирует их по времени"""
    heatmaps = []
    print(f"\nПоиск heatmap файлов в {data_folder}")
    
    for filename in os.listdir(data_folder):
        if filename.endswith('_heatmap.json'):
            file_path = os.path.join(data_folder, filename)
            print(f"Обработка файла: {filename}")
            
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    # Извлекаем имя экрана из имени файла
                    screen_name = filename.replace('_heatmap.json', '')
                    if 'home_page_' in screen_name:
                        screen_name = screen_name.replace('home_page_', '')
                    data['pageName'] = screen_name
                    data['pageId'] = screen_name
                    print(f"Успешно загружен файл: {filename}")
                    print(f"- Количество точек: {len(data.get('points', []))}")
                    print(f"- ID страницы: {data.get('pageId', 'не указан')}")
                    heatmaps.append(data)
            except Exception as e:
                print(f"Ошибка при загрузке файла {filename}: {e}")
    
    # Сортируем по времени создания
    sorted_heatmaps = sorted(heatmaps, key=lambda x: x.get('createdAt', ''))
    print(f"\nВсего загружено {len(sorted_heatmaps)} heatmap файлов")
    return sorted_heatmaps

def analyze_user_journey(heatmaps: List[Dict], api_key: str) -> str:
    """Анализирует полный путь пользователя по приложению"""
    if not heatmaps:
        return "Нет данных о взаимодействии пользователя с приложением."
    
    print("\n=== Начало анализа пути пользователя ===")
    
    # Собираем статистику по экранам
    screen_stats = defaultdict(lambda: {
        'clicks': 0,
        'time_spent': 0,
        'visits': 0,
        'from_screens': set(),
        'to_screens': set(),
        'click_coordinates': []
    })
    
    # Анализируем последовательность действий
    for i, heatmap in enumerate(heatmaps):
        screen_name = heatmap['pageName']
        stats = screen_stats[screen_name]
        
        # Обновляем статистику
        stats['clicks'] += len(heatmap['points'])
        stats['visits'] += 1
        stats['click_coordinates'].extend(heatmap['points'])
        
        if 'navigation' in heatmap:
            nav = heatmap['navigation']
            stats['time_spent'] += nav.get('time_spent', 0)
            if 'previous_screen' in nav:
                stats['from_screens'].add(nav['previous_screen'])
            if 'next_screen' in nav:
                stats['to_screens'].add(nav['next_screen'])
    
    # Находим наиболее посещаемые экраны
    most_visited = sorted(screen_stats.items(), key=lambda x: x[1]['visits'], reverse=True)
    
    # Находим экраны с наибольшим временем пребывания
    most_time = sorted(screen_stats.items(), key=lambda x: x[1]['time_spent'], reverse=True)
    
    # Подготавливаем данные для нейросети
    total_time = sum(stats['time_spent'] for stats in screen_stats.values())
    total_clicks = sum(stats['clicks'] for stats in screen_stats.values())
    
    # Добавляем анализ с помощью ChatGPT
    print("\nЗапуск анализа с помощью ChatGPT...")
    try:
        analyzer = OpenAIAnalyzer(api_key)
        # Подготавливаем данные для анализа
        ai_analysis_data = {
            'screen_stats': {k: {
                'clicks': v['clicks'],
                'time_spent': v['time_spent'],
                'visits': v['visits'],
                'from_screens': list(v['from_screens']),
                'to_screens': list(v['to_screens']),
                'click_coordinates': v['click_coordinates']
            } for k, v in screen_stats.items()},
            'total_time': total_time,
            'total_clicks': total_clicks,
            'most_visited': [(k, {'visits': v['visits'], 'clicks': v['clicks']}) for k, v in most_visited],
            'most_time': [(k, {'time_spent': v['time_spent']}) for k, v in most_time]
        }
        
        # Получаем рекомендации от ChatGPT
        ai_recommendations = analyzer.get_response(json.dumps(ai_analysis_data, indent=2))
        
        return ai_recommendations
        
    except Exception as e:
        print(f"Ошибка при анализе ChatGPT: {e}")
        return "Не удалось получить рекомендации от ChatGPT."

def analyze_data(tag: str = None, designs_folder: str = "page_designs", data_folder: str = "data/consilium", gpt_api_key: str = None) -> str:
    """Анализирует данные о кликах и создает тепловую карту и маску полезных зон"""
    try:
        print(f"\n=== Начало анализа для тега: {tag} ===\n")
        
        # Проверяем существование директорий
        if not os.path.exists(designs_folder):
            os.makedirs(designs_folder)
        if not os.path.exists(data_folder):
            os.makedirs(data_folder)
        if not os.path.exists("analytics"):
            os.makedirs("analytics")

        # Загружаем все heatmap
        print(f"Загрузка данных из директории: {data_folder}")
        heatmaps = load_all_heatmaps(data_folder)
        print(f"Загружено {len(heatmaps)} heatmap файлов")
        
        # Если указан конкретный тег, анализируем только его
        if tag:
            print(f"Фильтрация по тегу: {tag}")
            heatmaps = [h for h in heatmaps if h['pageId'] == tag]
            print(f"Найдено {len(heatmaps)} heatmap для тега {tag}")
        
        if not heatmaps:
            error_msg = f"Нет данных для анализа тега: {tag}"
            print(error_msg)
            return error_msg
            
        # Проверяем данные в heatmap
        for i, heatmap in enumerate(heatmaps):
            print(f"\nПроверка heatmap #{i+1}:")
            print(f"ID страницы: {heatmap.get('pageId', 'не указан')}")
            print(f"Имя страницы: {heatmap.get('pageName', 'не указано')}")
            print(f"Количество точек: {len(heatmap.get('points', []))}")
            print(f"Размер экрана: {heatmap.get('screenSize', {})}")
            if 'navigation' in heatmap:
                print(f"Данные навигации: {heatmap['navigation']}")
        
        # Анализируем полный путь пользователя
        print("\nАнализ пути пользователя...")
        analysis = analyze_user_journey(heatmaps, gpt_api_key)
        
        # Создаем тепловые карты для каждого экрана
        heatmap_results = []
        for heatmap in heatmaps:
            coordinates = [{'x': point['x'], 'y': point['y']} for point in heatmap['points']]
            screen_size = heatmap['screenSize']
            
            print(f"\nСоздание тепловой карты для {heatmap['pageName']}:")
            print(f"Количество координат: {len(coordinates)}")
            print(f"Размер экрана: {screen_size}")
            
            # Создаем тепловую карту
            heatmap_img = create_heatmap(coordinates, screen_size)
            
            # Сохраняем тепловую карту
            heatmap_image_path = os.path.join(
                designs_folder, 
                f"{heatmap['pageId']}_heatmap_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            )
            
            plt.figure(figsize=(10, 10))
            plt.imshow(heatmap_img.T, cmap='hot', interpolation='nearest', origin='lower')
            plt.colorbar()
            plt.title(f"Тепловая карта: {heatmap['pageName']}")
            plt.axis('off')
            plt.savefig(heatmap_image_path, bbox_inches='tight', pad_inches=0)
            plt.close()
            
            heatmap_results.append(f"Тепловая карта сохранена: {heatmap_image_path}")
            heatmap_results.append(f"Всего точек на экране {heatmap['pageName']}: {len(coordinates)}")

        # Формируем полный отчет
        full_report = f"""=== Аналитика ===\n\n{analysis}\n\n=== Тепловые карты ===\n\n{chr(10).join(heatmap_results)}"""
        
        # Сохраняем отчет в файл
        if tag:
            report_file = os.path.join("analytics", f"{tag}_analysis.txt")
        else:
            report_file = os.path.join("analytics", "overall_analysis.txt")
            
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(full_report)
        
        print(f"\nОтчет сохранен в файл: {report_file}")
        print("\n=== Анализ завершен ===\n")
        return full_report

    except Exception as e:
        error_msg = f"Ошибка при анализе данных: {e}"
        print(error_msg)
        return error_msg

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Анализ данных о кликах')
    parser.add_argument('--tag', type=str, help='Тег экрана (опционально)')
    parser.add_argument('--gpt-api-key', type=str, help='API ключ для ChatGPT')
    args = parser.parse_args()
    
    result = analyze_data(args.tag, gpt_api_key=args.gpt_api_key)
    print(result) 