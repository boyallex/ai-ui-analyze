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

class LocalAnalyzer:
    def __init__(self):
        # Используем русскоязычную GPT модель от Сбера
        self.model_name = "sberbank-ai/rugpt3small_based_on_gpt2"
        print("Загрузка модели...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
            print("Модель успешно загружена")
        except Exception as e:
            print(f"Ошибка при загрузке модели: {e}")
            raise
        
    def get_response(self, prompt: str) -> str:
        """Получение ответа от локальной модели"""
        try:
            print("Подготовка промпта для нейросети...")
            
            # Подготавливаем промпт
            system_prompt = """Ты опытный продукт-менеджер и UX-аналитик. 
            Проанализируй данные о кликах пользователей и дай рекомендации по улучшению интерфейса.
            Учитывай следующие аспекты:
            1. Распределение кликов по экрану
            2. Паттерны навигации
            3. Время пребывания на экране
            4. Последовательность действий
            5. Потенциальные проблемы UX
            6. Конкретные рекомендации по улучшению
            
            Дай структурированный ответ с разделами:
            - Общий анализ
            - Паттерны взаимодействия
            - Проблемные места
            - Рекомендации по улучшению
            """
            
            full_prompt = f"{system_prompt}\n\nДанные:\n{prompt}\n\nАнализ:"
            print("Промпт подготовлен")
            
            # Токенизируем и генерируем ответ
            print("Токенизация входных данных...")
            inputs = self.tokenizer.encode(full_prompt, return_tensors="pt", max_length=512, truncation=True)
            print(f"Размер входных токенов: {inputs.shape}")
            
            print("Генерация ответа...")
            outputs = self.model.generate(
                inputs,
                max_length=1024,
                num_return_sequences=1,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            print("Ответ сгенерирован")
            
            # Декодируем ответ
            print("Декодирование ответа...")
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Извлекаем только часть после "Анализ:"
            if "Анализ:" in response:
                response = response.split("Анализ:")[1].strip()
            
            print("Ответ успешно получен")
            return response
                
        except Exception as e:
            error_msg = f"Ошибка при анализе: {str(e)}"
            print(error_msg)
            return error_msg

def analyze_clicks(coordinates: list, screen_size: dict, tag: str, navigation_data: dict = None) -> str:
    """Анализирует данные о кликах и возвращает рекомендации"""
    total_points = len(coordinates)
    if total_points == 0:
        return "Нет данных о кликах пользователей для анализа."
    
    # Анализ распределения кликов
    x_coords = [point['x'] for point in coordinates]
    y_coords = [point['y'] for point in coordinates]
    
    # Разделим экран на зоны
    width = screen_size['width']
    height = screen_size['height']
    
    # Анализ горизонтальных зон
    top_clicks = sum(1 for y in y_coords if y < height/3)
    middle_clicks = sum(1 for y in y_coords if height/3 <= y < 2*height/3)
    bottom_clicks = sum(1 for y in y_coords if y >= 2*height/3)
    
    # Анализ вертикальных зон
    left_clicks = sum(1 for x in x_coords if x < width/3)
    center_clicks = sum(1 for x in x_coords if width/3 <= x < 2*width/3)
    right_clicks = sum(1 for x in x_coords if x >= 2*width/3)
    
    # Формируем анализ
    analysis = []
    analysis.append(f"Анализ взаимодействия пользователя с экраном '{tag}':\n")
    
    # Общая информация
    analysis.append(f"Всего кликов на экране: {total_points}")
    
    # Анализ навигации
    if navigation_data:
        analysis.append("\nАнализ навигации:")
        prev_screen = navigation_data.get('previous_screen')
        next_screen = navigation_data.get('next_screen')
        if prev_screen:
            analysis.append(f"- Пользователь пришел с экрана: {prev_screen}")
        if next_screen:
            analysis.append(f"- Пользователь перешел на экран: {next_screen}")
        
        # Анализ времени на экране
        time_spent = navigation_data.get('time_spent', 0)
        if time_spent:
            analysis.append(f"- Время, проведенное на экране: {time_spent:.1f} секунд")
    
    # Анализ вертикального распределения
    analysis.append("\nВертикальное распределение кликов:")
    analysis.append(f"- Верхняя часть экрана: {top_clicks} кликов ({top_clicks/total_points*100:.1f}%)")
    analysis.append(f"- Средняя часть экрана: {middle_clicks} кликов ({middle_clicks/total_points*100:.1f}%)")
    analysis.append(f"- Нижняя часть экрана: {bottom_clicks} кликов ({bottom_clicks/total_points*100:.1f}%)")
    
    # Анализ горизонтального распределения
    analysis.append("\nГоризонтальное распределение кликов:")
    analysis.append(f"- Левая часть экрана: {left_clicks} кликов ({left_clicks/total_points*100:.1f}%)")
    analysis.append(f"- Центральная часть: {center_clicks} кликов ({center_clicks/total_points*100:.1f}%)")
    analysis.append(f"- Правая часть экрана: {right_clicks} кликов ({right_clicks/total_points*100:.1f}%)")
    
    # Анализ паттернов взаимодействия
    analysis.append("\nПаттерны взаимодействия:")
    
    # Анализируем концентрацию кликов
    if top_clicks > middle_clicks and top_clicks > bottom_clicks:
        analysis.append("- Пользователи чаще взаимодействуют с верхней частью экрана. Возможно, там находятся важные элементы управления или навигации.")
    elif bottom_clicks > middle_clicks and bottom_clicks > top_clicks:
        analysis.append("- Высокая активность в нижней части экрана. Скорее всего, там расположены основные элементы управления или навигации.")
    
    if left_clicks > center_clicks and left_clicks > right_clicks:
        analysis.append("- Большинство кликов сконцентрировано в левой части. Возможно, там находятся основные элементы интерфейса или меню.")
    elif right_clicks > center_clicks and right_clicks > left_clicks:
        analysis.append("- Пользователи активно используют правую часть экрана. Вероятно, там расположены важные функции или элементы управления.")
    
    # Рекомендации
    analysis.append("\nРекомендации:")
    
    # Рекомендации по навигации
    if navigation_data:
        if time_spent and time_spent < 5:
            analysis.append("- Пользователи быстро покидают экран. Возможно, стоит упростить навигацию или добавить более заметные элементы управления.")
        if prev_screen and next_screen and prev_screen == next_screen:
            analysis.append("- Пользователи часто возвращаются на предыдущий экран. Стоит проверить, не вызывает ли текущий экран затруднений.")
    
    # Рекомендации по расположению элементов
    if center_clicks > left_clicks and center_clicks > right_clicks:
        analysis.append("- Большинство взаимодействий происходит в центре экрана. Рекомендуется размещать важные элементы управления в центральной части.")
    
    if total_points < 10:
        analysis.append("\nПримечание: Для более точного анализа рекомендуется собрать больше данных о взаимодействии пользователей с интерфейсом.")
    
    return "\n".join(analysis)

def analyze_with_gpt(report: str, api_key: str = None):
    """Анализирует данные о кликах с помощью нейросети и сохраняет результат в файл"""
    try:
        print("Начало анализа с помощью нейросети...")
        
        # Извлекаем данные из отчета
        data = json.loads(report)
        coordinates = data.get('points', [])
        screen_size = data.get('screenSize', {'width': 1920, 'height': 1080})
        tag = data.get('tag', 'unknown')
        navigation_data = data.get('navigation', {})
        
        print(f"Данные извлечены: {len(coordinates)} точек, экран: {tag}")
        
        # Создаем экземпляр анализатора
        print("Создание экземпляра анализатора...")
        analyzer = LocalAnalyzer()
        
        # Формируем данные для анализа
        analysis_data = {
            'screen_name': tag,
            'total_clicks': len(coordinates),
            'screen_size': screen_size,
            'click_distribution': {
                'top': sum(1 for p in coordinates if p['y'] < screen_size['height']/3),
                'middle': sum(1 for p in coordinates if screen_size['height']/3 <= p['y'] < 2*screen_size['height']/3),
                'bottom': sum(1 for p in coordinates if p['y'] >= 2*screen_size['height']/3),
                'left': sum(1 for p in coordinates if p['x'] < screen_size['width']/3),
                'center': sum(1 for p in coordinates if screen_size['width']/3 <= p['x'] < 2*screen_size['width']/3),
                'right': sum(1 for p in coordinates if p['x'] >= 2*screen_size['width']/3)
            },
            'navigation': navigation_data,
            'time_spent': navigation_data.get('time_spent', 0),
            'previous_screen': navigation_data.get('previous_screen'),
            'next_screen': navigation_data.get('next_screen')
        }
        
        print("Данные подготовлены для анализа:")
        print(json.dumps(analysis_data, indent=2))
        
        # Получаем анализ от нейросети
        print("Запрос к нейросети...")
        ai_analysis = analyzer.get_response(json.dumps(analysis_data, indent=2))
        print("Получен ответ от нейросети:")
        print(ai_analysis)
        
        # Добавляем базовый анализ
        print("Выполнение базового анализа...")
        basic_analysis = analyze_clicks(coordinates, screen_size, tag, navigation_data)
        
        # Объединяем анализы
        full_analysis = f"""=== Базовый анализ ===\n\n{basic_analysis}\n\n=== Анализ нейросети ===\n\n{ai_analysis}"""
        
        print("Анализ завершен")
        return full_analysis
        
    except Exception as e:
        error_msg = f"Ошибка при анализе: {e}"
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

def analyze_user_journey(heatmaps: List[Dict]) -> str:
    """Анализирует полный путь пользователя по приложению"""
    if not heatmaps:
        return "Нет данных о взаимодействии пользователя с приложением."
    
    print("\n=== Начало анализа пути пользователя ===")
    analysis = []
    analysis.append("Анализ полного пути пользователя по приложению:\n")
    
    # Собираем статистику по экранам
    screen_stats = defaultdict(lambda: {
        'clicks': 0,
        'time_spent': 0,
        'visits': 0,
        'from_screens': set(),
        'to_screens': set(),
        'click_coordinates': []
    })
    
    print("\nСбор статистики по экранам...")
    # Анализируем последовательность действий
    for i, heatmap in enumerate(heatmaps):
        screen_name = heatmap['pageName']
        print(f"\nОбработка экрана {screen_name}:")
        stats = screen_stats[screen_name]
        
        # Обновляем статистику
        stats['clicks'] += len(heatmap['points'])
        stats['visits'] += 1
        stats['click_coordinates'].extend(heatmap['points'])
        print(f"- Добавлено {len(heatmap['points'])} кликов")
        
        if 'navigation' in heatmap:
            nav = heatmap['navigation']
            stats['time_spent'] += nav.get('time_spent', 0)
            if 'previous_screen' in nav:
                stats['from_screens'].add(nav['previous_screen'])
                print(f"- Добавлен переход с экрана: {nav['previous_screen']}")
            if 'next_screen' in nav:
                stats['to_screens'].add(nav['next_screen'])
                print(f"- Добавлен переход на экран: {nav['next_screen']}")
    
    print("\nФормирование анализа...")
    # Анализ последовательности экранов
    analysis.append("Последовательность посещения экранов:")
    for i, heatmap in enumerate(heatmaps):
        screen_name = heatmap['pageName']
        nav = heatmap.get('navigation', {})
        prev_screen = nav.get('previous_screen', 'Начало сессии')
        next_screen = nav.get('next_screen', 'Конец сессии')
        
        analysis.append(f"\n{i+1}. {screen_name}:")
        analysis.append(f"   - Пришел с: {prev_screen}")
        analysis.append(f"   - Ушел на: {next_screen}")
        analysis.append(f"   - Кликов: {len(heatmap['points'])}")
        if 'navigation' in heatmap and 'time_spent' in heatmap['navigation']:
            analysis.append(f"   - Время на экране: {heatmap['navigation']['time_spent']:.1f} сек")
    
    # Анализ паттернов использования
    analysis.append("\nПаттерны использования:")
    
    # Находим наиболее посещаемые экраны
    most_visited = sorted(screen_stats.items(), key=lambda x: x[1]['visits'], reverse=True)
    analysis.append("\nНаиболее посещаемые экраны:")
    for screen, stats in most_visited:
        analysis.append(f"- {screen}: {stats['visits']} посещений, {stats['clicks']} кликов")
    
    # Находим экраны с наибольшим временем пребывания
    most_time = sorted(screen_stats.items(), key=lambda x: x[1]['time_spent'], reverse=True)
    analysis.append("\nЭкраны с наибольшим временем пребывания:")
    for screen, stats in most_time:
        if stats['time_spent'] > 0:
            analysis.append(f"- {screen}: {stats['time_spent']:.1f} секунд")
    
    # Анализ навигационных паттернов
    analysis.append("\nНавигационные паттерны:")
    for screen, stats in screen_stats.items():
        if stats['from_screens'] or stats['to_screens']:
            analysis.append(f"\n{screen}:")
            if stats['from_screens']:
                analysis.append(f"- Пользователи приходят с: {', '.join(stats['from_screens'])}")
            if stats['to_screens']:
                analysis.append(f"- Пользователи уходят на: {', '.join(stats['to_screens'])}")
    
    print("\nГенерация рекомендаций...")
    # Рекомендации
    analysis.append("\nРекомендации:")
    
    # Анализируем быстрые переходы
    for screen, stats in screen_stats.items():
        if stats['time_spent'] > 0 and stats['time_spent'] < 5:
            analysis.append(f"- Пользователи быстро покидают экран {screen}. Возможно, стоит упростить навигацию или улучшить контент.")
    
    # Анализируем частые возвраты
    for screen, stats in screen_stats.items():
        if len(stats['from_screens']) > 2:
            analysis.append(f"- Пользователи часто возвращаются на экран {screen}. Возможно, стоит оптимизировать навигацию или добавить быстрый доступ к часто используемым функциям.")
    
    # Анализируем распределение кликов
    for screen, stats in screen_stats.items():
        if stats['clicks'] > 0:
            print(f"\nАнализ кликов для экрана {screen}:")
            # Анализируем концентрацию кликов
            x_coords = [p['x'] for p in stats['click_coordinates']]
            y_coords = [p['y'] for p in stats['click_coordinates']]
            
            if x_coords and y_coords:
                avg_x = sum(x_coords) / len(x_coords)
                avg_y = sum(y_coords) / len(y_coords)
                print(f"- Средние координаты: x={avg_x:.1f}, y={avg_y:.1f}")
                
                # Определяем зоны концентрации кликов
                if avg_y < 300:  # Верхняя часть экрана
                    analysis.append(f"- На экране {screen} большинство кликов происходит в верхней части. Возможно, стоит разместить важные элементы управления в этой области.")
                elif avg_y > 600:  # Нижняя часть экрана
                    analysis.append(f"- На экране {screen} большинство кликов происходит в нижней части. Рекомендуется разместить основные элементы управления внизу экрана.")
                
                if avg_x < 150:  # Левая часть экрана
                    analysis.append(f"- На экране {screen} большинство кликов происходит в левой части. Возможно, стоит оптимизировать расположение элементов меню.")
                elif avg_x > 250:  # Правая часть экрана
                    analysis.append(f"- На экране {screen} большинство кликов происходит в правой части. Рекомендуется разместить важные элементы управления справа.")
    
    # Добавляем общие рекомендации
    analysis.append("\nОбщие рекомендации:")
    
    # Рекомендации на основе времени пребывания
    total_time = sum(stats['time_spent'] for stats in screen_stats.values())
    if total_time > 0:
        for screen, stats in screen_stats.items():
            if stats['time_spent'] > 0:
                time_percentage = (stats['time_spent'] / total_time) * 100
                if time_percentage > 50:
                    analysis.append(f"- Пользователи проводят {time_percentage:.1f}% времени на экране {screen}. Рекомендуется оптимизировать этот экран для повышения эффективности.")
    
    # Рекомендации на основе количества кликов
    total_clicks = sum(stats['clicks'] for stats in screen_stats.values())
    if total_clicks > 0:
        for screen, stats in screen_stats.items():
            if stats['clicks'] > 0:
                click_percentage = (stats['clicks'] / total_clicks) * 100
                if click_percentage > 30:
                    analysis.append(f"- На экране {screen} происходит {click_percentage:.1f}% всех кликов. Возможно, стоит упростить интерфейс этого экрана.")
    
    print("\n=== Анализ пути пользователя завершен ===")
    return "\n".join(analysis)

def analyze_data(tag: str = None, designs_folder: str = "page_designs", data_folder: str = "data/consilium", gpt_api_key: str = None) -> str:
    """Анализирует данные о кликах и создает тепловую карту и маску полезных зон"""
    try:
        print(f"\n=== Начало анализа для тега: {tag} ===\n")
        
        # Проверяем существование директорий
        if not os.path.exists(designs_folder):
            os.makedirs(designs_folder)
        if not os.path.exists(data_folder):
            os.makedirs(data_folder)

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
        analysis = analyze_user_journey(heatmaps)
        
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