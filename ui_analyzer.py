import json
import base64
import os
import tempfile
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.stats import gaussian_kde
from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks, Query, Body, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Optional
import io
from PIL import Image
from datetime import datetime
import asyncio
import glob
from scipy.ndimage import gaussian_filter
import sqlite3
from pathlib import Path
import logging

# Настройка логирования
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class UIData(BaseModel):
    app_id: str
    user_id: str
    tag: str
    coordinates: List[Dict[str, float]]
    screen_size: Dict[str, int]
    screenshot_base64: str
    timestamp: Optional[float] = None

class Database:
    def __init__(self, db_path: str = "analytics.db"):
        self.db_path = db_path
        self.init_db()

    def init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Таблица пользователей
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT PRIMARY KEY,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Таблица скриншотов
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS screenshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT,
                    screen_name TEXT,
                    file_path TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    is_sent BOOLEAN DEFAULT FALSE,
                    last_sent_date TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(user_id)
                )
            ''')
            
            # Таблица кликов
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS clicks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT,
                    screen_name TEXT,
                    x FLOAT,
                    y FLOAT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(user_id)
                )
            ''')
            
            conn.commit()

    def save_screenshot(self, user_id: str, screen_name: str, file_path: str):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO screenshots (user_id, screen_name, file_path)
                VALUES (?, ?, ?)
            ''', (user_id, screen_name, file_path))
            conn.commit()

    def save_click(self, user_id: str, screen_name: str, x: float, y: float):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO clicks (user_id, screen_name, x, y)
                VALUES (?, ?, ?, ?)
            ''', (user_id, screen_name, x, y))
            conn.commit()

    def get_user_screenshots(self, user_id: str) -> List[Dict]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT screen_name, file_path, created_at, is_sent, last_sent_date
                FROM screenshots
                WHERE user_id = ?
                ORDER BY created_at DESC
            ''', (user_id,))
            return [dict(zip(['screen_name', 'file_path', 'created_at', 'is_sent', 'last_sent_date'], row))
                   for row in cursor.fetchall()]

class UIAnalyzer:
    def __init__(self):
        self.app = FastAPI(title="UI Analyzer API")
        self.setup_routes()
        self.designs_folder = "page_designs"
        self.data_folder = "data"
        self.db = Database()
        
        # Создаем структуру директорий
        for folder in [self.designs_folder, self.data_folder]:
            Path(folder).mkdir(parents=True, exist_ok=True)
            
        # Создаем подпапки для каждого приложения
        for app_id in self._get_app_ids():
            Path(f"{self.designs_folder}/{app_id}").mkdir(exist_ok=True)
            Path(f"{self.data_folder}/{app_id}").mkdir(exist_ok=True)

    def _get_app_ids(self) -> List[str]:
        return [d for d in os.listdir(self.data_folder) 
                if os.path.isdir(os.path.join(self.data_folder, d))]

    def setup_routes(self):
        @self.app.post("/analyze")
        async def analyze_ui(
            app_id: str = Form(...),
            user_id: str = Form(...),
            tag: str = Form(...),
            coordinates: str = Form(...),
            screen_size: str = Form(...),
            timestamp: str = Form(...),
            screenshot_file: UploadFile = File(None)
        ):
            try:
                logger.debug(f"Получен запрос от пользователя {user_id} на странице {tag}")
                logger.debug(f"Параметры запроса: app_id={app_id}, coordinates={coordinates}, screen_size={screen_size}")
                
                # Парсим JSON данные
                coordinates_data = json.loads(coordinates)
                screen_size_data = json.loads(screen_size)
                timestamp_data = float(timestamp)

                # Сохраняем клики в БД
                for point in coordinates_data:
                    self.db.save_click(
                        user_id,
                        tag,
                        point['x'],
                        point['y']
                    )
                
                # Сохраняем скриншот если он есть
                base64_data = ''
                if screenshot_file:
                    logger.debug(f"Получен файл скриншота: {screenshot_file.filename}")
                    contents = await screenshot_file.read()
                    logger.debug(f"Размер файла: {len(contents)} байт")
                    base64_data = base64.b64encode(contents).decode('utf-8')
                    logger.debug(f"Скриншот закодирован в base64, длина: {len(base64_data)}")
                    
                    try:
                        screenshot_path = await self._save_screenshot(
                            app_id,
                            user_id,
                            tag,
                            base64_data
                        )
                        logger.debug(f"Скриншот сохранен по пути: {screenshot_path}")
                        self.db.save_screenshot(user_id, tag, screenshot_path)
                        logger.debug("Информация о скриншоте сохранена в БД")
                    except Exception as e:
                        logger.error(f"Ошибка при сохранении скриншота: {str(e)}")
                        raise
                else:
                    logger.debug("Скриншот не был предоставлен")
                
                return await self.save_click_data(UIData(
                    app_id=app_id,
                    user_id=user_id,
                    tag=tag,
                    coordinates=coordinates_data,
                    screen_size=screen_size_data,
                    screenshot_base64=base64_data,
                    timestamp=timestamp_data
                ))
            except Exception as e:
                logger.error(f"Ошибка при обработке запроса: {str(e)}")
                raise HTTPException(status_code=400, detail=str(e))

        @self.app.get("/user/{user_id}/screenshots")
        async def get_user_screenshots(user_id: str):
            try:
                screenshots = self.db.get_user_screenshots(user_id)
                return JSONResponse(screenshots)
            except Exception as e:
                raise HTTPException(
                    status_code=500,
                    detail=f"Ошибка при получении скриншотов: {str(e)}"
                )

        @self.app.post("/upload-design/{page_id}")
        async def upload_design(page_id: str, file: UploadFile = File(...)):
            try:
                if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    raise HTTPException(
                        status_code=400,
                        detail="Поддерживаются только изображения в форматах PNG и JPEG"
                    )

                file_extension = os.path.splitext(file.filename)[1]
                design_path = os.path.join(self.designs_folder, f"{page_id}{file_extension}")

                contents = await file.read()
                with open(design_path, "wb") as f:
                    f.write(contents)

                with Image.open(design_path) as img:
                    width, height = img.size

                return JSONResponse({
                    "message": "Дизайн успешно загружен",
                    "page_id": page_id,
                    "file_path": design_path,
                    "dimensions": {
                        "width": width,
                        "height": height
                    }
                })

            except Exception as e:
                raise HTTPException(
                    status_code=500,
                    detail=f"Ошибка при загрузке дизайна: {str(e)}"
                )

        @self.app.get("/heatmaps/{tag}")
        async def get_heatmap(tag: str, app_id: str = Query(...)):
            try:
                app_folder = os.path.join(self.data_folder, app_id)
                heatmap_path = os.path.join(app_folder, f"{tag}_heatmap.json")
                if not os.path.exists(heatmap_path):
                    raise HTTPException(
                        status_code=404,
                        detail=f"Тепловая карта с тегом {tag} не найдена"
                    )
                designs_folder = os.path.join(self.designs_folder, app_id)
                image_pattern = os.path.join(designs_folder, f"*{tag}*.png")
                image_files = glob.glob(image_pattern)
                if not image_files:
                    raise HTTPException(
                        status_code=404,
                        detail=f"Изображение с тегом {tag} не найдено"
                    )
                design_image = max(image_files, key=os.path.getctime)
                with open(heatmap_path, 'r') as f:
                    data = json.load(f)
                    data['designImageUrl'] = design_image
                return JSONResponse(data)
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(
                    status_code=500,
                    detail=f"Ошибка при получении тепловой карты: {str(e)}"
                )

        @self.app.get("/pages")
        async def get_all_pages(app_id: str = Query(...)):
            try:
                app_folder = os.path.join(self.data_folder, app_id)
                if not os.path.exists(app_folder):
                    return JSONResponse([])
                pages = []
                for filename in os.listdir(app_folder):
                    if filename.endswith('_heatmap.json'):
                        with open(os.path.join(app_folder, filename), 'r') as f:
                            data = json.load(f)
                            tag = data['tag']
                            pages.append({
                                'tag': tag,
                                'createdAt': data['createdAt'],
                                'updatedAt': data['updatedAt']
                            })
                return JSONResponse(pages)
            except Exception as e:
                raise HTTPException(
                    status_code=500,
                    detail=f"Ошибка при получении списка страниц: {str(e)}"
                )

    async def save_click_data(self, data: UIData) -> Dict:
        try:
            app_folder = os.path.join(self.data_folder, data.app_id)
            os.makedirs(app_folder, exist_ok=True)
            heatmap_path = os.path.join(app_folder, f"{data.tag}_heatmap.json")
            if os.path.exists(heatmap_path):
                with open(heatmap_path, 'r') as f:
                    heatmap_data = json.load(f)
                points = heatmap_data.get('points', [])
                new_points = [{
                    'x': point['x'],
                    'y': point['y'],
                    'timestamp': datetime.now().isoformat()
                } for point in data.coordinates]
                points.extend(new_points)
                heatmap_data.update({
                    'points': points,
                    'updatedAt': datetime.now().isoformat()
                })
            else:
                heatmap_data = {
                    'tag': data.tag,
                    'points': [{
                        'x': point['x'],
                        'y': point['y'],
                        'timestamp': datetime.now().isoformat()
                    } for point in data.coordinates],
                    'screenSize': data.screen_size,
                    'createdAt': datetime.now().isoformat(),
                    'updatedAt': datetime.now().isoformat()
                }
            with open(heatmap_path, 'w') as f:
                json.dump(heatmap_data, f, indent=2)
            return {
                "message": "Данные успешно сохранены",
                "total_points": len(heatmap_data['points'])
            }
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Ошибка при сохранении данных: {str(e)}"
            )

    def generate_heatmap(self, coordinates: list, screen_size: dict) -> np.ndarray:
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

    async def _save_screenshot(self, app_id: str, user_id: str, screen_name: str, base64_data: str) -> str:
        try:
            logger.debug(f"Начало сохранения скриншота: app_id={app_id}, user_id={user_id}, screen_name={screen_name}")
            
            # Декодируем base64
            image_data = base64.b64decode(base64_data)
            logger.debug(f"Декодирован base64, размер данных: {len(image_data)} байт")
            
            # Создаем директорию для скриншотов
            screenshots_dir = Path(f"{self.designs_folder}/{app_id}/screenshots")
            screenshots_dir.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Создана директория: {screenshots_dir}")
            
            # Генерируем имя файла
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{screen_name}_{user_id}_{timestamp}.png"
            file_path = screenshots_dir / filename
            logger.debug(f"Сгенерирован путь к файлу: {file_path}")
            
            # Сохраняем изображение
            with open(file_path, "wb") as f:
                f.write(image_data)
            logger.debug(f"Файл успешно сохранен: {file_path}")
            
            return str(file_path)
        except Exception as e:
            logger.error(f"Ошибка при сохранении скриншота: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Ошибка при сохранении скриншота: {str(e)}"
            )

def main():
    analyzer = UIAnalyzer()
    import uvicorn
    uvicorn.run(analyzer.app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main() 