#!/bin/bash

# Проверка количества аргументов
if [ "$#" -ne 2 ]; then
    echo "Использование: $0 <путь_к_изображению> <page_id>"
    echo "Пример: $0 ./designs/main_screen.png main_screen"
    exit 1
fi

IMAGE_PATH=$1
PAGE_ID=$2
BACKEND_URL="http://localhost:8000"

# Проверка существования файла
if [ ! -f "$IMAGE_PATH" ]; then
    echo "Ошибка: файл $IMAGE_PATH не существует"
    exit 1
fi

# Проверка расширения файла
if ! [[ "$IMAGE_PATH" =~ \.(png|jpg|jpeg)$ ]]; then
    echo "Ошибка: поддерживаются только файлы с расширениями .png, .jpg или .jpeg"
    exit 1
fi

# Загрузка файла
echo "Загрузка дизайна для страницы $PAGE_ID..."
response=$(curl -s -X POST -F "file=@$IMAGE_PATH" "$BACKEND_URL/upload-design/$PAGE_ID")

# Проверка ответа
if echo "$response" | grep -q "Дизайн успешно загружен"; then
    echo "Дизайн успешно загружен!"
    echo "Ответ сервера: $response"
else
    echo "Ошибка при загрузке дизайна:"
    echo "$response"
    exit 1
fi 