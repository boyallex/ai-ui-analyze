#!/bin/bash

# Цвета для вывода
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}Начало установки UI Analyzer Backend${NC}"

# Проверка Python
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Ошибка: Python3 не установлен${NC}"
    exit 1
fi

# Создание виртуального окружения
echo -e "${GREEN}Создание виртуального окружения...${NC}"
python3 -m venv venv
source venv/bin/activate

# Обновление pip
echo -e "${GREEN}Обновление pip...${NC}"
pip install --upgrade pip

# Установка системных зависимостей
echo -e "${GREEN}Установка системных зависимостей...${NC}"
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    if ! command -v brew &> /dev/null; then
        echo -e "${RED}Ошибка: Homebrew не установлен${NC}"
        exit 1
    fi
    brew install libjpeg libpng gcc
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Linux
    if command -v apt-get &> /dev/null; then
        sudo apt-get update
        sudo apt-get install -y libjpeg-dev zlib1g-dev gcc gfortran python3-dev
    elif command -v yum &> /dev/null; then
        sudo yum install -y libjpeg-devel zlib-devel gcc gfortran python3-devel
    else
        echo -e "${RED}Ошибка: Неизвестный менеджер пакетов${NC}"
        exit 1
    fi
fi

# Установка Python зависимостей
echo -e "${GREEN}Установка Python зависимостей...${NC}"
pip install -r requirements.txt

# Создание необходимых директорий
echo -e "${GREEN}Создание директорий...${NC}"
mkdir -p page_designs data

echo -e "${GREEN}Установка завершена!${NC}"
echo -e "${GREEN}Для запуска сервера выполните:${NC}"
echo -e "${GREEN}source venv/bin/activate${NC}"
echo -e "${GREEN}python3 ui_analyzer.py${NC}" 