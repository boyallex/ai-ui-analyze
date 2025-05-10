#!/bin/bash

# Цвета для вывода
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Функция для логирования
log() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

# Функция для обработки ошибок
handle_error() {
    log "${RED}Ошибка: $1${NC}"
    exit 1
}

# Проверяем наличие скрипта анализа
if [ ! -f "analyze_data.py" ]; then
    handle_error "Скрипт analyze_data.py не найден"
fi

# Проверяем наличие виртуального окружения
if [ ! -d "venv" ]; then
    handle_error "Виртуальное окружение не найдено. Сначала запустите setup.sh для создания виртуального окружения"
fi

# Активируем виртуальное окружение
log "${GREEN}Активация виртуального окружения...${NC}"
source venv/bin/activate || handle_error "Не удалось активировать виртуальное окружение"

# Создаем директории для аналитики
log "${GREEN}Создание директорий для аналитики...${NC}"
mkdir -p analytics/individual analytics/combined page_designs data/consilium || handle_error "Не удалось создать директории"

# Список экранов для анализа
screens=(
    "DialogsScreen"
    "WorldScreen"
    "ProfileScreen"
    "PublicProfileScreen"
    "NotificationsScreen"
    "AudioRoomCreator"
    "CreateWorldPost"
    "CommentsScreen"
    "AudioRoom"
    "ChatScreen"
    "MainScreen"
)

# Проверяем наличие данных
log "${YELLOW}Проверка наличия данных...${NC}"
missing_data=false
available_screens=()

for screen in "${screens[@]}"; do
    data_file="data/consilium/${screen}_heatmap.json"
    
    if [ ! -f "$data_file" ]; then
        log "${RED}Файл с данными не найден: $data_file${NC}"
        missing_data=true
    else
        log "${GREEN}Найден файл данных: $data_file${NC}"
        available_screens+=("$screen")
    fi
done

if [ "$missing_data" = true ]; then
    log "\n${YELLOW}Внимание: Некоторые файлы с данными отсутствуют.${NC}"
    log "Для сбора данных:"
    log "1. Запустите Flutter приложение"
    log "2. Перейдите на нужные экраны и совершите несколько кликов"
    log "3. Данные будут сохранены в папке data/consilium/"
    log -e "\nХотите продолжить анализ только для имеющихся файлов? (y/n)"
    read -r answer
    if [ "$answer" != "y" ]; then
        log "Анализ отменен"
        exit 1
    fi
fi

# Запускаем анализ для каждого экрана
log "\n${GREEN}Запуск индивидуального анализа экранов...${NC}"
for screen in "${available_screens[@]}"; do
    data_file="data/consilium/${screen}_heatmap.json"
    
    if [ -f "$data_file" ]; then
        log "\n${GREEN}Анализ экрана: $screen${NC}"
        log "----------------------------------------"
        
        # Запускаем анализ и сохраняем результат
        output_file="analytics/individual/${screen}_analysis.txt"
        
        # Выполняем анализ и сохраняем результат во временный файл
        temp_output=$(python3 analyze_data.py --tag "$screen" 2> "analytics/individual/${screen}_error.log")
        
        if [ $? -eq 0 ]; then
            # Выводим результат в консоль
            log "${GREEN}Результат анализа для $screen:${NC}"
            echo -e "${temp_output}"
            
            # Сохраняем результат в файл
            echo "$temp_output" > "$output_file"
            log "${GREEN}Анализ сохранен в: $output_file${NC}"
        else
            log "${RED}Ошибка при анализе экрана $screen. Проверьте лог: analytics/individual/${screen}_error.log${NC}"
        fi
    fi
done

# Запускаем общий анализ всех экранов
log "\n${GREEN}Запуск общего анализа всех экранов...${NC}"
log "----------------------------------------"

# Создаем временный файл со списком всех доступных экранов
echo "${available_screens[@]}" > "analytics/combined/screens_list.txt"

# Запускаем общий анализ
output_file="analytics/combined/overall_analysis.txt"

# Выполняем общий анализ и сохраняем результат во временный файл
temp_output=$(python3 analyze_data.py 2> "analytics/combined/overall_error.log")

if [ $? -eq 0 ]; then
    # Выводим результат в консоль
    log "${GREEN}Результат общего анализа:${NC}"
    echo -e "${temp_output}"
    
    # Сохраняем результат в файл
    echo "$temp_output" > "$output_file"
    log "${GREEN}Общий анализ сохранен в: $output_file${NC}"
else
    log "${RED}Ошибка при общем анализе. Проверьте лог: analytics/combined/overall_error.log${NC}"
fi

# Деактивируем виртуальное окружение
deactivate

log "\n${GREEN}Анализ завершен!${NC}"
log "Результаты сохранены в папках:"
log "- page_designs/ (тепловые карты и маски)"
log "- data/consilium/ (данные о кликах)"
log "- analytics/individual/ (индивидуальный анализ каждого экрана)"
log "- analytics/combined/ (общий анализ всех экранов)" 