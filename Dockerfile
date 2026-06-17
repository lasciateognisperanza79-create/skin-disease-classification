# Используем официальный образ Python
FROM python:3.9-slim

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем файл зависимостей и устанавливаем их
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем код приложения и модель
COPY app.py .
COPY skin_model.keras .

# Открываем порт для Streamlit
EXPOSE 8501

# Запускаем приложение
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]