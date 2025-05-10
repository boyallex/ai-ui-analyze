import os
import tensorflow as tf
from openai import OpenAI
from dotenv import load_dotenv

# Загрузка переменных окружения
load_dotenv()

class NeuralNetwork:
    def __init__(self):
        self.model = None
    
    def create_model(self):
        """Создание базовой модели нейронной сети"""
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu', input_shape=(784,)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(10, activation='softmax')
        ])
        
        self.model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        return self.model

class ChatGPT:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    
    def get_response(self, prompt):
        """Получение ответа от ChatGPT"""
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Ошибка при обращении к ChatGPT: {str(e)}"

def main():
    # Пример использования нейронной сети
    nn = NeuralNetwork()
    model = nn.create_model()
    print("Модель нейронной сети создана успешно!")
    
    # Пример использования ChatGPT
    chat = ChatGPT()
    response = chat.get_response("Привет! Как дела?")
    print("Ответ от ChatGPT:", response)

if __name__ == "__main__":
    main() 