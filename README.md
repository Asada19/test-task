## Требования
- Python 3.11+ или Docker

## Запуск

### Обычная установка
1. Создайте виртуальное окружение:
   ```bash
   python -m venv .venv
   ```

2. Активируйте виртуальное окружение:
   - **macOS/Linux**:
     ```bash
     source .venv/bin/activate
     ```
   - **Windows**:
     ```bash
     .venv\Scripts\activate
     ```

3. Установите зависимости:
   ```bash
   pip install -r requirements.txt
   ```

4. Создайте файл `.env`:
   ```
   ANTHROPIC_API_KEY=your_api_key_here
   ```

5. Запустите:
   ```bash
   langgraph dev
   ```

### Запуск через Docker
1. Создайте файл `.env`:
   ```
   ANTHROPIC_API_KEY=your_api_key_here
   ```

2. Соберите Docker образ:
   ```bash
   docker build -t chatbot .
   ```

3. Запустите контейнер:
   ```bash
   docker run -p 8123:8123 --env-file .env chatbot
   ```

4. Откройте в браузере: https://smith.langchain.com/studio/?baseUrl=http://0.0.0.0:8123

> **Примечание**: Если ваш основной браузер Safari, используйте флаг `--tunel`:
> ```bash
> langgraph dev --tunel
> ```
