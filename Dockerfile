FROM python:3.11-slim

WORKDIR /app

COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ ./backend/
COPY model/model.pth ./model/model.pth
COPY model/label_map.json ./model/label_map.json
COPY frontend/ ./frontend/

WORKDIR /app/backend

EXPOSE 7860

CMD ["python", "app.py"]