FROM python:3.12-slim-bookworm

# 작업 디렉토리 설정
WORKDIR /app

# 의존성 파일들을 컨테이너에 복사
COPY requirements.txt .

# 의존성 설치
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install python-multipart

# Flask 앱 코드를 컨테이너에 복사
COPY . /app

# CMD ["fastapi", "run", "FastAPI.py", "--port", "80"]
CMD ["uvicorn", "main:app", "--port", "8000", "--host", "0.0.0.0", "--log-level", "debug"]