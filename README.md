# HDBank Risk Management System

Hệ thống quản lý rủi ro cho HDBank với kiến trúc microservices.

## Kiến trúc

- **Gateway**: Node.js API Gateway (Port 3000)
- **Model**: Python FastAPI ML Service (Port 8000)

## Cài đặt và chạy

### Với Docker Compose (Khuyến nghị)

```bash
docker compose up --build
```

### Chạy riêng từng service

#### Gateway (Node.js)
```bash
cd gateway
npm install
npm start
```

#### Model (Python)
```bash
cd model
python -m uvicorn app:app --reload --port 8000
```

## API Endpoints

- Gateway: http://localhost:3000
- Model Service: http://localhost:8000

## Công nghệ sử dụng

- **Backend**: FastAPI (Python), Node.js
- **Containerization**: Docker, Docker Compose
- **ML/AI**: TensorFlow, scikit-learn, pandas

## Cấu trúc thư mục

```
hdbank-risk/
├── gateway/          # Node.js API Gateway
│   ├── src/
│   └── package.json
├── model/           # Python ML Service
│   ├── app.py
│   └── Dockerfile
├── docker-compose.yml
└── README.md
```

## Yêu cầu hệ thống

- Docker Desktop
- Node.js 16+
- Python 3.11+
