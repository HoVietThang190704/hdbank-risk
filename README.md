 # HDBank Risk Management System

Hệ thống thử nghiệm chấm điểm rủi ro tín dụng (PD – Probability of Default) dựa trên 2 nguồn dữ liệu công khai (Taiwan Credit Default & LendingClub) và cơ chế ensemble linh hoạt thông qua một API Gateway. Dự án mang tính minh hoạ kiến trúc microservices.

## 1. Kiến trúc tổng quan

```
┌────────────────────┐      ┌────────────────────┐
│  model_taiwan      │      │ model_lendingclub  │
│ (FastAPI + artifacts)     │ (FastAPI + artifacts)
└─────────▲──────────┘      └─────────▲──────────┘
		│                             │
		└─────── HTTP predict ────────┘
							(song song)
									│
							 ┌──────▼──────┐
							 │  gateway    │  (Ensemble: avg | weighted | tw | lc)
							 └──────▲──────┘
											│
								Client / UI / Batch
```

Services chính:
- `model_taiwan`: phục vụ mô hình logistic được train từ bộ Taiwan.
- `model_lendingclub`: phục vụ mô hình logistic từ dữ liệu LendingClub.
- `gateway`: gọi song song 2 model, hợp nhất PD theo chiến lược cấu hình (trung bình, trọng số theo AUC, hoặc chọn 1 model).

## 2. Đặc trưng & nhãn được dùng để train

Mỗi mẫu (application) sau tiền xử lý có 4 đặc trưng đầu vào thống nhất:
| Feature | Mô tả | Nguồn tạo (Taiwan) | Nguồn tạo (LendingClub) |
|---------|-------|--------------------|-------------------------|
| age | Tuổi (hoặc proxy) | AGE | 22 + emp_length (chuẩn hoá) |
| income | Thu nhập/tháng ước lượng | LIMIT_BAL | annual_inc / 12 |
| liabilities | Dư nợ bình quân | mean(BILL_AMT*) | loan_amnt |
| credit_history_months | Số tháng lịch sử tín dụng | 6 (giá trị giả lập) | issue_d - earliest_cr_line (tháng) |

Label: `target_default_12m` (1 = default / charged off / late nặng, 0 = tốt).

## 3. Quy trình huấn luyện

Thực hiện bằng script `training/train.py` với các bước:
1. Đọc file nguồn CSV (hoặc XLSX) theo `--flavor` (taiwan | lendingclub | custom).
2. Map về schema chung 4 feature + 1 label.
3. Train Pipeline: `StandardScaler` → `LogisticRegression(class_weight='balanced', max_iter=2000)`.
4. Đánh giá AUC sơ cấp (raw) trên tập test (stratified split).
5. Hiệu chỉnh xác suất bằng `CalibratedClassifierCV` (isotonic mặc định) → AUC calibrated, KS, Gini.
6. Lưu artifacts: `pd_model.pkl`, `calibrator.pkl`, `feature_order.json`, `metrics.json` vào `artifacts/<flavor>`.
7. Service online nạp và ưu tiên dùng calibrator để dự đoán PD chuẩn hoá.

Script hợp nhất tuỳ chọn: `training/preprocess_merge.py` ghép 2 nguồn thành `data/credit_risk_merged.csv` (dùng cho flavor `custom`).

## 4. Artifacts

Mỗi thư mục `artifacts/<flavor>` chứa:
- `pd_model.pkl` (Pipeline logistic gốc)
- `calibrator.pkl` (Sau hiệu chỉnh xác suất)
- `feature_order.json` (Thứ tự features: ["age","income","liabilities","credit_history_months"])
- `metrics.json` (auc_raw, auc_calibrated, ks, gini, flavor)
- `scaler.pkl`, `encoder.pkl` (placeholder – hiện không dùng)

Container model mount thư mục artifacts tương ứng ở đường dẫn chỉ định bởi biến `ARTIFACTS_DIR` (mặc định `/models`).

## 5. Chạy nhanh với Docker Compose

```powershell
docker compose up --build -d
docker compose ps
```

Các cổng:
- Gateway: http://localhost:3000
- Model Taiwan: http://localhost:8001
- Model LendingClub: http://localhost:8002

Healthcheck dùng `wget` nội bộ; hình ảnh base đã cài sẵn (xem `model/Dockerfile`).

Gỡ container mồ côi nếu đổi tên service:
```powershell
docker compose up -d --remove-orphans
```

## 6. Huấn luyện lại mô hình (tạo mới artifacts)

Tạo virtual env (tùy chọn) rồi cài requirements:
```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r training/requirements.txt
```

Ví dụ train Taiwan:
```powershell
python training/train.py --flavor taiwan --data data/taiwan_credit_default.csv --artifacts artifacts/taiwan
```

Ví dụ train LendingClub (giới hạn 150k dòng để nhanh):
```powershell
python training/train.py --flavor lendingclub --data data/lendingclub/accepted_2007_to_2018Q4.csv --artifacts artifacts/lendingclub --max_rows 150000
```

Sau khi train xong, rebuild lại container model:
```powershell
docker compose build model_taiwan model_lendingclub
docker compose up -d
```

## 7. Endpoints chi tiết

### Model Service (mỗi flavor)
- `GET /v1/healthz` – kiểm tra nạp artifacts (trả missing nếu thiếu file).  
- `GET /v1/metadata` – trả `feature_order`, `metrics`, `model_version`.  
- `POST /v1/predict` – body mẫu:
```json
{
	"customer_id": "abc123",
	"age": 35,
	"income": 5000,
	"liabilities": 12000,
	"credit_history_months": 24
}
```
Trả về: `pd` (xác suất default), `top_reasons` (gợi ý đơn giản), `latency_ms`.

### Gateway
- `GET /ensemble/weights` – xem chiến lược hiện tại, trọng số, metrics nguồn.
- `POST /applications/score?strategy=avg|weighted|tw|lc` – hợp nhất PD.

Ví dụ gọi gateway:
```powershell
curl -X POST http://localhost:3000/applications/score ^
	-H "Content-Type: application/json" ^
	-d '{"customer_id":"demo","age":40,"income":6000,"liabilities":10000,"credit_history_months":30}'
```

Trả JSON (rút gọn):
```json
{
	"strategy": "weighted",
	"pd": 0.1345,
	"weights": {"tw":0.52,"lc":0.48,"updated_at": 1730000000000},
	"components": { "taiwan": {"pd":0.15}, "lendingclub": {"pd":0.12} },
	"reasons": [ {"feature":"liabilities","impact":12000}, ... ],
	"latency_ms": 43
}
```

## 8. Biến môi trường quan trọng

Gateway:
- `MODEL_TAIWAN_URL` (mặc định http://model_taiwan:8000)
- `MODEL_LC_URL` (mặc định http://model_lendingclub:8000)
- `ENSEMBLE_REFRESH_SEC` (mặc định 600 giây; định kỳ tải lại metadata → cập nhật trọng số khi strategy=weighted)
- `ENSEMBLE_STRATEGY` (weighted | avg | tw | lc)

Model:
- `ARTIFACTS_DIR` – nơi mount các file `*.pkl`, `feature_order.json`, `metrics.json`.

## 9. Ensemble strategy
- `tw` / `lc`: dùng trực tiếp 1 model.
- `avg`: trung bình tuyến tính 2 PD.
- `weighted` (mặc định): kết hợp theo logit với trọng số tỉ lệ AUC (ưu tiên mô hình tốt hơn).

## 10. Khắc phục sự cố (Troubleshooting)
| Vấn đề | Nguyên nhân khả dĩ | Cách xử lý |
|--------|--------------------|------------|
| Container model unhealthy | Thiếu artifacts hoặc healthcheck không chạy | Kiểm tra `artifacts/<flavor>` đủ file, xem `docker logs hdbank-risk-model-...` |
| 404 /v1/predict | Service chưa ready | Đợi healthcheck ok hoặc giảm `retries`/`interval` nếu cần |
| PD lạ (toàn 0.5) | AUC thấp dữ liệu không đủ | Kiểm tra dữ liệu nguồn & feature engineering |
| Ensemble không cập nhật | `ENSEMBLE_REFRESH_SEC` quá lớn hoặc lỗi metadata | Gọi thủ công 2 endpoint model để chắc chắn trả `metrics` |

## 11. Công nghệ sử dụng
- FastAPI, Uvicorn
- scikit-learn, numpy, pandas
- Express.js, axios
- Docker, Docker Compose

## 12. Cấu trúc thư mục (rút gọn)
```
artifacts/
	taiwan/            # artifacts sau train flavor taiwan
	lendingclub/       # artifacts sau train flavor lendingclub
data/                # dữ liệu nguồn (không commit file lớn thực tế)
gateway/             # API Gateway (Node.js Express)
model/               # FastAPI model service
training/            # Scripts huấn luyện & tiền xử lý
docker-compose.yml
README.md
```

## 13. Giới hạn hiện tại & ý tưởng mở rộng
- Mới dùng 4 feature tối giản → cần mở rộng đặc trưng thực tế (lịch sử giao dịch, hành vi...).
- Chưa có tracking / versioning (gợi ý: MLflow hoặc DVC).
- Calibration dùng k-fold isotonic có thể chậm với tập rất lớn → tách hold-out hoặc giảm cv.
- Chưa có auth / rate limit nâng cao (mới cơ bản). Có thể tích hợp JWT / API key.
- Lý do (explanation) chỉ là placeholder; có thể tích hợp SHAP / feature contribution thật.

## 14. License & ghi chú
Mục đích học tập / demo. Không dùng trực tiếp cho quyết định tín dụng sản xuất nếu chưa qua kiểm định, kiểm toán mô hình và tuân thủ pháp lý (fair lending, explainability, privacy...).

---
Nếu cần thêm phần benchmark, test tự động, hoặc bổ sung explainability chi tiết hãy tạo issue hoặc yêu cầu tiếp.
