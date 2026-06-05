# Backend Microservices Setup Guide

Hệ thống Backend được chia làm 2 microservices:
1. **Node.js API Gateway** (Express.js - Port 3000)
2. **Python ML Service** (FastAPI - Port 8000)

## 1. Yêu cầu Hệ thống
- Node.js (v18+)
- Python (v3.9+)
- PostgreSQL (Đã cài đặt và đang chạy)
- Nginx

## 2. Cài đặt Database (PostgreSQL)
1. Hãy tạo database có tên `fabric_defect_db` trong PostgreSQL.
2. Cập nhật connection string trong file `app/backend/api_gateway/.env`:
   ```env
   DATABASE_URL="postgresql://<user>:<password>@localhost:5432/fabric_defect_db?schema=public"
   ```
3. Chạy lệnh sau để push schema vào database và sinh code Prisma client:
   ```bash
   cd api_gateway
   npm install
   npx prisma db push
   ```
*(Lưu ý: Nếu cần lấy file raw SQL để tự export/import sang server khác, hãy truy cập công cụ database client của bạn sau khi chạy lệnh trên)*

## 3. Chạy Node.js API Gateway
Service này phụ trách nhận Request từ người dùng, upload ảnh, gọi ML service, và lưu lịch sử.

```bash
cd api_gateway
npm install
npm run start
```
*API Gateway sẽ chạy ở `http://localhost:3000`*

## 4. Chạy Python ML Service
Service này phụ trách chạy Inference AI với các file đã được đóng băng.

```bash
cd ml_service
pip install -r requirements.txt
python main.py
```
*ML Service sẽ chạy ở `http://localhost:8000`*

## 5. Cấu hình Nginx
DevOps vui lòng copy cấu hình Nginx trong file `nginx.conf` và điều chỉnh.
**Lưu ý quan trọng:** Cập nhật đường dẫn tuyệt đối cho `alias` trong `location /uploads/` cho phù hợp với server thực tế để Nginx có thể render ảnh tĩnh với tốc độ cao.

---
**Troubleshooting:**
- Lỗi CORS: Hãy kiểm tra file `api_gateway/index.js`, danh sách các Domain được whitelist đang bao gồm `localhost:3000`, `localhost:5173`, và domain Storage Account của Azure.
- File model không tải được: Kiểm tra đường dẫn thư mục `models/` ở gốc có bị thay đổi vị trí so với `ml_service` không.
