# Sơ đồ Kiến trúc Hệ thống Tổng thể (System Architecture)

Tài liệu này chứa sơ đồ khối mô tả kiến trúc phần mềm Client-Server của dự án (phục vụ cho Mục 2.1 của Báo cáo). 
Hệ thống được thiết kế theo kiến trúc Microservices đơn giản, phân tách rõ ràng trách nhiệm của từng tầng.

```mermaid
graph TD
    %% User
    User(("Người dùng<br/>(QC Inspector)"))
    
    %% Frontend
    subgraph UI_Layer ["Tầng Trình diễn (Frontend)"]
        React["ReactJS Dashboard<br/>(Vite, Port: 5173)"]
    end
    
    %% API Gateway
    subgraph API_Layer ["Tầng Điều phối (API Gateway)"]
        NodeJS["Node.js Server<br/>(Express, Port: 3000)"]
        Storage[("Local Storage<br/>(Lưu ảnh tạm)")]
    end
    
    %% ML Service
    subgraph Core_Layer ["Tầng Trí tuệ Nhân tạo (ML Service)"]
        FastAPI["Python FastAPI<br/>(Uvicorn, Port: 8000)"]
        CV["Core Vision Lab<br/>(Morph/Directional)"]
        AI["Machine Learning Lab<br/>(SVM/RF)"]
        
        FastAPI --> CV
        CV --> AI
    end
    
    %% Data flow connections
    User -- "1. Upload ảnh bề mặt" --> React
    React -- "2. Gửi file qua REST API" --> NodeJS
    NodeJS -- "3. Lưu ảnh" --> Storage
    NodeJS -- "4. Bắn Request phân tích" --> FastAPI
    AI -- "5. Trả JSON Kết quả (Nhãn, Đặc trưng)" --> FastAPI
    FastAPI -- "6. Response Data" --> NodeJS
    NodeJS -- "7. Render lên UI" --> React
    React -- "8. Xem kết quả trực quan" --> User

    %% Styling
    classDef frontend fill:#61DAFB,stroke:#333,stroke-width:2px,color:#000;
    classDef nodejs fill:#339933,stroke:#333,stroke-width:2px,color:#fff;
    classDef python fill:#FFD43B,stroke:#333,stroke-width:2px,color:#000;
    
    class React frontend;
    class NodeJS,Storage nodejs;
    class FastAPI,CV,AI python;
```

## Diễn giải Sơ đồ
1. **Tầng Frontend:** Giao diện người dùng được xây dựng bằng ReactJS. Đóng vai trò tương tác trực tiếp với nhân viên kiểm định (QC), cho phép họ tải ảnh lỗi lên và xem kết quả báo cáo trực quan.
2. **Tầng API Gateway (Node.js):** Đóng vai trò là trạm trung chuyển (Router) và quản lý tải trọng (Payload). Nó nhận file từ Frontend, lưu trữ tạm thời và gửi lệnh kích hoạt luồng AI bên dưới. Việc tách Node.js giúp Frontend không phải gọi trực tiếp vào lõi AI, tăng cường bảo mật và dễ dàng mở rộng Database (MongoDB/MySQL) sau này.
3. **Tầng ML Service (FastAPI):** Là "bộ não" của hệ thống viết bằng Python. FastAPI nhận ảnh từ Node.js, đẩy qua mảng Xử lý ảnh (Core Vision) để trích xuất đặc trưng, rồi đưa vào mô hình học máy (Machine Learning) để dự đoán. Kết quả cuối cùng được đóng gói thành JSON và trả ngược về chuỗi.
