# Sơ đồ Luồng dữ liệu (Dataflow) - A/B Testing

Đây là sơ đồ kiến trúc hệ thống phản ánh chiến lược tách biệt 2 luồng xử lý (Morphological và Directional Gradient) để phục vụ cho việc huấn luyện Machine Learning và so sánh chéo hiệu năng phát hiện lỗi bề mặt vải.

```mermaid
graph TD
    %% Input Layer
    Input["Raw Fabric Image"] --> Gray["Grayscale Conversion"]
    
    %% Preprocessing
    subgraph S1 ["Tiền Xử Lý Chung (OOP)"]
        Gray --> Median["Median Filter (Khử nhiễu)"]
        Median --> TopHat["Illumination Correction<br>TopHat + BlackHat"]
        TopHat --> StatThresh["Statistical Thresholding<br>(Mean ± k*Std)"]
    end

    %% Split into two branches
    StatThresh -->|Mask Nhị Phân| BranchA{"Nhánh A:<br>Morphological"}
    Median -->|Ảnh Xám Median| BranchB{"Nhánh B:<br>Directional Gradient"}

    %% Branch A: Morphological
    subgraph S2 ["Nhánh A: Hình thái khối (Blobs)"]
        BranchA --> MorphOp["Morphological Operations"]
        MorphOp --> Opening("Opening (K:3x3): Tẩy đốm nhiễu")
        Opening --> Closing("Cascading Closing (K:5x5): Lấp khối lỗi")
        Closing --> ExtractA["Blob Feature Extractor"]
        ExtractA --> FeatA("Max/Total Area,<br>Min Eccentricity")
    end

    %% Branch B: Gradient Magnitude
    subgraph S3 ["Nhánh B: Cấu trúc Tuyến tính (Lines)"]
        BranchB --> Gradient["Gaussian Smoothing & Sobel Gradient"]
        Gradient --> StatThreshB["Statistical Thresholding<br>(Mean + k*Std)"]
        StatThreshB --> MorphOpenB("Morphological Opening<br>(Lọc đốm nhiễu)")
        MorphOpenB --> SplitAngle{"Lượng tử hóa Hướng"}
        SplitAngle --> DirFilterH["Horizontal Mask<br>& Directional Closing"]
        SplitAngle --> DirFilterV["Vertical Mask<br>& Directional Closing"]
        DirFilterH --> ExtractB["Line Feature Extractor"]
        DirFilterV --> ExtractB
        ExtractB --> FeatB("Horiz_Length, Vert_Length,<br>Diag_Length")
    end

    %% CSV Data Output (Bridges)
    FeatA --> CSV_A[("morph_features.csv")]
    FeatB --> CSV_B[("directional_features.csv")]

    %% Machine Learning Phase
    subgraph S4 ["Huấn luyện & Đánh giá (A/B Testing)"]
        CSV_A --> ML_A["Machine Learning Lab A"]
        CSV_B --> ML_B["Machine Learning Lab B"]
        ML_A --> TrainA("Train SVM / Random Forest")
        ML_B --> TrainB("Train SVM / Random Forest")
        TrainA --> EvalA["Phân tích Hiệu năng<br>(Confusion Matrix)"]
        TrainB --> EvalA
    end

    EvalA --> Report["Kết luận: Morphological ưu việt trên nền vải dệt"]
    
    %% Styling
    classDef branchA fill:#d4edda,stroke:#28a745,stroke-width:2px;
    classDef branchB fill:#cce5ff,stroke:#007bff,stroke-width:2px;
    classDef mlPhase fill:#fff3cd,stroke:#ffc107,stroke-width:2px;
    
    class BranchA,CSV_A,ExtractA,FeatA branchA;
    class BranchB,CSV_B,ExtractB,FeatB branchB;
    class ML_A,ML_B,TrainA,TrainB mlPhase;
```

## Diễn giải Sơ đồ

1. **Tiền xử lý chung:** Toàn bộ ảnh sẽ đi qua bộ lọc Trung vị (khử hạt) và phép biến đổi Top-Hat/Black-Hat để triệt tiêu hoàn toàn sự thiếu đồng đều của ánh sáng trên bề mặt vải.
2. **Nhánh A (Bên Trái):** Chuyên trách các lỗi dạng Đốm/Khối. Áp dụng Morphological (Opening/Closing) để lấp lỗ và tính toán Diện tích (Area), Chu vi (Perimeter). Đầu ra lưu vào tập dữ liệu riêng biệt `morph_features.csv`.
3. **Nhánh B (Bên Phải):** Chuyên trách các lỗi Xước/Đứt sợi. Thay thế Directional Gradient cũ bằng luồng Gradient Magnitude. Dùng đạo hàm Sobel kết hợp Ngưỡng thống kê để bắt lỗi tuyến tính. Sau đó dùng Lọc hình thái học có hướng (Directional Morphological) để nối liền vệt đứt gãy. Đầu ra đếm số pixel đứt gãy lưu vào `directional_features.csv`.
4. **Machine Learning:** Đưa 2 tập dữ liệu này vào huấn luyện độc lập. Cuối cùng, sinh ra 2 Ma trận nhầm lẫn (Confusion Matrix) để bảo vệ luận điểm khoa học trước hội đồng: Nhánh nào tối ưu cho loại khuyết tật nào.