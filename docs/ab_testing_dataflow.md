# Sơ đồ Luồng dữ liệu (Dataflow) - A/B Testing

Đây là sơ đồ kiến trúc hệ thống phản ánh chiến lược tách biệt 2 luồng xử lý (Morphological và Canny Edge) để phục vụ cho việc huấn luyện Machine Learning và so sánh chéo hiệu năng phát hiện lỗi bề mặt vải.

```mermaid
graph TD
    %% Input Layer
    Input["Raw Fabric Image"] --> Preprocess["Preprocessing Module"]
    
    %% Preprocessing
    subgraph S1 ["Tiền Xử Lý Chung"]
        Preprocess --> Gray["Grayscale Conversion"]
        Gray --> Median["Median Filter"]
        Median --> TopHat["Top-Hat / Black-Hat Transform<br>Cân bằng ánh sáng dệt"]
    end

    %% Split into two branches
    TopHat --> BranchA{"Nhánh A: Morphological"}
    TopHat --> BranchB{"Nhánh B: Canny Edge"}

    %% Branch A: Morphological
    subgraph S2 ["Nhánh A: Hình thái khối (Blobs)"]
        BranchA --> Otsu["Otsu / Statistical Binarization"]
        Otsu --> MorphOp["Morphological Operations"]
        MorphOp --> Opening("Opening: Tẩy đốm nhiễu / Tách Stain")
        MorphOp --> Closing("Closing: Vá lỗ thủng / Lấp Hole")
        Opening --> ExtractA["Trích xuất Đặc trưng Khối"]
        Closing --> ExtractA
        ExtractA --> FeatA("Max Area, Perimeter, Eccentricity")
        FeatA --> CSV_A[("morph_features.csv")]
    end

    %% Branch B: Canny Edge
    subgraph S3 ["Nhánh B: Cấu trúc Tuyến tính (Lines)"]
        BranchB --> Canny["Canny Edge Detection"]
        Canny --> Gradient["Calculate Gradient Angle"]
        Gradient --> SplitAngle{"Lượng tử hóa Hướng"}
        SplitAngle --> HorizMask["Horizontal Mask 0°"]
        SplitAngle --> VertMask["Vertical Mask 90°"]
        HorizMask --> DirFilterH["Directional Morph<br>SE Line 1x15"]
        VertMask --> DirFilterV["Directional Morph<br>SE Line 15x1"]
        DirFilterH --> ExtractB["Trích xuất Độ dài Tuyến tính"]
        DirFilterV --> ExtractB
        ExtractB --> FeatB("Horiz_Length, Vert_Length")
        FeatB --> CSV_B[("canny_features.csv")]
    end

    %% Machine Learning Phase
    CSV_A --> ML_A["Machine Learning Lab A"]
    CSV_B --> ML_B["Machine Learning Lab B"]

    subgraph S4 ["Huấn luyện & Đánh giá (A/B Testing)"]
        ML_A --> TrainA("Train SVM / Random Forest")
        ML_B --> TrainB("Train SVM / Random Forest")
        
        TrainA --> EvalA["Đánh giá mạnh: Lỗi Hole/Stain"]
        TrainB --> EvalB["Đánh giá mạnh: Lỗi Horizontal/Vertical"]
        
        EvalA --> Compare{"So Sánh Chéo Hiệu Năng<br>Confusion Matrix"}
        EvalB --> Compare
        Compare --> Report["Báo Cáo Đồ Án Cuối Kỳ"]
    end
    
    %% Styling
    classDef branchA fill:#d4edda,stroke:#28a745,stroke-width:2px;
    classDef branchB fill:#cce5ff,stroke:#007bff,stroke-width:2px;
    classDef mlPhase fill:#fff3cd,stroke:#ffc107,stroke-width:2px;
    
    class BranchA,CSV_A,ExtractA branchA;
    class BranchB,CSV_B,ExtractB branchB;
    class ML_A,ML_B,TrainA,TrainB mlPhase;
```

## Diễn giải Sơ đồ

1. **Tiền xử lý chung:** Toàn bộ ảnh sẽ đi qua bộ lọc Trung vị (khử hạt) và phép biến đổi Top-Hat/Black-Hat để triệt tiêu hoàn toàn sự thiếu đồng đều của ánh sáng trên bề mặt vải.
2. **Nhánh A (Bên Trái):** Chuyên trách các lỗi dạng Đốm/Khối. Áp dụng Morphological (Opening/Closing) để lấp lỗ và tính toán Diện tích (Area), Chu vi (Perimeter). Đầu ra lưu vào tập dữ liệu riêng biệt `morph_features.csv`.
3. **Nhánh B (Bên Phải):** Chuyên trách các lỗi Xước/Đứt sợi. Sử dụng góc Gradient từ Canny để tách các dải pixel nằm ngang/dọc, sau đó dùng Lọc hình thái học có hướng (Line SE) để loại bỏ nhiễu. Đầu ra đếm số pixel đứt gãy lưu vào `canny_features.csv`.
4. **Machine Learning:** Đưa 2 tập dữ liệu này vào huấn luyện độc lập. Cuối cùng, sinh ra 2 Ma trận nhầm lẫn (Confusion Matrix) để bảo vệ luận điểm khoa học trước hội đồng: Nhánh nào tối ưu cho loại khuyết tật nào.
