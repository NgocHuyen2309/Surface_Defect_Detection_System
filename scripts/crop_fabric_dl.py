"""
Deep Learning Fabric Cropper using U^2-Net
Script này sử dụng mô hình học sâu để tự động bóc tách miếng vải khỏi nền,
sau đó cắt ảnh sát vào viền của miếng vải và lưu lại theo đúng cấu trúc thư mục.
"""

import cv2
import numpy as np
from pathlib import Path
from rembg import remove
from PIL import Image

def crop_fabric_with_dl(image_path: Path, output_path: Path):
    try:
        # 1. Đọc ảnh gốc bằng OpenCV để dùng cho bước cắt cuối cùng (giữ nguyên chất lượng)
        original_img = cv2.imread(str(image_path))
        if original_img is None:
            print(f"⚠️ Lỗi: Không thể đọc ảnh {image_path.name}")
            return

        # 2. Đọc ảnh bằng PIL để đưa vào mô hình Deep Learning
        input_img = Image.open(image_path)

        # 3. Đi qua Deep Learning Model (U^2-Net) để xóa nền
        # Kết quả trả về là ảnh có nền trong suốt (RGBA)
        output_img = remove(input_img)

        # Chuyển RGBA sang mảng Numpy
        rgba_arr = np.array(output_img)

        # 4. Trích xuất kênh Alpha (kênh độ mờ) để làm Mask
        # Pixel nào thuộc miếng vải sẽ có giá trị > 0, nền sẽ là 0
        alpha_channel = rgba_arr[:, :, 3]

        # 5. Dùng OpenCV tìm viền (Contour) bao quanh vùng có ảnh
        contours, _ = cv2.findContours(alpha_channel, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # Lấy contour có diện tích lớn nhất (chính là miếng vải)
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Lấy tọa độ Bounding Box (x, y, chiều rộng, chiều cao)
            x, y, w, h = cv2.boundingRect(largest_contour)

            # 6. Cắt thẳng trên ảnh GỐC (original_img) để không bị viền đen/trong suốt
            cropped_fabric = original_img[y:y+h, x:x+w]

            # Lưu ảnh đã cắt
            cv2.imwrite(str(output_path), cropped_fabric)
            print(f"✅ Đã cắt và lưu: {output_path.name}")
        else:
            # Fallback: Nếu mô hình DL không tìm thấy gì, copy nguyên ảnh gốc sang
            print(f"⚠️ Không nhận diện được vải, giữ nguyên: {image_path.name}")
            cv2.imwrite(str(output_path), original_img)

    except Exception as e:
        print(f"❌ Lỗi xử lý {image_path.name}: {str(e)}")

def process_entire_dataset(input_dir: str, output_dir: str):
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    print(f"🚀 Khởi động Deep Learning Cropper...")
    print(f"📁 Thư mục nguồn: {input_path}")
    print(f"📁 Thư mục đích: {output_path}\n")

    # Đếm số lượng ảnh để theo dõi tiến độ
    total_images = list(input_path.rglob('*.*'))
    img_extensions = ['.jpg', '.jpeg', '.png']
    
    valid_images = [img for img in total_images if img.suffix.lower() in img_extensions]
    
    for idx, img_path in enumerate(valid_images, 1):
        # Giữ nguyên cấu trúc thư mục con (VD: train/Hole/..., test/Stain/...)
        relative_path = img_path.relative_to(input_path)
        save_path = output_path / relative_path

        # Tạo thư mục đích nếu chưa tồn tại
        save_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"[{idx}/{len(valid_images)}] ", end="")
        crop_fabric_with_dl(img_path, save_path)

if __name__ == "__main__":
    # ĐỊNH TUYẾN THƯ MỤC
    # Khuyến nghị: Lưu tạm ra một thư mục mới để kiểm tra trước khi ghi đè
    INPUT_DIRECTORY = "data/raw"
    OUTPUT_DIRECTORY = "data/raw_cropped" 

    process_entire_dataset(INPUT_DIRECTORY, OUTPUT_DIRECTORY)
    print("\n🎉 HOÀN THÀNH TIẾN TRÌNH CẮT ẢNH!")