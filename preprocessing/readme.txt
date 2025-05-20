Bước 1: Xử lý dữ liệu csv
dti: xử lý dữ liệu dti => dti.csv
mri: xử lý dữ liệu mri => mri.csv
merge: từ dti.csv, mri.csv => final.csv

Bước 2: Chia dữ liệu, xử lý ảnh, sinh dữ liệu.
(Các bước p1, p2, p3 cần tính ngẫu nhiên đã được cố định seed. Kết quả khi chạy từ DTI hay MRI đều sinh ra file csv giống nhau).
DTI:
    p1: Chia dữ liệu thành train, val, test => Train.csv, test.csv, val.csv
    p2: Sinh dữ liêu tăng cường cho nhãn 2, 3 trong tệp train => train_augmented.csv
    p3: Xóa 30% dữ liệu nhãn 1 trong tệp train. => Thay đổi train.csv
    p4: Xử lý ảnh
    p5: Tạo ảnh 3D tăng cường theo dữ liệu train_augmented.csv
    p6: Gộp file train.csv và train_augmented.csv => train.csv