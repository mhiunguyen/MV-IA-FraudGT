# Kế hoạch chạy thực nghiệm cuối vào ngày mai

## 1. Mục tiêu của lượt chạy

Chạy lại seed 43 bằng một cấu hình công bằng, tạo checkpoint có thể demo và
trả lời chính xác nhóm đặc trưng nào tạo ra mức tăng của H-FraudGT.

Có **bốn khối thuật toán** dự kiến trình bày trong đồ án:

1. Tính tám đặc trưng lịch sử chỉ từ giao dịch có thời gian nhỏ hơn giao dịch
   hiện tại (past-only history extraction).
2. Chuẩn hóa bằng thống kê tập train và ánh xạ nhóm R/F/M.
3. Ghép đặc trưng lịch sử vào cạnh rồi huấn luyện FraudGT (H-FraudGT).
4. Reliability gate `q=n/(n+kappa)` để giảm độ tin cậy của lịch sử ít quan sát
   (HG, phần phát triển tiếp theo).

Tám cấu hình A/R/F/M/RF/RM/FM/RFM là **ablation của thuật toán**, không phải
tám thuật toán khác nhau. HG là thí nghiệm mở rộng và phải được trình bày tách
khỏi bảng factorial.

## 2. Chuẩn bị trên Kaggle

1. Bật Internet.
2. Chọn accelerator **GPU T4 x2**.
3. Add Input bộ **IBM Transactions for Anti Money Laundering** và chắc chắn có
   `HI-Small_Trans.csv`.
4. Mở notebook
   `notebooks/kaggle/11_History_Final_Seed43_Checkpoints_T4x2.ipynb`.
5. Kiểm tra `REPO_URL`, sau đó chọn **Run All**.

Không sửa riêng batch size, fanout, epoch hoặc loss của bất kỳ biến thể nào.
Khác biệt duy nhất giữa A/R/F/M/RF/RM/FM/RFM là nhóm đặc trưng lịch sử; HG chỉ
thêm reliability gate đã khai báo.

## 3. Giao thức đánh giá đã khóa

- Dataset: AML Small-HI.
- Seed: 43.
- Epoch: 100.
- Neighbor fanout: `[25, 25]`.
- Batch size: 512; accumulation: 4.
- Iterations/epoch: 256.
- Loss: weighted cross entropy `[1, 6]`.
- Threshold chính: 0.50.
- Best epoch: chọn bằng **validation F1@0.50**.
- Test: chỉ đọc đúng một lần tại epoch đã chọn.
- Checkpoint tốt nhất: `best.ckpt`, cùng epoch với bảng kết quả.
- Checkpoint phục hồi: mỗi 5 epoch.

## 4. Thứ tự chạy trên hai GPU

1. A // H-RFM (đồng thời tạo hai cache gốc và history).
2. H-RF // H-RM.
3. H-FM // H-R.
4. H-F // H-M.
5. HG (cache reliability riêng).

Ước lượng tổng thời gian: khoảng 3–5 giờ. GPU memory thấp không có nghĩa cấu
hình bị hạ; mô hình dùng neighbor sampling nên chỉ đưa subgraph từng batch lên
GPU. Hai GPU chủ yếu giúp chạy hai mô hình song song.

## 5. Nếu Kaggle bị ngắt

- Không xóa `/kaggle/working/TH-FraudGT`.
- Mở lại cùng notebook và Run All.
- Runner tự bỏ qua model hoàn tất và tiếp tục model còn checkpoint.
- Nếu Kaggle đã hủy hẳn session và mất `/kaggle/working`, phải chạy lại model
  chưa kịp tải bằng chứng; checkpoint không thể sống qua một session đã bị xóa.

## 6. Bằng chứng bắt buộc phải tải về

File cuối cùng:

`/kaggle/working/H_FraudGT_Final_Seed43_Evidence.zip`

ZIP phải chứa:

- toàn bộ source liên quan và notebook;
- config gốc và chín config sinh thực tế;
- log của từng model;
- train/validation/test `stats.json`;
- `best.ckpt` và checkpoint phục hồi;
- summary CSV và biểu đồ;
- commit Git, `pip freeze`, thông tin runtime;
- SHA-256, kích thước và số dòng dataset;
- manifest SHA-256 của toàn bộ file trong ZIP.

Ngoài tải ZIP, chọn **Save Version** sau khi Run All thành công để giữ output
trực quan trên Kaggle.

## 7. Sau lượt chạy này

1. Chốt kết luận factorial trên seed 43 theo yêu cầu hiện tại của giảng viên.
2. Chỉ giữ HG nếu nó cải thiện hợp lý hoặc giải thích được độ ổn định; không ép
   đưa HG vào kết luận nếu kết quả kém.
3. Khi phương pháp đã khóa, nếu còn ngân sách GPU mới chạy A, H-RFM và phương
   pháp cuối trên seed 42–44 để báo cáo mean ± std. Không dùng seed tốt nhất để
   thay cho mean trong kết luận tổng quát.
4. Không đưa T/TH temporal cũ hoặc hướng B/C vào bảng chính vì các nhánh đó đã
   bị loại khỏi phạm vi phương pháp cuối.
