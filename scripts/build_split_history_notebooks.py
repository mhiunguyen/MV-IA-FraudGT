"""Derive two ready-to-run Kaggle notebooks from the audited master notebook."""

from __future__ import annotations

import copy
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MASTER = ROOT / "notebooks" / "kaggle" / \
    "11_History_Final_Seed43_Checkpoints_T4x2.ipynb"

PARTS = {
    "11A_History_Final_Part1_Seed43_T4x2.ipynb": {
        "models": ["A", "H-RFM", "H-RF", "H-RM", "H-FM"],
        "title": "# Thí nghiệm H-FraudGT — Phần 1: Mô hình gốc và các tổ hợp R/F/M",
        "description": (
            "Mục tiêu là so sánh **FraudGT gốc (A)** với bốn biến thể dùng "
            "đặc trưng lịch sử: **H-RFM, H-RF, H-RM và H-FM**."
        ),
        "purpose": (
            "A là mốc FraudGT gốc; bốn cấu hình còn lại lần lượt kiểm tra các "
            "tổ hợp Recency–Frequency–Monetary."
        ),
        "duration": "6,5–7 giờ",
        "output_base": "H_FraudGT_Final_Part1",
    },
    "11B_History_Final_Part2_Seed43_T4x2.ipynb": {
        "models": ["H-R", "H-F", "H-M", "HG"],
        "title": "# Thí nghiệm H-FraudGT — Phần 2: Đo riêng R/F/M và cổng độ tin cậy",
        "description": (
            "Phần này tách riêng **H-R, H-F và H-M** để đo đóng góp của từng "
            "nhóm đặc trưng, đồng thời chạy **HG** để kiểm nghiệm cơ chế giảm "
            "độ tin cậy khi lịch sử quan sát còn ít."
        ),
        "purpose": (
            "Kết quả của phần này sẽ được ghép với baseline A ở Phần 1; vì vậy "
            "cột chênh lệch so với A có thể để trống trong CSV tạm thời."
        ),
        "duration": "4,5–5 giờ",
        "output_base": "H_FraudGT_Final_Part2",
    },
}


def source_text(cell: dict) -> str:
    source = cell.get("source", [])
    return "".join(source) if isinstance(source, list) else str(source)


def set_source(cell: dict, text: str) -> None:
    cell["source"] = text.splitlines(True)


def replace_once(notebook: dict, old: str, new: str) -> None:
    matches = []
    for index, cell in enumerate(notebook["cells"]):
        text = source_text(cell)
        if old in text:
            matches.append((index, text))
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one notebook cell containing {old!r}; "
            f"found {len(matches)}"
        )
    index, text = matches[0]
    set_source(notebook["cells"][index], text.replace(old, new, 1))


def rewrite_markdown(notebook: dict, spec: dict) -> None:
    """Give every section a concise purpose and a concrete expected output."""
    intro = f"""{spec['title']}

{spec['description']}

> **Mục đích.** {spec['purpose']}

| Thiết lập | Giá trị |
|---|---|
| Dữ liệu | AML Small-HI |
| Seed | 43 |
| GPU | 2 × NVIDIA T4 |
| Chọn mô hình | Validation F1 tại threshold 0.50 |
| Thời gian dự kiến | {spec['duration']} |

Toàn bộ mô hình dùng cùng sampler, số epoch, loss và siêu tham số. Tập test
không được dùng để chọn epoch. Hãy chạy notebook bằng **Save Version → Save &
Run All** để Kaggle thực thi từ một môi trường sạch và giữ lại output.

**Luồng thực nghiệm:** kiểm tra môi trường → xác nhận dữ liệu → kiểm tra chống
rò rỉ → huấn luyện → chọn epoch bằng validation → đánh giá test → đóng gói.
"""
    set_source(notebook["cells"][0], intro)

    sections = {
        2: """## 1. Xác nhận tài nguyên thực nghiệm

Cell tiếp theo in phiên bản Python, PyTorch, CUDA và thông tin của hai GPU.
Giữ output này làm bằng chứng về môi trường chạy; nếu đưa vào báo cáo, chỉ cần
chụp phần tên GPU, VRAM và phiên bản PyTorch/CUDA.
""",
        4: """## 2. Cố định phiên bản mã nguồn

Repository được clone trực tiếp từ GitHub và commit hash được ghi lại. Commit
hash giúp xác định chính xác phiên bản code đã tạo ra kết quả, kể cả khi mã
nguồn tiếp tục được chỉnh sửa sau này.
""",
        6: """## 3. Chuẩn bị thư viện tương thích với GPU Kaggle

Các gói PyTorch Geometric phải khớp với phiên bản PyTorch và CUDA của session.
Cell này tự tạo đúng địa chỉ wheel rồi kiểm tra lại các import quan trọng. Nếu
cell chưa in đủ phiên bản thư viện thì không tiếp tục huấn luyện.
""",
        8: """## 4. Xác nhận dữ liệu AML Small-HI

Notebook tìm `HI-Small_Trans.csv` trong Kaggle Input, đưa file về đúng vị trí
mà FraudGT sử dụng, sau đó ghi số giao dịch, kích thước và SHA-256. Manifest
này chứng minh các lần chạy dùng cùng một bản dữ liệu.
""",
        10: """## 5. Kiểm tra tính hợp lệ trước khi huấn luyện

Trước khi dùng GPU, notebook kiểm tra cú pháp, chạy unit test cho đặc trưng
lịch sử và xác nhận giao dịch tại thời điểm `t` không nhìn thấy giao dịch cùng
hoặc sau `t`. Bảng cấu hình sau đó phải cho thấy mọi mô hình cùng seed, epoch,
checkpoint và tiêu chí `f1_t50`; chỉ nhóm R/F/M hoặc reliability được phép khác.
""",
        13: f"""## 6. Huấn luyện các cấu hình của phần này

Các mô hình được phân thành từng cặp để hai T4 hoạt động đồng thời. Dòng
`heartbeat` cho biết tiến trình vẫn sống dù cell không in log từng epoch. GPU
thấp trong lúc lấy mẫu hoặc tạo cache là bình thường. Thời gian dự kiến của
phần này là **{spec['duration']}**.
""",
        15: """## 7. Chọn epoch mà không nhìn vào tập test

Với mỗi mô hình, epoch tốt nhất được chọn bằng F1 trên validation tại threshold
0.50. Precision, recall, F1 và AUC trên test chỉ được đọc tại đúng epoch đó.
Cell cũng đối chiếu epoch trong `best.ckpt`; nếu `checkpoint_ok` không phải
`True`, chưa được sử dụng kết quả trong báo cáo.
""",
        17: """## 8. So sánh F1 giữa các cấu hình

Biểu đồ dùng cùng một thang đo và ghi trực tiếp F1 (%) trên từng cột. Đây là
hình nên lưu để trình bày ablation; bảng CSV vẫn là nguồn số liệu chính khi
viết báo cáo.
""",
        19: """## 9. Đóng gói khả năng tái lập

Ngoài kết quả, đồ án cần giữ được code, config, môi trường, log và trọng số.
Hai cell sau ghi commit, `pip freeze`, thông tin runtime rồi tạo một ZIP có
manifest SHA-256 cho toàn bộ bằng chứng.
""",
    }
    for index, content in sections.items():
        set_source(notebook["cells"][index], content)

    checklist = f"""## 10. Kiểm tra trước khi rời Kaggle

Chỉ xem phần này hoàn tất khi đã kiểm tra đủ các mục sau:

- [ ] Bảng summary hiển thị đủ các mô hình của phần này.
- [ ] Tất cả dòng `checkpoint_ok` đều là `True`.
- [ ] Biểu đồ F1 đã được tạo và không thiếu cột.
- [ ] Mỗi run có `best.ckpt`, checkpoint phục hồi và ba file `stats.json`.
- [ ] Đã tải **`/kaggle/working/{spec['output_base']}.zip`** về máy.
- [ ] Phiên bản notebook trên Kaggle có trạng thái `Successful`.

Nếu batch run thất bại, đọc phần cuối log để xác định model dừng ở đâu. Trong
cùng một Draft Session, runner có thể tiếp tục từ checkpoint; nếu Kaggle đã
xóa toàn bộ session thì cần chạy lại phần chưa có ZIP.
"""
    set_source(notebook["cells"][22], checklist)


def build(name: str, spec: dict, master: dict) -> Path:
    notebook = copy.deepcopy(master)
    rewrite_markdown(notebook, spec)

    models_literal = repr(spec["models"])
    replace_once(
        notebook,
        "MODELS = ['A', 'H-R', 'H-F', 'H-M', 'H-RF', 'H-RM', 'H-FM', 'H-RFM', 'HG']",
        f"MODELS = {models_literal}",
    )
    replace_once(
        notebook,
        "output_base = Path('/kaggle/working/H_FraudGT_Final_Seed43_Evidence')",
        f"output_base = Path('/kaggle/working/{spec['output_base']}')",
    )
    notebook["metadata"]["h_fraudgt_part"] = name[:3]
    notebook["metadata"]["h_fraudgt_models"] = spec["models"]

    output = MASTER.parent / name
    output.write_text(
        json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
    return output


def main() -> None:
    master = json.loads(MASTER.read_text(encoding="utf-8"))
    for name, spec in PARTS.items():
        print(build(name, spec, master))


if __name__ == "__main__":
    main()
