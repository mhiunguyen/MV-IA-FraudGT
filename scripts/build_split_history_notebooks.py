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
        "title": "# 11A — H-FraudGT Part 1 trên seed 43 (T4 x2)",
        "description": (
            "Chạy **A, H-RFM, H-RF, H-RM và H-FM** trong ba lượt song song. "
            "Thời gian dự kiến **6,5–7 giờ**. Đây là Part 1 trong thí nghiệm "
            "factorial; không thay đổi danh sách MODELS hoặc siêu tham số."
        ),
        "output_base": "H_FraudGT_Final_Part1",
    },
    "11B_History_Final_Part2_Seed43_T4x2.ipynb": {
        "models": ["H-R", "H-F", "H-M", "HG"],
        "title": "# 11B — H-FraudGT Part 2 trên seed 43 (T4 x2)",
        "description": (
            "Chạy **H-R, H-F, H-M và HG** trong hai lượt song song. "
            "Thời gian dự kiến **4,5–5 giờ**. Part 2 không có A nên "
            "`delta_f1_vs_A` có thể trống; hai CSV sẽ được ghép sau khi tải về."
        ),
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


def build(name: str, spec: dict, master: dict) -> Path:
    notebook = copy.deepcopy(master)
    protocol = (
        f"{spec['title']}\n\n{spec['description']}\n\n"
        "Notebook tự kiểm tra dependency, dữ liệu và leakage trước khi train. "
        "Best epoch cùng `best.ckpt` được chọn bằng **validation F1@0.50**; "
        "test không tham gia lựa chọn. Chạy bằng **Save Version → Save & Run All**."
    )
    set_source(notebook["cells"][0], protocol)

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
    replace_once(
        notebook,
        "`/kaggle/working/H_FraudGT_Final_Seed43_Evidence.zip`",
        f"`/kaggle/working/{spec['output_base']}.zip`",
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
