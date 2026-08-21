"""Build the self-contained Kaggle notebook for the final history study."""

from __future__ import annotations

import json
from pathlib import Path


def markdown(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(True)}


def code(source: str) -> dict:
    return {
        "cell_type": "code", "execution_count": None, "metadata": {},
        "outputs": [], "source": source.splitlines(True),
    }


cells = [
    markdown("""# 11 — Thực nghiệm cuối H-FraudGT trên seed 43 (T4 x2)

Notebook này chạy lại từ đầu theo một giao thức thống nhất và tự đóng gói bằng chứng.

- **A**: FraudGT gốc.
- **H-R, H-F, H-M, H-RF, H-RM, H-FM, H-RFM**: tám cấu hình factorial của ba nhóm Recency/Frequency/Monetary.
- **HG**: H-RFM có reliability gate `q=n/(n+kappa)`; đây là thuật toán mở rộng, được báo cáo tách khỏi ablation tám cấu hình.

Mọi cấu hình dùng seed 43, cùng sampler, số epoch và siêu tham số. Best epoch và `best.ckpt` đều được chọn bằng **validation F1 tại threshold 0.50**; test không tham gia lựa chọn. Checkpoint phục hồi được lưu mỗi 5 epoch. Nếu phiên bị ngắt, chạy lại đúng notebook để tiếp tục.

Thời gian dự kiến trên hai T4: khoảng 3–5 giờ, tùy thời gian tạo cache và tải dữ liệu. Chạy lần lượt từ trên xuống bằng **Run All**."""),
    code("""# Thiết lập duy nhất cần kiểm tra trước khi Run All
from pathlib import Path

REPO_URL = 'https://github.com/mhiunguyen/TH-FraudGT.git'
REPO = Path('/kaggle/working/TH-FraudGT')
MODELS = ['A', 'H-R', 'H-F', 'H-M', 'H-RF', 'H-RM', 'H-FM', 'H-RFM', 'HG']
GPU_INDICES = [0, 1]
RUN_TRAINING = True

EVIDENCE = Path('/kaggle/working/final_history_evidence')
EVIDENCE.mkdir(parents=True, exist_ok=True)
print('Models:', MODELS)
print('GPU indices:', GPU_INDICES)
print('RUN_TRAINING:', RUN_TRAINING)"""),
    markdown("## 1. Ghi nhận môi trường phần cứng"),
    code("""import platform, subprocess, sys
import torch

print('Python:', sys.version)
print('Platform:', platform.platform())
print('PyTorch:', torch.__version__)
print('CUDA runtime:', torch.version.cuda)
print('CUDA available:', torch.cuda.is_available())
print('GPU count:', torch.cuda.device_count())
for index in range(torch.cuda.device_count()):
    prop = torch.cuda.get_device_properties(index)
    print(f'GPU {index}: {prop.name}; VRAM={prop.total_memory / 1024**3:.2f} GiB')
subprocess.run(['nvidia-smi'], check=False)"""),
    markdown("## 2. Lấy đúng mã nguồn và ghi commit"),
    code("""import os, subprocess

os.chdir('/kaggle/working')
if not (REPO / '.git').exists():
    subprocess.run(['git', 'clone', REPO_URL, str(REPO)], check=True)
else:
    subprocess.run(['git', '-C', str(REPO), 'pull', '--ff-only'], check=True)

os.chdir(REPO)
commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
required = [
    REPO / 'scripts/run_final_history_experiments.py',
    REPO / 'scripts/summarize_final_history_experiments.py',
    REPO / 'scripts/package_final_history_evidence.py',
    REPO / 'configs/AML-Small-HI/AML-Small-HI-History-Final-Seed43.yaml',
]
missing = [str(path) for path in required if not path.exists()]
if missing:
    raise RuntimeError('Repository chưa có bộ chạy cuối: ' + str(missing))
print('Repository:', REPO)
print('Commit:', commit)"""),
    markdown("## 3. Cài dependency đúng với PyTorch/CUDA của Kaggle"),
    code("""import subprocess, sys, torch

torch_version = torch.__version__.split('+')[0]
cuda_tag = 'cu' + torch.version.cuda.replace('.', '') if torch.version.cuda else 'cpu'
wheel_url = f'https://data.pyg.org/whl/torch-{torch_version}+{cuda_tag}.html'
print('PyG wheel index:', wheel_url)
subprocess.run([
    sys.executable, '-m', 'pip', 'install', '-q',
    'pyg_lib', 'torch_scatter', 'torch_sparse', '-f', wheel_url,
], check=True)
subprocess.run([
    sys.executable, '-m', 'pip', 'install', '-q', '-r',
    str(REPO / 'requirements-kaggle.txt'),
], check=True)

import torch_geometric, torch_sparse, torch_scatter, yaml
print('torch_geometric:', torch_geometric.__version__)
print('torch_sparse:', torch_sparse.__version__)
print('torch_scatter:', torch_scatter.__version__)
print('PyYAML:', yaml.__version__)"""),
    markdown("## 4. Gắn AML Small-HI và lập manifest dữ liệu"),
    code("""import hashlib, json, shutil

candidates = list(Path('/kaggle/input').rglob('HI-Small_Trans.csv'))
if not candidates:
    raise FileNotFoundError(
        'Không tìm thấy HI-Small_Trans.csv. Hãy Add Input bộ IBM AML rồi chạy lại cell.'
    )
source = candidates[0]
destination = REPO / 'data/AML/HI-Small_Trans.csv'
destination.parent.mkdir(parents=True, exist_ok=True)
if not destination.exists() or destination.stat().st_size != source.stat().st_size:
    shutil.copy2(source, destination)

digest = hashlib.sha256()
with destination.open('rb') as stream:
    for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
        digest.update(block)
with destination.open('rb') as stream:
    rows = max(sum(1 for _ in stream) - 1, 0)
manifest = {
    'source': str(source), 'destination': str(destination),
    'bytes': destination.stat().st_size, 'transaction_rows': rows,
    'sha256': digest.hexdigest(),
}
(EVIDENCE / 'dataset_manifest.json').write_text(
    json.dumps(manifest, indent=2), encoding='utf-8')
print(json.dumps(manifest, indent=2))"""),
    markdown("## 5. Kiểm tra mã, chống rò rỉ và sinh ma trận cấu hình"),
    code("""import subprocess, sys

subprocess.run([sys.executable, '-m', 'compileall', '-q', 'fraudGT', 'scripts'], check=True)
subprocess.run([
    sys.executable, '-m', 'pytest', '-q',
    'tests/test_history_features.py', 'tests/test_threshold_selection.py',
], check=True)
subprocess.run([
    sys.executable, 'scripts/run_final_history_experiments.py',
    '--repo', str(REPO), '--prepare-only',
], check=True)
print('Kiểm tra logic và sinh config: OK')"""),
    code("""import pandas as pd, yaml

rows = []
cfg_dir = REPO / 'generated_configs_final_history'
for model in MODELS:
    path = cfg_dir / f'AML-Small-HI-Final-{model}-Seed43.yaml'
    cfg = yaml.safe_load(path.read_text(encoding='utf-8'))
    rows.append({
        'model': model,
        'history': cfg['dataset']['add_history'],
        'groups': '+'.join(cfg['dataset']['history_groups']),
        'reliability': cfg['dataset']['history_reliability'],
        'best_metric': cfg['metric_best'],
        'checkpoint': cfg['train']['enable_ckpt'],
        'epochs': cfg['optim']['max_epoch'],
    })
config_table = pd.DataFrame(rows)
display(config_table)
assert set(config_table['best_metric']) == {'f1_t50'}
assert config_table['checkpoint'].all()"""),
    markdown("## 6. Huấn luyện hoặc tiếp tục từ checkpoint"),
    code("""import subprocess, sys, time

if RUN_TRAINING:
    command = [
        sys.executable, '-u', 'scripts/run_final_history_experiments.py',
        '--repo', str(REPO), '--gpus', *map(str, GPU_INDICES),
        '--models', *MODELS,
    ]
    print('Command:', ' '.join(command))
    started = time.time()
    subprocess.run(command, cwd=REPO, check=True)
    print(f'Hoàn tất sau {(time.time() - started) / 3600:.2f} giờ')
else:
    print('Bỏ qua huấn luyện theo RUN_TRAINING=False')"""),
    markdown("## 7. Tổng hợp đúng best validation epoch và kiểm tra checkpoint"),
    code("""summary_path = EVIDENCE / 'summary_final_history_seed43.csv'
subprocess.run([
    sys.executable, 'scripts/summarize_final_history_experiments.py',
    '--results-root', str(REPO / 'results_final_history'),
    '--output', str(summary_path), '--models', *MODELS,
], cwd=REPO, check=True)
results = pd.read_csv(summary_path)
display(results[[
    'model', 'best_epoch_by_validation', 'val_f1', 'test_f1',
    'delta_f1_vs_A', 'test_precision', 'test_recall', 'test_auc',
    'checkpoint_ok',
]])
assert results['checkpoint_ok'].all()"""),
    markdown("## 8. Biểu đồ đơn giản để đưa vào báo cáo"),
    code("""import matplotlib.pyplot as plt

plot = results.set_index('model').loc[MODELS].reset_index()
fig, ax = plt.subplots(figsize=(10, 5.2))
colors = [
    '#777777' if model == 'A' else '#222222' if model == 'HG' else '#356a9a'
    for model in plot['model']
]
bars = ax.bar(plot['model'], plot['test_f1'] * 100, color=colors)
if 'A' in set(plot['model']):
    baseline = float(plot.loc[plot['model'] == 'A', 'test_f1'].iloc[0]) * 100
    ax.axhline(baseline, color='black', linewidth=1, linestyle='--',
               label=f'FraudGT gốc: {baseline:.2f}%')
for bar, value in zip(bars, plot['test_f1'] * 100):
    ax.text(bar.get_x() + bar.get_width()/2, value + 0.6,
            f'{value:.2f}%', ha='center', va='bottom', fontsize=9)
ax.set_ylabel('F1 trên tập kiểm thử (%)')
ax.set_xlabel('Cấu hình')
ax.set_title('Ablation đặc trưng lịch sử và reliability gate — seed 43')
ax.grid(axis='y', alpha=0.2)
if 'A' in set(plot['model']):
    ax.legend()
fig.tight_layout()
plot_path = EVIDENCE / 'final_history_f1_seed43.png'
fig.savefig(plot_path, dpi=180, bbox_inches='tight')
plt.show()
print('Saved:', plot_path)"""),
    markdown("## 9. Ghi môi trường, commit và đóng gói bằng chứng"),
    code("""import json, platform, subprocess, sys

commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
status = subprocess.check_output(['git', 'status', '--short'], cwd=REPO, text=True)
(EVIDENCE / 'source_manifest.json').write_text(json.dumps({
    'repository': REPO_URL, 'commit': commit, 'git_status': status,
}, indent=2), encoding='utf-8')
(EVIDENCE / 'pip_freeze.txt').write_text(
    subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True),
    encoding='utf-8')
(EVIDENCE / 'runtime.txt').write_text(
    f'Python: {sys.version}\\nPlatform: {platform.platform()}\\n'
    f'PyTorch: {torch.__version__}\\nCUDA runtime: {torch.version.cuda}\\n',
    encoding='utf-8')
print('Commit:', commit)
print('Git status sau run:\\n' + (status or '(clean)'))"""),
    code("""output_base = Path('/kaggle/working/H_FraudGT_Final_Seed43_Evidence')
subprocess.run([
    sys.executable, 'scripts/package_final_history_evidence.py',
    '--repo', str(REPO),
    '--results-root', str(REPO / 'results_final_history'),
    '--summary', str(summary_path),
    '--logs-dir', '/kaggle/working/final_history_logs',
    '--evidence-dir', str(EVIDENCE),
    '--output-base', str(output_base),
], cwd=REPO, check=True)
archive = output_base.with_suffix('.zip')
print('TẢI FILE NÀY VỀ MÁY:', archive)
print('Kích thước:', archive.stat().st_size / 1024**2, 'MiB')"""),
    markdown("""## 10. Checklist trước khi đóng Kaggle

Chỉ kết thúc khi đã có:

1. `summary_final_history_seed43.csv` và biểu đồ F1.
2. Mỗi run có `best.ckpt`, checkpoint phục hồi, ba file `stats.json` và config thực tế.
3. `dataset_manifest.json`, `source_manifest.json`, `runtime.txt`, `pip_freeze.txt`.
4. Đã tải `/kaggle/working/H_FraudGT_Final_Seed43_Evidence.zip` về máy.
5. Save Version của notebook để giữ output trực quan.

Nếu phiên bị ngắt giữa chừng: mở lại cùng notebook, gắn lại dataset, chạy từ đầu. Runner sẽ bỏ qua model hoàn tất và tiếp tục model còn checkpoint."""),
]

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

output = Path(__file__).resolve().parents[1] / "notebooks" / "kaggle" / \
    "11_History_Final_Seed43_Checkpoints_T4x2.ipynb"
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
print(output)
