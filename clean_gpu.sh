#!/bin/bash
# ==========================================
# 一键清理 GPU 缓存与僵尸进程
# 作者：Xing Wang（自动检测+清理）
# ==========================================

echo "=============================="
echo "🔍 检查 GPU 状态中..."
echo "=============================="
nvidia-smi

echo ""
echo "=============================="
echo "🧹 清理 PyTorch CUDA 缓存..."
echo "=============================="
python3 - <<'EOF'
import torch
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print("✅ 已清空 PyTorch CUDA 缓存")
else:
    print("⚠️ 当前系统未检测到 CUDA 设备")
EOF

echo ""
echo "=============================="
echo "💀 查找并终止僵尸 Python 进程..."
echo "=============================="
GPU_PROCS=$(nvidia-smi | grep python | awk '{print $5}' | grep -E '^[0-9]+$')

if [ -z "$GPU_PROCS" ]; then
    echo "✅ 未发现僵尸 python 进程。"
else
    echo "⚠️ 检测到以下 GPU 进程："
    echo "$GPU_PROCS"
    for PID in $GPU_PROCS; do
        echo "🧨 正在结束 PID $PID..."
        kill -9 $PID 2>/dev/null && echo "✅ 已结束进程 $PID" || echo "⚠️ 无法结束进程 $PID"
    done
fi

echo ""
echo "=============================="
echo "✨ 清理完成，当前 GPU 状态如下："
echo "=============================="
nvidia-smi
