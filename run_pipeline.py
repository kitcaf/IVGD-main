"""
完整训练管道脚本
依次执行预训练（pretrain.py）和主训练（main.py）
避免每次都需要手动运行两个文件
"""

import subprocess
import sys
import os
from pathlib import Path

# ==================== 集中配置 ====================
# 在这里修改要运行的数据集和使用的GPU设备
DATASET = 'android'  # 可选: 'karate', 'dolphins', 'jazz', 'netscience', 'cora_ml', 'power_grid'
DEVICE = 'cuda:0'    # 例如: 'cuda:0', 'cuda:1', 'cpu'
# =================================================

def run_pretrain():
    """运行预训练脚本"""
    print("=" * 80)
    print(f"开始预训练（i-DeepIS模型）- 数据集: {DATASET}, 设备: {DEVICE}")
    print("=" * 80)
    
    script_path = Path(__file__).parent / "pretrain.py"
    
    # 为子进程设置环境变量
    env = os.environ.copy()
    env['IVGD_DATASET'] = DATASET
    env['IVGD_DEVICE'] = DEVICE
    
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=Path(__file__).parent,
            check=True,
            env=env
        )
        print("\n✓ 预训练完成！")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ 预训练失败！错误代码: {e.returncode}")
        return False


def run_main():
    """运行主训练脚本"""
    print("\n" + "=" * 80)
    print(f"开始主训练（ALM网络校正和评估）- 数据集: {DATASET}, 设备: {DEVICE}")
    print("=" * 80 + "\n")
    
    script_path = Path(__file__).parent / "main.py"
    
    # 为子进程设置环境变量
    env = os.environ.copy()
    env['IVGD_DATASET'] = DATASET
    env['IVGD_DEVICE'] = DEVICE
    
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=Path(__file__).parent,
            check=True,
            env=env
        )
        print("\n✓ 主训练完成！")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ 主训练失败！错误代码: {e.returncode}")
        return False


def main():
    """主函数"""
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + f"IVGD 完整训练管道 (数据集: {DATASET})".center(84) + "║")
    print("║" + " " * 15 + "预训练 → 主训练 → 评估（端到端流程）" + " " * 28 + "║")
    print("╚" + "=" * 78 + "╝")
    
    # 运行预训练
    pretrain_success = run_pretrain()
    if not pretrain_success:
        print("\n⚠ 预训练失败，中断管道执行")
        sys.exit(1)
    
    # 运行主训练
    main_success = run_main()
    if not main_success:
        print("\n⚠ 主训练失败，中断管道执行")
        sys.exit(1)
    
    # 执行完成
    print("\n" + "=" * 80)
    print("✓ 完整训练管道执行成功！")
    print("=" * 80)
    print("\n关键输出文件：")
    print(f"  - 预训练模型: i-deepis_{DATASET}.pt")
    print("  - 校正后的预测和评估指标已在上述步骤中输出")
    

if __name__ == "__main__":
    main()
