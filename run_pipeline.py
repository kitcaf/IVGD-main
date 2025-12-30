"""
完整训练管道脚本
依次执行预训练（pretrain.py）和主训练（main.py）
避免每次都需要手动运行两个文件
"""

import subprocess
import sys
import os
from pathlib import Path

def run_pretrain():
    """运行预训练脚本"""
    print("=" * 80)
    print("开始预训练（i-DeepIS模型）")
    print("=" * 80)
    
    script_path = Path(__file__).parent / "pretrain.py"
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=Path(__file__).parent,
            check=True
        )
        print("\n✓ 预训练完成！")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ 预训练失败！错误代码: {e.returncode}")
        return False


def run_main():
    """运行主训练脚本"""
    print("\n" + "=" * 80)
    print("开始主训练（ALM网络校正和评估）")
    print("=" * 80 + "\n")
    
    script_path = Path(__file__).parent / "main.py"
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=Path(__file__).parent,
            check=True
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
    print("║" + " " * 20 + "IVGD 完整训练管道" + " " * 42 + "║")
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
    print("  - 预训练模型: i-deepis_<dataset>.pt")
    print("  - 校正后的预测和评估指标已在上述步骤中输出")
    

if __name__ == "__main__":
    main()
