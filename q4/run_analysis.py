#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第四问分析运行脚本
简化版本，直接运行分析
"""

import sys
import os

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main import main

if __name__ == "__main__":
    print("=" * 60)
    print("第四问：女胎异常判定模型分析")
    print("=" * 60)
    print()
    
    try:
        main()
        print()
        print("=" * 60)
        print("分析完成！请查看输出目录中的结果文件。")
        print("=" * 60)
    except Exception as e:
        print(f"分析过程中出现错误: {e}")
        print("请检查数据文件路径和依赖包是否正确安装。")
