#!/usr/bin/env python3
"""
简单的GIF生成脚本，专门用于blender_temp目录
"""

import os
from pathlib import Path
from PIL import Image
import glob

def create_gif_from_blender_temp(blender_temp_path, output_gif_path="animation.gif", duration=100):
    """
    从blender_temp目录创建GIF动画
    """
    blender_temp_path = Path(blender_temp_path)
    
    # 获取所有包含result.png的目录
    frame_dirs = []
    for item in blender_temp_path.iterdir():
        if item.is_dir() and (item / "result.png").exists():
            try:
                # 将目录名转换为浮点数进行排序
                frame_time = float(item.name)
                frame_dirs.append((frame_time, item / "result.png"))
            except ValueError:
                # 跳过非数字目录名（如init）
                continue
    
    # 按时间排序
    frame_dirs.sort(key=lambda x: x[0])
    
    print(f"找到 {len(frame_dirs)} 帧图像")
    
    # 加载所有帧
    frames = []
    for time, frame_path in frame_dirs:
        try:
            img = Image.open(frame_path)
            if img.mode != 'RGBA':
                img = img.convert('RGBA')
            
            # 创建白色背景
            white_bg = Image.new('RGBA', img.size, (255, 255, 255, 255))
            # 将原图合成到白色背景上
            img_with_bg = Image.alpha_composite(white_bg, img)
            frames.append(img_with_bg)
            print(f"加载帧 {time:.3f}: {frame_path}")
        except Exception as e:
            print(f"警告: 无法加载帧 {frame_path}: {e}")
    
    if not frames:
        print("错误: 没有成功加载任何帧")
        return
    
    # 创建GIF
    print(f"正在生成GIF: {output_gif_path}")
    frames[0].save(
        output_gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=0,
        optimize=True,
        background=255,  # 设置背景为白色
        disposal=2
    )
    
    print(f"✅ GIF已保存: {output_gif_path}")
    print(f"📊 信息: {len(frames)}帧, 每帧{duration}ms, 总时长{len(frames)*duration/1000:.1f}秒")

if __name__ == "__main__":
    # 使用示例
    blender_temp_path = "elog/final_output/ours/StorageFurniture_45940_3/0/gif/blender_temp"
    output_path = "elog/final_output/ours/StorageFurniture_45940_3/0/gif/animation.gif"
    
    create_gif_from_blender_temp(blender_temp_path, output_path, duration=100)
