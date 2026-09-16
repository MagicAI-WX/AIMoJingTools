"""
ComfyUI 自定义节点插件：秒哒 图像加载与遮罩输出
Plugin: ComfyUI-AIMoJingTools
"""
from .nodes import MiaodaImageLoadWithMask

# ComfyUI 官方自定义节点类名映射
NODE_CLASS_MAPPINGS = {
    "MiaodaImageLoadWithMask": MiaodaImageLoadWithMask,
}

# ComfyUI 界面显示名称映射
NODE_DISPLAY_NAME_MAPPINGS = {
    "MiaodaImageLoadWithMask": "秒哒 图像加载与遮罩输出",
}

# 前端扩展目录：ComfyUI 会自动加载 ./js 下的脚本，用于上游图像自动预览
WEB_DIRECTORY = "./js"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
