import os
import time
import torch
import numpy as np
from PIL import Image, ImageOps, ImageSequence

try:
    import folder_paths
except ImportError:
    class DummyFolderPaths:
        @staticmethod
        def get_input_directory():
            return os.path.join(os.path.dirname(os.path.dirname(__file__)), "input")
        @staticmethod
        def get_annotated_filepath(name):
            input_dir = DummyFolderPaths.get_input_directory()
            return os.path.join(input_dir, name)
    folder_paths = DummyFolderPaths()

SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif"}


class MiaodaImageLoadWithMask:
    """
    秒哒 图像加载与遮罩输出自定义节点 (ComfyUI-AIMoJingTools)
    
    核心升级能力：
    1. 输入源自动仲裁与切换开关 (source_mode)：
       - "auto (上游优先并屏蔽本地)"：若 image_in 已连接并有输入，自动彻底屏蔽本地图像，直接透传上游；
       - "image_in (强制上游)"：强制透传上游图像；
       - "local_file (强制本地)"：强制使用本地加载控件的图像。
    2. 严格的图像输出保障：确保无论何种情况下，主 image 端口始终输出清晰完整的 RGB 图像。
    3. 遮罩条件输出机制 (未编辑遮罩时不输出遮罩图层)：
       - 检测图像是否存在有效的遮罩编辑器编辑痕迹（如非全不透明的真实 Alpha 涂抹或 clipspace 遮罩）；
       - 若未通过遮罩编辑器编辑，不输出遮罩图层（返回全 0 遮罩或空张量），彻底避免白遮罩覆盖图像或引发错误重绘；
       - 仅在真实编辑或携带有效透明遮罩时，才输出精确的 Alpha 遮罩图层。
    """

    @classmethod
    def INPUT_TYPES(cls):
        input_dir = folder_paths.get_input_directory()
        files = []
        if os.path.exists(input_dir):
            files = [
                f for f in os.listdir(input_dir)
                if os.path.isfile(os.path.join(input_dir, f)) and any(f.lower().endswith(ext) for ext in SUPPORTED_IMAGE_EXTENSIONS)
            ]
        files = sorted(files)
        return {
            "required": {
                "image": (files, {"image_upload": True}),
                "source_mode": (
                    ["auto (上游优先并屏蔽本地)", "image_in (强制上游)", "local_file (强制本地)"],
                    {"default": "auto (上游优先并屏蔽本地)"}
                ),
                "mask_behavior": (
                    ["auto (仅编辑时输出遮罩)", "always_output (强制输出)", "disabled (不输出遮罩)"],
                    {"default": "auto (仅编辑时输出遮罩)"}
                ),
            },
            "optional": {
                "image_in": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "load_image"
    CATEGORY = "image/loaders"

    def _save_preview(self, image_tensor):
        """将图像保存为临时预览文件，供前端在节点内显示（模拟本地加载的预览体验）。"""
        try:
            temp_dir = folder_paths.get_temp_directory()
        except Exception:
            temp_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "temp")
        os.makedirs(temp_dir, exist_ok=True)
        arr = (image_tensor[0].clamp(0, 1).cpu().numpy() * 255).astype("uint8")
        pil_img = Image.fromarray(arr)
        fname = f"aimojing_preview_{int(time.time() * 1000)}.png"
        pil_img.save(os.path.join(temp_dir, fname))
        return fname

    def load_image(self, image, source_mode="auto (上游优先并屏蔽本地)", mask_behavior="auto (仅编辑时输出遮罩)", image_in=None):
        # 仲裁决策：是否采用上游输入
        use_upstream = False
        if source_mode == "image_in (强制上游)":
            if image_in is None:
                raise ValueError("[MiaodaImageLoadWithMask] 已选择强制使用上游图像模式，但 image_in 端口未接入任何有效图像！")
            use_upstream = True
        elif source_mode == "local_file (强制本地)":
            use_upstream = False
        else: # "auto (上游优先并屏蔽本地)"
            if image_in is not None:
                use_upstream = True

        # 分支 1：使用上游输入（自动屏蔽本地加载的图像）
        if use_upstream:
            if not isinstance(image_in, torch.Tensor):
                raise TypeError(f"Expected torch.Tensor for image_in, got {type(image_in).__name__}")
            
            img_tensor = image_in.unsqueeze(0) if image_in.ndim == 3 else image_in
            b, h, w, c = img_tensor.shape
            
            # 保证主图像端口始终输出标准的 3 通道 RGB 张量
            out_image = img_tensor[..., :3].contiguous()

            # 遮罩判断：若存在第 4 通道且非禁用
            if mask_behavior != "disabled" and c >= 4:
                raw_alpha = img_tensor[..., 3].contiguous()
                is_edited = torch.any(raw_alpha < 0.999) and torch.any(raw_alpha > 0.001)
                if is_edited or mask_behavior == "always_output":
                    out_mask = raw_alpha
                else:
                    # 未编辑遮罩时不输出遮罩图层（全 0 遮罩，不遮挡任何内容）
                    out_mask = torch.zeros((b, h, w), dtype=torch.float32, device=img_tensor.device)
            else:
                out_mask = torch.zeros((b, h, w), dtype=torch.float32, device=img_tensor.device)

            # 上游图像模式下，保存预览图并返回 ui，让前端在节点内显示（如同本地加载的预览）
            preview_fname = self._save_preview(out_image)
            return {
                "ui": {"images": [{"filename": preview_fname, "subfolder": "", "type": "temp"}]},
                "result": (out_image, out_mask),
            }

        # 分支 2：使用本地控件加载图片
        if not image:
            raise ValueError("[MiaodaImageLoadWithMask] 未选择或上传本地图片，且未连接上游 image_in 接口。")

        image_path = folder_paths.get_annotated_filepath(image)
        if not os.path.exists(image_path) or not os.path.isfile(image_path):
            raise ValueError(f"[MiaodaImageLoadWithMask] 指定的图片文件不存在: {image_path}")

        ext = os.path.splitext(image_path)[1].lower()
        if ext not in SUPPORTED_IMAGE_EXTENSIONS:
            raise ValueError(f"[MiaodaImageLoadWithMask] 不支持的格式 '{ext}'。仅支持: {', '.join(sorted(SUPPORTED_IMAGE_EXTENSIONS))}")

        try:
            img_obj = Image.open(image_path)
        except Exception as e:
            raise ValueError(f"[MiaodaImageLoadWithMask] 无法打开图像文件 '{image_path}': {str(e)}")

        # 动图提取首帧
        for frame in ImageSequence.Iterator(img_obj):
            img_obj = frame
            break

        # EXIF 旋转自动校正
        img_obj = ImageOps.exif_transpose(img_obj)

        # 始终转换出清晰的 RGB 图像，确保主 image 输出绝对正常
        rgb_img = img_obj.convert("RGB")
        np_rgb = np.array(rgb_img).astype(np.float32) / 255.0
        out_image = torch.from_numpy(np_rgb)[None,]
        b, h, w, _ = out_image.shape

        # 判定是否包含遮罩编辑器编辑痕迹（如非全不透明的实际透明区涂抹）
        has_edited_mask = False
        extracted_mask = None

        if mask_behavior != "disabled":
            if img_obj.mode in ("RGBA", "LA") or (img_obj.mode == "P" and "transparency" in img_obj.info):
                rgba_img = img_obj.convert("RGBA")
                alpha_channel = np.array(rgba_img.getchannel("A")).astype(np.float32) / 255.0
                if np.any(alpha_channel < 0.999):
                    has_edited_mask = True
                    extracted_mask = torch.from_numpy(alpha_channel)[None,]

        # 根据遮罩策略输出：未用遮罩编辑器编辑时，不输出遮罩图层（全 0 遮罩，不遮挡、不触发错误重绘）
        if has_edited_mask and extracted_mask is not None:
            out_mask = extracted_mask
        elif mask_behavior == "always_output":
            out_mask = torch.ones((b, h, w), dtype=torch.float32)
        else:
            out_mask = torch.zeros((b, h, w), dtype=torch.float32)

        return (out_image, out_mask)
