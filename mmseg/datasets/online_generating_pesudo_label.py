# from typing import Callable, List, Optional, Sequence, Union
# from mmengine.dataset import BaseDataset, Compose

# from mmseg.registry import DATASETS
# from typing import Any, Callable, List, Optional, Sequence, Tuple, Union

from typing import Any, List
from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset
from diffusers import StableDiffusionPipeline
from xformers.ops import MemoryEfficientAttentionFlashAttentionOp
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
from PIL import Image, ImageOps

import torch 
import gc
import numpy as np
import random

torch.backends.cuda.matmul.allow_tf32 = True


@DATASETS.register_module()
class OnlineGenDataset(BaseSegDataset):
    METAINFO = dict(
        classes=('background', 'aeroplane', 'bicycle', 'bird', 'boat',
                 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
                 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep',
                 'sofa', 'train', 'tvmonitor'),
        palette=[[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                 [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                 [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                 [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                 [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                 [0, 64, 128]])
    def __init__(self,
                 img_suffix='.png',
                 seg_map_suffix='.png',
                 stable_diffusion_repo_id = 'stabilityai/stable-diffusion-2-1',
                 data_type = torch.bfloat16,
                 device = 'cuda',
                 num_inference_steps = 34,
                 num_image_per_prompt = 1,
                 guidance_scale = 5.0,
                 
                 num_random_classes = 2,
                 min_thing_size = 100, 
                 max_thing_size = 220,
                 image_size = 512,
                 serialize_data = False,
                 **kwargs) -> None:
        
        super().__init__(
            img_suffix=img_suffix, seg_map_suffix=seg_map_suffix,
            serialize_data=serialize_data, **kwargs)

        sd_pipeline = StableDiffusionPipeline.from_pretrained(
         stable_diffusion_repo_id , torch_dtype = data_type,
        #  num_inference_steps = num_inference_steps
        )
        self.sd_device = device
        self.sd_pipeline = sd_pipeline.to(self.sd_device)
        # self.output_directory = output_direcotry
        # self.pipeline.enable_model_cpu_offload()
        # self.pipeline.enable_attention_slicing() 
        self.sd_pipeline.enable_xformers_memory_efficient_attention(attention_op=MemoryEfficientAttentionFlashAttentionOp)
        self.sd_pipeline.vae.enable_xformers_memory_efficient_attention(attention_op=None)
        self.num_inference_steps = num_inference_steps
        self.num_image_per_prompt = num_image_per_prompt
        self.guidance_scale = guidance_scale
        
        self.num_random_classes = num_random_classes
        self.min_thing_size = min_thing_size
        self.max_thing_size = max_thing_size
        self.image_size = image_size
        
        self.processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
        self.clipseg = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")

        
    def load_data_list(self) -> List[dict]:
        # fake data list
        data_list = []
        # data length 10000
        for i in range(10000):
            data_info = {'seg_map_path': "image_" + str(i) +'.png', 
                         'img_path': 'image_' + str(i) + '.png',
                        'label_map': self.label_map,
                        'reduce_zero_label': self.reduce_zero_label,
                        'seg_fields': []
                        }          
                        
            data_list.append(data_info)
        return data_list
    
    def prepare_data(self, idx) -> Any:
        data_info = self.get_data_info(idx)
        # data = super().prepare_data(idx)
        random_class = random.choices(self._metainfo.get("classes"), k = self.num_random_classes)
        
        if self.min_thing_size <= 60:
            self.min_thing_size = 60
        if self.max_thing_size >= self.image_size // 2:
            self.max_thing_size = self.image_size // 2
        assert self.max_thing_size >self. min_thing_size
        
        
        
        # prompts = []
        images_buffer = []
        masks_buffer = []
        # print('random class here', random_class)
        for cls in random_class:
            prompt = f"a photo of a {cls}"
            with torch.inference_mode():
                images = self.sd_pipeline(
                    prompt, num_images_per_prompt=self.num_image_per_prompt, 
                    guidance_scale=self.guidance_scale, 
                    num_inference_steps=self.num_inference_steps
                )
                gc.collect()
                if "cuda" in self.sd_device:
                    torch.cuda.empty_cache()
            images_buffer.append(images.images[0])
            h, w = images.images[0].size
            inputs = self.processor(text=cls, images=images.images[0], padding="max_length", return_tensors="pt")
            with torch.no_grad():
                # operation on cpu
                # even faster execution than kmeans clustering from unt feature maps
                outputs = self.clipseg(**inputs)
            preds = outputs.logits
            temp_mask = np.zeros_like(preds)
            # print( 'thing preds' , thing_preds.shape)
            # print(random_things[0])
            #class index with background index == class_name
            temp_idx = self._metainfo.get("classes").index(cls)
            temp_mask[(torch.sigmoid(preds) > 0.5)] = temp_idx
            temp_mask = Image.fromarray(temp_mask)
            temp_mask = temp_mask.resize((h, w), Image.NEAREST)
            temp_mask = ImageOps.grayscale(temp_mask)
            masks_buffer.append(temp_mask)
        
        if self.num_random_classes == 1 and len(images_buffer) == 1 and len(masks_buffer) == 1:
            data_info['img'] = np.array(images_buffer[0])
            data_info['seg_map']    = np.array(masks_buffer[0])
            return self.pipeline(data_info)
        
        # choose the first image as background and 
        # then remove the first image and mask from buffer            
        stuff_image = images_buffer[0]
        mask = masks_buffer[0]
        images_buffer.pop(0)
        masks_buffer.pop(0)
        
        random_sizes = random.sample(range(self.min_thing_size, self.max_thing_size), 
                                        len(images_buffer))
        random_positions_1 = random.choices([i for i in range(self.image_size)],
                                            k = len(images_buffer))
        random_positions_2 = random.choices([i for i in range(self.image_size)], 
                                            k = len(images_buffer))
        # compute the positions of things and random rotation degrees
        positions = []
        rotations = []
        for size_idx, size in enumerate(random_sizes):
            position1 = random_positions_1[size_idx]
            position2 = random_positions_2[size_idx ]
            
            # image = things_image_buffer[size_idx]
            
            if position1 + size < self.image_size:
                x_position = (position1, position1 + size)
            else:
                x_position = (position1 - size , position1)
            
            if position2 + size < self.image_size:
                y_position = (position2, position2 + size)
            else:
                y_position = (position2 - size, position2)
            
            positions.append((x_position[0],  y_position [0], x_position[1], y_position[1]))
            # random rotation
            rotation_degree = random.choice([90, 180, 270])
            rotations.append(rotation_degree)
        
        for i, thing_image in enumerate(images_buffer):
            thing_mask = masks_buffer[i]
            thing_mask = thing_mask.resize((random_sizes[i], random_sizes[i]), Image.NEAREST)
            thing_image = thing_image.resize((random_sizes[i], random_sizes[i]))
            
            thing_image.rotate(rotations[i])
            thing_mask.rotate(rotations[i])
            
            
            mask.paste(thing_mask, positions[i])
                
            stuff_image.paste(thing_image, positions[i])
            data_info['img'] = np.array(stuff_image)
            data_info['seg_map'] = np.array(mask)
        
        return self.pipeline(data_info)   
        # return super().prepare_data(idx)
        
    # def __init__(self, ann_file: str = '', metainfo: dict | None = None, data_root: str = '', data_prefix: dict = ..., filter_cfg: dict | None = None, indices: int | Sequence[int] | None = None, serialize_data: bool = True, pipeline: List[dict | Callable[..., Any]] = ..., test_mode: bool = False, lazy_init: bool = False, max_refetch: int = 1000):
    #     super().__init__(ann_file, metainfo, data_root, data_prefix, filter_cfg, indices, serialize_data, pipeline, test_mode, lazy_init, max_refetch