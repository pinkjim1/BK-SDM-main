import numpy as np
import os
import torch
from transformers import CLIPTokenizer

cliptokenizer = CLIPTokenizer.from_pretrained("model/CLIP")
client_id=0
round=0

tem_index = f"{client_id}_{round}"
text_inputs = cliptokenizer(tem_index, padding=True, return_tensors="pt")

tem_arr = []
for i in text_inputs['input_ids'][0][1:]:
    if i != 49407:
        tem_arr.append(i)
    else:
        break
tt='_'.join([str(t.item()) for t in tem_arr])
print(tt)

# 打印图像和标签的形状
