import sys
import time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torchvision.models.detection.faster_rcnn import fasterrcnn_resnet50_fpn

import matplotlib.pyplot as plt
from PIL import ImageDraw, ImageFont

# from nenepy.torch.utils import TorchSummary
from nenepy.torch.utils.summary.modules.network_architecture import NetworkArchitecture
from nenepy.torch.utils.summary.torchsummary import TorchSummary
from nenepy.torch.utils.summary.torchsummary_old import Summary

# x = torch.ones(size=(int(2.5 * 10 ** 8),)).cuda()
# print(x.element_size() * x.nelement() / 10 ** 6)
# del x
# for i in range(10):
#     time.sleep(1)
#
# sys.exit()
#

batch_size = 1
img = Image.open(f"{Path.home()}/Pictures/living.jpg")
img = np.array(img)
img = torch.from_numpy(img)
img = img.permute((2, 0, 1))
img = torch.stack([img] * batch_size, dim=0)
img = img.cuda() / 255.0
boxes = torch.tensor([[0, 0, 100, 100]]).cuda()
labels = torch.ones(1, dtype=torch.int64).cuda()

label = {"boxes": boxes, "labels": labels}

input_tensor = [img, [label] * batch_size]
# input_tensor = [img]

model = fasterrcnn_resnet50_fpn(pretrained=True)
model.cuda()
# model.eval()
# torch.save(model.state_dict(), "model.pth")
time.sleep(2)

# for i in range(100000):
#     print("A")
#     out = model(*input_tensor)
#     out["loss_classifier"].backward()
#     time.sleep(2)
#     print(i)
# sys.exit()
# time.sleep(1)

# summary = Summary(model, batch_size=1, is_validate=True)
summary = TorchSummary(model, batch_size=batch_size, is_validate=False, is_exit=False)
summary.forward_tensor(input_tensor)

sys.exit()
#
# out = model(img)
#
# for dic in out:
#     for key, value in dic.items():
#         print(key, value)

# # 結果の表示
#
#
# image = img[0].permute(1, 2, 0).cpu().numpy()
# image = Image.fromarray((image * 255).astype(np.uint8))
#
# boxes = out[0]["boxes"].data.cpu().numpy()
# scores = out[0]["scores"].data.cpu().numpy()
# labels = out[0]["labels"].data.cpu().numpy()
#
# boxes = boxes[scores >= 0.5].astype(np.int32)
# scores = scores[scores >= 0.5]
#
# for i, box in enumerate(boxes):
#     draw = ImageDraw.Draw(image)
#     label = "Unknown"
#     draw.rectangle([(box[0], box[1]), (box[2], box[3])], outline="red", width=3)
#
#     # ラベルの表示
#
#     from PIL import Image, ImageDraw, ImageFont
#
#     # fnt = ImageFont.truetype('/content/mplus-1c-black.ttf', 20)
#     fnt = ImageFont.truetype("UbuntuMono-R.ttf", 10)  # 40
#     text_w, text_h = fnt.getsize(label)
#     draw.rectangle([box[0], box[1], box[0] + text_w, box[1] + text_h], fill="red")
#     draw.text((box[0], box[1]), label, font=fnt, fill='white')
#
# # 画像を保存したい時用
# # image.save(f"resample_test{str(i)}.png")
#
# fig, ax = plt.subplots(1, 1)
# ax.imshow(np.array(image))
#
# plt.show()
