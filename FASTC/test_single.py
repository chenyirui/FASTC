import argparse
import os
import sys
import cv2
import numpy as np
import torch
import tqdm
from PIL import Image
from inference import BEVNetSingle
from bev_utils import Evaluator
from train_fixture_utils import make_label_vis, get_colormap
from test_sequence_factory import make
cmpcolor='kyxz4'
parser = argparse.ArgumentParser()
parser.add_argument('--model_file', type=str, required=True, help='Path to the model.')
parser.add_argument('--test_env', type=str, required=True,
                    help='Test environment. See test_sequence_factory.py')
parser.add_argument('--visualize', action='store_true',
                    help='True to visualize the predictions.')
opts = parser.parse_args()


VISUALIZATION = opts.visualize
MODEL_FILE = opts.model_file
TEST_ENV = opts.test_env
torch.cuda.set_device(1)

model = BEVNetSingle(MODEL_FILE)
test_data = make(TEST_ENV)

if model.g.include_unknown:
    ignore_idx = model.g.num_class - 1
else:
    ignore_idx = 255
e = Evaluator(num_classes=model.g.num_class, ignore_label=ignore_idx)

for i in tqdm.trange(len(test_data['scan_files'])):
    scan_fn = test_data['scan_files'][i]
    name = os.path.basename(os.path.splitext(scan_fn)[0])
    label_fn = os.path.join(test_data['label_dir'], name + '.png')

    scan = np.fromfile(scan_fn, dtype=np.float32).reshape(-1, 4)
    label = np.array(Image.open(label_fn), dtype=np.uint8)
    label_th = torch.as_tensor(label, device='cuda').long()
    if model.g.include_unknown:
        # Note that `num_class` already includes the unknown label
        label_th[label_th == 255] = model.g.num_class - 1
    logits = model.predict(scan)[0]
   
    pred = torch.argmax(logits, dim=0)

    predl=pred.cpu().numpy()
    label=label_th.cpu().numpy()

    e.append(pred[None], label_th[None])
    print(e.classwiseIoU()[-1])
    if VISUALIZATION:
        cmap = get_colormap(cmpcolor)
        label_vis = make_label_vis(label, cmap)
        pred_vis = make_label_vis(pred.cpu().numpy(), cmap)
        vis = np.concatenate([pred_vis, label_vis], axis=1)
        cv2.imshow('', cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
        cv2.waitKey(1)

ious = e.classwiseIoU()
oA=e.classwiseAcc()




if model.g.include_unknown:
    ious = ious[:-2]
    oA = oA[:-2]
print('ious:', ious)
print('miou:', np.mean(ious))
print('oa:', oA)
print('oA:', np.mean(oA))
