import torch
import numpy as np
import json
import sys

sys.path.append('ASL-citizen-code/ST-GCN')
from architecture.st_gcn import STGCN
from architecture.fc import FC
from architecture.network import Network

torch.set_default_dtype(torch.float64)

# Load gloss dictionary
with open('gloss_dict.json', 'r') as f:
    gloss_dict = json.load(f)
idx2gloss = {v: k for k, v in gloss_dict.items()}
n_classes = len(gloss_dict)

# Load and preprocess the captured sample (replicates asl_citizen_dataset_pose.py exactly)
def preprocess(npy_path, max_frames=128):
    data0 = np.load(npy_path)
    length = data0.shape[0]

    # pad or downsample to max_frames
    if length > max_frames:
        # simple downsample (matches training code's downsample logic closely enough for testing)
        indices = np.linspace(0, length - 1, max_frames).astype(int)
        data0 = data0[indices]
    elif length < max_frames:
        data0 = np.pad(data0, ((0, max_frames - length), (0, 0), (0, 0)))

    # normalize using shoulder distance
    shoulder_l = data0[:, 11, :]
    shoulder_r = data0[:, 12, :]
    center = np.mean((shoulder_l + shoulder_r) / 2, axis=0)
    mean_dist = np.mean(np.sqrt(((shoulder_l - shoulder_r) ** 2).sum(-1)))
    if mean_dist != 0:
        data0 = (data0 - center) * (1.0 / mean_dist)

    # select subset + reorder (pose, left hand, right hand)
    data0 = data0[:, 0:75, :]
    posedata = data0[:, 0:33, :]
    rhdata = data0[:, 33:54, :]
    lhdata = data0[:, 54:, :]
    data = np.concatenate([posedata, lhdata, rhdata], axis=1)

    keypoints = [0, 2, 5, 11, 12, 13, 14, 33, 37, 38, 41, 42, 45, 46, 49, 50, 53, 54,
                 58, 59, 62, 63, 66, 67, 70, 71, 74]
    data = data[:, keypoints, :]
    data = np.transpose(data, (2, 0, 1))  # (channels, frames, nodes)

    return torch.from_numpy(data).double().unsqueeze(0)  # add batch dimension


# Load model
graph_args = {'num_nodes': 27, 'center': 0,
              'inward_edges': [[2, 0], [1, 0], [0, 3], [0, 4], [3, 5],
                               [4, 6], [5, 7], [6, 17], [7, 8], [7, 9],
                               [9, 10], [7, 11], [11, 12], [7, 13], [13, 14],
                               [7, 15], [15, 16], [17, 18], [17, 19], [19, 20],
                               [17, 21], [21, 22], [17, 23], [23, 24], [17, 25], [25, 26]]}

stgcn = STGCN(in_channels=2, graph_args=graph_args, edge_importance_weighting=True)
fc = FC(n_features=256, num_class=n_classes, dropout_ratio=0.05)
model = Network(encoder=stgcn, decoder=fc)
model.load_state_dict(torch.load('models/ASL_citizen_stgcn_weights.pt', map_location='cuda'))
model.cuda()
model.eval()

# Run prediction
inputs = preprocess('live_sample.npy').cuda()

with torch.no_grad():
    predictions = model(inputs)
    probs = torch.softmax(predictions, dim=1)
    top5 = torch.topk(probs, 5, dim=1)

print("\nTop 5 predictions:")
for i in range(5):
    idx = top5.indices[0][i].item()
    conf = top5.values[0][i].item()
    print(f"  {idx2gloss[idx]}: {conf*100:.2f}%")
