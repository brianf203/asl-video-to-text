# ASL Video-to-Text Setup Instructions

This guide walks through setting up and running the isolated sign recognition
pipeline (live camera -> MediaPipe Holistic -> ST-GCN model -> predicted gloss)
on a Jetson Orin Nano running JetPack 6.2.

## 1. Clone this repository
```bash
git clone https://github.com/brianf203/asl-video-to-text.git
cd asl-video-to-text
```

## 2. Clone the ASL Citizen baseline code
```bash
git clone https://github.com/microsoft/ASL-citizen-code.git
```

## 3. Set up a Python virtual environment
```bash
python3 -m venv venv
source venv/bin/activate
```

## 4. Install PyTorch (Jetson-specific build)
```bash
pip install numpy==1.26.1
pip install torch --index-url https://pypi.jetson-ai-lab.io/jp6/cu126
```

## 5. Install remaining Python dependencies
```bash
pip install -r requirements.txt
```

## 6. Fix CUDA library paths (Jetson-specific issue)
The pip PyTorch build needs a few CUDA libraries pointed to manually. Add
these lines to the end of `venv/bin/activate`:
```bash
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.6/targets/aarch64-linux/lib:/home/YOUR_USERNAME/asl-video-to-text/venv/lib/python3.10/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH' >> venv/bin/activate
```
(Replace `YOUR_USERNAME` with your actual username, and adjust the Python
version folder if it's not 3.10.)

Then reactivate:
```bash
deactivate
source venv/bin/activate
```

## 7. Download the pretrained model weights
```bash
mkdir -p models
cd models
wget https://github.com/microsoft/ASL-citizen-code/releases/download/checkpoints_v1/ASL_citizen_stgcn_weights.zip
unzip ASL_citizen_stgcn_weights.zip
cd ..
```

## 8. Get the gloss vocabulary CSVs
```bash
pip install kagglehub
python3 -c "
import kagglehub
kagglehub.dataset_download('abd0kamel/asl-citizen', path='ASL_Citizen/splits/train.csv')
"
mkdir -p data_csv
cp ~/.cache/kagglehub/datasets/abd0kamel/asl-citizen/versions/1/ASL_Citizen/splits/train.csv data_csv/
```

## 9. Build the gloss dictionary
```bash
python3 build_gloss_dict.py
```
This should print "Total unique glosses: 2731" when done.

## 10. Set your Jetson to max performance mode
```bash
sudo nvpmodel -m 0
sudo jetson_clocks
```

## 11. Run the live pipeline
```bash
python3 capture_live.py
```
- Press `r` to start/stop recording a sign
- Press `q` to quit
- After recording, run `python3 predict.py` to see the top-5 predicted glosses

## Known limitation
On base Jetson Orin Nano hardware (non-Super), MediaPipe Holistic runs at
roughly 8fps on CPU. Signing slowly and holding each sign for 1-2+ seconds
gives more reliable results than signing at normal speed.

## Troubleshooting
If you hit `ImportError: libcudss.so.0` or similar CUDA library errors,
find the missing library and add its folder to `LD_LIBRARY_PATH`:
```bash
find venv -iname "libNAME.so*" 2>/dev/null
export LD_LIBRARY_PATH=/path/to/folder:$LD_LIBRARY_PATH
```
