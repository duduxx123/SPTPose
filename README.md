# Enhanced Human Pose Estimation via Self-Distilled and Token-Pruned Transformer

The code corresponds to the paper submitted to The Visual Computer.

## Installation & Quick Start
SPTPose was developed with reference to SDPose. To run the project, please refer to the <a href="https://github.com/MartyrPenink/SDPose">SDPose repository</a>.
```
conda create -n sdpose python=3.8 pytorch=1.7.0 torchvision -c pytorch -y
conda activate sdpose
pip3 install openmim
mim install mmcv-full==1.3.8
git submodule update --init
cd mmpose
git checkout v0.29.0
pip3 install -e .
cd ..
pip3 install -r requirements.txt
```

Train `python ./tools/train.py`

Test `python ./tools/test.py`

## Models
SPTPose-B comming soon
SPTPose-S-v1 comming soon
SPTPose-S-v2 comming soon
SPTPose-T comming soon
