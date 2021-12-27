```shell
# Our system were Windows10 and Windows Server
# At first, you should configure the CUDA(CUDA 10.0 + cudnn 7.4.1.5) environment.

# We used Anaconda3 to creating virtual environment
conda create –n tensorflow-gpu python=3.6

activate tensorflow-gpu
# install tensorflow-gpu
pip install tensorflow-gpu==1.13.2

# download the source code
mkdir -p /home/lymphdetect
cd /home/lymphdetect
git clone https://github.com/whucsss/LNdiagnosis-stage1.git
cd LNdiagnosis-stage1

# download the coco model
curl -L https://github.com/matterport/Mask_RCNN/releases/download/v1.0/mask_rcnn_coco.h5 -o model_data/

# install dependency
pip install -r requirements.txt

# start training
