conda create -n tnet python=3.7
conda activate tnet
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
conda install numpy
pip install tensorboardX
pip install tensorboard