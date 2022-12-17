오전T
12:12~ 이론쉅 듣기

가상환경 설정 
콘다 파이썬버전 3.8이상으로 설치


conda prompt 실행 >
v1.9.0
Conda

Linux and Windows
# CUDA 10.2 GPU2000
conda install pytorch==1.9.0 torchvision==0.10.0 torchaudio==0.9.0 cudatoolkit=10.2 -c pytorch

# CUDA 11.3 GPU3000
conda install pytorch==1.9.0 torchvision==0.10.0 torchaudio==0.9.0 cudatoolkit=11.3 -c pytorch -c conda-forge

# CPU Only
conda install pytorch==1.9.0 torchvision==0.10.0 torchaudio==0.9.0 cpuonly -c pytorch