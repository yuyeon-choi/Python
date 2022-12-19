오전T
12:12~ 이론쉅 듣기

### 면접준비시 참고!
https://github.com/JaeYeopHan/Interview_Question_for_Beginner


가상환경 설정 
콘다 파이썬버전 3.8이상으로 설치


conda prompt 실행 >
v1.9.0
Conda

Linux and Windows
# CUDA 10.2 GPU2000 - GTX 1660 본채
conda install pytorch==1.9.0 torchvision==0.10.0 torchaudio==0.9.0 cudatoolkit=10.2 -c pytorch

# CUDA 11.3 GPU3000
conda install pytorch==1.9.0 torchvision==0.10.0 torchaudio==0.9.0 cudatoolkit=11.3 -c pytorch -c conda-forge

# CPU Only  - 노트북
conda install pytorch==1.9.0 torchvision==0.10.0 torchaudio==0.9.0 cpuonly -c pytorch