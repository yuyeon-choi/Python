# Python

Python을 공부하는 동안 정리해 놓는 저장소 입니다.
 ### Terminal 코드
 git add * : 모든 변경사항을 깃허브에 반영

 git commit -m "Update README.md"

 git config --global user.name "yuyeon-choi" : 내 깃허브 레포지트리와 연결

 git push: 연결된 레포지트리로 보내기

파이썬 환경 충돌문제있으면 코랩으로 실행하기
https://colab 


[git bash]   
ls : 파일 확인   
git add 2022_10_18/numpy.md : 수정할 파일 지정

202

master 분기생성 및 push
git checkout -b master
git push origin master


local 에 저장소를 clone 한 경우 다음과 같이 기본 저장소 이름을 바꿔줘야 합니다.
git branch -m master main
git fetch origin
git branch -u origin/main main
-------------------
1. git init

2. [F1] git clone

repository 연결 후 fetch 작업
git push --set-upstream origin main

3. git remote add origin "http://"
    - 연결되었는지 확인! => git remote -v

+ 브런치 바꾸기
git checkout main