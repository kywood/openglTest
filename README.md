# 🎯 OpenGL Offscreen Vector Renderer (FastAPI)

## 📘 프로젝트 개요
이 서버는 **FastAPI + PyOpenGL + GLFW**를 사용해  
**화면을 띄우지 않고(OpenGL 오프스크린)**  
GPU로 원형 안에 다수의 화살표를 빠르게 렌더링하여  
**PNG 이미지로 HTTP 응답으로 반환**합니다.

---

## 🧩 주요 기능
✅ 클라이언트 요청으로 이미지 크기(`width`, `height`)와 화살표 개수(`arrows`) 지정  
✅ 10만~20만 개 이상의 화살표도 빠르게 렌더링 (NumPy 벡터화 최적화)  
✅ 화살표마다 **길이·색상 랜덤**  
✅ 난수 시드(`seed`) 파라미터로 동일한 결과 재현 가능  

---

## 🧱 설치 방법

### 1️⃣ Python 3.9 이상 권장  
(3.10 이상이면 `int | None` 문법도 그대로 사용 가능)

### 2️⃣ 가상환경 생성 (선택)
```bash
python -m venv venv
source venv/bin/activate     # macOS/Linux
venv\Scripts\activate        # Windows


pip install -r requirements.txt
```

### ️ 실행 방법
```
 gl3.py 를 실행이후 
 http://127.0.0.1:8000/render?width=800&height=800&arrows=200000
```

### 윈도우에서의 TEST
```commandline
PS> curl.exe -o NUL -s -w "%{time_total}\n" "http://127.0.0.1:8000/render?width=800&height=800&arrows=200000"
0.119874
```

### ️ 실행 결과 -> (20 만개의 화살표 )
![img.png](img.png)


### ️TODO
```
쉐이더를 통한 성능 향샹을 함

``` 


### nividia docker 설치하기
```commandline
1.저장소 및 GPG 키 설정

distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
   && curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add - \   && curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
   
  
2. Nvidia-docker install
sudo apt-get update
sudo apt-get install -y nvidia-docker2

3.Docker 서비스 재시작
sudo systemctl restart docker

3.5 사용자가 root 가 아니므로
sudo usermod -aG docker ${USER}

TEST
Test 전에 cuda 버전 에 따른 이미지 전재 여부 확인 하는 방법
docker search nvidia/cuda

>> sudo docker run --rm --gpus all nvidia/cuda:11.0.3-base-ubuntu18.04 nvidia-smi

-- 여기 싸이트 참고
https://hub.docker.com/r/nvidia/cuda/tagsdocker run --rm --gpus all nvidia/cuda:12.8.0-cudnn-runtime-ubuntu20.04 nvidia-smi

docker run --rm --gpus all nvidia/cuda:11.0.3-base-ubuntu18.04 nvidia-smi

nvidia-docker info

   
```

### 참고 URL
```commandline
https://www.notion.so/Docker-Nvidia-Docker-1e7b6e926e1e4ebdbe99feeae95bc1eb
```