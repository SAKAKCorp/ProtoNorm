# ProtoNorm

**Heterogeneous Federated Learning with Prototype Alignment and Upscaling**
논문: [arXiv:2507.04310](https://www.arxiv.org/abs/2507.04310)

이 저장소는 이기종(Heterogeneous) 연합학습(FL) 환경에서 클라이언트별로 서로 다른 모델 구조를 사용하면서도 프로토타입(Prototype) 정렬 및 업스케일링을 통해 성능을 개선하는 **ProtoNorm** 알고리즘의 공식 구현체입니다.

원본 저장소는 [regulationLee/ProtoNorm](https://github.com/regulationLee/ProtoNorm) 이며, 본 저장소는 재현성 확보를 위해 README를 재작성한 버전입니다. 코드 로직은 원본과 동일합니다.

---

## 목차

1. [프로젝트 구조](#1-프로젝트-구조)
2. [설치 가이드](#2-설치-가이드)
3. [데이터셋 준비](#3-데이터셋-준비)
4. [학습 실행](#4-학습-실행)
5. [주요 인자(argument) 설명](#5-주요-인자argument-설명)
6. [실행 결과 확인](#6-실행-결과-확인)
7. [License](#7-license)

---

## 1. 프로젝트 구조

```
ProtoNorm/
├── LICENSE
├── README.md
├── dataset/                      # 데이터셋 생성/전처리 스크립트
│   ├── generate_cifar10.py       # CIFAR-10 클라이언트별 분할 스크립트
│   ├── data_utils/               # 데이터 로더/유틸
│   └── utils/                    # 언어/HAR/공통 유틸
└── system/                       # 학습/평가 실행 코드
    ├── main.py                   # 학습 엔트리포인트
    ├── test_args.json            # 예시 인자 파일 (참고용)
    ├── get_mean_std.py           # 데이터 통계 계산 유틸
    ├── flcore/
    │   ├── clients/              # 클라이언트 구현 (clientbase, clientprotonorm)
    │   ├── servers/              # 서버 구현 (serverbase, serverprotonorm)
    │   └── trainmodel/           # 모델 정의 (resnet, mobilenet_v2, models)
    ├── utils/                    # 결과 저장/메모리/데이터 유틸
    └── results/                  # (자동 생성) 학습 결과가 저장되는 디렉토리
```

> **중요**: `main.py`는 반드시 `system/` 디렉토리 안에서 실행해야 합니다. 코드 내부에서 `os.path.dirname(os.getcwd())/dataset/<데이터셋명>/config.json` 경로로 설정 파일을 찾기 때문입니다.

---

## 2. 설치 가이드

### 2-1. 저장소 클론

```sh
git clone https://github.com/SAKAKCorp/ProtoNorm.git
cd ProtoNorm
```

### 2-2. Python 가상환경 생성

**Ubuntu / macOS (venv 사용)**
```sh
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
```

**Conda 사용 (선택)**
```sh
conda create -n protonorm python=3.10 -y
conda activate protonorm
pip install --upgrade pip
```

### 2-3. PyTorch 설치 (CUDA 버전에 맞게 선택)

```sh
# CUDA 12.1 환경
pip install torch torchvision torchtext --index-url https://download.pytorch.org/whl/cu121

# CUDA 11.8 환경
pip install torch torchvision torchtext --index-url https://download.pytorch.org/whl/cu118

# CPU 전용 / macOS (MPS 포함)
pip install torch torchvision torchtext
```

> 자신의 CUDA 버전은 `nvidia-smi` 명령의 우측 상단 `CUDA Version` 표기로 확인할 수 있습니다.

### 2-4. 추가 Python 패키지 설치

원본 저장소에는 `requirements.txt`가 포함되어 있지 않습니다. 아래 명령으로 실제 코드가 사용하는 의존성을 모두 설치합니다.

```sh
pip install numpy scipy matplotlib scikit-learn h5py ujson calmsize
```

### 2-5. 설치 확인

```sh
python -c "import torch; print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available())"
python -c "import torchvision, sklearn, matplotlib, h5py, ujson, calmsize; print('All imports OK')"
```

---

## 3. 데이터셋 준비

CIFAR-10을 20개 클라이언트에 non-IID (Dirichlet 분할) 방식으로 나누어 저장합니다.

### 3-1. 실행

```sh
cd dataset
python generate_cifar10.py noniid - dir
cd ..
```

### 3-2. 인자 설명

| 위치 인자 | 값 예시   | 의미                                                                 |
|-----------|-----------|----------------------------------------------------------------------|
| 1번째     | `noniid`  | non-IID 분할 (`iid` 입력 시 IID 분할)                                |
| 2번째     | `-`       | 클래스 균형 비활성화 (`balance` 입력 시 균형 분할)                    |
| 3번째     | `dir`     | Dirichlet 분할 사용 (`-` 입력 시 사용 안 함, `pat` 입력 시 pathological) |

> 원본 README에는 `generate_Cifar10.py`(대문자 C)로 표기되어 있으나, 실제 파일명은 소문자 `generate_cifar10.py` 입니다.

### 3-3. 생성 결과

명령 실행 후 `dataset/Cifar10/` 아래에 다음 구조가 만들어집니다.

```
dataset/Cifar10/
├── config.json                # 클래스 수, 파티션 정보, 클라이언트별 샘플 분포
├── rawdata/                   # 원본 CIFAR-10 (자동 다운로드, 약 170 MB)
├── train/
│   ├── 0.npz  ...  19.npz     # 클라이언트 0~19의 학습 데이터
└── test/
    ├── 0.npz  ...  19.npz     # 클라이언트 0~19의 테스트 데이터
```

- 총 클라이언트 수: **20** (`generate_cifar10.py` 내 `num_clients = 20`으로 하드코딩)
- 데이터 분할 시드: `random.seed(1)`, `np.random.seed(1)` 로 고정 → 재현 가능
- 이미 생성된 상태에서 다시 실행하면 기존 config와 옵션이 같은 경우 재생성하지 않고 종료합니다.

---

## 4. 학습 실행

### 4-1. 기본 실행 (원본 README와 동일)

```sh
cd system
python main.py -did 0 -data Cifar10 -m Ht0 -algo ProtoNorm -pa -pu -csf 80
```

### 4-2. GPU 지정 예시

```sh
# 0번 GPU 사용
python main.py -did 0 -data Cifar10 -m Ht0 -algo ProtoNorm -pa -pu -csf 80

# 1번 GPU 사용
python main.py -did 1 -data Cifar10 -m Ht0 -algo ProtoNorm -pa -pu -csf 80

# CPU 사용
python main.py -dev cpu -data Cifar10 -m Ht0 -algo ProtoNorm -pa -pu -csf 80
```

### 4-3. 동종(Homogeneous) 모델로 실행

```sh
# 모든 클라이언트가 동일한 CNN 사용
python main.py -did 0 -data Cifar10 -m HmCNN -algo ProtoNorm -pa -pu -csf 80

# 모든 클라이언트가 ResNet18 사용
python main.py -did 0 -data Cifar10 -m HmRes18 -algo ProtoNorm -pa -pu -csf 80
```

---

## 5. 주요 인자(argument) 설명

### 5-1. 일반 인자

| 인자         | 기본값     | 설명                                                                 |
|--------------|-----------|----------------------------------------------------------------------|
| `-go`        | `test`    | 실험 목적/이름 (결과 파일명에 포함)                                    |
| `-dev`       | `cuda`    | `cpu` 또는 `cuda` (macOS는 자동으로 `mps` 폴백)                       |
| `-did`       | `0`       | 사용할 GPU 인덱스 (`CUDA_VISIBLE_DEVICES` 로 지정됨)                  |
| `-data`      | `Cifar10` | 데이터셋 이름 (`dataset/<이름>/config.json` 이 있어야 함)              |
| `-m`         | `Ht0`     | 모델 계열: `Ht0`(이기종 4종) / `HmCNN` / `HmRes18`                    |
| `-algo`      | `FedCST`  | 알고리즘 (본 저장소에서는 `ProtoNorm`만 구현됨)                        |
| `-nc`        | `20`      | 총 클라이언트 수 (데이터 생성 시 값과 반드시 일치해야 함)              |
| `-gr`        | `300`     | 전체 라운드 수                                                        |
| `-ls`        | `1`       | 라운드당 로컬 학습 epoch                                              |
| `-lbs`       | `32`      | 로컬 배치 크기                                                        |
| `-lr`        | `0.01`    | 로컬 학습률                                                           |
| `-jr`        | `1.0`     | 라운드당 참여 클라이언트 비율                                          |
| `-t`         | `1`       | 반복 실험 횟수                                                        |
| `-eg`        | `1`       | 평가 주기(라운드 단위)                                                 |

### 5-2. ProtoNorm 전용 인자

| 인자     | 기본값 | 설명                                                       |
|----------|--------|-----------------------------------------------------------|
| `-pa`    | (플래그) | Prototype Alignment 활성화                                |
| `-pu`    | (플래그) | Prototype Upscaling 활성화                                |
| `-csf`   | `80.0` | Constant Scale Factor (프로토타입 리스케일 파라미터)       |
| `-wots`  | (플래그) | Thomson 최적화 비활성화                                    |
| `-uit`   | `500`  | Thomson 반복 횟수                                          |
| `-ulr`   | `0.1`  | Thomson 학습률                                             |
| `-umt`   | `0.9`  | Thomson 모멘텀                                             |
| `-uest`  | `10`   | Thomson early stop 기준                                    |

### 5-3. `-m Ht0` 사용 시 클라이언트별 모델 구성

이기종 학습 옵션(`Ht0`) 은 클라이언트에게 아래 4가지 모델을 순환 할당합니다.

1. `resnet8`
2. `mobilenet_v2`
3. `torchvision.models.shufflenet_v2_x1_0`
4. `torchvision.models.efficientnet_b0`

---

## 6. 실행 결과 확인

- 결과 디렉토리: `system/results/` (없으면 자동 생성)
- 저장 파일 예시:
  - `Cifar10_ProtoNorm_test_data_distribution.npy` — 클라이언트별 클래스 분포
  - 학습 곡선 그림 (`plotting_trial_result` 에 의해 저장)
- 콘솔에는 라운드별 정확도, 평균 실험 시간이 출력됩니다.

---

## 7. License

본 저장소는 원본 저장소의 라이선스(GPL v2)를 그대로 따릅니다. 자세한 내용은 [`LICENSE`](./LICENSE) 파일을 참고하세요.
