# MuJoCo NEMO → NVIDIA Isaac Lab 마이그레이션 (스타터)

MuJoCo MJCF 기반 **NEMO** 이족보행 로봇을 **Isaac Lab**에서 쓰기 위한 **에셋 설정·데모 스크립트**만 모은 저장소입니다.  
**MuJoCo / Brax / `legged_rl` 학습 코드는 포함하지 않습니다** (별도 프로젝트로 두는 것을 권장합니다).

- **Isaac Lab**: 공식 문서 [Isaac Lab](https://isaac-sim.github.io/IsaacLab/)  
- 이 저장소는 Isaac Lab 설치 트리 안에 **아래 경로 그대로 복사**하거나, 동일 구조로 파일을 두는 것을 전제로 합니다.

---

## 저장소에 포함된 것

| 경로 | 역할 |
|------|------|
| `source/isaaclab_assets/isaaclab_assets/robots/nemo.py` | MuJoCo `nemo5.xml`의 12개 다리 관절 이름·스티프니스·댐핑·토크 한계·home 자세에 맞춘 `NEMO_CFG` (`ArticulationCfg`). USD 파일은 **포함하지 않음** — Isaac Sim에서 MJCF 등을 임포트한 뒤 `nemo.usd`로 내보내 같은 경로에 두면 됨. |
| `source/isaaclab_assets/data/Robots/Nemo/.gitkeep` | `nemo.usd`를 둘 디렉터리 자리 표시 (빈 폴더 유지). 실제 `nemo.usd`는 사용자가 생성 후 배치. |
| `scripts/demos/nemo_viewer.py` | `nemo.usd`가 준비되면 Isaac Sim에서 NEMO를 스폰하고 기본 자세 유지·주기적 리셋으로 **시각 확인**하는 데모. |

### `nemo.py`가 하는 일 (요약)

- `ArticulationCfg`로 **스폰 USD 경로**, **초기 루트 높이·관절 초기값**, **암시적 액추에이터 그룹**(hip/knee/foot_pitch vs roll/yaw/foot_roll)을 정의합니다.
- 관절 이름은 MJCF 액추에이터와 동일하게 `l_*` / `r_*` 12축을 가정합니다.
- USD의 조인트 이름이 임포터 때문에 바뀌면, `NEMO_CFG`의 `joint_pos` 키나 액추에이터 `joint_names_expr`를 실제 이름에 맞게 수정해야 합니다.

### `nemo_viewer.py`가 하는 일 (요약)

- 지면·돔 라이트를 깔고 `NEMO_CFG`로 로봇을 스폰합니다.
- 시뮬레이션 루프에서 기본 관절 목표를 유지하고 일정 주기로 리셋합니다 (간단한 스탠딩 확인용).

---

## Isaac Lab에 넣는 방법

1. [Isaac Lab](https://github.com/isaac-sim/IsaacLab)을 클론·설치한 루트에서, 이 저장소의 다음 경로를 **동일 상대 경로**로 복사합니다.  
   - `source/isaaclab_assets/isaaclab_assets/robots/nemo.py`  
   - `source/isaaclab_assets/data/Robots/Nemo/` (`.gitkeep` 포함; 여기에 `nemo.usd` 추가)  
   - `scripts/demos/nemo_viewer.py`
2. `isaaclab_assets` 패키지에서 `nemo`를 자동으로 임포트하지 않도록 둔 경우가 많습니다. 데모에서는 다음처럼 직접 import 합니다.  
   `from isaaclab_assets.robots.nemo import NEMO_CFG`
3. Isaac Sim에서 MJCF(+STL)를 임포트해 **`source/isaaclab_assets/data/Robots/Nemo/nemo.usd`** 로 내보냅니다.
4. 실행 예 (Windows에서는 보통 `isaaclab.bat` 사용):

   ```bat
   isaaclab.bat -p scripts\demos\nemo_viewer.py
   ```

학습(Brax 등)과 동시에 Isaac Sim을 띄우면 VRAM 부담이 클 수 있어, **학습과 뷰어는 분리**하는 것을 권장합니다.

---

## 앞으로 할 일 (로드맵)

1. **`nemo.usd` 생성·검증** — Isaac Sim MJCF 임포트 후 조인트 이름·축 방향·접촉이 기대와 맞는지 확인.  
2. **`ManagerBasedRLEnv` (또는 동급 환경)** — MuJoCo 쪽 보상·관측·명령(예: 조이스틱 명령)을 Isaac Lab 관측/보상/액션 API에 이식.  
3. **RSL-RL (또는 Isaac Lab 튜토리얼 패턴)으로 학습 파이프라인** — 정책 체크포인트·평가 루프.  
4. (선택) **MuJoCo에서 학습한 정책 가중치**를 Isaac 쪽 네트워크에 이식할지, **처음부터 Isaac에서 재학습**할지 전략 결정.

---

## 라이선스

추가된 `nemo.py`, `nemo_viewer.py` 소스 상단의 SPDX 표기와 동일하게 **BSD-3-Clause** (Isaac Lab 프로젝트와 동일한 헤더)를 따릅니다.

---

## 관련 링크

- 이 저장소: [Migration-from-Mujoco-RL-to-Nvidia-Isaac-Lab](https://github.com/MoongiMin/Migration-from-Mujoco-RL-to-Nvidia-Isaac-Lab)
