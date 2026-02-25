# Call Flash Detector

웹캠에서 **red channel의 시간 변화(이전 프레임 평균 대비 증가)** 를 감지해 알림을 보냅니다.

핵심 규칙은 하나입니다.
- 화면에 빨간색이 "있다"는 이유만으로 ACTIVE가 되지 않음
- **이전 프레임 평균(red baseline) 대비 현재 red가 충분히 증가할 때만 ACTIVE**

즉, 빨간 물체가 계속 화면에 고정되어 있으면 baseline에 흡수되어 다시 `IDLE`로 돌아갑니다.

## 설치

```bash
uv sync
```

## 실행

```bash
uv run call-flash-detector --show-preview
```

## 주요 옵션

- `--history-size`: 이전 프레임 평균 창 크기(클수록 반응 느리고 안정적)
- `--pixel-delta-threshold`: 픽셀 단위 red 증가 임계값
- `--min-active-pixels`: 증가 픽셀 최소 개수
- `--activation-threshold`: 프레임 평균 red 증가 임계값 (`delta_mean`)
- `--deactivation-threshold`: ACTIVE 해제 임계값
- `--on-frames`, `--off-frames`: 상태 안정화 프레임 수
- `--cooldown`: 알림 재전송 쿨다운(초)

## 프리뷰 패널

- `DETECTION`: 최종 상태 오버레이
- `RED CHANNEL`: 현재 red 채널 강도
- `DELTA`: baseline 대비 red 증가량

윈도우 알림 메시지: `전화가 감지되었습니다`
