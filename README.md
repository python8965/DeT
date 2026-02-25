# DeT Tools

단일 uv 루트 프로젝트에서 두 도구를 함께 실행합니다.

- `call_flash_detector/`: LED 점멸/광량 변화 감지 + 알림
- `seat_presence_checker/`: 클릭 기반 자리 상태 관리 (EDIT/RUN)

## 설치

```bash
uv sync
```

## 실행

```bash
uv run call-flash-detector --show-preview
uv run seat-presence-checker
```
