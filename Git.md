# Codespaces 학습 → GitHub 반영 가이드

Codespaces에서 `train_free.js`로 학습한 결과(`model_free.json`)를 GitHub 레포와 사이트에 반영하는 방법과, 안 될 때 확인할 것들이에요.

## 0. 먼저 알아둘 것

- **Codespaces는 레포의 복사본이에요.** 그 안에서 만든 파일은 `git push`를 해야 GitHub 레포에 올라가요. 자동으로 올라가지 않아요.
- 순서는 항상 **`git add` → `git commit` → `git push`** 세 단계예요. 하나라도 빠지면 레포에는 안 보여요.
- 화면 아래 상태바의 `main*`에서 `*`는 **커밋하지 않은 변경이 있다**는 뜻이에요.
- 왼쪽 소스 제어 아이콘의 큰 숫자(예: `8K+`)는 대부분 `node_modules` 폴더 때문이에요. 그대로 `git add .` 하면 수천 개 파일이 올라가요. 아래 1번을 먼저 하세요.
- 올릴 파일: `model_free.json`, `index.html`, `train_free.js`, `gen_reason.js`, 그리고 레포에 아직 없는 파일들.
- 올리지 않을 것: `node_modules/`, `train.log`, 그리고 `gen_reason.js`가 만드는 `data/reason_*.jsonl` (몇 MB, 다시 만들 수 있어요).

## 1. 처음 한 번만: 올리지 말 파일 막기

터미널을 열어요. (메뉴 ≡ → **Terminal → New Terminal**. 채팅창이 아니라 터미널이에요.)

```bash
printf 'node_modules/\ntrain.log\nnohup.out\ndata/reason_train.jsonl\ndata/reason_eval.jsonl\ndata/reason_stats.json\ndata/free_eval.json\n' >> .gitignore
```

> `data/` 폴더 전체를 막지 마세요. `index.html`이 `data/` 안의 JSON을 지식으로 읽기 때문에, 직접 넣어 둔 지식 파일이 있으면 같이 막혀요.

`node_modules`가 이미 추적되고 있으면(소스 제어에 수천 개가 "추가됨"으로 보이면) 이렇게 빼요.

```bash
git rm -r --cached node_modules
```

## 2. 지금 상태 확인

```bash
git branch --show-current     # 어느 브랜치인지 (보통 main)
git remote -v                 # 어느 레포로 올라가는지
git status -s | head -20      # 바뀐 파일 목록
ls -la model_free.json        # 학습 결과 파일이 실제로 있는지
```

## 3. 올리기

`git add .`는 쓰지 말고 **파일 이름을 직접** 적어요. 없는 파일 이름을 적으면 오류가 나니, 2번의 `ls`로 있는 것만 적으세요.

```bash
git add .gitignore model_free.json index.html train_free.js gen_reason.js
git commit -m "자유 생성 모델 추가"
git push
```

끝났는지 확인해요.

```bash
git status -sb
```

첫 줄이 `## main...origin/main` 으로 끝나고 `[ahead 1]` 같은 표시가 **없으면** 올라간 거예요. 그다음 GitHub 레포 페이지를 새로고침해서 파일이 보이는지 확인하세요.

## 4. 안 될 때

| 증상 | 원인 | 해결 |
|---|---|---|
| `nothing to commit` | 파일이 없거나 `.gitignore`에 막혀 있어요 | `ls -la model_free.json` 로 존재 확인. 있으면 `git check-ignore -v model_free.json` 로 막는 규칙 확인 |
| `pathspec ... did not match` | `git add`에 적은 파일이 없어요 | `ls`로 이름 확인. 없는 파일은 빼고 다시 `git add` |
| `rejected ... non-fast-forward` | GitHub에 내 쪽에 없는 새 커밋이 있어요 | `git pull --rebase` 후 `git push` |
| `permission denied` / 403 | 쓰기 권한이 없는 레포(남의 레포를 포크하지 않고 연 경우 등)예요 | `git remote -v` 로 레포 확인. 내 레포가 맞는지 확인하고, 아니면 내 레포에서 Codespace를 새로 열어요 |
| 소스 제어에 수천 개 변경 | `node_modules`가 추적 대상이에요 | 1번의 `.gitignore`와 `git rm -r --cached node_modules` |
| `model_free.json`이 없음 | 학습이 아직 안 끝났거나 중간 저장(500스텝) 전이에요 | `tail -20 train.log` 로 진행 확인. `ps aux \| grep train_free` 로 실행 중인지 확인 |
| 학습이 중간에 끊김 | Codespace가 멈췄어요 | 다시 열고 `ls -la model_free.json`. 있으면 `node train_free.js train --resume model_free.json --steps 5000` 로 이어요 (학습률 스케줄은 처음부터 다시 시작해요) |
| 올렸는데 사이트에서 안 바뀜 | 배포 지연, 브라우저 캐시, 브랜치 불일치 | 아래 5번 |

## 5. 올렸는데 사이트에 반영이 안 될 때

1. GitHub 레포 페이지에서 **브랜치가 `main`인지**, 파일 목록에 `model_free.json`이 보이는지 확인해요. 안 보이면 push가 안 된 거예요(3번).
2. GitHub Pages를 쓴다면 레포의 **Settings → Pages**에서 어느 브랜치를 배포하는지 확인해요. 작업한 브랜치와 달라도 반영이 안 돼요.
3. Pages는 배포에 1~몇 분 걸려요. 레포 **Actions** 탭에서 배포가 끝났는지 볼 수 있어요.
4. 브라우저 캐시 때문일 수 있어요. 시크릿 탭으로 열거나 강력 새로고침을 해보세요.
5. 사이트 폴더에 `lm.js`, `brain.json`, `model.json`, `model_free.json`, `index.html`이 **같은 위치**에 있는지 확인해요.
6. `model_free.json`을 열었는데 `"kind":"free-v1"`이 없으면 `index.html`이 무시해요. `train_free.js`가 만든 파일이 맞는지 확인하세요. (`model.json`의 이름을 바꾼 파일은 안 돼요.)

## 6. 반대 방향: GitHub에서 직접 고친 걸 가져오기

GitHub 웹에서 파일을 고쳤다면 Codespace에서 먼저 받아와요.

```bash
git pull
```

## 7. Codespace를 지우기 전 체크

Codespace를 삭제하면 안에 있던 올리지 않은 파일이 사라져요. 지우기 전에 확인하세요.

```bash
git status -sb              # 바뀐 파일이 없고 [ahead]가 없어야 해요
ls -la model_free.json      # 학습 결과가 레포에 올라갔는지 GitHub에서도 확인
```

push가 어려우면 왼쪽 파일 탐색기에서 `model_free.json`을 길게 누르거나 우클릭 → **Download**로 폰/PC에 직접 저장할 수도 있어요.

## 8. 자주 쓰는 명령 한눈에

```bash
git status -sb                      # 상태 + 올릴 게 남았는지
git add <파일>                       # 올릴 파일 담기
git commit -m "메시지"               # 기록 만들기
git push                            # GitHub로 올리기
git pull --rebase                   # GitHub 변경 먼저 받기
tail -f train.log                   # 학습 진행 보기 (Ctrl+C로 보기만 중단, 학습은 계속)
ps aux | grep train_free            # 학습이 돌고 있는지
```
