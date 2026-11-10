# Чек-лист перед отправкой в `main` (не выполнять без явного разрешения владельца)

Проверяется последний коммит в `main` на момент дедлайна, 11 ноября 2026 (#646, SCORING.md).

1. [ ] Получено явное разрешение владельца на push в `nssr-hackathon/nssr-sol-team-048`, и только туда.
2. [ ] Выбран коммит DeftSort, который сдаётся. В сдаваемое дерево не входят `research/` и `HANDOFF.md`.
3. [ ] `python3 -m pytest -q` из корня проходит с модулем `team048_accel` и с `TEAM048_ACCEL=0`.
4. [ ] `bash hackathon/tools/solution.sh <путь к решению>` в симуляторе: установка и запуск в чистом контейнере (SCORING.md).
5. [ ] Прогоны `task.sh sort2 --pool 3` и `sort3 --pool 3`. В логе симулятора две строки `[hack] left hand force limits` и `[hack] right hand force limits`; в логе узла `hand settings at end` с обеими руками; в TRACE найдены все ящики и ближний край стола, нет `WARNING uncertified path`.
6. [ ] Один прогон с `HACK_SELF_COLLISION=1` (#629): рука не проходит через корпус и другую руку.
7. [ ] Отладочные топики не используются (тест `test_subscribed_topics` / `Survey.test_subscriptions`).
8. [ ] В коде нет констант раскладки сцены (тест `NoSceneConstants`), `test/scene_fixture.py` не импортируется пакетом.
9. [ ] `setup.sh` офлайн ставит wheel (SHA256), без сети падения нет. `run.sh` не требует интернета.
10. [ ] В дереве нет секретов, токенов и личных путей (`git grep -nIiE 'ghp_|github_pat_|token|secret|password'`), нет ссылок на `research/` из `src/`, `run.sh`, `setup.sh`.
11. [ ] README и `docs/` актуальны. Лицензии: URDF Apache-2.0, код команды свой, чужой код не копировался.
12. [ ] После push проверить, что последний коммит в `main` на GitHub совпадает с локальным (`git ls-remote`).

## Сборка сдаваемого коммита

История DeftSort содержит `research/`, поэтому в командный репозиторий уходит отдельный коммит без родителей (orphan), а не коммит из этой истории:

```bash
SRC=<sha>                                    # выбранный коммит DeftSort
git worktree add --detach /tmp/submit-048 "$SRC"
cd /tmp/submit-048
git checkout -q --orphan submit-048
git rm -r -q --cached research HANDOFF.md
rm -rf research HANDOFF.md
git commit -q -m "Team 048 solution (DeftSort $SRC)"
git ls-files | grep -E '^(research/|HANDOFF.md)' && echo "STOP: research in tree"
python3 -m pytest -q && TEAM048_ACCEL=0 python3 -m pytest -q
# только после явного разрешения владельца:
# git push <team-repo> submit-048:refs/heads/main
```
