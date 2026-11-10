# Чек-лист перед сдачей

Организаторы проверяют последний коммит в `main` командного репозитория на момент дедлайна, 11 ноября 2026 (#646, SCORING.md). Сдаётся одно дерево решения: этот репозиторий без истории разработки.

## Офлайн (выполняется на CPU, без симулятора)

1. [ ] `python3 -m pytest -q` из корня проходит с модулем `team048_accel` и с `TEAM048_ACCEL=0`.
2. [ ] `cd rust/team048_accel && cargo test --release` и `cargo clippy --all-targets --features python` без ошибок и предупреждений.
3. [ ] Чистая сборка по командам README: `pip install -r requirements.txt`, `bash setup.sh`, `colcon build`. Без сети `setup.sh` ставит wheel из `vendor/` (SHA256) и не падает, `run.sh` интернета не требует.
4. [ ] Отладочные топики не используются (`test_compliance.py`: `test_subscriptions`; `test_node_stub.py`: `test_subscribed_topics`; `test_v14.py`: `Rules`).
5. [ ] В коде нет констант раскладки сцены (`test_compliance.py`), `test/scene_fixture.py` пакетом не импортируется.
6. [ ] В дереве нет секретов, токенов, личных путей и ссылок на файлы вне дерева: `git grep -nIiE 'ghp_|github_pat_|token|secret|password'`, все относительные ссылки markdown ведут на существующие файлы и заголовки.
7. [ ] README и `docs/` соответствуют коду. Лицензии: URDF Apache-2.0 (UBTECH), код команды Apache-2.0, чужой код не копировался.

## В симуляторе (нужна GPU; для текущей версии не выполнялось)

8. [ ] `bash hackathon/tools/solution.sh <путь к решению>`: установка и запуск в чистом контейнере (SCORING.md).
9. [ ] Прогоны `task.sh sort2 --pool 3` и `sort3 --pool 3`. Что должно быть в логах:
   - в логе симулятора две строки, `[hack] left hand force limits` и `[hack] right hand force limits`;
   - в логе узла строка `hand settings at end` с обеими руками;
   - в TRACE найдены все ящики и ближний край стола;
   - нет `WARNING uncertified path`.
10. [ ] Один прогон с `HACK_SELF_COLLISION=1` (#629): рука не проходит через корпус и другую руку.

## Отправка

11. [ ] Push в командный репозиторий только с явного разрешения владельца.
12. [ ] После push последний коммит `main` на GitHub совпадает с локальным (`git ls-remote`).
