# v9: чек-лист перед push (сессии A и B)

Сейчас ничего не отправлено. Push разрешает только родительский агент или пользователь.

## Перед сборкой
1. В master слиты compliance (4cb594b), rust-accel2 (68d6b61) и v8fixes (959edcf, d5eef56). Без этого `make_v9.py` откажется собирать ядро.
2. В master пройдены тесты, в двух режимах:
   - `cd src/team048_solution/test && /workspace/venv310/bin/python -m pytest -q`;
   - то же с `TEAM048_ACCEL=0`.
3. `git -C /workspace/solution/nssr-sol-team-048 status` чистый: собирается только закоммиченное, через git archive.
4. В vendor/ лежит wheel той версии, что в master (у rust-accel2 это 0.2.0), и SHA256SUMS с ним совпадает.

## Сборка, локально и без push
- `cd /workspace/fixes/kaggle-sim/v9 && python3 make_v9.py A --ref master && python3 make_v9.py B --ref master`
- Проверить, что в выводе стоит `missing []` и у wheel нужная версия. Код ядер ≈ 0,64 МБ.
- Ядра: kA → dmitrijkv/team048-sim-v9a (sort3, seed 103, HACK_SELF_COLLISION=1, RUNTO 2400), kB → dmitrijkv/team048-sim-v9b (sort2, seed 101, RUNTO 1600). Оба private, T4, общий лимит сессии 95 мин.

## Квота
- `kaggle quota`: GPU должно быть не меньше 2 ч. На 14:33 МСК было 3,43 ч.
- Оценка на сессию: A ≈ 78 мин (setup 12 + Isaac ≤ 22 + прогон ≤ 40 + оценка 3), B ≈ 65 мин. Обе параллельно (лимит 2 одновременно) ≈ 2,4 ч по верхней оценке.
- Если GPU меньше 2,5 ч, запускать только A.

## Push (только с разрешения)
- `kaggle kernels push -p kA`, `kaggle kernels push -p kB`. Токен не печатать, kernel id private.

## Что смотреть в выводе
- `ACCEL_SETUP: accel import OK`, а в логе решения строки `accel on ...` и `accel at end on ...`.
- `tray sources`: у всех трёх лотков (A) или двух (B) источник `camera`.
- `self-collisions ON (sim log)` в sort.log (только A).
- `run rc=0`, не 124. `done at` < 405 (A) или < 270 (B).
- В TRACE и HAND:
  - `guard` (где сработал, какие пальцы);
  - `jam`;
  - у `lifted` поля `verdict`, `verdict_past`, `verdict_v7` против факта `over_tray`/счёта;
  - строки `no plan:` с причиной;
  - `re-observe after contact`.
