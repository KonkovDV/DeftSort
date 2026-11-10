# Аудит v14: функции, стыки, слепые пятна, сравнение с лёгкими методами из литературы

Ветка `fix/v14` (от `main` = `1cdd15e`). Все числа ниже получены скриптами из этого каталога на коде ветки; push и Kaggle не запускались.
Скрипты: `audit_inventory.py` (перечень функций и покрытие), `ffi_edges.py` (шов Rust ↔ NumPy), `accel_bitwise.py` и `accel_tol.py` (62 кадра), `doc_refs.py` (ссылки файл:строка в документации).

## 1. Перечень функций и покрытие

- Функций и методов в `src/team048_solution/team048_solution` (обход AST, вложенные тоже): **249**.
- Покрытие тестами (coverage.py 7.17, `--cov-branch`; объединены прогон с Rust и прогон `TEAM048_ACCEL=0`, в каждом 204 теста):
  - строки: 2427 исполняемых, не исполнены 303, то есть **87.5 %**;
  - строки и ветви вместе (TOTAL coverage.py): **84 %**;
  - отдельно: Rust 82 %, NumPy 84 % (строки+ветви).
- Полностью покрыты 159 функций, частично 69, ни одной исполненной строки у 21.

| Модуль | Функций | Покрыты полностью | Не исполнены | Строки+ветви |
|---|---:|---:|---:|---:|
| `_accel.py` | 19 | 16 | 0 | 93 % |
| `arm.py` | 50 | 41 | 0 | 92 % |
| `budget.py` | 8 | 6 | 0 | 93 % |
| `geometry.py` | 10 | 5 | 2 | 65 % |
| `grasp.py` | 6 | 4 | 0 | 97 % |
| `grip.py` | 19 | 10 | 1 | 87 % |
| `kinematics.py` | 17 | 10 | 0 | 89 % |
| `node.py` | 68 | 39 | 11 | 78 % |
| `perception.py` | 17 | 5 | 4 | 82 % |
| `planning.py` | 20 | 11 | 2 | 81 % |
| `sensing.py` | 6 | 6 | 0 | 100 % |
| `trays.py` | 9 | 6 | 1 | 87 % |

Почему 21 функция не исполнена ни разу:

- В продакшене не вызываются (мёртвый код, на поведение не влияют): `geometry.fresh`, `geometry.project`, `perception.plane_inliers` (только импорт в `node.py:31`), `perception.RangeClassifier` (`__init__`, `classify`), `perception.tray_color`, `planning.cartesian_waypoints`, `grip.GraspMonitor.calibrated`. `trays._components` вызывается только без scipy, а scipy объявлен зависимостью.
- Ветка `planning.CollisionModel._certify_escape_full` (`planning.py:216`) работает, только если старт уже в пересечении (`gs0 <= 0`). В тестах она не исполнялась.
- Методы узла, которые шевелят робота и ждут ответа симулятора: `on_js`, `on_hand`, `hand_log`, `sleep`, `hand_close`, `measure_near_edge`, `waist`, `go_home`, `retreat`, `approach_guard`, `main` (`node.py:125–898`). Заглушки ROS в тестах их подменяют. Я прочитал их построчно, замечания в разделе 3; поведение подтверждено только логами Kaggle v7–v11b.

Полная таблица по каждой функции (файл:строка, число исполняемых строк, покрытие, тесты, где функция названа) — в приложении в конце файла.
Ревью единиц, систем координат и границ входов сделано по модулям. Ниже указаны только найденные дефекты и сомнения. Для функций, которых нет в разделах 2–3, замечаний нет.

## 2. Стыки между модулями

| Стык | Что передаётся | Проверено | Итог |
|---|---|---|---|
| ROS → `sensing.PairBuffer` → `node.capture` | Image rgb8/bgr8 и depth 16UC1 (мм), пары по метке времени | замок буфера (`test_v13`), `bgr8` → RGB (`test_v14.ColourEncoding`) | ОК. Другая кодировка depth (32FC1) или `step ≠ width·bpp` дают исключение в `reshape`, а не тихую ошибку. Тогда `observe` повторяется, после неудачи публикуется `done` |
| `capture` → `perception.deproject` | dmm float64 (мм) → м (`/1000`), K из `camera_info.k` (3×3, построчно), T pelvis←камера из `/tf` на метку кадра | формула `node.py:375`, FK головы против `/tf` (`test_ideas`), реальные кадры v5/v7 | ОК. Оптическая система камеры (x вправо, y вниз, z вперёд) подтверждается тем, что на 62 кадрах находится горизонтальная плоскость стола |
| `/tf` → `quat_to_T` | кватернион ROS (x, y, z, w) | та же формула, что в `geometry.quaternion_transform` (порядок xyzw) | ОК. Нормировки нет (`node.py:91`): при |q| = 1 ± 1e-7 ошибка поворота порядка 1e-7, пренебрежимо |
| `perception` → `arm` | плоскость (n, d), n·p + d = 0, n_z ≥ 0.96, pelvis | `plane_plausible`, `plane_height` | ОК |
| `trays.detect` → `arm.tray_geom` → `node.pick_place` | dict: center, size (короткая, длинная), axis (единичная, длинная ось), height | `tray_geom` (`arm.py:314`), `slot_offset` | Знак оси приведён к канону (`trays.canonical_axis`, v14). Частичные детекции в планы не попадают (`node.py:412`) |
| `grasp.candidates` → `arm.grasp_pose` | центр xy, approach (2D), close_axis, kind | `test_v10_clear`, `test_yaw_release` | ОК |
| `arm` → `kinematics` → `node.cmd` | порядок суставов `ARM` (`arm.py:23`), id моторов 11–17 и 21–27 (`node.py:36`) | `chain.names == [f'{n}_{side}_joint' for n in ARM]` для обеих рук (проверено), id из ТЗ (`research/REQUIREMENTS_RU.md` B4) | ОК. `q_now` молча подставляет 0 для сустава, которого нет в `/joint_states` (`node.py:157`), см. риски |
| Углы | радианы везде; градусы только в логах, документации и `grasp.RIM_ANGLE_DEG` | grep `deg2rad`/`radians`/`/180`: единственное преобразование — `grasp.py:110` (`np.deg2rad(deg)` для углов щипка за бортик); `HEAD_SCAN_YAWS` в радианах (`node.py:50`) | ОК |
| `grip` → SetForce | 6 значений: DOF 1–5 = 0.4, DOF 6 = 1.0 | единственный путь `grip.force_vector` → `Sol.hand_requests` (`node.py:236`); `/inspire_hand/ctrl` — только позиции (JointState) | ОК: сжатие 0.3–0.5 на любом пути и при любом env (`test_v14.ForceRange`) |
| Rust ↔ NumPy (FFI) | массивы через `np.ascontiguousarray(..., float)` | `ffi_edges.py`: 50 граничных входов (NaN, inf, нули, отрицательные, пустые, не-contiguous, Fortran-порядок, uint16, float32) | 48 совпадают точно, 1 в пределах округления (вырожденный отрезок под 45°, ширина 5.8e-17 против 4.8e-17), 1 расходится (float32 в `table_plane`; в узел float32 не попадает, `deproject` всегда даёт float64). **Дефект:** пустой кадр глубины ронял Rust-`deproject` (паника `chunk size must be non-zero`, perc.rs:337), и Rust отключался на весь процесс. Исправлено (`perception.py`, проверка `dmm.size`), тест `test_v14.FfiEdges` |
| ИК/столкновения в Rust | q проходит `validate_q` (конечность, пределы) до FFI | `kinematics.py:99`, `planning.py:111` | ОК: NaN и inf до Rust не доходят |

## 3. Слепые пятна

| Тема | Что есть | Остаток |
|---|---|---|
| Бюджет 270/405 с | `budget.can_start`, `place_time`, `hold_time_bound` до подъёма; тесты `test_audit`, `test_v10_budget`, `test_compliance` | Время переноса откалибровано на TRACE v8; при другом RTF симулятора — не проверено |
| Потоки ROS | `PairBuffer` под замком (v13), `js` под `s.lock` | `hpos`/`hvel` пишет колбэк, а читает основной поток без замка (`node.py:127–131`). В CPython это не исключение, в худшем случае пропуск одного отсчёта |
| Потеря сенсора | `capture` ждёт 5 с, потом берёт самую свежую пару с предупреждением в логе (`node.py:367`); если пары нет — исключение → `observe` повторяется → `done` | Свежесть «самой свежей» пары не ограничена. В логах v7–v11b не встречалось |
| Потеря `/joint_states` | `while not s.js` без таймаута (теперь внутри try) | Без `/joint_states` узел ждёт бесконечно и `done` не публикует. Ничего не поднято, так что штрафа нет. Оставлено как есть |
| Остановка `/clock` | `Sol.sleep` и все ожидания идут по симвремени | Если `/clock` встал, циклы ожидания не кончаются. Это и есть остановка симулятора |
| «Ничего в руке в конце» (−10) | `end_safe` раскрывает обе кисти и уводит руки домой на любом выходе, включая ошибку настройки (v14) и исключение в `pick_place` (v13). Проверка `lift_allowed`/`hold_time_bound` до подъёма | Выпадение на переносе (v11b Utana) — физика, а не логика. Без GPU не проверить |
| Сила 0.3–0.5 | раздел 2 | DOF 6 (поворот большого пальца) = 1.0 — осознанное допущение, `docs/COMPLIANCE.md` |
| Запреты (#645, #595) | AST-тесты `test_v14.Rules`, `test_compliance`; grep по всему дереву сдачи (раздел 5) | — |
| Ничья по размеру компонент | `trays._largest_component`: `np.argsort(-sizes)` не устойчива при равных размерах | Две крупнейшие компоненты одного цвета с равным числом пикселей (≥ 150) на 62 кадрах не встречались |
| Ось ящика вдоль x | для k ≥ 1 «внутренний» слот определён детерминированно (a_x > 0), но вдоль x это «ближе/дальше от робота», а не «к середине» | Полных детекций с такой ничьей на 62 кадрах нет (у всех 28 изменившихся кадров `partial: True`) |

## 4. Сравнение с лёгкими методами из литературы

Условия: только CPU, только сенсоры робота, без тяжёлых моделей (VLA, GraspGen, FoundationPose, YOLO под AGPL) и без чужого кода. Ссылки проверены поиском 2026-11-10.

| Узел | DeftSort | Лёгкие методы из литературы | Вывод |
|---|---|---|---|
| Плоскость стола | RANSAC по горизонтальным кандидатам (n_z ≥ 0.96), затем уточнение по инлаерам: полная МНК-плоскость через SVD (`_refine_plane`, `perception.py:131`) | RANSAC: Fischler, Bolles, *Comm. ACM* 24(6), 1981. Шаг уточнения по сути совпадает с локальной оптимизацией LO-RANSAC: Chum, Matas, Kittler, DAGM 2003 | Соответствует. На 62 кадрах плоскость найдена везде; улучшать нечего |
| Сегментация | связные компоненты организованного облака (4-соседство, скачок глубины < 2.5 см) над плоскостью | Trevor, Gedikli, Rusu, Christensen, «Efficient Organized Point Cloud Segmentation with Connected Components», SPME 2013 | Тот же класс методов. Сравнение нормалей (у Trevor и др.) разделило бы касающиеся предметы. Только после GPU-проверки: меняет сегменты и, значит, планы |
| Опознание и «поза» | признаки (высота q98, размеры по главным осям), венгерский алгоритм (Kuhn 1955) против априорных размеров; ось — минимальный прямоугольник по 91 углу с квантилями | Точный минимальный прямоугольник вращающимися калиперами: Freeman, Shapira, *Comm. ACM* 18(7), 1975; Toussaint, MELECON 1983 | Скан с квантилями устойчивее к выбросам на краях, чем точный калипер; шаг 1° даёт ошибку оси ≤ 0.5°. Оставить |
| Выбор захвата | аналитические кандидаты (сбоку, бортик, сверху) по выпуклой оболочке, ширине кисти, моменту m·g·r, зазору до соседей | Антиподальные захваты и force-closure: Nguyen, *IJRR* 7(3), 1988. Выборка кандидатов из облака + CNN-оценка: ten Pas и др., «Grasp Pose Detection in Point Clouds», *IJRR* 2017 (arXiv:1706.09911) | Геометрическая часть совпадает с генерацией кандидатов у ten Pas и др.; их CNN — обученная модель, её не берём. Своя оценка антиподальности по нормалям облака — кандидат после GPU-проверки (меняет выбор захвата) |
| ИК | DLS с нуль-пространством, несколько затравок, ограничение шага | DLS: Nakamura, Hanafusa, *J. Dyn. Sys. Meas. Control* 108(3), 1986; Wampler, *IEEE T-SMC* 16(1), 1986. TRAC-IK (параллельно Ньютон-Якоби и SQP со случайными перезапусками): Beeson, Ames, Humanoids 2015 | Наши несколько затравок — аналог перезапусков TRAC-IK. Отказов ИК на достижимых целях в логах не видно, план ограничивают столкновения. Не менять |
| Проверка пути | сферы робота и сцены; сертификат отрезка через оценку сверху длины дуги точек (рычаги) и шаг 3 мм | Адаптивная динамическая проверка с гарантией: Schwarzer, Saha, Latombe, *IEEE T-RO* 21(3), 2005. Векторизованные проверки сфер и FK на CPU (SIMD): Thomason, Kingston, Kavraki, «Motions in Microseconds via Vectorized Sampling-Based Planning», ICRA 2024 (arXiv:2309.14545) | Сертификат — тот же принцип, что у Schwarzer и др. (оценка дуги против расстояния). Ядра Rust по духу близки VAMP. Полноценный планировщик (RRT-Connect) вместо прямых путей с промежуточной точкой «вверх» вернул бы часть «нет плана» у торса. Большая правка, только после GPU-проверки |
| Контроль захвата | двухфазное сжатие: до остановки пальцев по `/inspire_hand/state`, затем на 0.15 хода дальше; проверка удержания по отклонению и усилию; охрана подхода | Контроллер с событиями Close/Load/Lift-and-Hold и реакцией на проскальзывание: Romano и др., «Human-Inspired Robotic Grasp Control With Tactile Sensing», *IEEE T-RO* 27(6), 2011 | Наша схема — упрощённый вариант без тактильных датчиков (их нет в интерфейсе). Наращивать силу при проскальзывании нельзя: сила ограничена 0.3–0.5 (#648). Детектор проскальзывания по росту скорости пальцев на переносе — после GPU-проверки |

Что внедрено до заморозки: ничего из этой таблицы. Каждый пункт меняет сегменты, выбор захвата или пути, а значит планы на 62 кадрах. Без GPU-прогона нельзя показать, что это не ухудшает результат в симуляторе. Правки v14 касаются только надёжности и соответствия правилам. Захваты и укладки они не меняют (раздел 5).

## 5. Итоги проверок v14

- Планы `main` (`1cdd15e`) → `fix/v14` на 62 кадрах, Rust, одна среда (Python 3.13, NumPy 2.x): 5358 записей. Различаются 28, все `tray:tray_left` с `partial: True`: ось `(-1, 6e-17)` → `(1, -6e-17)`, то есть та же прямая с другим знаком; центр, размер, высота и пиксели те же. В планирование частичные детекции не идут. Различий в захватах, укладках и путях домой 0.
- Rust ↔ NumPy в целевой среде (Python 3.10.22, NumPy 1.26.4, OpenBLAS 1 поток): 5358 записей. Депроекция, плоскость, ящики, маски и сегменты совпадают бит-в-бит. Признаки сегментов отличаются до 5.6e-17, кандидаты до 5.6e-17, захваты до 1.1e-14, укладки до 2.3e-14. Дискретных различий 0.
- Ссылки файл:строка в `docs/MATH_PHYSICS.md` (39) и `docs/ORG_ANSWERS.md` (22 группы) указывают на нужные строки (`doc_refs.py --check`). Значения `docs/PARAMETERS.md` сверены с атрибутами модулей.

## 6. Что нельзя проверить без GPU-симуляции

- Удержание и проскальзывание при силе 0.4 (тяжёлые предметы ≥ 0.22 кг, тонкие коробки, перенос).
- Поворот большого пальца с силой 1.0: принимает ли его судья (#648) и держит ли палец противопоставление при 0.4.
- Работу `go_home`, `retreat`, `waist`, `approach_guard` в новом коде v13–v14 (`recover`, `end_safe`, повтор `observe`) в реальном цикле симулятора.
- Время циклов при другом RTF; ожидание `/sim/task` и `/joint_states` при медленном старте.
- Новый знак оси ящика (v14) в раскладке, где полная детекция даёт ничью 0°/90° (на 62 кадрах такой нет).

## Приложение: все функции

| Файл:строка | Функция | Строк (исполн.) | Покрыто | Тесты, где упомянута |
|---|---|---:|---:|---|
| `_accel.py:31` | `enabled` | 1 | 100% |  |
| `_accel.py:35` | `enabled2` | 1 | 100% | test_accel_phase2, test_v14 |
| `_accel.py:39` | `failed` | 3 | 100% | test_v13, test_v14, test_v8fixes |
| `_accel.py:47` | `guard` | 2 | 50% | test_v8fixes, test_v9fix |
| `_accel.py:53` | `fast_chain` | 6 | 67% |  |
| `_accel.py:65` | `fast_collision` | 14 | 86% |  |
| `_accel.py:85` | `plane_inlier_counts` | 1 | 100% |  |
| `_accel.py:89` | `flood_fill` | 1 | 100% |  |
| `_accel.py:97` | `pcg_state` | 6 | 100% |  |
| `_accel.py:107` | `ransac_plane` | 1 | 100% | test_accel_phase2 |
| `_accel.py:112` | `deproject` | 1 | 100% | test_accel_phase2, test_v14 |
| `_accel.py:117` | `tray_window_mask` | 1 | 100% | test_accel_phase2 |
| `_accel.py:123` | `rot_rect_mask` | 1 | 100% | test_accel_phase2 |
| `_accel.py:127` | `segment_components` | 1 | 100% | test_accel_phase2 |
| `_accel.py:133` | `min_caliper` | 1 | 100% | test_accel_phase2 |
| `_accel.py:137` | `min_area_rect_scan` | 1 | 100% | test_accel_phase2 |
| `_accel.py:141` | `min_dist_xy` | 1 | 100% | test_accel_phase2 |
| `_accel.py:145` | `fast_env` | 1 | 100% |  |
| `_accel.py:149` | `table_mask` | 1 | 100% | test_accel_phase2 |
| `arm.py:26` | `load_urdf` | 15 | 87% | test_compliance |
| `arm.py:45` | `joints` | 5 | 100% | test_v8fixes |
| `arm.py:55` | `inertial` | 4 | 100% |  |
| `arm.py:60` | `link_radius` | 1 | 100% |  |
| `arm.py:62` | `fk_link` | 7 | 100% | test_accel_equivalence, test_accel_phase2, test_compliance, test_ideas … |
| `arm.py:72` | `fingertip` | 3 | 100% |  |
| `arm.py:77` | `hand_opening` | 11 | 100% |  |
| `arm.py:92` | `arm_chain` | 2 | 100% |  |
| `arm.py:101` | `mesh_spheres` | 6 | 100% | test_ideas |
| `arm.py:111` | `hand_mesh` | 9 | 89% | test_accel_phase2, test_ideas |
| `arm.py:135` | `hand_open_points` | 11 | 100% | test_v10_clear |
| `arm.py:153` | `box_coords` | 2 | 100% | test_v10_clear |
| `arm.py:158` | `box_inside` | 4 | 100% |  |
| `arm.py:164` | `grasp_clear_shift` | 25 | 100% | test_v10_clear |
| `arm.py:195` | `arm_spheres` | 19 | 74% |  |
| `arm.py:226` | `torso_spheres` | 10 | 90% | test_v9fix |
| `arm.py:241` | `body_env` | 12 | 100% |  |
| `arm.py:261` | `_cached_entry` | 6 | 100% |  |
| `arm.py:270` | `_cached` | 1 | 100% |  |
| `arm.py:272` | `_key` | 1 | 100% |  |
| `arm.py:274` | `plane_height` | 1 | 100% | test_v9fix |
| `arm.py:277` | `plane_plausible` | 2 | 100% | test_compliance, test_node_stub |
| `arm.py:281` | `_table_args` | 4 | 100% |  |
| `arm.py:286` | `_table_entry` | 2 | 100% |  |
| `arm.py:289` | `table_env` | 1 | 100% | test_accel_phase2, test_arm, test_compliance, test_ideas … |
| `arm.py:296` | `table_env_build` | 11 | 100% | test_accel_phase2 |
| `arm.py:314` | `tray_geom` | 4 | 100% | test_accel_phase2, test_v11, test_v12 |
| `arm.py:322` | `rotate_trays` | 4 | 100% | test_accel_phase2 |
| `arm.py:328` | `_tray_entry` | 3 | 100% |  |
| `arm.py:332` | `tray_env` | 1 | 100% | test_accel_phase2, test_ideas |
| `arm.py:334` | `tray_env_build` | 12 | 100% | test_accel_phase2 |
| `arm.py:351` | `home_dict` | 2 | 100% |  |
| `arm.py:354` | `other_home` | 1 | 100% |  |
| `arm.py:358` | `Arm.__init__` | 11 | 100% | test_node_stub, test_v11, test_v13 |
| `arm.py:372` | `Arm.set_scene` | 21 | 95% | test_accel_equivalence, test_accel_phase2, test_arm, test_compliance … |
| `arm.py:399` | `Arm._body_entry` | 5 | 100% |  |
| `arm.py:405` | `Arm.set_other` | 8 | 88% | test_accel_phase2, test_v11 |
| `arm.py:415` | `Arm.path_home` | 8 | 62% | test_v9fix |
| `arm.py:437` | `Arm.roadmap_nodes` | 14 | 100% |  |
| `arm.py:452` | `Arm.roadmap_path` | 19 | 100% |  |
| `arm.py:474` | `Arm.tcp` | 1 | 100% | test_ideas, test_node_stub, test_v10_clear, test_v12 … |
| `arm.py:476` | `Arm.lift` | 4 | 100% | test_audit, test_compliance, test_ideas, test_v10_budget … |
| `arm.py:482` | `Arm.correct` | 7 | 100% | test_ideas |
| `arm.py:492` | `Arm.position_error` | 1 | 100% | test_compliance, test_ideas |
| `arm.py:495` | `Arm.grasp_pose` | 4 | 100% | test_arm, test_v10_clear |
| `arm.py:503` | `Arm._hand_frame` | 11 | 100% | test_v12 |
| `arm.py:517` | `Arm.solve` | 5 | 100% | test_accel_phase2 |
| `arm.py:524` | `Arm._solve_fast` | 16 | 81% |  |
| `arm.py:546` | `Arm.solve_py` | 6 | 100% | test_accel_phase2 |
| `arm.py:554` | `Arm.plan_pick` | 24 | 92% | test_arm, test_v10_clear, test_v12 |
| `budget.py:16` | `TimeBudget.__init__` | 7 | 86% | test_node_stub, test_v11, test_v13 |
| `budget.py:25` | `TimeBudget.elapsed` | 1 | 100% |  |
| `budget.py:28` | `TimeBudget.remaining` | 1 | 100% | test_compliance, test_ideas |
| `budget.py:31` | `TimeBudget.record_cycle` | 5 | 100% | test_ideas |
| `budget.py:39` | `TimeBudget.can_start` | 2 | 100% | test_compliance, test_ideas, test_v13 |
| `budget.py:45` | `start_time` | 2 | 100% | test_compliance, test_ideas, test_v13 |
| `budget.py:56` | `launch_age_s` | 21 | 90% | test_compliance |
| `budget.py:83` | `time_boot` | 2 | 100% |  |
| `geometry.py:5` | `finite` | 4 | 100% |  |
| `geometry.py:12` | `rotation` | 6 | 83% | test_node_stub |
| `geometry.py:21` | `pose` | 4 | 100% | test_compliance, test_node_stub, test_v10_clear, test_v8fixes |
| `geometry.py:31` | `_close` | 1 | 100% |  |
| `geometry.py:36` | `check_transform` | 4 | 100% | test_accel_equivalence |
| `geometry.py:43` | `quaternion_transform` | 7 | 86% | test_ideas |
| `geometry.py:55` | `fresh` | 2 | 0% | test_v11 |
| `geometry.py:60` | `project` | 13 | 0% |  |
| `geometry.py:76` | `check_rotation` | 3 | 100% |  |
| `geometry.py:82` | `rotation_error` | 12 | 67% | test_accel_equivalence |
| `grasp.py:40` | `_caliper_py` | 5 | 100% |  |
| `grasp.py:47` | `footprint` | 9 | 89% | test_ideas, test_v11 |
| `grasp.py:58` | `circle_fit` | 1 | 100% | test_v11 |
| `grasp.py:61` | `_box` | 2 | 100% |  |
| `grasp.py:69` | `rim_closed` | 2 | 100% | test_v11 |
| `grasp.py:78` | `candidates` | 50 | 96% | test_accel_phase2, test_ideas, test_v10_clear, test_v11 … |
| `grip.py:38` | `_env_ratio` | 6 | 67% |  |
| `grip.py:59` | `force_vector` | 2 | 100% | test_compliance, test_v11 |
| `grip.py:65` | `finger_stroke_time` | 1 | 100% | test_audit, test_compliance |
| `grip.py:70` | `effort_held_threshold` | 3 | 100% | test_compliance |
| `grip.py:81` | `thumb_rotated` | 3 | 67% |  |
| `grip.py:93` | `jammed` | 5 | 80% | test_v8fixes, test_v9fix |
| `grip.py:103` | `contact_target` | 6 | 100% | test_compliance, test_v8fixes |
| `grip.py:127` | `held_rule` | 10 | 90% | test_v8fixes |
| `grip.py:156` | `approach_deviation` | 10 | 80% | test_v8fixes, test_v9fix |
| `grip.py:176` | `fingers_stalled` | 8 | 100% | test_v8fixes |
| `grip.py:189` | `contact_empty` | 5 | 80% | test_compliance, test_v8fixes |
| `grip.py:198` | `past_verdict` | 10 | 70% | test_compliance, test_v8fixes |
| `grip.py:218` | `force_vec` | 2 | 100% | test_ideas |
| `grip.py:224` | `GraspMonitor.__init__` | 4 | 100% | test_node_stub, test_v11, test_v13 |
| `grip.py:231` | `GraspMonitor.calibrated` | 1 | 0% | test_audit |
| `grip.py:234` | `GraspMonitor.baseline` | 1 | 100% | test_ideas, test_node_stub |
| `grip.py:237` | `GraspMonitor.verdict` | 10 | 100% | test_ideas, test_node_stub |
| `grip.py:251` | `finger_verdict` | 12 | 83% | test_ideas, test_v8fixes, test_yaw_release |
| `grip.py:269` | `fingers_settled` | 3 | 100% | test_ideas |
| `kinematics.py:21` | `numbers` | 1 | 100% | test_compliance, test_v11 |
| `kinematics.py:26` | `Chain.__init__` | 13 | 92% | test_node_stub, test_v11, test_v13 |
| `kinematics.py:43` | `Chain.from_urdf` | 34 | 76% |  |
| `kinematics.py:79` | `Chain.lock_joints` | 13 | 85% |  |
| `kinematics.py:99` | `Chain.validate_q` | 4 | 100% |  |
| `kinematics.py:105` | `Chain._rot` | 2 | 100% |  |
| `kinematics.py:110` | `Chain._fallback` | 1 | 100% |  |
| `kinematics.py:113` | `Chain.frames` | 8 | 88% | test_accel_equivalence, test_accel_phase2, test_arm, test_ideas … |
| `kinematics.py:123` | `Chain.frames_py` | 7 | 100% | test_accel_equivalence, test_accel_phase2 |
| `kinematics.py:132` | `Chain.fk` | 5 | 100% | test_accel_equivalence, test_arm, test_ideas |
| `kinematics.py:139` | `Chain.fk_py` | 1 | 100% | test_accel_equivalence, test_accel_phase2 |
| `kinematics.py:142` | `Chain.jacobian` | 5 | 80% | test_accel_equivalence |
| `kinematics.py:150` | `Chain.jacobian_py` | 8 | 100% | test_accel_equivalence |
| `kinematics.py:160` | `Chain.ik` | 16 | 88% | test_accel_equivalence, test_arm, test_ideas |
| `kinematics.py:188` | `Chain.ik_py` | 9 | 78% | test_accel_equivalence |
| `kinematics.py:201` | `Chain._ik_loop` | 28 | 100% |  |
| `kinematics.py:203` | `Chain._ik_loop.residual` | 2 | 100% |  |
| `node.py:91` | `quat_to_T` | 3 | 100% | test_node_stub |
| `node.py:96` | `Sol.__init__` | 26 | 100% | test_node_stub, test_v11, test_v13 |
| `node.py:125` | `Sol.on_js` | 1 | 0% |  |
| `node.py:127` | `Sol.on_hand` | 4 | 0% |  |
| `node.py:132` | `Sol.hand_log` | 6 | 0% | test_v8fixes |
| `node.py:144` | `Sol.now` | 1 | 100% | test_ideas, test_node_stub, test_v11, test_v13 |
| `node.py:145` | `Sol.sleep` | 2 | 0% | test_compliance, test_node_stub, test_v8fixes |
| `node.py:148` | `Sol.log` | 1 | 100% | test_audit, test_v11, test_v12, test_v13 … |
| `node.py:150` | `Sol.cmd` | 4 | 100% | test_node_stub, test_v8fixes |
| `node.py:155` | `Sol.q_now` | 2 | 100% | test_compliance, test_v11 |
| `node.py:158` | `Sol.move_arm` | 18 | 78% | test_compliance, test_v10_budget, test_v8fixes |
| `node.py:179` | `Sol.hand_set` | 1 | 100% | test_compliance, test_v8fixes, test_yaw_release |
| `node.py:181` | `Sol.hand_open` | 1 | 100% | test_v13, test_yaw_release |
| `node.py:182` | `Sol.fingers_open` | 3 | 100% | test_v13, test_yaw_release |
| `node.py:188` | `Sol.closed_target` | 1 | 100% |  |
| `node.py:192` | `Sol.hand_close` | 1 | 0% |  |
| `node.py:194` | `Sol.wait_thumb` | 7 | 100% | test_compliance |
| `node.py:204` | `Sol.grasp_close` | 10 | 100% | test_compliance, test_v8fixes |
| `node.py:219` | `Sol.wait_fingers` | 7 | 86% | test_compliance, test_v8fixes |
| `node.py:231` | `Sol.hand_requests` | 9 | 100% | test_v14 |
| `node.py:242` | `Sol.hand_settings_step` | 26 | 92% | test_compliance, test_node_stub, test_v11 |
| `node.py:273` | `Sol.hand_settings` | 9 | 89% | test_node_stub, test_v11 |
| `node.py:278` | `Sol.hand_settings.loop` | 5 | 80% | test_accel_phase2, test_node_stub, test_v13 |
| `node.py:285` | `Sol.hand_force_applied` | 2 | 100% | test_v11 |
| `node.py:290` | `Sol.hand_cfg_summary` | 1 | 100% | test_v11 |
| `node.py:292` | `Sol.hand_verdict` | 4 | 100% | test_v8fixes |
| `node.py:300` | `Sol.hand_shadow` | 5 | 100% | test_v8fixes |
| `node.py:307` | `Sol.look` | 14 | 93% | test_compliance, test_node_stub, test_v13, test_v8fixes |
| `node.py:314` | `Sol.look.cur` | 1 | 100% |  |
| `node.py:325` | `Sol.trace` | 12 | 50% | test_accel_equivalence, test_accel_phase2, test_compliance, test_node_stub … |
| `node.py:349` | `Sol.tf` | 8 | 88% | test_node_stub, test_v14 |
| `node.py:359` | `Sol.capture` | 15 | 93% | test_compliance, test_v14 |
| `node.py:377` | `Sol.measure_plane` | 4 | 100% | test_compliance |
| `node.py:383` | `Sol.tray_mask` | 5 | 100% | test_accel_phase2 |
| `node.py:390` | `Sol.observe` | 8 | 100% | test_audit, test_node_stub, test_v13, test_v8fixes … |
| `node.py:399` | `Sol.tray_names` | 3 | 100% |  |
| `node.py:404` | `Sol.locate_trays` | 10 | 60% | test_compliance, test_node_stub |
| `node.py:418` | `Sol.survey` | 27 | 93% | test_audit, test_compliance, test_node_stub, test_v13 … |
| `node.py:449` | `Sol.measure_near_edge` | 10 | 0% | test_compliance |
| `node.py:461` | `Sol.identify` | 20 | 100% | test_accel_phase2, test_audit, test_node_stub, test_v13 … |
| `node.py:484` | `Sol.run` | 70 | 94% | test_audit, test_compliance, test_node_stub, test_v11 … |
| `node.py:568` | `Sol.end_safe` | 7 | 86% |  |
| `node.py:578` | `Sol.recover` | 8 | 75% | test_v13 |
| `node.py:590` | `Sol.observe_retry` | 3 | 100% |  |
| `node.py:595` | `Sol.use_robot_description` | 9 | 78% |  |
| `node.py:608` | `Sol.waist` | 6 | 0% | test_v10_budget |
| `node.py:616` | `Sol.go_home` | 7 | 0% | test_audit, test_node_stub, test_v13, test_v8fixes … |
| `node.py:627` | `Sol.retreat` | 9 | 0% | test_audit, test_compliance, test_node_stub, test_v10_budget … |
| `node.py:638` | `Sol.finger_dev` | 2 | 100% |  |
| `node.py:642` | `Sol.approach_guard` | 3 | 0% |  |
| `node.py:647` | `Sol.guard_abort` | 3 | 100% | test_v8fixes |
| `node.py:652` | `Sol.pick_place` | 101 | 54% | test_audit, test_compliance, test_node_stub, test_v10_budget … |
| `node.py:769` | `accel_status` | 6 | 67% | test_v8fixes |
| `node.py:777` | `replan_infeasible` | 1 | 100% | test_yaw_release |
| `node.py:784` | `held_offset` | 4 | 100% | test_v12 |
| `node.py:794` | `plan_place` | 15 | 93% | test_node_stub, test_v12, test_yaw_release |
| `node.py:817` | `rotz` | 1 | 100% | test_accel_phase2, test_yaw_release |
| `node.py:824` | `yaw_candidates` | 6 | 100% | test_yaw_release |
| `node.py:834` | `far_tray` | 1 | 100% | test_yaw_release |
| `node.py:839` | `plan_place_yaw` | 21 | 100% | test_compliance, test_v12, test_yaw_release |
| `node.py:865` | `move_time` | 1 | 100% | test_audit, test_v10_budget |
| `node.py:868` | `heavy` | 1 | 100% | test_audit, test_v10_clear |
| `node.py:869` | `place_time` | 5 | 100% | test_audit, test_compliance, test_v10_budget |
| `node.py:878` | `move_bound` | 1 | 100% | test_v10_budget |
| `node.py:882` | `hold_time_bound` | 6 | 100% | test_v10_budget |
| `node.py:893` | `lift_allowed` | 1 | 100% | test_v10_budget |
| `node.py:897` | `self_w` | 1 | 100% |  |
| `node.py:898` | `main` | 4 | 0% | test_accel_equivalence, test_accel_phase2, test_arm, test_audit … |
| `perception.py:24` | `table_plane` | 40 | 92% | test_accel_equivalence, test_accel_phase2, test_compliance, test_v9fix |
| `perception.py:70` | `_plane_from_best` | 7 | 86% |  |
| `perception.py:83` | `table_region` | 7 | 86% | test_accel_phase2 |
| `perception.py:95` | `rect_mask` | 5 | 80% | test_accel_phase2 |
| `perception.py:105` | `rect_mask_py` | 5 | 100% | test_accel_phase2 |
| `perception.py:113` | `deproject` | 5 | 100% | test_accel_phase2, test_v14 |
| `perception.py:126` | `deproject_py` | 4 | 100% | test_accel_phase2 |
| `perception.py:133` | `_refine_plane` | 11 | 82% |  |
| `perception.py:147` | `table_plane_py` | 31 | 87% | test_accel_equivalence, test_accel_phase2 |
| `perception.py:181` | `plane_inliers` | 2 | 0% |  |
| `perception.py:187` | `near_edge` | 17 | 100% | test_compliance, test_v9fix |
| `perception.py:213` | `segment` | 33 | 91% | test_accel_equivalence, test_accel_phase2, test_v13 |
| `perception.py:254` | `_components_py` | 12 | 100% |  |
| `perception.py:269` | `segment_py` | 28 | 93% | test_accel_equivalence, test_accel_phase2 |
| `perception.py:306` | `RangeClassifier.__init__` | 8 | 0% | test_node_stub, test_v11, test_v13 |
| `perception.py:316` | `RangeClassifier.classify` | 9 | 0% |  |
| `perception.py:328` | `tray_color` | 8 | 0% |  |
| `planning.py:18` | `Sphere.__post_init__` | 4 | 75% |  |
| `planning.py:25` | `spheres_bulk` | 6 | 100% |  |
| `planning.py:38` | `CollisionModel.__init__` | 27 | 89% | test_node_stub, test_v11, test_v13 |
| `planning.py:71` | `CollisionModel.margin` | 1 | 100% | test_accel_equivalence, test_arm, test_ideas, test_v10_clear … |
| `planning.py:75` | `CollisionModel.margin` | 3 | 100% | test_accel_equivalence, test_arm, test_ideas, test_v10_clear … |
| `planning.py:80` | `CollisionModel._fallback` | 1 | 100% |  |
| `planning.py:84` | `CollisionModel.environment` | 1 | 100% | test_accel_phase2, test_arm, test_yaw_release |
| `planning.py:88` | `CollisionModel.environment` | 9 | 89% | test_accel_phase2, test_arm, test_yaw_release |
| `planning.py:100` | `CollisionModel.install_environment` | 6 | 83% |  |
| `planning.py:111` | `CollisionModel.valid` | 7 | 100% | test_accel_equivalence, test_accel_phase2, test_arm, test_compliance … |
| `planning.py:120` | `CollisionModel.valid_vec` | 11 | 100% | test_accel_equivalence, test_accel_phase2 |
| `planning.py:134` | `CollisionModel.valid_py` | 15 | 100% | test_accel_equivalence |
| `planning.py:153` | `CollisionModel.certify` | 9 | 78% | test_accel_equivalence, test_accel_phase2, test_v9fix |
| `planning.py:165` | `CollisionModel.gaps` | 10 | 90% | test_v9fix |
| `planning.py:179` | `CollisionModel.certify_escape` | 22 | 100% | test_v9fix |
| `planning.py:216` | `CollisionModel._certify_escape_full` | 11 | 0% |  |
| `planning.py:230` | `CollisionModel.certify_py` | 4 | 100% | test_accel_equivalence |
| `planning.py:237` | `CollisionModel.certify_box` | 6 | 100% | test_accel_equivalence |
| `planning.py:257` | `joint_path` | 16 | 81% |  |
| `planning.py:277` | `cartesian_waypoints` | 7 | 0% |  |
| `sensing.py:7` | `stamp_s` | 2 | 100% |  |
| `sensing.py:17` | `PairBuffer.__init__` | 1 | 100% | test_node_stub, test_v11, test_v13 |
| `sensing.py:20` | `PairBuffer.add_rgb` | 2 | 100% | test_ideas, test_node_stub, test_v13, test_v14 |
| `sensing.py:24` | `PairBuffer.add_depth` | 2 | 100% | test_ideas, test_node_stub, test_v13, test_v14 |
| `sensing.py:28` | `PairBuffer.snapshot` | 2 | 100% |  |
| `sensing.py:32` | `PairBuffer.pair` | 13 | 100% | test_ideas, test_v13 |
| `trays.py:34` | `chroma_mask` | 6 | 100% | test_ideas |
| `trays.py:44` | `_components` | 7 | 0% |  |
| `trays.py:57` | `_largest_component` | 8 | 100% |  |
| `trays.py:77` | `min_area_rect` | 12 | 83% | test_accel_phase2, test_v14 |
| `trays.py:96` | `min_area_rect_py` | 11 | 100% | test_accel_phase2 |
| `trays.py:110` | `_rect_result` | 4 | 100% |  |
| `trays.py:117` | `detect` | 37 | 89% | test_accel_phase2, test_compliance, test_ideas, test_v14 |
| `trays.py:168` | `canonical_axis` | 4 | 100% | test_v14 |
| `trays.py:180` | `slot_offset` | 4 | 100% | test_ideas, test_v12 |
