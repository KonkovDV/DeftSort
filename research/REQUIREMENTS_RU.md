# Чек-лист требований team048 (пул 3) — статус на 2026-11-08 МСК
Источники: TASKS.md, SCORING.md (+уточнения 2–4 ноября), INTERFACES.md, pools.yaml, tasks.yaml.
Статусы: ✅ подтверждено доказательством · 🟡 частично/только mock · ❌ не сделано · ⛔ заблокировано.

## A. Стенд (инфраструктура, Kaggle T4)
| # | Требование | Статус | Доказательство |
|---|---|---|---|
| A1 | Isaac Sim 5.0 стартует headless | ✅ | kaggle lifecycle-native / joint-v2-retry: startup=true |
| A2 | RTX RGB-D 640x360, depth 16UC1 мм | ✅ | joint-v2-retry: rgb, depth_mm, camera_info=true |
| A3 | ROS2 Humble DDS, /clock, use_sim_time, TF | ✅ | joint-v2-retry: clock, use_sim_time, tf, stamps=true |
| A4 | Receiver JSON (numpy.bool_) — fix 1 | ✅ | receiver_exit0/no_error=true; fixes/test_json_value.py PASS |
| A5 | Isaac Lab 2.2 ≥10 шагов — fix 2 | ✅ | lab10steps=true |
| A6 | Закрытие симулятора — fix 3 | ✅ (паритет) | v2: disable расширений не помогает (SIGSEGV в simulation_manager.plugin при unload_all_plugins — баг порядка выгрузки Isaac 5.0); официальный sim_runner.py работает с fast_shutdown=True по умолчанию → принимаем его. Было: диагноз: SIGSEGV в unload_all_plugins (simulation_manager→carb.events); обход в fixes/kaggle-safe-close; GPU-прогон v1 не дошёл до close (зависание старта RTX >180 с), v2 в очереди |
| A7 | Код симулятора организаторов | ✅ скачан | /workspace/project2 (347/347): hack_sim, nssr_polygon sim_node/robot_commands/camera_bridge, tools, docker, bodyctrl deb |
| A8 | USD робота + склад (nssr-polygon-assets.tar) | ⛔ | только в GitHub release nssr-hackathon/nssr-uwt-polygon, нужен GH_TOKEN |
| A9 | Stage 2 совместный acceptance | ✅ функционально | safe-close v2 joint: 17/17 функциональных проверок true; не прошли только graceful/exit0 (см. A6) |

## B. Интерфейсы робота (INTERFACES.md)
| B1 | ROS_DOMAIN_ID=146, fastdds профиль | 🟡 | domain 146 используется в kaggle; fastdds_hack.xml нет |
| B2 | /{arm,head,waist}/cmd_pos (CmdSetMotorPosition: name,pos,spd,cur) | 🟡 | декларации из SDK deb (sdk-static); runtime-сверка ⛔ (нет симулятора) |
| B3 | Скорость ≤0.8 рад/с; spd rpm vs рад/с | ✅ по исходнику | robot_commands.py: spd в рад/с, MAX_SPEED=HACK_MAX_JOINT_SPEED=0.8; runtime-проверка — check_interfaces.py после ассетов |
| B4 | Пределы суставов (ID 1–3, 11–17, 21–27, 31) | 🟡 | таблица из ТЗ; URDF assets/robots/tiangong_pro скачивается |
| B5 | Кисти Inspire: /inspire_hand/ctrl, 6 DoF, 0..1 | ❌ | |
| B6 | Камеры cam_head/cam_torso, HFOV 86°, TF pelvis→camera_*_link | ❌ | |
| B7 | /arm_6dof_{l,r} WrenchStamped | ❌ | |
| B8 | /sim/task, /sim/objects/names latched (TRANSIENT_LOCAL) | ❌ | |

## C. Правила решения
| C1 | Позы только из датчиков; запрет USD/физики/внутр. API | 🟡 | заложено в v4 (mock) |
| C2 | Не использовать /sim/objects/set, randomize, task/check в финале | 🟡 | только для отладки |
| C3 | Без самопересечений, реализуемые траектории | ❌ | |
| C4 | Свой код (антиплагиат) | ✅ | |

## D. Задачи и баллы
| D1 | sort2 пул 3 (dish, box), 6 объектов, ≤270 с sim | ❌ | итог sort2+sort3 >0 обязателен для бонусов |
| D2 | sort3 пул 3 (dish, box, jar), 9 объектов, ≤405 с | ❌ | |
| D3 | sort2 пул 6 (доп.) | ❌ | |
| D4 | tubes easy/standard/color/lying | ❌ | после сортировки |
| D5 | −10 за взятый и брошенный → abstain при неуверенности | 🟡 | стратегия v4 |
| D6 | Загрузка моделей входит в лимит; run.sh на каждую раскладку | 🟡 | цель cold start p95 ≤10 с (предл.) |

## E. Сдача
| E1 | Репозиторий nssr-sol-team-048 main: src/, run.sh, setup.sh, README.md | ⛔ | GitHub-выгрузки запрещены пользователем |
| E2 | Скринкасты в Releases | ⛔ | |
| E3 | Проверка hackathon/tools/solution.sh в чистом контейнере | ⛔ | нужен код организаторов |
| E4 | Дедлайн | ❌ | не указан в доступных файлах |

## F. Официальное описание трека (добавлено)
| F1 | Ноги закреплены; только верх: голова, корпус (yaw), руки, пальцы | 🟡 | учтено в ТЗ-интерфейсах |
| F2 | 2 RGB-D камеры (голова, торс) | ❌ runtime | |
| F3 | Isaac Sim 5.0 + Isaac Lab 2.2, ROS 2 Humble, топики как на реальном роботе | 🟡 | версии совпадают на Kaggle (pip 5.0.0.0 / Lab 2.2.0) |
| F4 | Категория→ящик из /sim/task; позиции только камерами | ❌ | |
| F5 | Ничьи решаются скоростью → оптимизировать время sim/real | ❌ | |
| F6 | Пул 3 = (48−1) mod 5 + 1; доп. sort2 пул 6 | ✅ | расчёт |
| F7 | Машина: Linux, Docker, NVIDIA Container Toolkit, RTX ≥8 ГБ, RAM 16 ГБ, ~50 ГБ | ⛔ на Kaggle | см. риски |
| F8 | Репозиторий организаторов nssr-hackathon/nssr-uwt-polygon (hackathon/docs/*) | ⛔ | приватный/нет доступа с box |
| F9 | Сдача только в nssr-sol-team-048; без форков/своих repo; код не в публичных Issues | ✅ соблюдается | GitHub-действий нет |

## Риски паритета Kaggle ↔ официальный стенд
- Нет Docker/NVIDIA Container Toolkit: ставим Isaac через pip + свой драйвер user-space (580.178.04) — образ
  `hackathon/docker/Dockerfile` не воспроизведён; пути, версии apt, fastdds-профиль могут отличаться.
- T4 (Turing, 16 ГБ, есть RT-ядра) ≠ RTX-десктоп: рендер медленнее, старт RTX бывает >180 с (видно в safe-close v1),
  тайминги/скорости sim и real-time не переносятся; фпс камер ниже.
- Без X/окна только headless; скринкасты для сдачи на Kaggle не снять.
- Сеть Kaggle и общая память DDS: профиль SHM не проверен; ROS_LOCALHOST_ONLY=1.
- Сессия ≤12 ч, квота GPU ~30 ч/нед, очереди (v1 ждал ~35 мин).

## G. Сверка с репозиторием организаторов, коммит 08247fc (08.11.2026, 20:50 MSK)
TASKS/INTERFACES/SCORING.md побайтно совпадают с ранее изученными копиями. Явно фиксирую уточнения:
- [ ] 2 ноя: «взят» = поднят; сдвинутый/опрокинутый без подъёма = 0; итог по положению на момент /sim/task/done; объект в руке/вне ящика к концу = −10 → перед done обязательно разжать кисть над ящиком.
- [ ] 3 ноя: самоколлизии в симе ОТКЛЮЧЕНЫ, но траектории проверяются вручную → собственная проверка самопересечений (ik/arm.py, сферы по URDF) обязательна.
- [ ] 4 ноя: run.sh запускается заново на каждую раскладку; без состояния между раскладками; время старта и загрузки моделей входит в лимит → быстрый старт (RF ≈0.9 с).
- [ ] Формат: src/ (colcon), run.sh, README.md обязателен; apt.txt/requirements.txt/setup.sh опц.; файлов репозитория симулятора в контейнере решения нет (URDF нужно положить в свой пакет — лицензия Apache-2.0, указать атрибуцию).
- [ ] Скринкасты на каждую решаемую задачу — в Releases репозитория команды (делает пользователь; агент не публикует).
