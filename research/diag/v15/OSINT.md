> Копия `/workspace/osint-v15/OSINT.md` (10.11.2026). Папка `raw/` (скачанные сторонние PDF и тексты) в репозиторий не копировалась: упоминания `raw/…` ниже относятся к рабочей копии, источники доступны по ссылкам. Как выводы учтены в коде и документах: `MATH_REVIEW.md`, раздел «Учёт OSINT».

# OSINT v15: DeftSort (команда 048, ROS Humanoid Hack 2026, трек манипуляции)

Дата: 10.11.2026, около 22:40 (MSK, UTC+3). Использованы только публичные источники. Код других команд не открывался, ничего не публиковалось, /workspace/DeftSort и /tmp/submit-048 не трогались.
Все ссылки ниже проверены 10.11.2026 через curl, все вернули HTTP 200. Скачанные копии источников (PDF мануалов, URDF, тексты страниц) лежат в `/workspace/osint-v15/raw/`.
Пометка **[не подтверждено]** означает, что публичного источника для утверждения нет. Пометка **[вывод]** означает наше умозаключение из источников, а не цитату.

---

## Краткое резюме

1. **Порядок 6 каналов Inspire RH56 подтверждён официальным мануалом:** 0 мизинец, 1 безымянный, 2 средний, 3 указательный, 4 сгиб большого пальца, 5 поворот большого пальца. Тот же порядок у Unitree G1 и в датасетах TienKung.
2. **Сила задаётся в «сырых» единицах 0–1000.** По мануалу единица равна грамму на кончике пальца, то есть 1000 ≈ 9.8 Н. Независимая калибровка (arXiv 2603.08988, 2026) даёт меньше: указательный ≈ 0.0075 Н на единицу (1000 → ≈ 7.1 Н), сгиб большого ≈ 0.0125 Н на единицу (1000 → ≈ 12.9 Н). Паспортная сила хвата: 4 пальца до 10 Н (DF) или 4 Н (BF), большой палец до 15 Н (DF) или 6 Н (BF).
3. **У канала поворота большого пальца есть свой регистр силы FORCE_SET(5) с диапазоном 0–1000,** то есть «сила поворота» существует аппаратно. Поворот большого пальца отвечает за отведение и противопоставление (90°–165°, в URDF ≈ 0–1.25…1.31 рад). Основную силу сжатия он не создаёт, её даёт канал 4 (сгиб).
4. **Правила хакатона (issues #595, #625, #645, #648) публично недоступны.** У организации github.com/nssr-hackathon 0 публичных репозиториев. Наши подключённые GitHub-аккаунты её не видят, gh CLI на боксе не авторизован (401). Поэтому нельзя подтвердить ни лимит 0.3–0.5, ни то, распространяется ли он на поворот большого пальца. Публичный сайт хакатона подробностей не даёт.
5. **Isaac Sim 5.0 работает на Kit 107.3.1, что соответствует PhysX SDK 5.6.0** (подтверждено косвенно). В PhysX 5.6 осталась только patch-модель трения: модели ONE_/TWO_DIRECTIONAL удалены. В Isaac Lab 2.2 по умолчанию стоят: friction combine `average`, solver TGS, `friction_correlation_distance` 0.025 м, `friction_offset_threshold` 0.04 м, `enable_ccd` False.
6. **Combine mode у пары материалов выбирается по приоритету average < min < multiply < max.** Если на пальцах стоит `max` с μ ≈ 1.0, это перекрывает материал объекта при любом его режиме. Для захвата NVIDIA советует поднять трение на пальцах и объекте, проверить массы и увеличить жёсткость приводов кисти. Сообщество добавляет 16–32 итерации солвера, contact_offset около 5 мм и rest_offset ≈ 0.
7. **Литература 2024–2026 без тяжёлых моделей:** главная для нас работа arXiv 2603.08988 (Inspire RH56DFX). В ней аналитический планировщик «ширина → поза захвата» и гибридный регулятор «скорость–сила», который убирает перерегулирование силы (до +1618% на высокой скорости). К ней добавляются оптимизационные синтезаторы захватов (BODex, SpringGrasp, Dexonomy) и регуляторы силы по проприоцепции и тактильным данным.
8. **Главное для нас:** публичных данных о том, что сила 1.0 на канале поворота большого пальца нарушает правило 0.3–0.5, нет. Самих правил тоже нет в открытом доступе. Физически сила на повороте почти не нужна для хвата. Безопаснее всего привести и этот канал в диапазон 0.3–0.5 (например 0.5): захват от этого почти ничего не теряет, а риск штрафа уходит.

---

## 1. Кисть Inspire RH56 (DFX/DFQ/BRL) и кисти TienKung 2.0

### Факты

**1.1. Порядок каналов (официальный мануал Inspire).**
- *RH56 Series User Manual*, Beijing Inspire-Robots, PRJ-02-TS-U-001, V1.0.9 (январь 2024), раздел 2.4.12 FORCE_SET(m):
  > «FORCE_SET(0) Set value of force control for the little finger … FORCE_SET(1) … ring finger … FORCE_SET(2) … middle finger … FORCE_SET(3) … index finger … FORCE_SET(4) Set value of bending force for the thumb … FORCE_SET(5) Set value of rotation force for the thumb … short 0-1000»
  Ссылка: https://sonnyrobotics.com/cdn/shop/files/Inspire_RH56DFX_Manual.pdf?v=3196307057052803410 (копия: raw/rh56dfx.pdf). Версия V1.0.8 (32 стр., есть CAN): https://static.generation-robots.com/media/rh56-series-user-manual.pdf
- Тот же порядок задан для ANGLE_SET, SPEED_SET, FORCE_ACT и CURRENT.
- Unitree G1 + RH56DFX, таблица «Joint Motor Sequence»: правая кисть Id 0–5 и левая Id 6–11 идут в порядке «pinky, ring, middle, index, thumb-bend, thumb-rotation». https://support.unitree.com/home/en/G1_developer/inspire_dfx_dexterous_hand. Страница рендерится JS-скриптом, цитата взята из поискового снимка, сам URL отвечает 200.
- Датасет TienKung + Inspire (Hugging Face, `robot_type: "Tienkung_Inspire"`), порядок суставов кисти: `L_pinky_proximal_joint, L_ring_proximal_joint, L_middle_proximal_joint, L_index_proximal_joint, L_thumb_proximal_pitch_joint, L_thumb_proximal_yaw_joint`. https://huggingface.co/datasets/Humandroid/tienkung_put_coca
- UFACTORY (RH56DFX): «Finger: Little finger, Ring finger, Middle finger, Index finger, Thumb Flexion, Thumb Rotation … Force: 0–1000». https://docs.supportarticle.ufactory.cc/support_articles/developer/ufactory-RH56DFX.html

**1.2. Единицы и диапазон силы.**
- Мануал V1.0.9, раздел 2.4.12:
  > «This register value indicates the gripping force provided by the fingertip of the corresponding finger. For example, when DEFAULT_FORCE_SET(1) is set to 800, it means that the fingertip of the ring finger is allowed to provide the gripping force of 800 g. (If the portion of a finger that touches an object is not the fingertip, the available gripping force will be different…)»
  > «…Unit: g.»
  Пересчёт: 1 единица = 1 г ≈ 0.0098 Н, значит 1000 ≈ 9.8 Н, 500 ≈ 4.9 Н, 300 ≈ 2.9 Н.
- Важная оговорка из мануала: FORCE_SET работает как **порог остановки**, а не как регулятор силы:
  > «set FORCE_SET(3) to 300 … the index finger will bend towards the palm. Once it is detected that the actual force (i.e., FORCE_ACT(3)) … reaches 300, the index finger will stop moving; otherwise, the index finger will move to the preset angle.»
- DEFAULT_FORCE_SET(4) (сгиб большого пальца) имеет диапазон **0–1500**, остальные каналы 0–1000. FORCE_SET(4) указан как 0–1000.
- Паспорт (раздел 1.3 Performance parameters): «Gripping force of four fingers DF: 10 N / BF: 4 N; Gripping force of the thumb DF: 15 N / BF: 6 N; Resolution of force sensor 0.5 N».
- Независимая калибровка: Tan, Xie, Correll, *Characterization, Analytical Planning, and Hybrid Force Control for the Inspire RH56DFX Hand*, arXiv 2603.08988 (09.03.2026), Table I:
  > «Raw load feedback L_raw ∈ [0,1000] is converted to Newtons with a per-finger linear model F = a·L_raw + b … Index 0.0075 / −0.414 / R² 0.987; Middle 0.0065 / 0.018 / 0.986; Thumb bend 0.0125 / 0.384 / 0.993»
  Пересчёт [вывод]: 1000 единиц ≈ 7.1 Н (указательный), ≈ 6.5 Н (средний), ≈ 12.9 Н (сгиб большого). Для поворота большого пальца калибровки в статье нет. https://arxiv.org/abs/2603.08988
- Там же: «exposes raw, unitless signals for force (0 to 1000) and exhibits severe contact overshoot at high speed … up to +1618% force limit overshoot».

**1.3. Что делает канал поворота большого пальца.**
- Мануал, таблица углов (раздел 2.4.11): «Rotation angle of the thumb … Metacarpal plane … 90° to 165°», «Bending angle of the thumb … −13° to 53.6°», «Little/Ring/Middle/Index finger … 19° to 176.7°».
- ANGLE_SET в диапазоне 0–1000, −1 означает «не двигать». ANGLE_SET(5) (поворот) принимает только 0–1000, без −1. Значение 1000 = «fully open» (пример с ANGLE_ACT(3)=1000).
- arXiv 2603.08988 описывает «2-DoF thumb yaw and pitch actuation» и пишет: «limited region of intersection with the other fingers and subsequent constrained grasping».
- Сторонний драйвер `inspire-hand-rh56-py` (только описание API): «for thumb_rotation, 0 = retracted, 1000 = across palm». https://github.com/yuzhench/inspire-hand-rh56-py **[не подтверждено официально]**: направление шкалы взято у третьей стороны.

**1.4. Диапазоны суставов в URDF (рад).**
- TienKung 2.0 Pro с кистями, официальный репозиторий X-Humanoid (`tiangong2.0_complete_with_hands.xacro`, `tiangong2.0_pro_with_hands.urdf`), https://github.com/Open-X-Humanoid/TienKung_URDF :
  | Сустав | lower–upper, рад | Примечание |
  |---|---|---|
  | `left/right_thumb_1_joint` (поворот, axis z) | 0 – 1.246165 | effort 50, velocity 1 |
  | `*_thumb_2_joint` (сгиб) | 0 – 0.48 | |
  | `*_thumb_3_joint` | 0 – 0.3578 | mimic thumb_2 ×1.1425 |
  | `*_thumb_4_joint` | 0 – 0.2775 | mimic thumb_3 ×0.7508 |
  | `*_index/middle/ring/little_1_joint` | 0 – 1.333 | |
  | `*_…_2_joint` | 0 – 1.527 | mimic _1 ×1.1169 |
  | `wrist_pitch_*` | −0.785 – 1.047 | effort 6.3 |
  | `wrist_roll_l` / `wrist_roll_r` | −1.658 – 1.309 / −1.309 – 1.658 | |
- Unitree, Inspire DFQ (`DFQ_right_hand.urdf`): `R_thumb_proximal_yaw_joint` −0.1…1.3; `R_thumb_proximal_pitch_joint` −0.1…0.6 (mimic intermediate ×1.6, distal ×2.4); пальцы proximal 0…1.7 (intermediate mimic ×1.0). https://github.com/unitreerobotics/unitree_ros/blob/master/robots/g1_description/inspire_hand/DFQ_right_hand.urdf
- Сторонний пакет RH56DFX: thumb_1 (ab/adduction) 0…1.310, thumb_2 0…0.523, пальцы 0…1.344. https://github.com/ookkshirsagar/rh56dfx_description
- [вывод] Ход поворота по мануалу 165° − 90° = 75° = 1.309 рад, это совпадает с URDF (1.25–1.31 рад). Для сгиба большого пальца и остальных пальцев градусы мануала напрямую не сравниваются с URDF: углы определены по-разному (у мануала угол всего пальца, у URDF проксимальный сустав плюс mimic).
- **[не подтверждено]** Точное соответствие «0–1000 ↔ радианы» в симуляции хакатона и направление шкалы (0 = сжато или 0 = открыто) для их ROS-топиков. Это задаётся приватным пакетом организаторов.

### Что это меняет для нас (п. 1)
- Порядок каналов [мизинец, безымянный, средний, указательный, сгиб большого, поворот большого] подтверждён. Если у нас другой порядок, это ошибка.
- Если нормированная «сила» хакатона 0..1 линейно отображается в 0..1000 (**[не подтверждено]**), то 0.3–0.5 соответствует 300–500 г ≈ 2.9–4.9 Н по мануалу. По калибровке Correll это ≈ 1.8–3.3 Н на указательном и ≈ 4.1–6.6 Н на сгибе большого. Для лёгких объектов сортировки (десятки–сотни граммов, **[не подтверждено]**, масс объектов в публичных источниках нет) при μ ≈ 1 этого достаточно.
- Поворот большого пальца — позиционный канал противопоставления: он ставит большой палец напротив указательного и среднего. Нормальную силу сжатия создаёт канал 4 (сгиб). FORCE_SET(5) аппаратно ограничивает момент поворота при упоре, например если палец упрётся в объект или в ладонь. Высокая сила здесь не помогает удержанию, но повышает риск контакта с большой силой и самопересечения.
- На реальной кисти будет сильное перерегулирование силы на высокой скорости. Для переноса на робота (23–27.11) подходят малая скорость на подходе и порог силы (гибридная схема из 2603.08988).

---

## 2. Isaac Sim 5.0 / PhysX 5: трение, offsets, устойчивый захват

### Факты

**2.1. Версия PhysX.** В Isaac Sim v5.0.0 файл `deps/kit-sdk.packman.xml` указывает `kit-kernel version="107.3.1+isaac…"` (https://github.com/isaac-sim/IsaacSim/blob/v5.0.0/deps/kit-sdk.packman.xml). Релиз PhysX называется «107.0-physx-5.6.0» (https://github.com/NVIDIA-Omniverse/PhysX/releases/tag/107.0-physx-5.6.0). [вывод] Isaac Sim 5.0 использует ветку PhysX 5.6. Подтверждено косвенно: прямой строки «Isaac Sim 5.0 = PhysX 5.6.0» в release notes не нашлось. Документация Isaac Sim 5.0.0 на docs.isaacsim.omniverse.nvidia.com сейчас отдаёт 404, поэтому ниже цитируется версия 5.1.0.

**2.2. Модель трения: только patch.**
- PhysX 5.6 Migration Guide: «PxFrictionType::eONE_DIRECTIONAL and ::eTWO_DIRECTIONAL were deprecated and have been removed. Please use PxFrictionType::ePATCH instead … since PxFrictionType::ePATCH is the only supported type left». https://nvidia-omniverse.github.io/PhysX/physx/5.6.1/docs/MigrationTo56.html
- PhysX 5.6.1, Rigid Body Dynamics: «Friction uses the coulomb friction model … static friction coefficient and the dynamic friction coefficient … PhysX computes friction per contact patch. Up to two contact points lying in the contact patch area are selected as friction anchors … The normal impulses of all contact points in the contact patch are accumulated and distributed equally among the friction anchors». https://nvidia-omniverse.github.io/PhysX/physx/5.6.1/docs/RigidBodyDynamics.html
- Там же: «TGS applies friction throughout all position and all velocity iterations, while PGS by default applies friction throughout the last 3 position iterations and all velocity iterations».
- Там же: «the solver permits a maximum of 32 friction patches per contact manager (pair of shapes). If more than 32 … which may be due to very complex collision geometry or very large contact offsets, the solver will ignore the remaining friction patches».
- Там же: «TGS is generally the recommended solver»; «The solver iteration count defaults to 4 position iterations and 1 velocity iteration … If you find a need to use a setting higher than 30, you may wish to reconsider the configuration».

**2.3. Combine mode.**
- Omniverse Physics, Rigid Bodies: «Four combine modes are supported: average (a+b)/2, min, multiply (a*b), max … the effective combine mode of a contact is the maximum value of the combine modes of the material pair … Static friction and dynamic friction are governed by a single combine mode per material … the default behaviour … is therefore also average». https://docs.omniverse.nvidia.com/kit/docs/omni_physics/latest/dev_guide/rigid_bodies_articulations/rigid_bodies.html
- PxCombineMode (PhysX 5.6.1 API): «The effective combine mode for the pair is maximum(material0.combineMode, material1.combineMode)». Про compliant: «If a compliant (restitution < 0) material interacts with a rigid … material, the compliant behavior will be chosen independent of combine mode». https://nvidia-omniverse.github.io/PhysX/physx/5.6.1/_api_build/structPxCombineMode.html
- Isaac Lab 2.2 `RigidBodyMaterialCfg`: «friction_combine_mode: Literal['average','min','multiply','max'] … Defaults to "average"»; «compliant_contact_stiffness … Defaults to 0.0 … The compliant contact model is only enabled if the stiffness is larger than 0». https://isaac-sim.github.io/IsaacLab/v2.2.0/source/api/lab/isaaclab.sim.spawners.html

**2.4. Contact / rest offset.**
- PhysX, Advanced Collision Detection: «Collision detection starts to generate contact points as soon as two shapes get closer than the sum of their contact offsets … contactOffset has to be non-negative … A shape's contactOffset also has to be larger than its restOffset … If a pair of shapes has a positive restDistance, the shapes will be separated at rest by an air gap. If … negative restDistance, the shape geometries will at rest penetrate by that distance». https://nvidia-omniverse.github.io/PhysX/physx/5.4.1/docs/AdvancedCollisionDetection.html
- Isaac Lab 2.2 `CollisionPropertiesCfg`: те же определения, плюс `torsional_patch_radius`: «Radius of the contact patch for applying torsional friction … If the radius is zero, no torsional friction is applied». https://isaac-sim.github.io/IsaacLab/v2.2.0/source/api/lab/isaaclab.sim.schemas.html

**2.5. Параметры сцены Isaac Lab 2.2 (`PhysxCfg`)** — https://isaac-sim.github.io/IsaacLab/v2.2.0/source/api/lab/isaaclab.sim.html
- «solver_type … Default is 1 (TGS)»; `max_position_iteration_count` и `max_velocity_iteration_count` по умолчанию 255. Каждый актор задаёт свои итерации, сцена задаёт min/max.
- «enable_ccd … Default is False»; «enable_stabilization … Default is False … recommend setting this flag to true only when the simulation step size is large»; «bounce_threshold_velocity … 0.5 m/s»; «friction_offset_threshold … 0.04 m»; «friction_correlation_distance … Distance threshold for merging contacts into a single friction anchor point … 0.025 m».
- `RigidBodyPropertiesCfg`: `solver_position_iteration_count`, `solver_velocity_iteration_count`, `max_depenetration_velocity`, `max_contact_impulse`.

**2.6. Рекомендации NVIDIA по захвату.**
- Isaac Sim 5.1 Robot Simulation Tips: «My gripper are not picking up the object. You can increase the friction parameters on both the fingers and object … Use the Physics Authoring Toolbar … especially the Mass Distribution Tool to make sure the weight of the object and weight of the arms are reasonable. Increase the stiffness of your joints at the gripper». https://docs.isaacsim.omniverse.nvidia.com/5.1.0/robot_simulation/robot_simulation_tips.html
- Isaac Sim 5.1 Tutorial 7 (Configure a Manipulator): «set the static friction to 1.0 and dynamic friction to 1.0 … Apply the physics material to the gripper finger tip». https://docs.isaacsim.omniverse.nvidia.com/5.1.0/robot_setup_tutorials/tutorial_configure_manipulator.html
- Форум NVIDIA (совет участника сообщества, **не официальный ответ NVIDIA**): «resolved it by increasing solver_position_iteration_count and solver_velocity_iteration_count to around 16 or 32 and setting static_friction and dynamic_friction … to 1.0». В его конфиге: `contact_offset: 0.005`, `rest_offset: 0.0`, `friction_correlation_distance: 0.025`, `max_depenetration_velocity: 5.0`, dt 1/120. https://forums.developer.nvidia.com/t/physics-and-simulation-parameters-for-grasping-and-picking-tasks-objects-slipping-away/259716
- PhysX, Compliant Contacts: включаются отрицательной restitution у материала, нормальная сила считается через неявную пружину-демпфер (raw/rbd56.txt).

### Что это меняет для нас (п. 2)
- Материал на подушечках пальцев с `friction_combine_mode="max"` и μs = μd ≈ 1.0 перекрывает режим `average` объекта: эффективный режим пары станет `max`, и μ будет не ниже 1.0. Можно ли нам менять материалы сцены и объектов по правилам хакатона, **[не подтверждено]**. Правила приватные, а #645 (позы из сцены) может касаться и этого.
- Держать patch-трение стабильным: 16–32 итерации позиции для кисти и объектов (TGS), contact_offset малый (≈ 2–5 мм), rest_offset ≈ 0. Большой contact_offset порождает лишние patch-и (лимит 32 на пару) и «зависание в воздухе».
- Для переноса: плавные траектории запястья (ограничение ускорения), CCD только при быстрых движениях. Числа по скорости и ускорению в официальных источниках не найдены, **[не подтверждено]**.
- Если удержание слабое, сначала поднимите μ и итерации. Силу пальцев поднимать в последнюю очередь, ведь её ограничивает правило 0.3–0.5.

---

## 3. Публичные правила и ответы организаторов (nssr-hackathon / ROS Humanoid Hack)

### Факты
- **Репозиторий с issues #595, #625, #645, #648 публично найти не удалось.**
  - `https://api.github.com/orgs/nssr-hackathon`: `"public_repos": 0`, followers 27, создана 2025-09-02 (страница https://github.com/nssr-hackathon показывает то же).
  - Пробы имён репозиториев (`ros-humanoid-hack`, `hackathon`, `tasks`, `docs`, `rules` и т.п.) без авторизации возвращают 404.
  - Подключённый к боту GitHub-коннектор (Cursor GitHub App) не видит `org:nssr-hackathon`: «GitHub cannot see org:nssr-hackathon with the connected account: they may not exist, be private, or not be granted to Cursor's GitHub App». Среди видимых репозиториев только наш `KonkovDV/DeftSort` (private, его содержимое не открывалось).
  - `gh` на боксе отвечает `Bad credentials (HTTP 401)`.
  - [вывод] Репозиторий с задачами и issues у организаторов приватный и доступен только участникам. Сайт это подтверждает: «Участники добавляют код в приватный репозиторий на GitHub»; «Полный перечень появится на сайте, а также в закрытом чате участников хакатона».
- Публичный сайт https://hackathon.rusindustrial.ai/ (организатор Codenrock по AllConf, почта roshumanoidhack@mail.ru):
  - Сроки: «29.10 — 11.11 Решение задач», «11.11 Дедлайн решения задачи», «16.11 … объявление финалистов», «23.11 — 27.11 Победители переносят код на реального робота … в Кронштадте».
  - Кисти: «Научиться управлять шестиприводными кистями: разные типы захвата, скорость и сила сжатия, обратная связь по положению и усилию пальцев и по датчикам силы в запястьях».
  - Сортировка: «двойная: 2 категории, 6 объектов, 2 лотка; тройная: 3 категории, 9 объектов, 3 лотка».
  - IK: «соблюдать пределы суставов и не задевать стол».
  - **Штрафов, лимитов времени (270/405 с), лимита силы 0.3–0.5, правил о самопересечениях, о позах из сцены и о штрафе −10 на сайте нет.**
- **#648 (сила 0.3–0.5): [не подтверждено].** Охватывает ли лимит канал поворота большого пальца, по публичным данным неизвестно.
- **#625 (самопересечения), #645 (запрет поз из сцены), #595 (штраф −10): [не подтверждено],** тексты недоступны.

### Что это меняет для нас (п. 3)
- Проверить формулировку #648 через OSINT нельзя. Текст нужно взять из приватного репозитория организаторов или из закрытого чата участников. Сделать это может только Shaundra или член команды с доступом, например подключив доступ к org для Cursor GitHub App по https://github.com/apps/cursor/installations/new или авторизовав `gh`.
- До проверки исходите из худшего случая: лимит действует на **все 6 каналов**, включая поворот большого пальца.
- Если доступ появится, нужно прочитать только тексты issues и ответы организаторов (без кода других команд): #648, #625, #645, #595.

---

## 4. Свежие работы 2024–2026: захват и перенос многопальцевой рукой без тяжёлых моделей

Все работы проверены через arXiv API (даты — первая версия на arXiv).

1. **Tan, Xie, Correll — Characterization, Analytical Planning, and Hybrid Force Control for the Inspire RH56DFX Hand** (arXiv 2603.08988, 09.03.2026). https://arxiv.org/abs/2603.08988
   > «(1) hardware characterization (force calibration, latency, and overshoot), (2) a sim2real validated MuJoCo model for analytical width-to-grasp planning, and (3) a hybrid, closed-loop speed-force grasp controller … 87% success [300 grasps, 15 objects]»
   Для нас самая релевантная: та же кисть, аналитический планировщик по ширине объекта, гибридный регулятор («Hybrid (approach + 25)»: быстрый подход, затем скорость 25 у контакта), латентность ≈ 66 мс.
2. **BODex** (arXiv 2412.16490, ICRA 2025). https://arxiv.org/abs/2412.16490
   > «We formulate grasp synthesis as a bilevel optimization problem, combining a novel lower-level quadratic programming (QP) with an upper-level gradient descent process»
   Синтез захватов без обучения: QP по силам контакта плюс градиентный спуск по позе.
3. **SpringGrasp** (arXiv 2404.13532, 2024). https://arxiv.org/abs/2404.13532
   > «We introduce an analytical and differentiable metric, SpringGrasp metric, that evaluates the dynamic behavior of the entire compliant grasping process»
   Аналитический планировщик податливого захвата с учётом неопределённости формы объекта.
4. **Dexonomy** (arXiv 2504.18829, RSS 2025). https://arxiv.org/abs/2504.18829
   > «Starting from a single human-annotated template for each hand and grasp type … optimize the object to fit the hand template first, and then locally refine the hand to fit the object in simulation … contact-aware control strategy»
   Шаблоны типов захвата (pinch, power и т.п.) с локальной оптимизацией, близко к нашему подходу «предзаписанные позы плюс подгонка».
5. **Current as Touch** (arXiv 2607.03529, 03.07.2026). https://arxiv.org/abs/2607.03529
   > «learns contact-aware compliance cues from motor current and joint states … predicts a compliance reference position: an ideal joint-position target for a standard PD controller whose induced position error generates appropriate grasping force»
   Сила хвата задаётся через «недоход» по позиции в PD без тактильных датчиков. Модель обучаемая, но лёгкая; идея переносится на сим без обучения.
6. **Underactuated Robotic Hand with Grasp State Estimation Using Tendon-Based Proprioception** (arXiv 2509.12969, 2025). https://arxiv.org/abs/2509.12969
   > «using only tendon-based proprioception … By coupling accurate proprioceptive measurements with potential energy-based modeling, the system estimates multiple key grasp state variables»
   Модельная (энергетическая) оценка состояния захвата у недоприводной кисти.
7. **Robust Adaptive Safe Robotic Grasping with Tactile Sensing** (arXiv 2411.07833, ECC 2025). https://arxiv.org/abs/2411.07833
   > «design contact force and force closure constraints, which are enforced by a safety filter [Control Barrier Functions] … to prevent a grasped object from being damaged or slipping»
   Аналитический CBF-фильтр сил: удобно, когда есть жёсткие пределы силы, как наше правило 0.3–0.5.
8. *(дополнительно, с оговоркой)* **Grasping Force Control and Adaptation for a Cable-Driven Robotic Hand** (arXiv 2407.19279, 2024): Youla-параметризация плюс ILC, классическое управление силой пятипалой кисти. https://arxiv.org/abs/2407.19279. **Shear-based Grasp Control** (arXiv 2503.17501, IEEE T-RO) и **Tracing Energy Flow** (arXiv 2512.21043, RA-L) решают задачу контроля проскальзывания, но используют обучаемые модели и тактильные датчики. Как «лёгкие» методы их можно брать только для идей.

### Что это меняет для нас (п. 4)
- Методика «ширина объекта → поза» (2603.08988) и шаблоны типов захвата (Dexonomy) подтверждают выбранный подход: детерминированные позы под класс объекта, без обучения.
- Контроль удержания без датчиков: сравнивать заданные и фактические углы пальцев. Если палец «не дошёл», значит есть контакт (идея из Current as Touch и из порога FORCE_SET мануала). Если во время переноса недоход пропал, объект выскользнул и нужен повторный захват.
- Подход к объекту на малой скорости пальцев (как Hybrid в 2603.08988) снижает выброс силы и важен на реальном роботе.

---

## Отдельно: сила канала поворота большого пальца = 1.0

**Факты:**
- В железе Inspire есть отдельный порог силы для поворота: FORCE_SET(5), 0–1000, в граммах (мануал V1.0.9, 2.4.12).
- Поворот большого пальца отвечает за отведение и противопоставление: 90°–165° по мануалу, `thumb_1_joint` / `thumb_proximal_yaw` 0…≈1.25–1.31 рад в URDF. Сжатие создаёт сгиб, канал 4.
- Правило 0.3–0.5 (#648) и его область действия публично недоступны: **[не подтверждено]**.

**Вывод [вывод]:**
1. **Нарушает ли 1.0 правило?** Подтвердить по открытым источникам нельзя. Буквально «сила 0.3–0.5» относится к любому силовому каналу. Раз у поворота есть свой регистр силы, проверяющий скрипт организаторов вполне может проверять все 6 значений force. Поэтому 1.0 на повороте **с высоким риском** будет засчитано как нарушение.
2. **Есть ли смысл в силе 1.0 для поворота?** Почти нет. При захвате поворот лишь удерживает большой палец в позиции противопоставления. Нагрузка на него боковая и небольшая, нормальную силу даёт сгиб. Высокий порог на повороте увеличивает силу удара, если большой палец заденет объект, ладонь или указательный палец. Это повышает риск самопересечения (#625) и выброса объекта при быстром движении.
3. **Рекомендация:** поставить поворот большого пальца в те же 0.3–0.5 (например 0.5, верхняя граница). Позицию поворота задавать заранее, **до** сжатия: сначала повернуть палец при открытой кисти, потом сжимать. Тогда ему не придётся преодолевать сопротивление объекта. Если удержание слабое, повышать трение и итерации солвера (п. 2), а не силу.
4. Когда появится доступ к приватному репозиторию, проверить в тексте #648, перечислены ли каналы явно и есть ли ответ организатора про thumb rotation.

---

## Что осталось «не подтверждено»
- Тексты issues #595 (штраф −10), #625 (самопересечения), #645 (позы из сцены), #648 (сила 0.3–0.5), а также точный репозиторий, где они лежат: у nssr-hackathon 0 публичных репозиториев, доступа нет.
- Распространяется ли лимит 0.3–0.5 на поворот большого пальца.
- Как нормированная сила 0..1 в API хакатона отображается в 0..1000 и в ньютоны в симуляции; соответствие 0–1000 ↔ радианы и направление шкалы в их ROS-интерфейсе.
- Направление шкалы поворота (0 = отведён, 1000 = поперёк ладони): есть только у стороннего драйвера.
- Калибровка «Н на единицу» для канала поворота и для мизинца и безымянного (в 2603.08988 есть только указательный, средний и сгиб большого).
- Прямое заявление NVIDIA «Isaac Sim 5.0 = PhysX 5.6.0»: подтверждено косвенно через Kit 107.3.1 и тег PhysX 107.0-physx-5.6.0. Документация Isaac Sim 5.0.0 сейчас отдаёт 404, цитировалась 5.1.0.
- Числовые рекомендации NVIDIA по итерациям (16–32), contact_offset (5 мм) и скоростям переноса: это советы сообщества, не официальные.
- Массы и материалы объектов в сцене хакатона и право менять материалы.
