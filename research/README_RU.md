# Исправления этапа 1 (team048), 2026-11-08 МСК

Только локальные копии и приватный Kaggle-ноутбук. Без GitHub, без cloud agents.

## Что найдено в аккаунте Kaggle (dmitrijkv)
Isaac-ноутбуки (все приватные, GPU T4): ros-environment-check, vulkan-*, nvidia-*, isaac500-neutral-startup,
egl-glu-repair, clean-close, ros-lab-neutral, ros311-repair, dds-context (7 ноя), joint-v2, joint-v2-retry,
lifecycle-native, native-stacks (8 ноя). Остальное — другой проект koz3 (не трогал). Модели — нет.
Выводы скачаны в /workspace/kaggle/<kernel>/out.

## Fix 1 — JSON receiver / numpy.bool_
Уже исправлено предыдущей итерацией: `json_value()` (fix1_receiver_json_value.py) конвертирует numpy-скаляры
(.item), массивы (.tolist), NaN/inf → null, неизвестный тип → TypeError (fail-closed); запись атомарная.
GPU-подтверждение: lifecycle-native и joint-v2-retry — receiver_exit0/receiver_no_error/report_finished = true.
Локальный тест: test_json_value.py — PASS.

## Fix 2 — Isaac Lab 0 шагов (kit layout)
Уже исправлено: собственный kit кладётся в `isaaclab/apps/neutral_joint_v2.kit` рядом с официальным
`apps/rendering_modes/balanced.kit` (из колеса Lab 2.2), путь пресета берётся из токена `${app}`;
SimulationContext(device=cpu) + reset + 10 шагов; на выходе clear_all_callbacks/clear_instance.
GPU-подтверждение: lab10steps = true (lifecycle-native, joint-v2-retry).

## Fix 3 — SIGSEGV при закрытии
Диагноз по логам native-stacks (gdb):
- clock_only / both: SIGSEGV в `simulation_app.py:693` = `self._framework.unload_all_plugins()` (после того как
  `self._app.shutdown()` вернулся). Стек: libcarb.events.plugin ← libisaacsim.core.simulation_manager.plugin ←
  libcarb (выгрузка плагинов). Т.е. simulation_manager при выгрузке отписывается/шлёт событие в уже
  разрушенный поток carb.events (use-after-free порядка выгрузки). Триггер — активный ROS2-граф /clock
  (OnPlaybackTick → ROS2PublishClock); граф удаляли до close, не помогло.
- rclpy_only / bridge_inactive: close не падает, но процесс зависает (timeout) — в close или при финализации Python.
- bare (lifecycle-native): close вернулся, затем rc=-11 при финализации интерпретатора.
Корень в нативном коде NVIDIA (без символов), точнее без исходников не установить.

Обходной путь (fix3_safe_close_probe.diff, kaggle-safe-close/):
1. до app.close(): выключить `isaacsim.ros2.bridge`, затем `isaacsim.core.simulation_manager`
   через публичный ExtensionManager.set_extension_enabled_immediate(..., False) + app.update();
2. обычный app.close() (fast_shutdown=False сохранён);
3. после возврата close: flush и `os._exit(code)` — пропуск финализации Python над уже выгруженными плагинами.
Это честно помечено в отчёте (`post_close_exit`). Пункт 3 по сути эквивалентен fast-shutdown после
нормального close; если критерий «graceful» должен включать финализацию — это исключение, его надо принять явно.

## Проверка на GPU
Приватный ноутбук dmitrijkv/isaac500-safe-close-v1 (режимы clock_only, both, joint + полный acceptance).

## Итог GPU-проверки fix 3 (isaac500-safe-close-v1, версия 2)
- joint: все функциональные проверки true (startup, RGB-D, физика, receiver, DDS ACK, clock, use_sim_time,
  camera_info, tf, stamps, Lab 10 шагов). SIGSEGV при close остаётся: стек libcarb.events ← simulation_manager.plugin
  во время unload_all_plugins, хотя расширение выключено (carb-плагин не выгружается при disable).
- Вывод: это баг порядка выгрузки в Isaac Sim 5.0, публичными API не обходится. Официальный симулятор
  (scripts/sim_runner.py, AppLauncher без fast_shutdown → дефолт True) сам закрывается через fast shutdown.
  Рекомендация: fast_shutdown=True (паритет с организаторами). На решение не влияет: решение — отдельный ROS-узел.
- Вывод ноутбука: /workspace/kaggle/isaac500-safe-close-v2/out (v1-результат: .../isaac500-safe-close-v1/out — старт RTX >180 с, таймаут).
