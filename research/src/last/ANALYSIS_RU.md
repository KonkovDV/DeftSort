# Что установлено до нового GPU-результата

- Прочитан точный исполненный собственный joint-v2 runner: AppLauncher создаёт единственный SimulationApp; run_lab_context создаёт только SimulationContext. Предположение о двух SimulationApp не подтверждается.
- SimulationApp.close 5.0 (локальная извлечённая wheel-копия, новая сессия сохраняет и сверяет установленную): stop/wait Replicator -> close USD stage -> дождаться загрузок до 300 updates -> остановить Tracy -> выключить logging -> IApp.shutdown -> framework.unload_all_plugins. SIGSEGV на последнем Python frame не идентифицирует native библиотеку.
- SimulationApp.__del__ не выполняет автоматический close; повторное создание приложения в одном процессе недопустимо для такого эксперимента. Каждый case — отдельный конечный child.
- AppLauncher хранит один app и устанавливает сигнальные handlers; SIGSEGV handler зовёт close, что может скрыть исходный crash рекурсивным shutdown. Probe заменяет SIGSEGV/SIGABRT handler на системный default плюс faulthandler только для диагностики исходного отказа. Это не исправление и не подавление ошибки: сигнал остаётся failure.
- Lab SimulationContext clear_instance публично отписывает STOP callback, затем вызывает базовую очистку singleton. Его callback при STOP может крутить render до возобновления; поэтому сохранён исходный public clear_instance-before-stop порядок. Он уже был в joint-v2, новой находкой/исправлением не выдаётся.
- Предыдущая chronology: normal shutdown всех расширений, затем SIGSEGV на unload_all_plugins, а bare normal close вернулся, но процесс не завершился до100s. Это два разных предиката и возможны разные причины.
- Новая гипотеза: жизненный цикл собственных handles и отложенная обработка stage/renderer. В новом own runner все scene-ресурсы ограничены функцией, stage закрывается публичным API и10frames drain до close, полный Replicator wait. Не патчим close/библиотеки, не обходятся unload, не используется fast shutdown.
- Набор не является строгой абляцией загруженных расширений: один и тот же neutral.kit загружает ROS2/Replicator во всех cases. Cases различают владение и реально выполненные операции/контексты, а не гарантируют отсутствие расширений. Поэтому по их успеху/падению нельзя единолично назвать виновный plugin.

Публичный аналог: https://github.com/isaac-sim/IsaacLab/issues/3730 — headless close5.0 SIGSEGV. В комментариях не доказана универсальная причина или исправление. Официальный совет/форум об игнорировании shutdown segfault/пропуске close не удовлетворяет нашему критерию и НЕ применяется.

Ограничение наблюдения: официальный SDK GetKernelSessionLogsStream вернул HTTP404; обходов через внутренние endpoint/cookies/browser не делалось. Проверяем status и итоговые малые artifacts штатным CLI.

# Фактические результаты новой сессии

- Все3точные установленные source-файла совпали byte-for-byte с прочитанными до GPU. Исходник выгруженного notebook совпал с локальным. Критерии full_acceptance AST не изменены.
- Bare: close вернулся, затем atexit-checkpoint сохранён и SIGSEGV с no Python frame. Значит native exit нельзя подменять close-return.
- AppLauncher/render/Lab: close вернулся и atexit-checkpoint сохранён; после него процесс не вышел до deadline220s (rc-15). Послеclose threading.enumerate показывает только MainThread. Это не доказательство отсутствия native threads и не локализует hang.
- ROS и joint: SIGSEGV rc-11 на unload_all_plugins; close не вернулся, atexit-checkpoint отсутствует. IApp.shutdown вернулся (следующий trace line693 достигнут).
- В render/ROS/Lab/joint stage_closed=true до ordinaryclose. Следовательно явно закрытый stage и drain10frames, function-scoped handles и полное ожидание Replicator НЕДОСТАТОЧНЫ, чтобы устранить этот сбой. Нельзя говорить, что ошибка stage-order доказана или исправлена.
- Упражнение ROS2 коррелирует с crash внутри unload против hang после close в launcher/render/Lab. Это зацепка для отдельного разделения bridge OmniGraphClock и Pythonrclpy; ещё НЕ причинная атрибуция DDS/ROS или конкретной библиотеке. Всеcases имеют одинkit с bridge-зависимостью.
- Gdb отсутствует; C/C++ native stack не получен. Pythonfaulthandler и maps — не его замена. Не заявляем установленный vendorbug, только остающийся native lifecycle defect в этой конфигурации.
- Все17ранее true joint-predicates сохранены, graceful_return/native_exit0 false. Независимыйchecker вернул1. Новые7дополнительных проверок true, включая PNG/wire SHA, stageclose, effective_fast_shutdown=false; это не компенсирует дваfalse.
- Повтор sanity не запускался: полный joint не прошёл.

Следующий информативный цикл (НЕ запущен): сначала обеспечить штатно работающий user-space gdb из официального distro-пакета и проверить его на собственном tinychild безGPU, не меняя system/ptracepolicy; затем собственный native stack для минимального app и отдельно4комбинации: bridge без активностей / только OmniGraphClock / только Pythonrclpy init-publish-destroy-shutdown / обе. Истинное отсутствие bridge проверять отдельным диагностическим kit, но итоговыйjoint со всемикомпонентами и прежнимикритериями. До native stack новые вариантыclose вслепую не обоснованы.

Публичная документация5.1 KnownIssues открыта в этой работе, но описывает другие version/context; её разрешение игнорироватьshutdowncrash неприменимо к строгойприёмке. Bot/Xпоиск не дал подтверждённого метода ремонта.
