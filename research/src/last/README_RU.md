# Isaac Sim 5.0: исследование штатного завершения

## Итог: FAIL
Исправление не подтверждено. Полностью принимаемая система не получена. Выполнена одна новая приватная конечная GPU-сессия с шестью последовательными дочерними процессами и общим кэшем, а не шесть холодных установок.

Notebook: https://www.kaggle.com/code/dmitrijkv/isaac500-lifecycle-native-dc2ccbf7c2
Статус ERROR. После завершения активных GPU-запусков 0, зарезервированная квота 0.
Версии: Isaac Sim 5.0.0.0, Isaac Lab 2.2.0, Python 3.11, torch 2.7 CPU. CUDA-обучение не проверялось.

ТЗ, частные SDK и полигоны не передавались: использованы только собственные нейтральные примитивы. Драйвер 580.178.04, ядро, устройства и runtime хоста не изменены. Записей в GitHub/Origin не было.

## Что подтверждено заново
- Настоящие RTX RGB-D 640×360 и физическое падение куба.
- Независимый ROS2 DDS receiver: сохранённый отчёт и exit 0; ACK/checkpoint/PNG/wire SHA; clock, use_sim_time, CameraInfo, TF, timestamps, глубина 16UC1 в миллиметрах с invalid=0.
- Десять реальных шагов Lab SimulationContext.
- В joint 17 из 19 прежних предикатов true. graceful_return=false и native_exit0=false. Независимый строгий checker вернул 1.

## Кандидат исправления не сработал
Сцена ограничена функцией, чтобы собственные USD/render/timeline handles освобождались до unload. Добавлены публичное закрытие stage, 10 updates для обработки завершения и полное ожидание Replicator. Код vendor close не менялся; fast_shutdown=False, пропуск unload и принудительный выход как успех не используются.

ROS/joint по-прежнему падают SIGSEGV (-11) в unload_all_plugins. Bare вернул close, но упал после atexit-checkpoint. AppLauncher/render/Lab вернули close, сохранили atexit-checkpoint, затем процесс завис до deadline (SIGTERM -15 считается FAIL). Эти исходы различаются; close-return не равен native exit 0.

Корневая причина не доказана. Активный ROS2 обмен коррелирует с падением внутри unload, но это не доказывает виновность конкретного компонента. У всех диагностических cases один kit с bridge-зависимостью. Gdb отсутствует; C/C++ stack не получен. Python fatal frames и memory maps не выдаются за native backtrace.

## Содержимое
- source/: точный отправленный bootstrap, собственный probe, исполненный receiver, lock и строгий checker.
- server-metadata/: повторно полученные server code и метаданные private/empty sources; исходник совпадает с отправленным.
- output/: только малые JSON/PNG/logs, точные установленные dependency sources, memory maps и chronology close. Binary environment, драйверные архивы и core dumps отсутствуют.
- tests/: локальные регрессии и отрицательные тесты, включая точный исполненный receiver с принудительной ошибкой импорта.
- RESULTS.json, strict_verification.json, ANALYSIS_RU.md: результаты, ограничения атрибуции и следующий ограниченный эксперимент.
- research/: публичный issue 3730 и поиск X. Они не доказывают причину нашего отказа.

## Проверка пакета
Из корня пакета, Python с numpy и Pillow:
    python3 source/verify_result.py output strict_recheck.json
Ожидается exit 1: строгая проверка корректно обнаруживает неполную приёмку.
    cd tests
    python3 test_reporting.py
    python3 test_receiver_failure_integration.py
Проверка SHA256SUMS не требует GPU. GPU-bootstrap нельзя запускать автоматически при распаковке: он тратит квоту и требует принятых лицензий и подходящего хоста с драйвером 580.178.04.

Сессия: 1800.66 s; холодный старт: 655.95 s; pip: 125.82 s; peak scratch: 9.99 GB (9.31 GiB); прирост учтённой квоты: 1814.035 s. Scratch удалён. RSS в этой сессии не измерялся. Старый joint-v2 ZIP не изменён, SHA проверен.

Полигон, SDK, geometry, calibration, units и сортировка не валидированы. Полная приёмка не пройдена.
