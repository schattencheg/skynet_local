# Skynet Local

Skynet Local — Windows-safe модульная desktop-платформа для распознавания лиц, голосовой биометрии, нескольких GUI-режимов и расширяемого real-time интерфейса, построенная вокруг ONNX/OpenCV-first подхода вместо `dlib` как обязательной базы.[cite:178][cite:19][cite:183]

## Назначение

Проект предназначен для локального desktop-приложения, которое обнаруживает и распознаёт лица, отдельно определяет владельца голоса, объединяет результаты в один scene model и отображает их через несколько визуальных режимов, включая cinematic Terminator mode.[cite:178][cite:16][cite:183]

## Почему это имя и этот стек

Название `skynet_local` отражает sci-fi стиль проекта и при этом хорошо подходит для локального desktop runtime без облачной зависимости. Техническая база выбрана так, чтобы минимизировать проблемы на Windows 11: ONNX Runtime и OpenCV дают более предсказуемую кроссплатформенную установку, чем стек, завязанный на `dlib`.[cite:178][cite:107][cite:156]

## Архитектурные принципы

1. Core отделён от библиотечных адаптеров.
2. Face identity и voice identity обрабатываются независимо.
3. GUI backend и presentation mode — это разные уровни расширения.
4. Windows-safe backend идёт первым, а тяжёлые или нестабильные зависимости остаются optional.
5. Все ключевые модули имеют docstring в начале файла.

## Структура проекта

```text
skynet_local/
├── pyproject.toml
├── README.md
├── configs/
│   └── app.yaml
├── assets/
│   ├── hud/
│   └── themes/
├── models/
│   ├── face/
│   │   ├── detectors/
│   │   ├── recognizers/
│   │   ├── landmarks/
│   │   └── attributes/
│   ├── voice/
│   │   ├── speaker/
│   │   └── vosk/
│   └── pose/
├── data/
│   ├── cache/
│   ├── embeddings/
│   ├── faces/
│   ├── logs/
│   ├── profiles/
│   └── voices/
├── src/
│   └── skynet_local/
│       ├── main.py
│       ├── bootstrap.py
│       ├── config/
│       ├── domain/
│       ├── application/
│       ├── infrastructure/
│       ├── presentation/
│       └── utils/
└── tests/
```

## Основные имена в коде

После переименования используются более точные и нейтральные имена:
- `SkynetRuntime` — тонкая runtime-обёртка, запускающая цикл обработки;
- `SceneOrchestrator` — координатор, собирающий `SceneState`;
- `skynet_local` — import package и src package;
- `skynet-local` — distribution name и console script.

## Слои

### `domain/`

Содержит сущности сцены, наблюдения лица и голоса, enum’ы GUI-режимов и протоколы интерфейсов. Этот слой не должен знать о конкретных реализациях OpenCV, Vosk, Qt или ONNX Runtime.

### `application/`

Содержит orchestration и application services. `SceneOrchestrator` координирует detector, recognizer и сборку `SceneState`, а вспомогательные сервисы отвечают за cooldown и future business rules.[cite:232][cite:243]

### `infrastructure/`

Содержит адаптеры к конкретным технологиям: захват видео, CV backend’ы, голосовые backend’ы, SQLite repository и GUI backend’ы.[cite:19][cite:183]

### `presentation/`

Содержит режимы отображения поверх scene model: `classic`, `minimal`, `diagnostic`, `terminator`. Терминатор-режим остаётся presentation-layer расширением, а не частью core detection pipeline.[cite:161][cite:163]

## GUI backend’ы

| Backend | Назначение |
|---|---|
| OpenCV | Быстрый preview и простое overlay-окно |
| Qt / PySide6 | Основной desktop shell |
| PyQtGraph | Диагностические real-time графики |

PyQtGraph хорошо подходит для технических real-time панелей, а Qt — для полноценных desktop UI поверх видеопотока.[cite:183][cite:179][cite:182]

## Режимы отображения

| Mode | Что делает |
|---|---|
| `classic` | Базовые bbox, имена, сообщения |
| `minimal` | Минимум графического шума |
| `diagnostic` | Инженерные метрики и отладка |
| `terminator` | Красный фильтр, боковая панель и HUD |

## Установка

```bash
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -e .
```

Для GUI и аудио:

```bash
pip install -e .[qt,diagnostics,audio]
```

## Запуск

```bash
python -m skynet_local.main
```

или

```bash
skynet-local
```

## Следующие шаги

1. Подключить реальные ONNX detector и recognizer.
2. Добавить speaker embedding backend.
3. Реализовать реальный Qt shell.
4. Добавить горячее переключение GUI mode.
5. Довести Terminator mode до полноценного cinematic HUD.
