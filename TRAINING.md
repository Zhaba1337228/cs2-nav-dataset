# CS2 Navigation Model Training

Обучение модели имитационного обучения для навигации в CS2 на 2x RTX 3090.

## Быстрый старт

### 0. One-click подготовка окружения (Ubuntu/SSH, в фоне)

```bash
bash setup_all.sh
```

Скрипт в фоне создаёт `.venv`, ставит зависимости для обучения, ставит PyTorch и подготавливает датасет.

Логи установки:
```bash
tail -f logs/setup_*.log
```

### 1. Подготовка данных

Убедись, что у тебя есть манифесты:
```bash
dataset/manifests/train_manifest.jsonl
dataset/manifests/val_manifest.jsonl
```

### 2. Запуск обучения (2 GPU)

```bash
.venv/bin/python -m training.train \
    --train-manifest dataset/manifests/train_manifest.jsonl \
    --val-manifest dataset/manifests/val_manifest.jsonl \
    --dataset-root dataset \
    --checkpoint-dir checkpoints \
    --backbone resnet18 \
    --batch-size 64 \
    --epochs 100 \
    --lr 1e-4 \
    --num-workers 8 \
    --world-size 2
```

### 3. Запуск обучения (1 GPU для теста)

```bash
.venv/bin/python -m training.train \
    --train-manifest dataset/manifests/train_manifest.jsonl \
    --val-manifest dataset/manifests/val_manifest.jsonl \
    --batch-size 32 \
    --epochs 10 \
    --world-size 1
```

## Параметры модели

### Backbone архитектуры
- `resnet18` - быстрая, легкая (11M параметров) ✅ рекомендуется для старта
- `resnet34` - средняя (21M параметров)
- `resnet50` - тяжелая (25M параметров)
- `efficientnet_b0` - эффективная (5M параметров)
- `efficientnet_b1` - эффективная средняя (7M параметров)

### Temporal modeling
```bash
--history-len 4 --use-temporal
```
Использует LSTM для обработки последовательности кадров (лучше для динамики).

## Оптимизация для 2x 3090

### Рекомендуемые настройки

**ResNet18 (быстро, хорошее качество):**
```bash
python -m training.train \
    --train-manifest dataset/manifests/train_manifest.jsonl \
    --val-manifest dataset/manifests/val_manifest.jsonl \
    --backbone resnet18 \
    --batch-size 64 \
    --grad-accum-steps 2 \
    --lr 2e-4 \
    --num-workers 8 \
    --epochs 100 \
    --world-size 2
```
- Эффективный batch size: 64 × 2 GPU × 2 accum = 256
- ~200-300 samples/sec на 2x 3090

**ResNet34 (лучшее качество):**
```bash
python -m training.train \
    --train-manifest dataset/manifests/train_manifest.jsonl \
    --val-manifest dataset/manifests/val_manifest.jsonl \
    --backbone resnet34 \
    --batch-size 48 \
    --grad-accum-steps 2 \
    --lr 1e-4 \
    --num-workers 8 \
    --epochs 100 \
    --world-size 2
```
- Эффективный batch size: 48 × 2 GPU × 2 accum = 192

**EfficientNet-B0 (максимальная эффективность):**
```bash
python -m training.train \
    --train-manifest dataset/manifests/train_manifest.jsonl \
    --val-manifest dataset/manifests/val_manifest.jsonl \
    --backbone efficientnet_b0 \
    --batch-size 96 \
    --grad-accum-steps 1 \
    --lr 1e-4 \
    --num-workers 8 \
    --epochs 100 \
    --world-size 2
```

**С temporal modeling (для последовательностей):**
```bash
python -m training.train \
    --train-manifest dataset/manifests/train_manifest.jsonl \
    --val-manifest dataset/manifests/val_manifest.jsonl \
    --backbone resnet18 \
    --history-len 4 \
    --use-temporal \
    --batch-size 32 \
    --grad-accum-steps 4 \
    --lr 1e-4 \
    --num-workers 8 \
    --epochs 100 \
    --world-size 2
```

## Веса лоссов

Настрой веса для разных действий:
```bash
--loss-weight-move 1.0 \
--loss-weight-turn 1.0 \
--loss-weight-jump 0.5 \
--loss-weight-crouch 0.5 \
--loss-weight-fire 0.5 \
--loss-weight-mouse 0.3
```

## Inference (предсказание)

### Одно изображение
```bash
python -m training.inference \
    --checkpoint checkpoints/checkpoint_best.pt \
    --image path/to/frame.jpg
```

### Batch inference
```bash
python -m training.inference \
    --checkpoint checkpoints/checkpoint_best.pt \
    --image-dir dataset/raw_sessions/session_0001/frames \
    --output predictions.json
```

## Мониторинг

Чекпоинты сохраняются в `checkpoints/`:
- `checkpoint_latest.pt` - последний чекпоинт
- `checkpoint_best.pt` - лучший по validation loss
- `checkpoint_epoch_N.pt` - каждые 10 эпох
- `metrics.json` - история метрик

## Возобновление обучения

```bash
python -m training.train \
    --train-manifest dataset/manifests/train_manifest.jsonl \
    --val-manifest dataset/manifests/val_manifest.jsonl \
    --resume checkpoints/checkpoint_latest.pt
```

## Архитектура модели

```
Input: (B, history_len, 3, 224, 224)
  ↓
CNN Backbone (ResNet/EfficientNet)
  ↓
[Optional] LSTM (if use_temporal=True)
  ↓
Shared FC (512)
  ↓
Multi-head outputs:
  - action_move: 9 classes (stop, forward, left, right, etc.)
  - action_turn: 9 classes (no_turn, turn_left_small, etc.)
  - action_jump: binary
  - action_crouch: binary
  - action_fire: binary
  - mouse_dx: regression
  - mouse_dy: regression
```

## Troubleshooting

### Out of Memory
- Уменьши `--batch-size`
- Увеличь `--grad-accum-steps`
- Используй меньший backbone (`resnet18` вместо `resnet50`)

### Медленное обучение
- Увеличь `--num-workers` (рекомендуется 4-8 на GPU)
- Проверь, что данные на быстром SSD
- Используй `--no-amp` если есть проблемы с mixed precision

### Низкая точность
- Увеличь размер модели (`resnet34` или `resnet50`)
- Добавь temporal modeling (`--use-temporal --history-len 4`)
- Настрой веса лоссов
- Увеличь learning rate или используй warmup
