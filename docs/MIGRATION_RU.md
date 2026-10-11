# Переход на папку gimb

Это отдельная копия проекта, подготовленная для GitHub. Рабочий каталог, из которого сделана копия, не изменён.

1. Распакуйте архив в отдельный каталог.
2. Сохраните рабочее окружение Orange Pi: переустанавливать SDK ради переименования не нужно.
3. Скопируйте `config/local.example.py` в `config/local.py`.
4. Перенесите в локальный конфиг свои пути и выбранные режимы. Например:

```python
detector_backend = "hailo"
mot_backend = "rknn"
frame_source = "video"
Hailo.yolo_hef = "/absolute/path/yolov11s_per.hef"
DatasetVideoPath.tv_n = "/absolute/path/input.mp4"
mot_settings["reid_model_path"] = "/absolute/path/osnet_dynamic.rknn"
Tracking.nanotrack_backbone = "/absolute/path/nanotrackv3_backbone_dyn.rknn"
Tracking.nanotrack_head = "/absolute/path/nanotrackv3_head_fp16.onnx"
```

5. Запустите `python -m src.gimb` из корня новой папки.
6. Обновите команду в systemd/ярлыках, если они используются вне проекта.

Веса и исходное видео можно оставить на прежнем месте: локальный конфиг принимает абсолютные пути. Публичный конфиг намеренно не содержит персональных путей. Свой прежний конфиг можно использовать как источник значений, но не заменяйте им новый файл целиком: в новой версии добавлены пути NanoTrack и загрузка локальных настроек.

Изменения алгоритмов RKNN детекции/ReID и контроллера MOT → SOT не вносились. Сохранены последние изменения Hailo, видеочтения и статистики. Для переносимости нормализованы импорты и убрана обязательная загрузка STARK при использовании NanoTrack.
