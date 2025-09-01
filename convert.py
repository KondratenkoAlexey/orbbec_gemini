from ultralytics import YOLO
import os
import glob


def convert_models_to_onnx(folder_path='.', overwrite=False, specific_models=None):
    """
    Конвертирует все модели YOLO в папке в формат ONNX.

    Args:
        folder_path (str): Путь к папке с моделями
        overwrite (bool): Перезаписывать существующие ONNX файлы
        specific_models (list): Список конкретных моделей для конвертации
    """

    # Определяем какие файлы искать
    if specific_models:
        model_files = [os.path.join(folder_path, f"{model}.pt") for model in specific_models]
    else:
        file_patterns = ['yolo*.pt', 'yolov*.pt']
        model_files = []
        for pattern in file_patterns:
            model_files.extend(glob.glob(os.path.join(folder_path, pattern)))
        model_files = sorted(set(model_files))

    print(f"Найдено моделей для конвертации: {len(model_files)}")

    successful = 0
    skipped = 0
    failed = 0

    for model_path in model_files:
        try:
            if not os.path.exists(model_path):
                print(f"❌ Файл не найден: {model_path}")
                failed += 1
                continue

            base_name = os.path.splitext(model_path)[0]
            onnx_path = f"{base_name}.onnx"

            if os.path.exists(onnx_path) and not overwrite:
                print(f"⏭️  Пропускаем (уже существует): {onnx_path}")
                skipped += 1
                continue

            print(f"\n🔧 Конвертируем: {os.path.basename(model_path)}")
            model = YOLO(model_path)
            model.export(format='onnx')

            # Проверяем что файл создался
            if os.path.exists(onnx_path):
                print(f"✅ Успешно: {os.path.basename(onnx_path)}")
                successful += 1
            else:
                print(f"⚠️  Файл не создан: {onnx_path}")
                failed += 1

        except Exception as e:
            print(f"❌ Ошибка с {os.path.basename(model_path)}: {e}")
            failed += 1
            continue

    print(f"{'=' * 60}\n"
          f"📊 ИТОГИ КОНВЕРТАЦИИ:\n"
          f"   ✅ Успешно: {successful}\n"
          f"   ⏭️ Пропущено: {skipped}\n"
          f"   ❌ Ошибок: {failed}\n"
          f"   📁 Всего обработано: {successful + skipped + failed}"
          )


if __name__ == "__main__":
    # Вариант 1: Конвертировать все модели в текущей папке
    convert_models_to_onnx(folder_path='./YOLO')

    # Вариант 2: Конвертировать с перезаписью существующих файлов
    # convert_models_to_onnx(overwrite=True)

    # Вариант 3: Конвертировать только конкретные модели
    # convert_models_to_onnx(specific_models=['yolo11n', 'yolo11s'])
