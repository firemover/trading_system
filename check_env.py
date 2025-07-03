import sys
try:
    from openvino.runtime import Core
    print("OpenVINO доступен:", Core().available_devices)
except ImportError:
    print("Ошибка: Системный OpenVINO не найден")
    sys.exit(1)