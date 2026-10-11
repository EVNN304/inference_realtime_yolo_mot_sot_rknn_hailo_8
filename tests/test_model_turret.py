from app_rook.model_turret import TurretModel


def test_model_turret():
    sut = TurretModel(threshold_deg=0.5)

    # Симулируем последовательность измерений с шумом и реальными изменениями
    test_angles = [
        (0, False),  # начальное положение
        (0.100, False),  # шум (< 0.5°)
        (0.200, False),  # шум (< 0.5°)
        (0, False),  # возврат в начальное положение
        (0.600, True),  # реальное изменение (> 0.5°)
        (359.900, True),  # почти полный круг
        (0.100, False),  # переход через 0° (большое изменение)
        (359.800, False),  # обратный переход
        (0.200, False),  # шум
    ]

    for i in range(len(test_angles)):
        azimuth, expected = test_angles[i]
        sut.update_angles(azimuth, 0.0)
        assert expected == sut.azimuth_changed, (
            f"{test_angles[i - 1][0]} and {test_angles[i][0]} expected to be "
            f"{'Changed' if expected else 'Not Changed'}"
        )
