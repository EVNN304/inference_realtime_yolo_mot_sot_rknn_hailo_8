"""Independent selection of detector and ReID acceleration; imports stay lazy."""

def selected_backends(config):
    detector = getattr(config, 'detector_backend', 'rknn').lower()
    mot = getattr(config, 'mot_backend', 'rknn').lower()
    for name, value in (('detector_backend', detector), ('mot_backend', mot)):
        if value not in ('rknn', 'hailo'):
            raise ValueError(f'{name}={value!r}; expected rknn or hailo')
    return detector, mot


def create_detection_loop(state, controller):
    import config.config as config
    detector, mot = selected_backends(config)
    if detector == mot == 'rknn':
        from src.gimb.ctrls.det_loop_two import TrOsDetectionTwoChannelLoop
        return TrOsDetectionTwoChannelLoop(state, controller)
    from src.gimb.ctrls.det_loop_hailo import HailoDetectionLoop
    return HailoDetectionLoop(state, controller, config.Hailo)
