from io import StringIO

from src.framework.cameras.alignment import CameraIntrinsics, Point
from src.framework.cameras.alignment_config import AlignmentConfig

if __name__ == '__main__':
    ini_text = """
    hor_px=12.5
    ver_px=-3.0
    roll_deg=1.75
    fov_h_deg=45
    fov_v_deg=11
    """

    input_io = StringIO(ini_text)
    config = AlignmentConfig.load(input_io)

    base_intrinsics = CameraIntrinsics(
        f_x=1200.0,
        f_y=1024.0,
        c_x=640.0,
        c_y=512.0,
    )

    pipeline = config.to_pipeline(base_intrinsics)


    angles = pipeline.pixel_to_angular(Point(x=800.0, y=400.0))
    print("angles:", angles)

    output_io = StringIO()
    config.dump(output_io)

    serialized_text = output_io.getvalue()
    print("--- serialized ---")
    print(serialized_text)
