import cv2

from src.framework.frame.gst.appsink_reader import GstAppSinkReader

if __name__ == "__main__":
    # app_sink_reader = GstAppSinkReader(
    #     pipeline_str="v4l2src ! "
    #                  "video/x-raw,width=640,height=480,framerate=30/1 ! "
    #                  "videoconvert ! "
    #                  "video/x-raw,format=BGR ! "
    #                  "appsink name=sink"
    # )

    app_sink_reader = GstAppSinkReader(
        # pipeline_str="rtspsrc location=rtsp://127.0.0.1:8554/test latency=0 ! "
        pipeline_str="rtspsrc location=rtsp://192.168.75.179:8554/operator/h264/tv latency=0 ! "
                     
                     "rtph264depay !"
                     "avdec_h264 ! "
                     "videoconvert ! "
                     "video/x-raw,format=BGR ! "
                     "appsink name=sink"
    )

    while True:
        sample = app_sink_reader.wait_for_sample()
        if sample is None:
            continue

        cv2.imshow("frame", sample.arr())
        cv2.waitKey(30)
