#!/usr/bin/python3
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from picamera import PiCamera


class MjpegMixin:
    """
    Add MJPEG features to a subclass of BaseHTTPRequestHandler.
    """

    mjpegBound = 'eb4154aac1c9ee636b8a6f5622176d1fbc08d382ee161bbd42e8483808c684b6'
    frameBegin = 'Content-Type: image/jpeg\n\n'.encode('ascii')
    frameBound = ('\n--' + mjpegBound + '\n').encode('ascii') + frameBegin

    def mjpegBegin(self):
        self.send_response(200)
        self.send_header('Content-Type',
                         'multipart/x-mixed-replace;boundary=' + MjpegMixin.mjpegBound)
        self.end_headers()
        self.wfile.write(MjpegMixin.frameBegin)

    def mjpegEndFrame(self):
        self.wfile.write(MjpegMixin.frameBound)


class SmoothedFpsCalculator:
    """
    Provide smoothed frame per second calculation.
    """

    def __init__(self, alpha=0.1):
        self.t = time.time()
        self.alpha = alpha
        self.sfps = None

    def __call__(self):
        t = time.time()
        d = t - self.t
        self.t = t
        fps = 1.0 / d
        if self.sfps is None:
            self.sfps = fps
        else:
            self.sfps = fps * self.alpha + self.sfps * (1.0 - self.alpha)
        return self.sfps


class Handler(BaseHTTPRequestHandler, MjpegMixin):
    def do_GET(self):
        if self.path == '/contour.mjpeg':
            self.handleContourMjpeg()
        else:
            self.send_response(404)
            self.end_headers()

    def handleContourMjpeg(self):
        import cv2
        import numpy as np
        width, height, blur, sigma = 640, 480, 2, 0.33
        fpsFont, fpsXY = cv2.FONT_HERSHEY_SIMPLEX, (0, height-1)
        self.mjpegBegin()
        with PiCamera() as camera:
            camera.resolution = (width, height)
            camera.video_denoise = False
            camera.image_effect = 'blur'
            camera.image_effect_params = (blur,)
            yuv = np.empty((int(width * height * 1.5),), dtype=np.uint8)
            sfps = SmoothedFpsCalculator()
            for x in camera.capture_continuous(yuv, format='yuv', use_video_port=True):
                image = yuv[:width*height].reshape((height, width))
                v = np.median(image)
                lower = int(max(0, (1.0 - sigma) * v))
                upper = int(min(255, (1.0 + sigma) * v))
                image = cv2.Canny(image, lower, upper)
                cv2.putText(image, '%0.2f fps' %
                            sfps(), fpsXY, fpsFont, 1.0, 255)
                self.wfile.write(cv2.imencode('.jpg', image)[1])
                self.mjpegEndFrame()


def run(port=8000):
    httpd = HTTPServer(('', port), Handler)
    httpd.serve_forever()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='HTTP streaming camera.')
    parser.add_argument('--port', type=int, default=8000,
                        help='listening port number')
    args = parser.parse_args()
    run(port=args.port)
