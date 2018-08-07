from picamera import PiCamera
from time import sleep

camera = PiCamera()

# Start PiCamera preview
# camera.start_preview(alpha=200)
# alpha can be any value between 0 and 255. alpha for transparency
camera.start_preview()
""" Note that the camera preview only works when a monitor is connected
to the Pi, so remote access (such as SSH and VNC) will not allow you to see
the camera preview """
for i in range(5):
    sleep(1)
    # PiCamera.capture for capture image
    camera.capture('/home/pi/Desktop/image%s.jpg' % i)
    print("Capture image%s.jpg ..." % i)
# Stop PiCamera preview
camera.stop_preview()

camera.rotation = 180
""" You can rotate the image by 90, 180, or 270 degrees,
or you can set it to 0 to reset. """

camera.start_preview()
camera.start_recording('/home/pi/Desktop/video.h264')
print("Recording ...")
sleep(10)
camera.stop_recording()
camera.stop_preview()

""" The video should play. It may actually play at a faster speed than what has
been recorded, due to omxplayer’s fast frame rate.
$ omxplayer video.h264 """

camera.start_preview()
camera.resolution = (3280, 2464)
camera.framerate = 39 # max framerate
sleep(10)
camera.capture('/home/pi/Desktop/max.jpg')
camera.stop_preview()

"""camera.start_preview()
for effect in camera.IMAGE_EFFECTS:
    camera.image_effect = effect
    camera.annotate_text = "Effect: %s" % effect
    sleep(2)
camera.stop_preview()"""

# camera.EXPOSURE_MODES include off, auto, night, nightpreview, backlight,
# spotlight, sports, snow, beach, verylong, fixedfps, antishake
