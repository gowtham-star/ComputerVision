import cv2
import numpy as np
import numba as nb
import depthai as dai
#Captures the image oakd camera feed
def integral_image(image, *, dtype=None):
    if dtype is None and image.real.dtype.kind == 'f':
        dtype = np.promote_types(image.dtype, np.float64)

    S = image
    for i in range(image.ndim):
        S = S.cumsum(axis=i, dtype=dtype)
    return S
streams = []
# Enable one or both streams
streams.append('isp')
# Optimized with 'numba' as otherwise would be extremely slow (55 seconds per frame!)
@nb.njit(nb.uint16[::1] (nb.uint8[::1], nb.uint16[::1], nb.boolean), parallel=True, cache=True)
def unpack_raw10(input, out, expand16bit):
    lShift = 6 if expand16bit else 0
    for i in nb.prange(input.size // 5): # around  5ms per frame
        b4 = input[i * 5 + 4]
        out[i * 4]     = ((input[i * 5]     << 2) | ( b4       & 0x3)) << lShift
        out[i * 4 + 1] = ((input[i * 5 + 1] << 2) | ((b4 >> 2) & 0x3)) << lShift
        out[i * 4 + 2] = ((input[i * 5 + 2] << 2) | ((b4 >> 4) & 0x3)) << lShift
        out[i * 4 + 3] = ((input[i * 5 + 3] << 2) |  (b4 >> 6)       ) << lShift
    return out
print("The depthai version:", dai.__version__)
pipeline = dai.Pipeline()
cam = pipeline.createColorCamera()
cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_12_MP)
if 'isp' in streams:
    xout_isp = pipeline.createXLinkOut()
    xout_isp.setStreamName('isp')
    cam.isp.link(xout_isp.input)
device = dai.Device(pipeline)
device.startPipeline()
queue_list = []
for s in streams:
    q = device.getOutputQueue(name=s, maxSize=3, blocking=True)
    queue_list.append(q)
    # Make window resizable, and configure initial size
    cv2.namedWindow(s, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(s, (960, 540))
capture_flag = False
img_counter = 0
while True:
    for q in queue_list:
        name = q.getName()
        data = q.get()
        width, height = data.getWidth(), data.getHeight()
        payload = data.getData()
        capture_file_info_str = f"capture_{name}_{img_counter}"
        if name == 'isp':
            shape = (height * 3 // 2, width)
            yuv420p = payload.reshape(shape).astype(np.uint8)
            bgr = cv2.cvtColor(yuv420p, cv2.COLOR_YUV2BGR_IYUV)
            # grayscale_img =  cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY)
            frame= cv2.copyMakeBorder(bgr, 50, 50, 50, 50, cv2.BORDER_CONSTANT, (0,0,0))
            frame=integral_image(frame)
            frame = frame/np.amax(frame)
            frame = np.clip(frame, 0,255)
        if capture_flag:  # Save to disk if 'space' was pressed
            filename = capture_file_info_str + '.png'
            print("Saving to file:", filename)
            grayscale_img = np.ascontiguousarray(grayscale_img)  # just in case
            cv2.imwrite(filename, grayscale_img)
        bgr = np.ascontiguousarray(bgr)  # just in case
        cv2.imshow(name, frame)
    # Reset capture_flag after iterating through all streams
    capture_flag = False
    key = cv2.waitKey(5)
    if key%256 == 27:
        #escape 
        print("Operation over")
        break
    elif key%256 == 32:
        #space to click picture
        capture_flag = True
        img_counter += 1
