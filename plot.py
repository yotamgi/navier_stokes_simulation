import numpy as np
import cv2
import time

FPS_INTERVAL = 1

class VideoWindow:

    def __init__(self, window_name, frame_size, figsize=(600, 600)):
        self._window_name = window_name
        self._frame_size = frame_size
        cv2.namedWindow(self._window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self._window_name, *figsize)

        self._last_fps = time.time()
        self._fps_counter = 0
    
    def init_frame(self):
        self._frame = np.array(self._frame_size)
    
    def set_image(self, image, vmin=0, vmax=1):
        image_shape = image.shape
        coeff = np.round(np.array(self._frame_size) / image_shape).astype(int)
        
        image = np.minimum(image, vmax)
        image = np.maximum(image, vmin) - vmin
        self._frame = np.kron(image / (vmax - vmin),
                              np.ones(coeff))
            
    def plot_vector(self, field, downscale):
        downscaled = np.array(field.shape[1:]) / downscale
        field_to_frame = np.array(self._frame_size) / np.array(field.shape[1:])
        for x in range(int(downscaled[0])):
            for y in range(int(downscaled[1])):
                field_coords = np.array([x, y]) * downscale
                frame_coords = field_coords * field_to_frame
                vec = -field[:, field_coords.astype(int)[1], field_coords.astype(int)[0]][::-1]
                cv2.arrowedLine(self._frame,
                                frame_coords.astype(int), 
                                (frame_coords + vec).astype(int),
                                (255, 0, 0),
                                5)
            
    def show(self):
        curr_time = time.time()
        if self._last_fps + FPS_INTERVAL < curr_time:
            print('FPS:', self._fps_counter / (curr_time - self._last_fps))
            self._last_fps = curr_time
            self._fps_counter = 0
        
        self._fps_counter += 1
        cv2.imshow(self._window_name, (self._frame*255).astype(np.uint8))

        # Press Q on keyboard to exit
        #key = cv2.waitKey()
        key = cv2.waitKey(25)
        if key & 0xFF == ord('q'):
            # Closes all the frames
            cv2.destroyAllWindows()
            return False
        return True
  

