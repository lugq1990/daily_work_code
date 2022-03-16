import win32api
import time
import win32con

while True:
    win32api.mouse_event(win32con.MOUSE_MOVED, 10, 10, 0, 0)
    time.sleep(120)