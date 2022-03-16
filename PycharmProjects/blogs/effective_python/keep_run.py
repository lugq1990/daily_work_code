from turtle import circle
import win32api
import win32con
import time
import tkinter as tk

root = tk.Tk()
width = root.winfo_screenwidth()
height = root.winfo_screenheight()

mid_width, mid_height = int(width/2), int(height/2)

cycle_size = 15

while True:
    x, y = win32api.GetCursorPos()
    if width - x <= cycle_size or height - x <= cycle_size:
        win32api.SetCursorPos(mid_width, mid_height)
    else:
        win32api.mouse_event(win32con.MOUSE_MOVED, 10, 10, 0, 0)
    time.sleep(120)