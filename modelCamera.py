from ultralytics import YOLO

model = YOLO("runs/detect/train18/weights/best.pt")  # path to trained weights

results = model("Screenshot 2026-02-19 at 2.38.52 PM.png")  # can also be a path or list of images

results[0].show() 

# pops up a window with predictions
# results[0].save("output/")  

# from PIL import Image

# # Open image
# img = Image.open("Screenshot 2026-02-19 at 2.42.47 PM.png")

# # Get width and height
# width, height = img.size
# print(f"Width: {width} px, Height: {height} px")
# print(f"Total pixels: {width * height}")
