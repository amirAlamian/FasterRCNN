import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
def draw_boxes (img_address, boxes):
  img_np = cv2.cvtColor(cv2.imread(img_address), cv2.COLOR_BGR2RGB)
  plt.figure(figsize=(8, 8))
  plt.imshow(img_np)
  for box in boxes:
      x1, y1, x2, y2 = box.cpu()
      rect = patches.Rectangle(
          (x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor="red", facecolor="none"
      )
      plt.gca().add_patch(rect)
  plt.show()
