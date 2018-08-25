from PIL import Image
import sys,os

def sketch(img, threshold):
    if threshold < 0: threshold = 0
    if threshold > 100: threshold = 100
    width, height = img.size
    img = img.convert('L')  # convert to grayscale mode
    pix = img.load()  # get pixel matrix
    for w in range(width):
        for h in range(height):
            if w == width - 1 or h == height - 1:
                continue
            src = pix[w, h]
            dst = pix[w + 1, h + 1]
            diff = abs(src - dst)

            if diff >= threshold:
                pix[w, h] = 0
            else:
                pix[w, h] = 255
    return img

path = 'f1-001-01.jpg'
threshold = 15
if len(sys.argv) == 2:
    try:
        threshold = int(sys.argv[1])
    except ValueError:
        path = sys.argv[1]
elif len(sys.argv) == 3:
    path = sys.argv[1]
    threshold = int(sys.argv[2])
img = Image.open(path)
img = sketch(img, threshold)
img.save('layout.jpg', 'JPEG')