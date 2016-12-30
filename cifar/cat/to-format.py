import numpy as np
from PIL import Image

CAT_LABEL = 3

im = Image.open("cat-small.png") #Can be many different formats.
pix = im.load()
width, height = im.size
out = np.zeros((3, 32, 32), int)
for y in xrange(height):
    for x in xrange(width):
        for c, val in enumerate(pix[x,y]):
            out[c, 4 + y, 4 + x] = val
raw = [CAT_LABEL] + out.flatten().tolist() 
with open("cat-small.bin", "wb") as f:
    f.write(bytearray(raw))
