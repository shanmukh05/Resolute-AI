from collections import defaultdict
from PIL import Image, ImageDraw,ImageFont
from math import pi, cos, sin
from canny import canny_edge_detector
import os

# Load Input image:
input_image = Image.open("input.jpeg")

# Output image:
output_image = Image.new("RGB", input_image.size)
output_image.paste(input_image)
draw_result = ImageDraw.Draw(output_image)

# Find circles
rmin = 18
rmax = 20
steps = 100
threshold = 0.4

points = []
for r in range(rmin, rmax + 1):
    for t in range(steps):
        points.append((r, int(r * cos(2 * pi * t / steps)), int(r * sin(2 * pi * t / steps))))

acc = defaultdict(int)
for x, y in canny_edge_detector(input_image):
    for r, dx, dy in points:
        a = x - dx
        b = y - dy
        acc[(a, b, r)] += 1

circles = []
circle_count = 0
for k, v in sorted(acc.items(), key=lambda i: -i[1]):
    x, y, r = k
    if v / steps >= threshold and all((x - xc) ** 2 + (y - yc) ** 2 > rc ** 2 for xc, yc, rc in circles):
        print(v / steps, x, y, r)
        circles.append((x, y, r))
        circle_count+=1

for x, y, r in circles:
    draw_result.ellipse((x-r, y-r, x+r, y+r), outline=(255,0,0,0))
#write pipe count on image
fontsize= 25
font = ImageFont.truetype("arial.ttf", fontsize)
draw_result.text((10,10),"Pipes count = "+str(circle_count),fill=(255,0,0),font=font)
# Save output image as "output.png"
output_image.save("output.png")

print("Number of Pipes in the given image =",circle_count)
file1 = open("output.txt","w")
file1.write("Number of Pipes in the given image = "+str(circle_count))
file1.close()