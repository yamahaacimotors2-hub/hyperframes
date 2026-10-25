"""Render the TFT dashboard texture used on the bike (system python3 + Pillow)."""

import math

from PIL import Image, ImageDraw, ImageFilter, ImageFont

W, H = 1024, 640
img = Image.new("RGB", (W, H), (2, 6, 12))
d = ImageDraw.Draw(img)
F = "/usr/share/fonts/opentype/inter/Inter-{}.otf"
big = ImageFont.truetype(F.format("Bold"), 200)
mid = ImageFont.truetype(F.format("SemiBold"), 54)
small = ImageFont.truetype(F.format("Medium"), 36)
cyan, white, dim = (40, 215, 255), (235, 245, 255), (40, 70, 90)

# rev arc
cx, cy, r = 300, 380, 250
for i in range(60):
    a0 = math.radians(150 + i * 4)
    col = cyan if i < 22 else dim
    x0, y0 = cx + r * math.cos(a0), cy + r * math.sin(a0)
    x1, y1 = cx + (r - 34) * math.cos(a0), cy + (r - 34) * math.sin(a0)
    d.line((x0, y0, x1, y1), fill=col, width=9)
d.text((cx, cy - 10), "0", font=big, fill=white, anchor="mm")
d.text((cx, cy + 105), "km/h", font=small, fill=(150, 180, 200), anchor="mm")

# hybrid panel
d.rounded_rectangle((610, 110, 970, 250), 24, outline=cyan, width=5)
d.text((790, 180), "HYBRID", font=mid, fill=cyan, anchor="mm")
d.text((790, 330), "SMG  ASSIST", font=small, fill=white, anchor="mm")
for i in range(8):
    d.rounded_rectangle((625 + i * 43, 375, 655 + i * 43, 425), 6, fill=cyan if i < 5 else dim)
d.text((790, 505), "STOP & START  ●", font=small, fill=(120, 255, 170), anchor="mm")

glow = img.filter(ImageFilter.GaussianBlur(10))
img = Image.blend(img, glow, 0.35)
img.save("tft.png")
print("tft.png")
