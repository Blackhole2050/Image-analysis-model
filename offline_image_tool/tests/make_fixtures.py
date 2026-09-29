from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).resolve().parent / "fixtures"

def main():
    OUT.mkdir(exist_ok=True)
    font = ImageFont.load_default(size=28)
    im = Image.new("RGB", (720, 320), "white")
    d = ImageDraw.Draw(im)
    rows = [["Item", "Quantity", "Price"], ["Apples", "12", "3.50"], ["Pears", "7", "2.25"]]
    for y in (40, 120, 200, 280):
        d.line((30, y, 690, y), fill="black", width=3)
    for x in (30, 250, 470, 690):
        d.line((x, 40, x, 280), fill="black", width=3)
    for r, row in enumerate(rows):
        for c, text in enumerate(row):
            d.text((45 + c * 220, 63 + r * 80), text, fill="black", font=font)
    im.save(OUT / "table.png")
    im = Image.new("RGB", (720, 600), "white")
    d = ImageDraw.Draw(im)
    for y, text in ((40, "Start"), (240, "Check input"), (440, "Save result")):
        d.rectangle((170, y, 550, y + 100), outline="black", width=3)
        d.text((205, y + 30), text, fill="black", font=font)
    for y in (140, 340):
        d.line((360, y, 360, y + 90), fill="black", width=4)
        d.polygon([(350, y + 85), (370, y + 85), (360, y + 100)], fill="black")
    im.save(OUT / "flowchart.png")
    im = Image.new("RGB", (720, 520), "white")
    d = ImageDraw.Draw(im)
    d.text((190, 15), "Units by month", fill="black", font=font)
    d.line((100, 80, 100, 420, 650, 420), fill="black", width=3)
    for value in (0, 10, 20):
        y = 420 - value * 14
        d.text((40, y - 15), str(value), fill="black", font=font)
        d.line((95, y, 650, y), fill="lightgray", width=1)
    for x, name, value in ((200, "Jan", 10), (420, "Feb", 20)):
        y = 420 - value * 14
        d.rectangle((x, y, x + 95, 419), fill="#4682B4", outline="black")
        d.text((x + 25, y - 40), str(value), fill="black", font=font)
        d.text((x + 18, 432), name, fill="black", font=font)
    d.text((290, 478), "Month", fill="black", font=font)
    d.text((10, 35), "Units", fill="black", font=font)
    im.save(OUT / "chart.png")

if __name__ == "__main__":
    main()

