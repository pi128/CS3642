from PIL import Image
import struct
import sys

def png_to_fwi(png_file, fwi_file):
    im = Image.open(png_file).convert("RGBA")
    w, h = im.size
    pixels = im.load()

    with open(fwi_file, "wb") as f:
        # Write header: 'FWI' + width + height (little endian)
        f.write(b'FWI')
        f.write(struct.pack("<H", w))
        f.write(struct.pack("<H", h))

        for y in range(h):
            for x in range(w):
                r,g,b,a = pixels[x,y]
                # premultiply alpha if you want transparency; or just drop alpha
                r5 = r >> 3
                g6 = g >> 2
                b5 = b >> 3
                rgb565 = (r5 << 11) | (g6 << 5) | b5
                f.write(struct.pack("<H", rgb565))

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 png2fwi.py input.png output.fwi")
    else:
        png_to_fwi(sys.argv[1], sys.argv[2])