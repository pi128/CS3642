package main

import (
	"image"
	"image/color"
	"image/png"
	"os"
)

func dotPNG(filename string, size int, rgba color.RGBA) error {
	img := image.NewNRGBA(image.Rect(0, 0, size, size)) // transparent by default
	cx, cy := float64(size-1)/2, float64(size-1)/2
	r := float64(size) * 0.5 // full circle; use 0.45 for softer edge

	for y := 0; y < size; y++ {
		for x := 0; x < size; x++ {
			dx, dy := float64(x)-cx, float64(y)-cy
			if dx*dx+dy*dy <= r*r {
				img.Set(x, y, rgba)
			}
		}
	}
	f, _ := os.Create(filename)
	defer f.Close()
	return png.Encode(f, img)
}

func main() {
	_ = dotPNG("fish.png", 5, color.RGBA{255, 255, 255, 255}) // white 5x5
	_ = dotPNG("tail.png", 5, color.RGBA{180, 180, 180, 255}) // light gray
	_ = dotPNG("shark.png", 9, color.RGBA{220, 40, 40, 255})  // red 9x9
}
