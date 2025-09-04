//go:build js && wasm

package main

import (
	"sync"
	"syscall/js"
	"time"
)

type control struct {
	id      int
	x, y    int
	visible bool
	key     string // filename key -> shared image
	w, h    int
}

var (
	doc         = js.Global().Get("document")
	win         = js.Global()
	canvas      js.Value
	ctx         js.Value
	mu          sync.Mutex
	started     bool
	controls    = map[int]*control{}
	images      = map[string]js.Value{} // filename -> shared <img>
	rafCallback js.Func
)

func AddPanel(panel, visible, inRotation, useTile, tileID, bgR, bgG, bgB, showMenu int) {
	ensureCanvas()
}
func ShowPanel(panel int) {}

func AddControlPictureFromFile(panel, controlID, x, y int, filename string, visible int) {
	ensureCanvas()

	img, ok := images[filename]
	if !ok {
		img = doc.Call("createElement", "img")
		img.Set("src", filename)
		img.Set("style", "display:none")
		doc.Get("body").Call("appendChild", img)
		images[filename] = img
	}

	c := &control{id: controlID, x: x, y: y, visible: visible != 0, key: filename}
	img.Call("addEventListener", "load", js.FuncOf(func(this js.Value, args []js.Value) any {
		if c.w == 0 || c.h == 0 {
			c.w = img.Get("naturalWidth").Int()
			c.h = img.Get("naturalHeight").Int()
		}
		return nil
	}))

	mu.Lock()
	controls[controlID] = c
	mu.Unlock()

	startLoop()
}

func SetControlXY(panel, controlID, x, y int) {
	mu.Lock()
	if c, ok := controls[controlID]; ok {
		c.x, c.y = x, y
	}
	mu.Unlock()
}
func SetControlValue(panel, controlID, newVal int) {} // not used in web
func WaitMS(ms int)                                { time.Sleep(time.Duration(ms) * time.Millisecond) }

func ensureCanvas() {
	if canvas.Truthy() && ctx.Truthy() {
		return
	}
	canvas = doc.Call("getElementById", "c")
	if !canvas.Truthy() {
		canvas = doc.Call("createElement", "canvas")
		canvas.Set("id", "c")
		canvas.Set("width", 480)
		canvas.Set("height", 320)
		doc.Get("body").Call("appendChild", canvas)
	}
	ctx = canvas.Call("getContext", "2d")
}

func startLoop() {
	if started {
		return
	}
	started = true
	rafCallback = js.FuncOf(func(this js.Value, args []js.Value) any {
		draw()
		win.Call("requestAnimationFrame", rafCallback)
		return nil
	})
	win.Call("requestAnimationFrame", rafCallback)
}

func draw() {
	if !ctx.Truthy() {
		return
	}
	w := canvas.Get("width").Int()
	h := canvas.Get("height").Int()

	ctx.Set("fillStyle", "#0a0f1a")
	ctx.Call("fillRect", 0, 0, w, h)

	mu.Lock()
	for _, c := range controls {
		if !c.visible {
			continue
		}
		img := images[c.key]
		if !img.Truthy() {
			continue
		}
		cx := c.x
		cy := c.y
		if c.w > 0 && c.h > 0 {
			ctx.Call("drawImage", img, cx-c.w/2, cy-c.h/2, c.w, c.h)
		} else {
			ctx.Call("drawImage", img, cx-8, cy-8)
		}
	}
	mu.Unlock()
}
