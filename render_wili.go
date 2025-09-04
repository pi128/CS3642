/*
//go:build wili

package main

// NOTE: These import names ("fwwasm", "AddPanel", etc.) must match the host's exported symbols.
// If your header uses different names, change the strings after //go:wasmimport.

import "unsafe"

// Panel / screen
//
//go:wasmimport fwwasm addPanel
func wili_addPanel(panelIndex, visible, inRotation, useTile, tileID, bgR, bgG, bgB, showMenu int32)

//go:wasmimport fwwasm showPanel
func wili_showPanel(panelIndex int32)

// Pictures (sprites)
//
//go:wasmimport fwwasm addControlPictureFromFile
func wili_addControlPictureFromFile(panel, controlID, x, y int32, namePtr unsafe.Pointer, nameLen int32, visible int32)

// Control updates
//
//go:wasmimport fwwasm setControlValue
func wili_setControlValue(panel, controlID, value int32)

// Timing
//
//go:wasmimport fwwasm waitms
func wili_waitms(ms int32)

// --------- Thin Go wrappers you call from main.go ---------
func AddPanel(panel, visible, inRotation, useTile, tileID, bgR, bgG, bgB, showMenu int) {
	wili_addPanel(int32(panel), int32(visible), int32(inRotation), int32(useTile), int32(tileID), int32(bgR), int32(bgG), int32(bgB), int32(showMenu))
}
func ShowPanel(panel int) {
	wili_showPanel(int32(panel))
}
func AddControlPictureFromFile(panel, controlID, x, y int, filename string, visible int) {
	ptr := unsafe.Pointer(&[]byte(filename)[0])
	wili_addControlPictureFromFile(int32(panel), int32(controlID), int32(x), int32(y), ptr, int32(len(filename)), int32(visible))
}
func SetControlValue(panel, controlID, newVal int) {
	wili_setControlValue(int32(panel), int32(controlID), int32(newVal))
}
func WaitMS(ms int) { wili_waitms(int32(ms)) }
*/