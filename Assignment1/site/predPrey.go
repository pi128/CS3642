package main

import (
	"math"
	"math/rand"
	"time"
)

/* ================= math + types ================= */

type Vec2 struct{ X, Y float64 }
type Fish struct {
	Pos, Vel Vec2
	Tail     [TailLength]Vec2
	TailHead int
}
type Shark struct{ Pos, Vel Vec2 }

func (a Vec2) Add(b Vec2) Vec2    { return Vec2{a.X + b.X, a.Y + b.Y} }
func (a Vec2) Sub(b Vec2) Vec2    { return Vec2{a.X - b.X, a.Y - b.Y} }
func (a Vec2) Mul(s float64) Vec2 { return Vec2{a.X * s, a.Y * s} }
func (a Vec2) Len2() float64      { return a.X*a.X + a.Y*a.Y }
func (a Vec2) Len() float64       { return math.Sqrt(a.Len2()) }
func (a Vec2) Normalize() Vec2 {
	l2 := a.Len2()
	if l2 == 0 {
		return Vec2{}
	}
	inv := 1 / math.Sqrt(l2)
	return Vec2{a.X * inv, a.Y * inv}
}
func (a Vec2) SafeNormalize() Vec2 { return a.Normalize() }
func (a Vec2) Limit(max float64) Vec2 {
	if max <= 0 {
		return Vec2{}
	}
	l2, m2 := a.Len2(), max*max
	if l2 > m2 {
		inv := max / math.Sqrt(l2)
		return Vec2{a.X * inv, a.Y * inv}
	}
	return a
}
func (a Vec2) PerpCW() Vec2  { return Vec2{a.Y, -a.X} }
func (a Vec2) PerpCCW() Vec2 { return Vec2{-a.Y, a.X} }

func Wrap(p Vec2, w, h float64) Vec2 {
	x := math.Mod(p.X+w, w)
	y := math.Mod(p.Y+h, h)
	return Vec2{x, y}
}

/* ================= sim constants ================= */

const (
	// world = canvas
	W  = 480.0
	H  = 320.0
	dt = 1.0 / 60.0 // seconds per tick

	NumFish      = 80
	TailLength   = 10
	MaxNeighbors = 14

	// radii (px)
	Rsep    = 12.0
	Rn      = 80.0
	Rthreat = 80.0 // smaller radius => less constant panic

	// schooling weights
	wSep   = 0.8
	wAlign = 2.2
	wCoh   = 1.4
	wTan   = 0.35
	drag   = 0.05

	// baseline spiral + soft wall
	wSpin = 0.22
	wWall = 0.9

	// flee (px/s)
	fleeMax = 90.0
	fleeExp = 2.0

	// speeds (px/s)
	vFishMax  = 150.0
	vFishMin  = 45.0 // <- speed floor so they never stall
	vBoost    = 18.0 // <- mild re-energizer if too slow
	vSharkMax = 110.0

	SwirlClockwise = true

	// IDs
	PanelID     = 0
	SharkCID    = 10
	BaseFishCID = 1000
	BaseTailCID = 5000
)

/* ================= forces ================= */

func SeparationForce(i int, flock []Fish, rsep float64, maxN int) Vec2 {
	const eps = 1e-6
	r2 := rsep * rsep
	self := flock[i].Pos
	force := Vec2{}
	seen := 0
	for j := range flock {
		if j == i {
			continue
		}
		d := self.Sub(flock[j].Pos)
		d2 := d.Len2()
		if d2 >= r2 || d2 == 0 {
			continue
		}
		dist := math.Sqrt(d2)
		strength := (rsep - dist) / (rsep + eps)
		force = force.Add(d.Normalize().Mul(strength))
		seen++
		if maxN > 0 && seen >= maxN {
			break
		}
	}
	return force
}

func AlignmentForce(i int, flock []Fish, rn float64, maxN int) Vec2 {
	r2 := rn * rn
	self := flock[i]
	sum := Vec2{}
	count := 0
	for j := range flock {
		if j == i {
			continue
		}
		if self.Pos.Sub(flock[j].Pos).Len2() <= r2 {
			sum = sum.Add(flock[j].Vel)
			count++
			if maxN > 0 && count >= maxN {
				break
			}
		}
	}
	if count == 0 {
		return Vec2{}
	}
	avg := sum.SafeNormalize()
	cur := self.Vel.SafeNormalize()
	return avg.Sub(cur)
}

func CohesionForce(i int, flock []Fish, rn float64, maxN int) Vec2 {
	r2 := rn * rn
	self := flock[i]
	sum := Vec2{}
	count := 0
	for j := range flock {
		if j == i {
			continue
		}
		if self.Pos.Sub(flock[j].Pos).Len2() <= r2 {
			sum = sum.Add(flock[j].Pos)
			count++
			if maxN > 0 && count >= maxN {
				break
			}
		}
	}
	if count == 0 {
		return Vec2{}
	}
	centroid := sum.Mul(1.0 / float64(count))
	desired := centroid.Sub(self.Pos).SafeNormalize()
	return desired.Sub(self.Vel.SafeNormalize())
}

func TangentialSwirlForce(i int, flock []Fish, rn float64, maxN int, clockwise bool) Vec2 {
	r2 := rn * rn
	self := flock[i]
	sum := Vec2{}
	count := 0
	for j := range flock {
		if j == i {
			continue
		}
		if self.Pos.Sub(flock[j].Pos).Len2() <= r2 {
			sum = sum.Add(flock[j].Pos)
			count++
			if maxN > 0 && count >= maxN {
				break
			}
		}
	}
	if count == 0 {
		return Vec2{}
	}
	centroid := sum.Mul(1.0 / float64(count))
	toC := centroid.Sub(self.Pos).SafeNormalize()
	if toC.X == 0 && toC.Y == 0 {
		return Vec2{}
	}
	var tangent Vec2
	if clockwise {
		tangent = toC.PerpCW()
	} else {
		tangent = toC.PerpCCW()
	}
	return tangent.SafeNormalize().Sub(self.Vel.SafeNormalize())
}

// global swirl toward scene center (baseline spin)
func GlobalSwirlForce(pos, center Vec2, clockwise bool) Vec2 {
	toC := center.Sub(pos).SafeNormalize()
	if toC.X == 0 && toC.Y == 0 {
		return Vec2{}
	}
	if clockwise {
		return toC.PerpCW()
	}
	return toC.PerpCCW()
}

// soft wall to keep fish inside (within margin m)
func BorderForce(pos Vec2, w, h, m float64) Vec2 {
	fx, fy := 0.0, 0.0
	if pos.X < m {
		fx += (m - pos.X) / m
	} else if pos.X > w-m {
		fx -= (pos.X - (w - m)) / m
	}
	if pos.Y < m {
		fy += (m - pos.Y) / m
	} else if pos.Y > h-m {
		fy -= (pos.Y - (h - m)) / m
	}
	// scale to velocity-like magnitude
	return Vec2{fx, fy}.Mul(140.0)
}

func FleeForce(i int, flock []Fish, sharkPos Vec2, rThreat, maxStrength, exponent float64) Vec2 {
	if rThreat <= 0 {
		return Vec2{}
	}
	d := flock[i].Pos.Sub(sharkPos)
	d2 := d.Len2()
	r2 := rThreat * rThreat
	if d2 == 0 || d2 >= r2 {
		return Vec2{}
	}
	dist := math.Sqrt(d2)
	t := 1.0 - dist/rThreat
	if t < 0 {
		t = 0
	}
	strength := maxStrength * math.Pow(t, exponent)
	return d.SafeNormalize().Mul(strength)
}

func clamp01(x float64) float64 {
	if x < 0 {
		return 0
	}
	if x > 1 {
		return 1
	}
	return x
}

/* ================ shark helpers ================ */

func SchoolCenter(flock []Fish) Vec2 {
	if len(flock) == 0 {
		return Vec2{}
	}
	sum := Vec2{}
	for i := range flock {
		sum = sum.Add(flock[i].Pos)
	}
	return sum.Mul(1.0 / float64(len(flock)))
}

func UpdateSharkOU(s *Shark, dt, kappa, sigma, beta, gamma, sMax float64, sceneCenter, schoolCenter Vec2) {
	reversion := s.Vel.Mul(-kappa * dt)
	noise := Vec2{rand.NormFloat64(), rand.NormFloat64()}.Mul(sigma * math.Sqrt(dt))
	centerPull := sceneCenter.Sub(s.Pos).Mul(beta * dt)
	schoolPull := schoolCenter.Sub(s.Pos).Mul(gamma * dt)
	s.Vel = s.Vel.Add(reversion).Add(noise).Add(centerPull).Add(schoolPull).Limit(sMax)
	s.Pos = s.Pos.Add(s.Vel.Mul(dt))
}

/* ===================== main ===================== */

func main() {
	rand.Seed(time.Now().UnixNano())

	// init fish
	fish := make([]Fish, NumFish)
	center0 := Vec2{W * 0.5, H * 0.5}
	for i := range fish {
		fish[i].Pos = Vec2{rand.Float64() * W, rand.Float64() * H}
		// give a slight initial tangential bias so a spiral forms quickly
		toC := center0.Sub(fish[i].Pos).SafeNormalize()
		tan := toC.PerpCW()
		dir := Vec2{rand.Float64()*2 - 1, rand.Float64()*2 - 1}.Normalize()
		fish[i].Vel = dir.Mul(40).Add(tan.Mul(90)).Limit(120.0)
		for j := 0; j < TailLength; j++ {
			fish[i].Tail[j] = fish[i].Pos
		}
	}

	// shark patrol
	shark := Shark{Pos: Vec2{W * 0.5, H * 0.5}}

	// UI setup (PNG files live next to index.html)
	AddPanel(PanelID, 1, 0, 0, 0, 0, 0, 0, 0)
	ShowPanel(PanelID)
	AddControlPictureFromFile(PanelID, SharkCID, int(shark.Pos.X), int(shark.Pos.Y), "shark.png", 1)
	for i := 0; i < NumFish; i++ {
		AddControlPictureFromFile(PanelID, BaseFishCID+i, int(fish[i].Pos.X), int(fish[i].Pos.Y), "fish.png", 1)
		for j := 0; j < TailLength; j++ {
			AddControlPictureFromFile(PanelID, BaseTailCID+i*TailLength+j, int(fish[i].Tail[j].X), int(fish[i].Tail[j].Y), "tail.png", 1)
		}
	}

	frame := 0
	for {
		// --- SIM ---
		center := SchoolCenter(fish)

		// calmer shark patrol (kappa↑, sigma↓, gentle pulls)
		UpdateSharkOU(&shark, dt, 0.9, 25.0, 0.22, 0.08, vSharkMax, Vec2{W * 0.5, H * 0.5}, center)
		shark.Pos = Wrap(shark.Pos, W, H)

		for i := range fish {
			// schooling first
			sep := SeparationForce(i, fish, Rsep, MaxNeighbors).Mul(wSep)
			aln := AlignmentForce(i, fish, Rn, MaxNeighbors).Mul(wAlign)
			coh := CohesionForce(i, fish, Rn, MaxNeighbors).Mul(wCoh)
			tan := TangentialSwirlForce(i, fish, Rn, MaxNeighbors, SwirlClockwise).Mul(wTan)

			// baseline swirl + soft wall
			spin := GlobalSwirlForce(fish[i].Pos, center, SwirlClockwise).Mul(wSpin)
			wall := BorderForce(fish[i].Pos, W, H, 24.0).Mul(wWall)

			// gentle flee (close + curved)
			flee := FleeForce(i, fish, shark.Pos, Rthreat, fleeMax, fleeExp)

			acc := sep.Add(aln).Add(coh).Add(tan).Add(spin).Add(wall).Add(flee).Sub(fish[i].Vel.Mul(drag))

			// integrate
			fish[i].Vel = fish[i].Vel.Add(acc.Mul(dt)).Limit(vFishMax)

			// speed floor: nudge forward if too slow so they never stall
			speed := fish[i].Vel.Len()
			if speed < vFishMin {
				dir := fish[i].Vel
				if dir.Len2() == 0 {
					// random tiny nudge if exactly zero
					dir = Vec2{rand.Float64()*2 - 1, rand.Float64()*2 - 1}.Normalize()
				} else {
					dir = dir.SafeNormalize()
				}
				fish[i].Vel = dir.Mul(vFishMin + vBoost*rand.Float64()*0.5)
			}

			fish[i].Pos = Wrap(fish[i].Pos.Add(fish[i].Vel.Mul(dt)), W, H)

			// tail ring push
			fish[i].TailHead = (fish[i].TailHead + 1) % TailLength
			fish[i].Tail[fish[i].TailHead] = fish[i].Pos
		}

		// --- RENDER (batched XY) ---
		SetControlXY(PanelID, SharkCID, int(shark.Pos.X), int(shark.Pos.Y))
		for i := 0; i < NumFish; i++ {
			SetControlXY(PanelID, BaseFishCID+i, int(fish[i].Pos.X), int(fish[i].Pos.Y))
			// update tail markers every other frame
			if frame%2 == 0 {
				for j := 0; j < TailLength; j++ {
					idx := (fish[i].TailHead - j + TailLength) % TailLength
					p := fish[i].Tail[idx]
					id := BaseTailCID + i*TailLength + j
					SetControlXY(PanelID, id, int(p.X), int(p.Y))
				}
			}
		}

		frame++
		WaitMS(16) // ~60 fps
	}
}
