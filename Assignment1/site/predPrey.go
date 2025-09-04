package main

import (
	"math"
	"math/rand"
	"time"
)

// ===== math types =====
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

// ===== sim constants =====
const (
	W  = 480.0
	H  = 320.0
	dt = 1.0 / 60.0 // seconds per tick

	NumFish      = 80
	TailLength   = 10
	MaxNeighbors = 14

	// radii (px)
	Rsep    = 14.0
	Rn      = 70.0
	Rthreat = 110.0

	// weights (dimensionless steering gains)
	wSep   = 1.2
	wAlign = 1.6  // ↑ stronger alignment
	wCoh   = 1.0  // ↑ stronger cohesion
	wTan   = 0.20 // ↓ swirl so flock holds together
	drag   = 0.04 // a touch more drag

	// flee & speeds (px/s)
	fleeMax   = 240.0
	fleeExp   = 1.4
	vFishMax  = 170.0
	vSharkMax = 160.0

	SwirlClockwise = true

	// UI IDs
	PanelID     = 0
	SharkCID    = 10
	BaseFishCID = 1000
	BaseTailCID = 5000
)

// ===== forces =====
func SeparationForce(i int, flock []Fish, rsep float64, maxN int) Vec2 {
	const eps = 1e-6
	rsep2 := rsep * rsep
	self := flock[i].Pos
	force := Vec2{}
	seen := 0
	for j := 0; j < len(flock); j++ {
		if j == i {
			continue
		}
		d := self.Sub(flock[j].Pos)
		dist2 := d.Len2()
		if dist2 >= rsep2 || dist2 == 0 {
			continue
		}
		dist := math.Sqrt(dist2)
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
	rn2 := rn * rn
	self := flock[i]
	sumHeading := Vec2{}
	count := 0
	for j := 0; j < len(flock); j++ {
		if j == i {
			continue
		}
		if self.Pos.Sub(flock[j].Pos).Len2() <= rn2 {
			sumHeading = sumHeading.Add(flock[j].Vel)
			count++
			if maxN > 0 && count >= maxN {
				break
			}
		}
	}
	if count == 0 {
		return Vec2{}
	}
	avg := sumHeading.SafeNormalize()
	cur := self.Vel.SafeNormalize()
	return avg.Sub(cur)
}

func CohesionForce(i int, flock []Fish, rn float64, maxN int) Vec2 {
	rn2 := rn * rn
	self := flock[i]
	sumPos := Vec2{}
	count := 0
	for j := 0; j < len(flock); j++ {
		if j == i {
			continue
		}
		if self.Pos.Sub(flock[j].Pos).Len2() <= rn2 {
			sumPos = sumPos.Add(flock[j].Pos)
			count++
			if maxN > 0 && count >= maxN {
				break
			}
		}
	}
	if count == 0 {
		return Vec2{}
	}
	centroid := sumPos.Mul(1.0 / float64(count))
	desired := centroid.Sub(self.Pos).SafeNormalize()
	cur := self.Vel.SafeNormalize()
	return desired.Sub(cur)
}

func CohesionForceBiased(i int, flock []Fish, rn float64, maxN int, sharkPos Vec2, kBias float64) Vec2 {
	base := CohesionForce(i, flock, rn, maxN)
	if (base.X == 0 && base.Y == 0) || kBias == 0 {
		return base
	}
	away := flock[i].Pos.Sub(sharkPos).SafeNormalize()
	desired := base.Add(away.Mul(kBias)).SafeNormalize()
	return desired.Sub(flock[i].Vel.SafeNormalize())
}

func TangentialSwirlForce(i int, flock []Fish, rn float64, maxN int, clockwise bool) Vec2 {
	rn2 := rn * rn
	self := flock[i]
	sumPos := Vec2{}
	count := 0
	for j := 0; j < len(flock); j++ {
		if j == i {
			continue
		}
		if self.Pos.Sub(flock[j].Pos).Len2() <= rn2 {
			sumPos = sumPos.Add(flock[j].Pos)
			count++
			if maxN > 0 && count >= maxN {
				break
			}
		}
	}
	if count == 0 {
		return Vec2{}
	}
	centroid := sumPos.Mul(1.0 / float64(count))
	cohDir := centroid.Sub(self.Pos).SafeNormalize()
	if cohDir.X == 0 && cohDir.Y == 0 {
		return Vec2{}
	}
	var tangent Vec2
	if clockwise {
		tangent = cohDir.PerpCW()
	} else {
		tangent = cohDir.PerpCCW()
	}
	tangent = tangent.SafeNormalize()
	return tangent.Sub(self.Vel.SafeNormalize())
}

func TangentialSwirlForceWithThreat(i int, flock []Fish, rn float64, maxN int, clockwise bool, sharkPos Vec2, rThreat, minFactor float64) Vec2 {
	base := TangentialSwirlForce(i, flock, rn, maxN, clockwise)
	if (base.X == 0 && base.Y == 0) || rThreat <= 0 {
		return base
	}
	d2 := flock[i].Pos.Sub(sharkPos).Len2()
	r2 := rThreat * rThreat
	var t float64
	if d2 >= r2 {
		t = 1.0
	} else {
		t = math.Sqrt(d2) / rThreat
	}
	scale := minFactor + (1.0-minFactor)*clamp01(t)
	return base.Mul(scale)
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

// helpers
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

// ===== main loop =====
func main() {
	rand.Seed(time.Now().UnixNano())

	// fish
	fish := make([]Fish, NumFish)
	for i := range fish {
		fish[i].Pos = Vec2{rand.Float64() * W, rand.Float64() * H}
		fish[i].Vel = Vec2{rand.Float64()*2 - 1, rand.Float64()*2 - 1}.Normalize().Mul(140.0)
		for j := 0; j < TailLength; j++ {
			fish[i].Tail[j] = fish[i].Pos
		}
	}
	// shark
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
		// kappa=0.7 (mean reversion), sigma=70 px/sqrt(s), beta=0.6 (to scene center), gamma=0.18 (to school)
		UpdateSharkOU(&shark, dt, 0.7, 70.0, 0.6, 0.18, vSharkMax, Vec2{W * 0.5, H * 0.5}, center)
		shark.Pos = Wrap(shark.Pos, W, H)

		for i := range fish {
			sep := SeparationForce(i, fish, Rsep, MaxNeighbors).Mul(wSep)
			aln := AlignmentForce(i, fish, Rn, MaxNeighbors).Mul(wAlign)
			coh := CohesionForceBiased(i, fish, Rn, MaxNeighbors, shark.Pos, 0.15).Mul(wCoh)
			tan := TangentialSwirlForceWithThreat(i, fish, Rn, MaxNeighbors, SwirlClockwise, shark.Pos, Rthreat, 0.6).Mul(wTan)
			flee := FleeForce(i, fish, shark.Pos, Rthreat, fleeMax, fleeExp)

			acc := sep.Add(aln).Add(coh).Add(tan).Add(flee).Sub(fish[i].Vel.Mul(drag))
			fish[i].Vel = fish[i].Vel.Add(acc.Mul(dt)).Limit(vFishMax)
			fish[i].Pos = Wrap(fish[i].Pos.Add(fish[i].Vel.Mul(dt)), W, H)

			// tail ring push
			fish[i].TailHead = (fish[i].TailHead + 1) % TailLength
			fish[i].Tail[fish[i].TailHead] = fish[i].Pos
		}

		// --- RENDER (batched XY) ---
		SetControlXY(PanelID, SharkCID, int(shark.Pos.X), int(shark.Pos.Y))

		for i := 0; i < NumFish; i++ {
			SetControlXY(PanelID, BaseFishCID+i, int(fish[i].Pos.X), int(fish[i].Pos.Y))

			// update tail markers every other frame to cut work in half
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
