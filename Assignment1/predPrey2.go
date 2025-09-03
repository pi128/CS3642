package main

import (
	"fmt"
	"math"
	"math/rand"
	"time"
)

/* ==============================
   Types
================================*/

type Vec2 struct{ X, Y float64 }

type Fish struct {
	Pos Vec2
	Vel Vec2
}

type Shark struct {
	Pos Vec2
	Vel Vec2
}

/* ==============================
   Vector helpers
================================*/

func (a Vec2) Add(b Vec2) Vec2    { return Vec2{a.X + b.X, a.Y + b.Y} }
func (a Vec2) Sub(b Vec2) Vec2    { return Vec2{a.X - b.X, a.Y - b.Y} }
func (a Vec2) Mul(s float64) Vec2 { return Vec2{a.X * s, a.Y * s} }
func (a Vec2) Dot(b Vec2) float64 { return a.X*b.X + a.Y*b.Y }
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
	l2 := a.Len2()
	m2 := max * max
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

func clamp01(x float64) float64 {
	if x < 0 {
		return 0
	}
	if x > 1 {
		return 1
	}
	return x
}

/* ==============================
   Sim constants (QVGA-ish world)
================================*/

const (
	W  = 320.0
	H  = 240.0
	dt = 1.0 / 60.0

	NumFish = 80

	// Radii
	Rsep    = 12.0
	Rn      = 50.0
	Rthreat = 95.0

	// Weights
	wSep   = 1.3
	wAlign = 0.9
	wCoh   = 0.5
	wTan   = 0.35
	drag   = 0.03

	// Flee shaping
	fleeMax = 2.6
	fleeExp = 1.4

	// Speeds
	vFishMax  = 1.85
	vSharkMax = 1.60

	// Swirl options
	SwirlClockwise = true

	// Neighbor cap for perf/stability
	MaxNeighbors = 14
)

/* ==============================
   Forces
================================*/

// Separation: short-range repulsion
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
		strength := (rsep - dist) / (rsep + eps) // linear falloff
		force = force.Add(d.Normalize().Mul(strength))

		seen++
		if maxN > 0 && seen >= maxN {
			break
		}
	}
	return force
}

// Alignment: steer toward average neighbor heading
func AlignmentForce(i int, flock []Fish, rn float64, maxN int) Vec2 {
	rn2 := rn * rn
	self := flock[i]

	sumHeading := Vec2{}
	count := 0

	for j := 0; j < len(flock); j++ {
		if j == i {
			continue
		}
		d := self.Pos.Sub(flock[j].Pos)
		if d.Len2() <= rn2 {
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

	avgHeading := sumHeading.SafeNormalize()
	myHeading := self.Vel.SafeNormalize()
	return avgHeading.Sub(myHeading)
}

// Cohesion: steer toward local neighbor centroid
func CohesionForce(i int, flock []Fish, rn float64, maxN int) Vec2 {
	rn2 := rn * rn
	self := flock[i]

	sumPos := Vec2{}
	count := 0

	for j := 0; j < len(flock); j++ {
		if j == i {
			continue
		}
		d := self.Pos.Sub(flock[j].Pos)
		if d.Len2() <= rn2 {
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
	current := self.Vel.SafeNormalize()
	return desired.Sub(current)
}

// Cohesion biased a bit away from the shark
func CohesionForceBiased(i int, flock []Fish, rn float64, maxN int, sharkPos Vec2, kBias float64) Vec2 {
	base := CohesionForce(i, flock, rn, maxN)
	if (base.X == 0 && base.Y == 0) || kBias == 0 {
		return base
	}
	away := flock[i].Pos.Sub(sharkPos).SafeNormalize()
	desired := base.Add(away.Mul(kBias)).SafeNormalize()
	return desired.Sub(flock[i].Vel.SafeNormalize())
}

// Tangential swirl (milling): perpendicular to local cohesion direction
func TangentialSwirlForce(i int, flock []Fish, rn float64, maxN int, clockwise bool) Vec2 {
	rn2 := rn * rn
	self := flock[i]

	sumPos := Vec2{}
	count := 0
	for j := 0; j < len(flock); j++ {
		if j == i {
			continue
		}
		d := self.Pos.Sub(flock[j].Pos)
		if d.Len2() <= rn2 {
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

// Optional: taper swirl near shark, but never to zero
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

// Flee: radial push away from shark within Rthreat
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

/* ==============================
   Shark helpers
================================*/

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

// OU drift with tiny biases
func UpdateSharkOU(s *Shark, dt, kappa, sigma, beta, gamma, sMax float64, sceneCenter, schoolCenter Vec2) {
	reversion := s.Vel.Mul(-kappa * dt)
	noise := Vec2{rand.NormFloat64(), rand.NormFloat64()}.Mul(sigma * math.Sqrt(dt))
	centerPull := sceneCenter.Sub(s.Pos).Mul(beta * dt)
	schoolPull := schoolCenter.Sub(s.Pos).Mul(gamma * dt)

	s.Vel = s.Vel.Add(reversion).Add(noise).Add(centerPull).Add(schoolPull)
	s.Vel = s.Vel.Limit(sMax)
	s.Pos = s.Pos.Add(s.Vel.Mul(dt))
}

/* ==============================
   Main (minimal loop)
================================*/

func main() {
	rand.Seed(time.Now().UnixNano())

	// init fish
	fish := make([]Fish, NumFish)
	for i := range fish {
		fish[i].Pos = Vec2{rand.Float64() * W, rand.Float64() * H}
		// small random initial velocity
		fish[i].Vel = Vec2{rand.Float64()*2 - 1, rand.Float64()*2 - 1}.Normalize().Mul(0.8)
	}

	// init shark at center
	shark := Shark{Pos: Vec2{W * 0.5, H * 0.5}}

	// run a tiny headless loop (prints a heartbeat)
	frames := 600 // ~10s at 60 FPS
	for f := 0; f < frames; f++ {
		// 1) shark update
		center := SchoolCenter(fish)
		UpdateSharkOU(&shark, dt, 1.0, 0.6, 1.5e-4, 7.5e-5, vSharkMax, Vec2{W * 0.5, H * 0.5}, center)
		shark.Pos = Wrap(shark.Pos, W, H)

		// 2) fish updates
		for i := range fish {
			// forces
			sep := SeparationForce(i, fish, Rsep, MaxNeighbors).Mul(wSep)
			aln := AlignmentForce(i, fish, Rn, MaxNeighbors).Mul(wAlign)
			coh := CohesionForceBiased(i, fish, Rn, MaxNeighbors, shark.Pos, 0.15).Mul(wCoh)
			tan := TangentialSwirlForceWithThreat(i, fish, Rn, MaxNeighbors, SwirlClockwise, shark.Pos, Rthreat, 0.6).Mul(wTan)
			flee := FleeForce(i, fish, shark.Pos, Rthreat, fleeMax, fleeExp) // already has magnitude, no extra weight here

			// total acceleration (drag is velocity-proportional)
			acc := Vec2{}
			acc = acc.Add(sep).Add(aln).Add(coh).Add(tan).Add(flee)
			acc = acc.Sub(fish[i].Vel.Mul(drag))

			// integrate
			fish[i].Vel = fish[i].Vel.Add(acc.Mul(dt)).Limit(vFishMax)
			fish[i].Pos = Wrap(fish[i].Pos.Add(fish[i].Vel.Mul(dt)), W, H)
		}

		// 3) heartbeat print every second
		if f%60 == 0 {
			fmt.Printf("t=%.1fs shark(%.1f,%.1f) firstFish(%.1f,%.1f)\n",
				float64(f)*dt, shark.Pos.X, shark.Pos.Y, fish[0].Pos.X, fish[0].Pos.Y)
		}

		// pacing only for the console demo; remove when driving a real render loop
		time.Sleep(time.Second / 60)
	}

	fmt.Println("Done.")
}
