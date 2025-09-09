package main

import (
	"math"
	"math/rand"
	"strconv"
	"syscall/js"
	"time"
)

// tiny vectors

type Vec2 struct{ X, Y float64 }

func (a Vec2) Add(b Vec2) Vec2    { return Vec2{a.X + b.X, a.Y + b.Y} }
func (a Vec2) Sub(b Vec2) Vec2    { return Vec2{a.X - b.X, a.Y - b.Y} }
func (a Vec2) Mul(s float64) Vec2 { return Vec2{a.X * s, a.Y * s} }
func (a Vec2) Len2() float64      { return a.X*a.X + a.Y*a.Y }
func (a Vec2) Len() float64 {
	l2 := a.Len2()
	if l2 == 0 {
		return 0
	}
	return math.Sqrt(l2)
}
func (a Vec2) Norm() Vec2 {
	l2 := a.Len2()
	if l2 == 0 {
		return Vec2{}
	}
	inv := 1 / math.Sqrt(l2)
	return Vec2{a.X * inv, a.Y * inv}
}
func (a Vec2) Dot(b Vec2) float64 { return a.X*b.X + a.Y*b.Y }

// sim types

type Boid struct {
	Pos, Vel Vec2
	Tail     [TailLen]Vec2
	Head     int
	Dead     bool // stays in slice; sim/render skip when true
}
type Shark struct{ Pos, Vel Vec2 }

// constants

const (
	// world / timing
	W, H = 480.0, 320.0
	dt   = 1.0 / 60.0

	// population & tails
	N       = 48
	TailLen = 6

	// off-canvas sentinel (always positive)
	Offscreen = 1_000_000

	// perception
	RsepBase = 20.0
	Rnei     = 110.0
	FOVcos   = -0.5 // cos(120°)

	// boids rule weights
	WsepBase = 1.60
	WaliBase = 1.75
	WcohBase = 0.45

	// damping & tiny noise
	Drag   = 0.045
	Jitter = 0.7

	// speed & steering (smooth)
	Vmax      = 118.0
	Vmin      = 46.0
	TargetSpd = 90.0
	MaxForce  = 140.0
	MaxTurn   = 3.0

	// predictive wall avoidance (corner-safe)
	WallTau      = 0.50
	WallMargin   = 38.0
	WallFriction = 0.65
	WallBlendPow = 2.0

	// gentle “home” field (prefer middle)
	HomePushMax = 100.0
	HomeExp     = 1.9
	CenterPull  = 0.08

	// shark (PD pursuit ellipse, re-centers toward flock)
	SharkPeriod         = 18.0
	SharkRXBase         = 0.28 // * W via slider
	SharkRYBase         = 0.20 // * H via slider (keeps aspect)
	SharkTargetV        = 150.0
	SharkMaxTurn        = 1.8
	SharkMaxForce       = 110.0
	SharkKp             = 1.8
	SharkKd             = 0.9
	SharkCenterBiasBase = 0.65
	SharkCenterTauSec   = 1.2
	SharkSeekGroupBase  = 0.35

	// flee
	ThreatR  = 80.0
	FleeGain = 120.0
	FleeEase = 2.2

	// evil mode (intercept + eat)
	EatRadius      = 10.0
	EatCooldownSec = 0.35
	LeadFactor     = 0.85
	SweepBias      = 0.35
	ConeCos        = 0.5

	// UI IDs  assets lazily but in the site
	PanelID     = 0
	BaseBoidCID = 1000
	BaseTailCID = 5000
	SpritePath  = "fish.png"
	TailPath    = "tail.png"
	SharkCID    = 10
	SharkSprite = "shark.png"
)

// helpers

func clampPos(p Vec2) Vec2 {
	if p.X < 1 {
		p.X = 1
	}
	if p.Y < 1 {
		p.Y = 1
	}
	if p.X > W-1 {
		p.X = W - 1
	}
	if p.Y > H-1 {
		p.Y = H - 1
	}
	return p
}
func ii(x float64) int { return int(math.Round(x)) }

func rotate(v Vec2, ang float64) Vec2 {
	c, s := math.Cos(ang), math.Sin(ang)
	return Vec2{v.X*c - v.Y*s, v.X*s + v.Y*c}
}

// slider & checkbox readers

func getSlider(id string, def float64) float64 {
	el := js.Global().Get("document").Call("getElementById", id)
	if !el.Truthy() {
		return def
	}
	f, err := strconv.ParseFloat(el.Get("value").String(), 64)
	if err != nil {
		return def
	}
	return f
}
func getCheckbox(id string) bool {
	el := js.Global().Get("document").Call("getElementById", id)
	if !el.Truthy() {
		return false
	}
	return el.Get("checked").Bool()
}

// falloffs & FOV

func quadKernel(d, R float64) float64 {
	if d <= 0 {
		return 1
	}
	if d >= R {
		return 0
	}
	x := d / R
	y := 1 - x*x
	return y * y
}
func inFOV(selfPos, selfVel, otherPos Vec2) bool {
	dir := selfVel.Norm()
	if dir.Len2() == 0 {
		return true
	}
	toN := otherPos.Sub(selfPos).Norm()
	return dir.Dot(toN) >= FOVcos
}

// core boids

func ruleSeparationR(i int, b []Boid, Rsep float64) Vec2 {
	if b[i].Dead {
		return Vec2{}
	}
	self := b[i].Pos
	out := Vec2{}
	for j := range b {
		if j == i || b[j].Dead {
			continue
		}
		dv := self.Sub(b[j].Pos)
		d := dv.Len()
		if d == 0 || d > Rsep {
			continue
		}
		w := (1 - d/Rsep)
		out = out.Add(dv.Norm().Mul(w * w * (Rsep / (d + 1e-6))))
	}
	return out
}
func ruleAlignment(i int, b []Boid) Vec2 {
	if b[i].Dead {
		return Vec2{}
	}
	self := b[i]
	sum := Vec2{}
	ws := 0.0
	for j := range b {
		if j == i || b[j].Dead {
			continue
		}
		if !inFOV(self.Pos, self.Vel, b[j].Pos) {
			continue
		}
		d := self.Pos.Sub(b[j].Pos).Len()
		if d > Rnei {
			continue
		}
		w := quadKernel(d, Rnei)
		sum = sum.Add(b[j].Vel.Mul(w))
		ws += w
	}
	if ws == 0 {
		return Vec2{}
	}
	return sum.Mul(1 / ws).Norm()
}
func ruleCohesion(i int, b []Boid) Vec2 {
	if b[i].Dead {
		return Vec2{}
	}
	self := b[i]
	sum := Vec2{}
	ws := 0.0
	for j := range b {
		if j == i || b[j].Dead {
			continue
		}
		if !inFOV(self.Pos, self.Vel, b[j].Pos) {
			continue
		}
		d := self.Pos.Sub(b[j].Pos).Len()
		if d > Rnei {
			continue
		}
		w := quadKernel(d, Rnei)
		sum = sum.Add(b[j].Pos.Mul(w))
		ws += w
	}
	if ws == 0 {
		return Vec2{}
	}
	center := sum.Mul(1 / ws)
	return center.Sub(self.Pos).Norm()
}

// home fiels and walls

func homeVel(pos Vec2) Vec2 {
	c := Vec2{W * 0.5, H * 0.5}
	d := c.Sub(pos)
	r := d.Len()
	if r == 0 {
		return Vec2{}
	}
	rMax := 0.5 * math.Min(W, H)
	if rMax < 1 {
		rMax = 1
	}
	t := r / rMax
	if t < 0 {
		t = 0
	}
	if t > 1 {
		t = 1
	}
	k := HomePushMax * math.Pow(t, HomeExp)
	return d.Mul(k / r)
}

func blendedWallNormal(p Vec2) Vec2 {
	dL := p.X
	dR := W - p.X
	dT := p.Y
	dB := H - p.Y
	wL := 1.0 / math.Pow(dL+1.0, WallBlendPow)
	wR := 1.0 / math.Pow(dR+1.0, WallBlendPow)
	wT := 1.0 / math.Pow(dT+1.0, WallBlendPow)
	wB := 1.0 / math.Pow(dB+1.0, WallBlendPow)
	n := Vec2{1, 0}.Mul(wL).Add(Vec2{-1, 0}.Mul(wR)).Add(Vec2{0, 1}.Mul(wT)).Add(Vec2{0, -1}.Mul(wB))
	return n.Norm()
}

func confineDelta(pos, vel Vec2) Vec2 {
	look := pos.Add(vel.Mul(WallTau))
	dx := math.Min(look.X, W-look.X)
	dy := math.Min(look.Y, H-look.Y)
	d := math.Min(dx, dy)
	if d >= WallMargin {
		return Vec2{}
	}
	n := blendedWallNormal(look)
	if n.Len2() == 0 {
		return Vec2{}
	}
	vn := vel.Dot(n) // inward +, outward -
	if vn >= 0 {
		return Vec2{}
	}
	s := 1 - d/WallMargin
	if s < 0 {
		s = 0
	}
	if s > 1 {
		s = 1
	}
	outPart := n.Mul(vn)
	vt := vel.Sub(outPart)
	alongDamp := vt.Mul(WallFriction * s)
	target := vel.Sub(outPart.Mul(1 + 0.7*s)).Sub(alongDamp)
	return target.Sub(vel)
}

//steering (turn-rate limited)

func steerTowards(curVel, desired Vec2, maxTurn, maxForce, targetSpd float64) Vec2 {
	if desired.Len2() == 0 {
		return curVel
	}
	wantDir := desired.Norm()
	curSpd := curVel.Len()
	if curSpd == 0 {
		curSpd = targetSpd
	}
	wantVel := wantDir.Mul(targetSpd)

	curDir := curVel.Norm()
	dot := curDir.Dot(wantDir)
	if dot > 1 {
		dot = 1
	}
	if dot < -1 {
		dot = -1
	}
	ang := math.Acos(dot)
	maxStep := maxTurn * dt
	sign := curDir.X*wantDir.Y - curDir.Y*wantDir.X
	if sign < 0 {
		ang = -ang
	}
	if ang > maxStep {
		ang = maxStep
	} else if ang < -maxStep {
		ang = -maxStep
	}

	newDir := rotate(curDir, ang)
	newVel := newDir.Mul(curSpd)

	steer := wantVel.Sub(newVel)
	maxKick := maxForce * dt
	if s := steer.Len(); s > maxKick {
		steer = steer.Mul(maxKick / s)
	}
	return newVel.Add(steer)
}

// shark helpers, flee, evil mode

func sharkTargetPosAt(t float64, C Vec2, rx, ry float64) Vec2 {
	ω := 2 * math.Pi / SharkPeriod
	return Vec2{C.X + rx*math.Cos(ω*t), C.Y + ry*math.Sin(ω*t)}
}
func sharkTargetVel(t float64, rx, ry float64) Vec2 {
	ω := 2 * math.Pi / SharkPeriod
	return Vec2{-rx * ω * math.Sin(ω*t), ry * ω * math.Cos(ω*t)}
}
func fleeVel(pos, shark Vec2) Vec2 {
	dv := pos.Sub(shark)
	d := dv.Len()
	if d == 0 || d >= ThreatR {
		return Vec2{}
	}
	t := 1 - d/ThreatR
	g := FleeGain * math.Pow(t, FleeEase)
	return dv.Mul(g / (d + 1e-6))
}
func pickTarget(shark Shark, boids []Boid) (int, bool) {
	best := -1
	bestScore := -1e9
	dir := shark.Vel.Norm()
	for i := range boids {
		if boids[i].Dead {
			continue
		}
		rel := boids[i].Pos.Sub(shark.Pos)
		d := rel.Len()
		if d == 0 {
			continue
		}
		toN := rel.Mul(1 / d)
		if dir.Dot(toN) < ConeCos {
			continue
		}
		s := 0.7*(1.0/d) + 0.3*dir.Dot(toN)
		if s > bestScore {
			bestScore, best = s, i
		}
	}
	return best, best >= 0
}
func evilPursuit(shark Shark, target Boid, targetSpeed float64) Vec2 {
	rel := target.Pos.Sub(shark.Pos)
	d := rel.Len()
	if d == 0 {
		return Vec2{}
	}
	tEst := d / (targetSpeed + 1e-6)
	aim := target.Pos.Add(target.Vel.Mul(tEst * LeadFactor))
	c := Vec2{W * 0.5, H * 0.5}
	toC := c.Sub(aim)
	r := toC.Len()
	sweep := Vec2{}
	if r > 0 {
		n := toC.Mul(1 / r)
		sweep = Vec2{-n.Y, n.X}.Mul(SweepBias * targetSpeed)
	}
	return aim.Add(sweep).Sub(shark.Pos)
}

// sprite helpers

func clearAllSprites() {
	offX, offY := Offscreen, Offscreen
	for i := 0; i < N; i++ {
		SetControlXY(PanelID, BaseBoidCID+i, offX, offY)
		for k := 0; k < TailLen; k++ {
			SetControlXY(PanelID, BaseTailCID+i*TailLen+k, offX, offY)
		}
	}
}

func chompMark(shark *Shark, boids []Boid) int {
	for i := range boids {
		if boids[i].Dead {
			continue
		}
		if shark.Pos.Sub(boids[i].Pos).Len() <= EatRadius {
			boids[i].Dead = true
			boids[i].Pos = Vec2{float64(Offscreen), float64(Offscreen)} // exile state
			boids[i].Vel = Vec2{}
			return 1
		}
	}
	return 0
}

// main

func main() {
	rand.Seed(time.Now().UnixNano())
	// can i just delete it?

	center := Vec2{W * 0.5, H * 0.5}
	boids := make([]Boid, N)
	for i := range boids {
		p := center.Add(Vec2{rand.NormFloat64() * 22, rand.NormFloat64() * 16})
		ang := rand.Float64() * 2 * math.Pi
		v := Vec2{math.Cos(ang), math.Sin(ang)}.Mul(80 + 20*rand.Float64())
		boids[i] = Boid{Pos: p, Vel: v, Dead: false}
		for k := 0; k < TailLen; k++ {
			boids[i].Tail[k] = p
		}
	}

	// shark + patrol center
	t := 0.0
	patrolC := center
	shark := Shark{Pos: sharkTargetPosAt(t, patrolC, SharkRXBase*W, SharkRYBase*H)}

	// UI bootstrap
	AddPanel(PanelID, 1, 0, 0, 0, 0, 0, 0, 0)
	ShowPanel(PanelID)
	AddControlPictureFromFile(PanelID, SharkCID, ii(shark.Pos.X), ii(shark.Pos.Y), SharkSprite, 1)
	for i := 0; i < N; i++ {
		AddControlPictureFromFile(PanelID, BaseBoidCID+i, ii(boids[i].Pos.X), ii(boids[i].Pos.Y), SpritePath, 1)
		for k := 0; k < TailLen; k++ {
			AddControlPictureFromFile(PanelID, BaseTailCID+i*TailLen+k, ii(boids[i].Tail[k].X), ii(boids[i].Tail[k].Y), TailPath, 1)
		}
	}

	frame := 0
	score := 0
	eatCD := 0.0

	for {
		t += dt

		// live sliders
		WsepLive := getSlider("sepW", WsepBase)
		WaliLive := getSlider("aliW", WaliBase)
		WcohLive := getSlider("cohW", WcohBase)
		RsepLive := getSlider("rsep", RsepBase)

		SharkTargetVLive := getSlider("shv", SharkTargetV)
		patScale := getSlider("pat", SharkRXBase)
		SharkRXLive := patScale * W
		SharkRYLive := (SharkRYBase / SharkRXBase) * patScale * H
		SharkCenterBiasLive := getSlider("bias", SharkCenterBiasBase)
		SharkSeekGroupLive := getSlider("seek", SharkSeekGroupBase)
		tailDraw := int(getSlider("tail", float64(TailLen)))
		if tailDraw < 0 {
			tailDraw = 0
		}
		if tailDraw > TailLen {
			tailDraw = TailLen
		}

		evil := getCheckbox("evil")
		if eatCD > 0 {
			eatCD -= dt
			if eatCD < 0 {
				eatCD = 0
			}
		}

		// centroid + patrol center smoothing
		sum := Vec2{}
		alive := 0
		for i := range boids {
			if boids[i].Dead {
				continue
			}
			sum = sum.Add(boids[i].Pos)
			alive++
		}
		gc := center
		if alive > 0 {
			gc = sum.Mul(1.0 / float64(alive))
		}

		desiredC := center.Add(gc.Sub(center).Mul(SharkCenterBiasLive))
		alpha := 1 - math.Exp(-dt/SharkCenterTauSec)
		patrolC = patrolC.Add(desiredC.Sub(patrolC).Mul(alpha))

		// shark: PD pursuit (normal)
		var wantVel Vec2
		if !evil {
			ω := 2 * math.Pi / SharkPeriod
			stPos := Vec2{patrolC.X + SharkRXLive*math.Cos(ω*t), patrolC.Y + SharkRYLive*math.Sin(ω*t)}
			stVel := Vec2{-SharkRXLive * ω * math.Sin(ω*t), SharkRYLive * ω * math.Cos(ω*t)}
			seek := gc.Sub(shark.Pos).Mul(SharkSeekGroupLive)
			wantVel = stVel.Add(stPos.Sub(shark.Pos).Mul(SharkKp)).Sub(shark.Vel.Sub(stVel).Mul(SharkKd)).Add(seek)
		} else {
			if idx, ok := pickTarget(shark, boids); ok {
				wantVel = evilPursuit(shark, boids[idx], SharkTargetVLive)
			} else {
				ω := 2 * math.Pi / SharkPeriod
				stPos := Vec2{patrolC.X + SharkRXLive*math.Cos(ω*t), patrolC.Y + SharkRYLive*math.Sin(ω*t)}
				stVel := Vec2{-SharkRXLive * ω * math.Sin(ω*t), SharkRYLive * ω * math.Cos(ω*t)}
				wantVel = stVel.Add(stPos.Sub(shark.Pos).Mul(SharkKp)).Sub(shark.Vel.Sub(stVel).Mul(SharkKd))
			}
		}
		snext := steerTowards(shark.Vel, wantVel, SharkMaxTurn, SharkMaxForce, SharkTargetVLive)
		shark.Vel = snext
		shark.Pos = clampPos(shark.Pos.Add(shark.Vel.Mul(dt)))

		// evil eating with cooldown — mark Dead (no slice reindex)
		if evil && eatCD == 0 {
			if n := chompMark(&shark, boids); n > 0 {
				score += n
				eatCD = EatCooldownSec
				js.Global().Call("setScore", score)
			}
		}

		// boids update
		for i := range boids {
			if boids[i].Dead {
				continue
			}

			sepDir := ruleSeparationR(i, boids, RsepLive).Mul(WsepLive)
			aliDir := ruleAlignment(i, boids).Mul(WaliLive)
			cohDir := ruleCohesion(i, boids).Mul(WcohLive)

			toC := center.Sub(boids[i].Pos).Norm().Mul(CenterPull)
			avoidDelta := confineDelta(boids[i].Pos, boids[i].Vel)
			homeDelta := homeVel(boids[i].Pos).Sub(boids[i].Vel).Mul(0.5)
			flee := fleeVel(boids[i].Pos, shark.Pos)
			fleeDelta := flee.Sub(boids[i].Vel).Mul(0.7)

			want := sepDir.Add(aliDir).Add(cohDir).Add(toC).Add(avoidDelta).Add(homeDelta).Add(fleeDelta)
			want = want.Add(Vec2{rand.NormFloat64(), rand.NormFloat64()}.Mul(Jitter))
			want = want.Sub(boids[i].Vel.Mul(Drag))

			next := steerTowards(boids[i].Vel, boids[i].Vel.Add(want), MaxTurn, MaxForce, TargetSpd)
			if s := next.Len(); s > Vmax {
				next = next.Mul(Vmax / s)
			} else if s < Vmin {
				if s == 0 {
					next = Vec2{1, 0}.Mul(Vmin)
				} else {
					next = next.Mul(Vmin / s)
				}
			}
			boids[i].Vel = next
			boids[i].Pos = clampPos(boids[i].Pos.Add(boids[i].Vel.Mul(dt)))

			boids[i].Head = (boids[i].Head + 1) % TailLen
			boids[i].Tail[boids[i].Head] = boids[i].Pos
		}

		//render: CLEAR EVERYTHING
		clearAllSprites()

		// shark
		SetControlXY(PanelID, SharkCID, ii(shark.Pos.X), ii(shark.Pos.Y))

		// alive boids
		for i := 0; i < len(boids); i++ {
			if boids[i].Dead {
				continue
			}
			// fish
			SetControlXY(PanelID, BaseBoidCID+i, ii(boids[i].Pos.X), ii(boids[i].Pos.Y))
			// tails: draw first tailDraw segments (others remain cleared)
			for k := 0; k < tailDraw; k++ {
				idx := (boids[i].Head - k + TailLen) % TailLen
				p := boids[i].Tail[idx]
				SetControlXY(PanelID, BaseTailCID+i*TailLen+k, ii(p.X), ii(p.Y))
			}
		}

		frame++
		WaitMS(16)
	}
}
