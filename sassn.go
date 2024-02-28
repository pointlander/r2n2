// Copyright 2024 The R2N2 Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"math"
	"math/rand"
	"os"
	"path"
	"strconv"
	"strings"
	"time"

	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
	"gonum.org/v1/plot/vg/draw"

	"github.com/pointlander/datum/bible"
	"github.com/pointlander/gradient/tf64"
	"github.com/pointlander/matrix"
)

// GraphSASSN graphs the 64 bit self attention ssn weight files
func GraphSASSN(directory string) {
	input, err := os.Open(directory)
	if err != nil {
		panic(err)
	}
	defer input.Close()
	type Pair struct {
		Epoch int
		Cost  float64
	}
	pairs := make(map[int]*Pair)
	names, err := input.Readdirnames(-1)
	if err != nil {
		panic(err)
	}
	for _, name := range names {
		if strings.HasSuffix(name, ".w") {
			set := tf64.NewSet()
			cost, epoch, err := set.Open(path.Join(directory, name))
			if err != nil {
				panic(err)
			}
			fmt.Println(epoch, cost)
			pair := Pair{
				Epoch: epoch,
				Cost:  cost,
			}
			pairs[epoch] = &pair
		}
	}

	points := make(plotter.XYs, 0, len(pairs))
	for _, pair := range pairs {
		points = append(points, plotter.XY{X: float64(pair.Epoch), Y: pair.Cost})
	}
	p := plot.New()

	p.Title.Text = "epochs vs cost"
	p.X.Label.Text = "epochs"
	p.Y.Label.Text = "cost"

	scatter, err := plotter.NewScatter(points)
	if err != nil {
		panic(err)
	}
	scatter.GlyphStyle.Radius = vg.Length(1)
	scatter.GlyphStyle.Shape = draw.CircleGlyph{}
	p.Add(scatter)

	err = p.Save(8*vg.Inch, 8*vg.Inch, "epochs.png")
	if err != nil {
		panic(err)
	}
}

// LearnSASSN learns 64bit self attention ssn r2n2 model
func LearnSASSN(name string) {
	seed, err := strconv.Atoi(name)
	if err != nil {
		panic(err)
	}
	rng := rand.New(rand.NewSource(int64(seed) + 1))
	bible, err := bible.Load()
	if err != nil {
		panic(err)
	}
	verses := bible.GetVerses()

	markov := Markov(verses)

	input := tf64.NewV(Symbols, 1)
	input.X = input.X[:cap(input.X)]
	output := tf64.NewV(Symbols, 1)
	output.X = output.X[:cap(output.X)]
	feedback := tf64.NewV(Space, 1)
	feedback.X = feedback.X[:cap(feedback.X)]
	feedbackcp := tf64.NewV(Space, 1)
	feedbackcp.X = feedbackcp.X[:cap(feedbackcp.X)]
	T := tf64.NewV(Symbols, 256)
	T.X = T.X[:cap(T.X)]
	set := tf64.NewSet()
	set.Add("w1", Symbols, Symbols)
	set.Add("b1", Symbols)
	set.Add("w1a", Symbols, Symbols)
	set.Add("b1a", Symbols)
	set.Add("w2", Space, Space)
	set.Add("w3", Space, Space)
	set.Add("b3", Space)
	set.Add("w3a", Space, Symbols)
	set.Add("b3a", Symbols)
	for i := range set.Weights {
		w := set.Weights[i]
		size := w.S[0] * w.S[1]
		if strings.HasPrefix(w.N, "b") {
			w.X = w.X[:size]
			w.States = make([][]float64, StateTotal)
			for i := range w.States {
				w.States[i] = make([]float64, len(w.X))
			}
			continue
		}
		factor := math.Sqrt(2.0 / float64(w.S[0]))
		for i := 0; i < size; i++ {
			w.X = append(w.X, rng.NormFloat64()*factor)
		}
		w.States = make([][]float64, StateTotal)
		for i := range w.States {
			w.States[i] = make([]float64, len(w.X))
		}
	}
	dropout := map[string]interface{}{
		"rng": rng,
	}

	A := tf64.Mul(T.Meta(), T.Meta())
	l1 := tf64.Dropout(tf64.Sigmoid(tf64.Add(tf64.Mul(set.Get("w1"), input.Meta()), set.Get("b1"))), dropout)
	l1a := tf64.Add(tf64.Mul(set.Get("w1a"), l1), set.Get("b1a"))
	l2 := tf64.Copy(feedbackcp.Meta(),
		tf64.Sigmoid(tf64.Add(tf64.Mul(set.Get("w2"), feedback.Meta()), l1a)))
	l3 := tf64.Dropout(tf64.Sigmoid(tf64.Add(tf64.Mul(set.Get("w3"), l2), set.Get("b3"))), dropout)
	l3a := tf64.CrossEntropy(tf64.Softmax(tf64.Add(tf64.Mul(set.Get("w3a"), l3), set.Get("b3a"))), output.Meta())

	iterations := 100
	points := make(plotter.XYs, 0, iterations)
	start := time.Now()
	for i := 0; i < iterations; i++ {
		pow := func(x float64) float64 {
			y := math.Pow(x, float64(i+1))
			if math.IsNaN(y) || math.IsInf(y, 0) {
				return 0
			}
			return y
		}

		for i := range verses {
			j := i + rng.Intn(len(verses)-i)
			verses[i], verses[j] = verses[j], verses[i]
		}

		total := 0.0
		for i := 0; i < len(verses); i++ {
			verse := "^" + verses[i].Verse + "$"
			for i := range feedback.X {
				feedback.X[i] = 0
			}
			feedback.Zero()
			for i := range feedbackcp.X {
				feedbackcp.X[i] = 0
			}
			feedbackcp.Zero()
			set.Zero()
			cost := 0.0
			last := 0
			buffer := [256][Symbols]float32{}
			entry := 0
			for l, symbol := range verse[:len(verses[i].Verse)-1] {
				input.Zero()
				for i := range input.X {
					input.X[i] = float64(markov[symbol][last][i])
				}
				copy(buffer[entry][:], markov[symbol][last][:])
				entry = (entry + 1) % len(buffer)
				for e := 0; e < len(buffer); e++ {
					for f := 0; f < Symbols; f++ {
						T.X[e*Symbols+f] = float64(buffer[(e+entry)%len(buffer)][f])
					}
				}
				A(func(a *tf64.V) bool {
					copy(set.ByName["w2"].X, a.X)
					return true
				})
				last = int(symbol)
				for i := range output.X {
					output.X[i] = 0
				}
				output.Zero()
				output.X[int(verse[l+1])] = 1
				cost += tf64.Gradient(l3a).X[0]

				copy(feedback.X, feedbackcp.X)
			}
			norm := 0.0
			for _, p := range set.Weights {
				for _, d := range p.D {
					norm += d * d
				}
			}
			norm = math.Sqrt(norm)
			b1, b2 := pow(B1), pow(B2)
			if norm > 1 {
				scaling := 1 / norm
				for _, w := range set.Weights {
					if w.N == "w2" {
						continue
					}
					for l, d := range w.D {
						g := d * scaling
						m := B1*w.States[StateM][l] + (1-B1)*g
						v := B2*w.States[StateV][l] + (1-B2)*g*g
						w.States[StateM][l] = m
						w.States[StateV][l] = v
						mhat := m / (1 - b1)
						vhat := v / (1 - b2)
						if vhat < 0 {
							vhat = 0
						}
						w.X[l] -= Eta * mhat / (math.Sqrt(vhat) + 1e-8)
					}
				}
			} else {
				for _, w := range set.Weights {
					if w.N == "w2" {
						continue
					}
					for l, d := range w.D {
						g := d
						m := B1*w.States[StateM][l] + (1-B1)*g
						v := B2*w.States[StateV][l] + (1-B2)*g*g
						w.States[StateM][l] = m
						w.States[StateV][l] = v
						mhat := m / (1 - b1)
						vhat := v / (1 - b2)
						if vhat < 0 {
							vhat = 0
						}
						w.X[l] -= Eta * mhat / (math.Sqrt(vhat) + 1e-8)
					}
				}
			}
			cost /= float64(len(verses[i].Verse))
			total += cost
			fmt.Println(cost)
		}
		fmt.Printf("\n")

		err := set.Save(fmt.Sprintf("weights_%d_%d.w", seed, i), total, i)
		if err != nil {
			panic(err)
		}

		fmt.Println(i, total, time.Now().Sub(start))
		start = time.Now()
		points = append(points, plotter.XY{X: float64(i), Y: float64(total)})
		if total < .001 {
			fmt.Println("stopping...")
			break
		}
	}

	p := plot.New()

	p.Title.Text = "epochs vs cost"
	p.X.Label.Text = "epochs"
	p.Y.Label.Text = "cost"

	scatter, err := plotter.NewScatter(points)
	if err != nil {
		panic(err)
	}
	scatter.GlyphStyle.Radius = vg.Length(1)
	scatter.GlyphStyle.Shape = draw.CircleGlyph{}
	p.Add(scatter)

	err = p.Save(8*vg.Inch, 8*vg.Inch, "epochs.png")
	if err != nil {
		panic(err)
	}
}

// InferenceSASSN inference 64 bit self attention ssn r2n2 model
func InferenceSASSN() {
	rng := rand.New(rand.NewSource(1))

	bible, err := bible.Load()
	if err != nil {
		panic(err)
	}
	verses := bible.GetVerses()
	markov := Markov(verses)

	in := []rune{'^'}
	type Net struct {
		name     string
		set      tf64.Set
		A        tf64.Meta
		l2       tf64.Meta
		l3a      tf64.Meta
		input    tf64.V
		feedback tf64.V
		T        tf64.V
		last     int
		buffer   [256][Symbols]float32
		entry    int
	}
	names := strings.Split(*FlagInference, ",")
	nets := make([]Net, len(names))
	for i := range nets {
		set := tf64.NewSet()
		name := names[i]
		cost, epoch, err := set.Open(name)
		if err != nil {
			panic(err)
		}
		fmt.Println(name, cost, epoch)
		input := tf64.NewV(Space, 1)
		input.X = input.X[:cap(input.X)]
		feedback := tf64.NewV(Space, 1)
		feedback.X = feedback.X[:cap(feedback.X)]
		T := tf64.NewV(Symbols, 256)
		T.X = T.X[:cap(T.X)]

		t := .5
		temp := tf64.NewV(Symbols, 1)
		for i := 0; i < Symbols; i++ {
			temp.X = append(temp.X, 1/t)
		}

		A := tf64.Mul(T.Meta(), T.Meta())
		l1 := tf64.Sigmoid(tf64.Add(tf64.Mul(set.Get("w1"), input.Meta()), set.Get("b1")))
		l1a := tf64.Add(tf64.Mul(set.Get("w1a"), l1), set.Get("b1a"))
		l2 := tf64.Sigmoid(tf64.Add(tf64.Mul(set.Get("w2"), feedback.Meta()), l1a))
		l3 := tf64.Sigmoid(tf64.Add(tf64.Mul(set.Get("w3"), l2), set.Get("b3")))
		l3a := tf64.Softmax(tf64.Hadamard(tf64.Add(tf64.Mul(set.Get("w3a"), l3), set.Get("b3a")), temp.Meta()))
		for i := range in[:len(in)-1] {
			current := in[i]
			distribution := markov[current][nets[i].last]
			for j := range input.X {
				input.X[j] = float64(distribution[j])
			}
			l2(func(a *tf64.V) bool {
				copy(feedback.X, a.X)
				return true
			})
			nets[i].last = int(current)
		}

		nets[i].name = name
		nets[i].set = set
		nets[i].A = A
		nets[i].l2 = l2
		nets[i].l3a = l3a
		nets[i].input = input
		nets[i].feedback = feedback
		nets[i].T = T
	}
	for i := 0; i < 4*128; i++ {
		symbols := make([][]float64, len(nets))
		selected, sum := rng.Float64(), 0.0
		for n := range nets {
			current := in[len(in)-1]
			distribution := markov[current][nets[n].last]
			for j := range nets[n].input.X {
				nets[n].input.X[j] = float64(distribution[j])
			}
			copy(nets[n].buffer[nets[n].entry][:], distribution[:])
			nets[n].entry = (nets[n].entry + 1) % len(nets[n].buffer)
			for e := 0; e < len(nets[n].buffer); e++ {
				for f := 0; f < Symbols; f++ {
					nets[n].T.X[e*Symbols+f] = float64(nets[n].buffer[(e+nets[n].entry)%len(nets[n].buffer)][f])
				}
			}
			nets[n].A(func(a *tf64.V) bool {
				copy(nets[n].set.ByName["w2"].X, a.X)
				return true
			})
			nets[n].l3a(func(a *tf64.V) bool {
				symbols[n] = a.X
				return true
			})
			nets[n].l2(func(a *tf64.V) bool {
				copy(nets[n].feedback.X, a.X)
				return true
			})
			nets[n].last = int(current)
		}
		router := matrix.NewMatrix(Symbols, len(nets))
		for _, symbol := range symbols {
			for _, s := range symbol {
				router.Data = append(router.Data, float32(s))
			}
		}
		entropy := matrix.SelfEntropy(router, router, router)
		index, min := 0, float32(math.MaxFloat32)
		for i, e := range entropy {
			if e < min {
				index, min = i, e
			}
		}
		for j := 0; j < Symbols; j++ {
			/*for _, s := range symbols {
				sum += s[j] / float64(len(symbols))
			}*/
			sum += symbols[index][j]
			if sum > selected {
				in = append(in, rune(j))
				break
			}
		}
	}
	fmt.Println(in)
	fmt.Println(string(in))
}
