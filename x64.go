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

// Graph64 graphs the 64 bit weight files
func Graph64(directory string) {
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

// Learn2X64 learns 64bit 2X the r2n2 model
func Learn2X64(name string) {
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
	set := tf64.NewSet()
	set.Add("w1", Symbols, Symbols)
	set.Add("b1", Symbols)
	set.Add("w1a", Symbols, Symbols)
	set.Add("b1a", Symbols)
	set.Add("w2", Width, Space)
	set.Add("b2", Space)
	set.Add("w2d", Space, Width)
	set.Add("b2d", Width)
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
	{
		pow := func(x float64) float64 {
			y := math.Pow(x, 1.0)
			if math.IsNaN(y) || math.IsInf(y, 0) {
				return 0
			}
			return y
		}
		for i := range feedback.X {
			feedback.X[i] = 0
		}
		feedback.Zero()
		for i := 0; i < 256; i++ {
			set.Zero()
			inputs := make([]*tf64.V, 0, 8)
			input := tf64.NewV(Symbols, 1)
			input.X = input.X[:cap(input.X)]
			for i := range input.X {
				e := math.Exp(rng.NormFloat64())
				input.X[i] = e / (e + 1)
			}
			inputs = append(inputs, &input)
			l1 := tf64.Sigmoid(tf64.Add(tf64.Mul(set.Get("w2"),
				tf64.Concat(input.Meta(), feedback.Meta())), set.Get("b2")))
			length := rng.Intn(32) + 1
			for j := 0; j < length; j++ {
				input := tf64.NewV(Symbols, 1)
				input.X = input.X[:cap(input.X)]
				for i := range input.X {
					e := math.Exp(rng.NormFloat64())
					input.X[i] = e / (e + 1)
				}
				inputs = append(inputs, &input)
				l1 = tf64.Sigmoid(tf64.Add(tf64.Mul(set.Get("w2"),
					tf64.Concat(input.Meta(), l1)), set.Get("b2")))
			}
			x := 0
			y := Symbols
			z := Width
			options := map[string]interface{}{
				"begin": &x,
				"end":   &y,
			}
			options1 := map[string]interface{}{
				"begin": &y,
				"end":   &z,
			}
			l1d := tf64.Sigmoid(tf64.Add(tf64.Mul(set.Get("w2d"), l1), set.Get("b2d")))
			cost := tf64.Avg(tf64.Quadratic(tf64.Slice(l1d, options), inputs[0].Meta()))
			for j := 0; j < length; j++ {
				l1d = tf64.Sigmoid(tf64.Add(tf64.Mul(set.Get("w2d"), tf64.Slice(l1d, options1)), set.Get("b2d")))
				cost = tf64.Add(cost, tf64.Avg(tf64.Quadratic(tf64.Slice(l1d, options), inputs[j+1].Meta())))
			}
			total := tf64.Gradient(cost).X[0]
			norm := 0.0
			for _, p := range set.Weights {
				for _, d := range p.D {
					norm += d * d
				}
			}
			norm = math.Sqrt(norm)
			b1, b2 := pow(B1), pow(B2)
			const Eta = .01
			if norm > 1 {
				scaling := 1 / norm
				for _, w := range set.Weights {
					if w.N != "w2" && w.N != "b2" && w.N != "w2d" && w.N != "b2d" {
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
						w.X[l] -= Eta * mhat / (math.Sqrt(vhat) + 1e-8)
					}
				}
			} else {
				for _, w := range set.Weights {
					if w.N != "w2" && w.N != "b2" && w.N != "w2d" && w.N != "b2d" {
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
						w.X[l] -= Eta * mhat / (math.Sqrt(vhat) + 1e-8)
					}
				}
			}
			fmt.Println("pre", length, total/float64(length))
		}
	}

	l1 := tf64.Sigmoid(tf64.Add(tf64.Mul(set.Get("w1"), input.Meta()), set.Get("b1")))
	l1a := tf64.Add(tf64.Mul(set.Get("w1a"), l1), set.Get("b1a"))
	l2 := tf64.Copy(feedbackcp.Meta(),
		tf64.Sigmoid(tf64.Add(tf64.Mul(set.Get("w2"), tf64.Concat(l1a, feedback.Meta())), set.Get("b2"))))
	l3 := tf64.Sigmoid(tf64.Add(tf64.Mul(set.Get("w3"), l2), set.Get("b3")))
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
			for l, symbol := range verse[:len(verses[i].Verse)-1] {
				input.Zero()
				for i := range input.X {
					input.X[i] = float64(markov[symbol][last][i])
				}
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
					if w.N == "w2" || w.N == "b2" || w.N == "w2d" || w.N == "b2d" {
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
					if w.N == "w2" || w.N == "b2" || w.N == "w2d" || w.N == "b2d" {
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

// Inference2X64 inference 64 bit 2X r2n2 model
func Inference2X64() {
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
		l2       tf64.Meta
		l3a      tf64.Meta
		input    tf64.V
		feedback tf64.V
		last     int
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

		t := .5
		temp := tf64.NewV(Symbols, 1)
		for i := 0; i < Symbols; i++ {
			temp.X = append(temp.X, 1/t)
		}

		l1 := tf64.Sigmoid(tf64.Add(tf64.Mul(set.Get("w1"), input.Meta()), set.Get("b1")))
		l1a := tf64.Add(tf64.Mul(set.Get("w1a"), l1), set.Get("b1a"))
		l2 := tf64.Sigmoid(tf64.Add(tf64.Mul(set.Get("w2"), tf64.Concat(l1a, feedback.Meta())), set.Get("b2")))
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
		nets[i].l2 = l2
		nets[i].l3a = l3a
		nets[i].input = input
		nets[i].feedback = feedback
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

// Inference2X64SE inference 64 bit self entropy 2X r2n2 model
func Inference2X64SE() {
	in := []rune{'^'}
	set := tf64.NewSet()
	name := *FlagInference
	cost, epoch, err := set.Open(name)
	if err != nil {
		panic(err)
	}
	fmt.Println(name, cost, epoch)
	input := tf64.NewV(Space, 1)
	input.X = input.X[:cap(input.X)]
	feedback := tf64.NewV(Space, 1)
	feedback.X = feedback.X[:cap(feedback.X)]

	t := 1.0
	temp := tf64.NewV(Symbols, 1)
	for i := 0; i < Symbols; i++ {
		temp.X = append(temp.X, 1/t)
	}

	l1 := tf64.Sigmoid(tf64.Add(tf64.Mul(set.Get("w1"), input.Meta()), set.Get("b1")))
	l1a := tf64.Add(tf64.Mul(set.Get("w1a"), l1), set.Get("b1a"))
	l2 := tf64.Sigmoid(tf64.Add(tf64.Mul(set.Get("w2"), tf64.Concat(l1a, feedback.Meta())), set.Get("b2")))
	l3 := tf64.Sigmoid(tf64.Add(tf64.Mul(set.Get("w3"), l2), set.Get("b3")))
	l3a := tf64.Softmax(tf64.Hadamard(tf64.Add(tf64.Mul(set.Get("w3a"), l3), set.Get("b3a")), temp.Meta()))
	symbols := make([][]float64, 0, 8)
	for i := range in {
		for j := range input.X {
			input.X[j] = 0
		}
		input.X[int(in[i])] = 1
		l3a(func(a *tf64.V) bool {
			symbols = append(symbols, a.X)
			return true
		})
		l2(func(a *tf64.V) bool {
			copy(feedback.X, a.X)
			return true
		})
	}
	for i := 0; i < 256; i++ {
		context := matrix.NewMatrix(Symbols, len(symbols)+1)
		for _, symbol := range symbols {
			for _, s := range symbol {
				context.Data = append(context.Data, float32(s))
			}
		}
		for j := 0; j < 256; j++ {
			context.Data = append(context.Data, 0)
		}
		symbol, min, best := 0, math.MaxFloat64, []float64{}
		for s := 0; s < 256; s++ {
			for j := range input.X {
				input.X[j] = 0
			}
			input.X[int(s)] = 1
			l3a(func(a *tf64.V) bool {
				dst := context.Data[256*len(symbols):]
				for i := range dst {
					dst[i] = float32(a.X[i])
				}
				entropy := matrix.SelfEntropy(context, context, context)
				sum := 0.0
				for _, e := range entropy {
					sum += float64(e)
				}
				if sum < min {
					symbol, min, best = s, sum, a.X
				}
				return true
			})
		}
		for j := range input.X {
			input.X[j] = 0
		}
		input.X[int(symbol)] = 1
		l2(func(a *tf64.V) bool {
			copy(feedback.X, a.X)
			return true
		})
		in = append(in, rune(symbol))
		symbols = append(symbols, best)
		fmt.Println(string(in))
	}
}
