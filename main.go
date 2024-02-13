// Copyright 2024 The R2N2 Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"flag"
	"fmt"
	"math"
	"math/rand"
	"os"
	"path"
	"runtime"
	"strings"
	"time"

	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
	"gonum.org/v1/plot/vg/draw"

	"github.com/pointlander/datum/bible"
	"github.com/pointlander/gradient/tf32"
	"github.com/pointlander/gradient/tf64"
)

const (
	// NumberOfVerses is the number of verses in the bible
	NumberOfVerses = 31102
	// Symbols is the number of symbols
	Symbols = 256
	// Space is the state space of the Println
	Space = 256
	// Width is the width of the neural network
	Width = Symbols + Space
	// Batch is the batch size
	Batch = 256
	// Scale scales the neural network
	Scale = 2
	// B1 exponential decay of the rate for the first moment estimates
	B1 = 0.8
	// B2 exponential decay rate for the second-moment estimates
	B2 = 0.89
	// Eta is the learning rate
	Eta = .0001
)

const (
	// StateM is the state for the mean
	StateM = iota
	// StateV is the state for the variance
	StateV
	// StateTotal is the total number of states
	StateTotal
)

var (
	// Nets the number of nets to run in parallel
	Nets = runtime.NumCPU()
	// FlagVerbose enables verbose mode
	FlagVerbose = flag.Bool("verbose", false, "verbose mode")
	// FlagLearn learn the model
	FlagLearn = flag.String("learn", "", "learning mode")
	// Flag2X use the 2X model
	Flag2X = flag.Bool("2X", false, "learn the 2X model")
	// Flag2X64 use the 64 bit 2X model
	Flag2X64 = flag.Bool("2X64", false, "learn the 64 bit 2X model")
	// FlagGraph graphs the model files
	FlagGraph = flag.String("graph", "", "graph mode")
	// FlagInference load weights and generate probable strings
	FlagInference = flag.String("inference", "", "inference mode")
)

func main() {
	flag.Parse()

	if *FlagLearn != "" {
		if *Flag2X {
			Learn2X()
			return
		} else if *Flag2X64 {
			Learn2X64()
			return
		}
		Learn()
		return
	} else if *FlagInference != "" {
		if *Flag2X {
			Inference2X()
			return
		} else if *Flag2X64 {
			Inference2X64()
			return
		}
		Inference()
		return
	} else if *FlagGraph != "" {
		if *Flag2X64 {
			Graph64(*FlagGraph)
			return
		}
		Graph(*FlagGraph)
		return
	}

	bible, err := bible.Load()
	if err != nil {
		panic(err)
	}
	verses := bible.GetVerses()
	maxVerse, maxWords := 0, 0
	for _, verse := range verses {
		if length := len(verse.Verse); length > maxVerse {
			maxVerse = length
		}
		if length := len(verse.Words); length > maxWords {
			maxWords = length
		}
	}
	sentences := bible.GetSentences()
	words := bible.GetWords()
	maxWord := 0
	for _, word := range words {
		if length := len(word); length > maxWord {
			maxWord = length
		}
	}
	fmt.Printf("number of verses %d\n", len(verses))
	fmt.Printf("number of sentences %d\n", len(sentences))
	fmt.Printf("max verse length %d\n", maxVerse)
	fmt.Printf("max words in verse %d\n", maxWords)
	fmt.Printf("number of unique words %d\n", len(words))
	fmt.Printf("max word length %d\n", maxWord)
	return
}

// Graph graphs the weight files
func Graph(directory string) {
	input, err := os.Open(directory)
	if err != nil {
		panic(err)
	}
	defer input.Close()
	type Pair struct {
		Epoch int
		Cost  float32
	}
	pairs := make(map[int]*Pair)
	names, err := input.Readdirnames(-1)
	if err != nil {
		panic(err)
	}
	for _, name := range names {
		if strings.HasSuffix(name, ".w") {
			set := tf32.NewSet()
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
		points = append(points, plotter.XY{X: float64(pair.Epoch), Y: float64(pair.Cost)})
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

// Learn learns the r2n2 model
func Learn() {
	rng := rand.New(rand.NewSource(1))
	bible, err := bible.Load()
	if err != nil {
		panic(err)
	}
	verses := bible.GetVerses()

	input := tf32.NewV(Symbols, 1)
	input.X = input.X[:cap(input.X)]
	output := tf32.NewV(Symbols, 1)
	output.X = output.X[:cap(output.X)]
	feedback := tf32.NewV(Space, 1)
	feedback.X = feedback.X[:cap(feedback.X)]
	set := tf32.NewSet()
	set.Add("w1", Symbols, Symbols)
	set.Add("b1", Symbols)
	set.Add("w2", Width, Space)
	set.Add("b2", Space)
	set.Add("w2d", Space, Width)
	set.Add("b2d", Width)
	set.Add("w3", Space, Symbols)
	set.Add("b3", Symbols)
	for i := range set.Weights {
		w := set.Weights[i]
		if strings.HasPrefix(w.N, "b") {
			w.X = w.X[:cap(w.X)]
			continue
		}
		factor := float32(math.Sqrt(float64(w.S[0])))
		for i := 0; i < cap(w.X); i++ {
			w.X = append(w.X, Random32(rng, -1, 1)/factor)
		}
	}
	{
		deltas := make([][]float32, 0, len(set.Weights))
		for _, p := range set.Weights {
			deltas = append(deltas, make([]float32, len(p.X)))
		}

		for i := range feedback.X {
			feedback.X[i] = 0
		}
		feedback.Zero()
		alpha, eta := float32(.9), float32(.1)
		for i := 0; i < 128; i++ {
			set.Zero()
			inputs := make([]*tf32.V, 0, 8)
			input := tf32.NewV(Symbols, 1)
			input.X = input.X[:cap(input.X)]
			for i := range input.X {
				e := math.Exp(rng.NormFloat64())
				input.X[i] = float32(e / (e + 1))
			}
			inputs = append(inputs, &input)
			l1 := tf32.Sigmoid(tf32.Add(tf32.Mul(set.Get("w2"),
				tf32.Concat(input.Meta(), feedback.Meta())), set.Get("b2")))
			length := rng.Intn(32) + 1
			for j := 0; j < length; j++ {
				input := tf32.NewV(Symbols, 1)
				input.X = input.X[:cap(input.X)]
				for i := range input.X {
					e := math.Exp(rng.NormFloat64())
					input.X[i] = float32(e / (e + 1))
				}
				inputs = append(inputs, &input)
				l1 = tf32.Sigmoid(tf32.Add(tf32.Mul(set.Get("w2"),
					tf32.Concat(input.Meta(), l1)), set.Get("b2")))
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
			l1d := tf32.Sigmoid(tf32.Add(tf32.Mul(set.Get("w2d"), l1), set.Get("b2d")))
			cost := tf32.Avg(tf32.Quadratic(tf32.Slice(l1d, options), inputs[0].Meta()))
			for j := 0; j < length; j++ {
				l1d = tf32.Sigmoid(tf32.Add(tf32.Mul(set.Get("w2d"), tf32.Slice(l1d, options1)), set.Get("b2d")))
				cost = tf32.Add(cost, tf32.Avg(tf32.Quadratic(tf32.Slice(l1d, options), inputs[j+1].Meta())))
			}
			total := tf32.Gradient(cost).X[0]
			norm := float32(0)
			for _, p := range set.Weights {
				for _, d := range p.D {
					norm += d * d
				}
			}
			norm = float32(math.Sqrt(float64(norm)))
			if norm > 1 {
				scaling := 1 / norm
				for k, p := range set.Weights {
					if p.N != "w2" && p.N != "b2" && p.N != "w2d" && p.N != "b2d" {
						continue
					}
					for l, d := range p.D {
						deltas[k][l] = alpha*deltas[k][l] - eta*d*scaling
						p.X[l] += deltas[k][l]
					}
				}
			} else {
				for k, p := range set.Weights {
					if p.N != "w2" && p.N != "b2" && p.N != "w2d" && p.N != "b2d" {
						continue
					}
					for l, d := range p.D {
						deltas[k][l] = alpha*deltas[k][l] - eta*d
						p.X[l] += deltas[k][l]
					}
				}
			}
			fmt.Println("pre", length, total/float32(length))
		}
	}

	deltas := make([][]float32, 0, len(set.Weights))
	for _, p := range set.Weights {
		deltas = append(deltas, make([]float32, len(p.X)))
	}

	l1 := tf32.Sigmoid(tf32.Add(tf32.Mul(set.Get("w1"), input.Meta()), set.Get("b1")))
	l2 := tf32.Sigmoid(tf32.Add(tf32.Mul(set.Get("w2"), tf32.Concat(l1, feedback.Meta())), set.Get("b2")))
	l3 := tf32.Quadratic(tf32.Sigmoid(tf32.Add(tf32.Mul(set.Get("w3"), l2), set.Get("b3"))), output.Meta())

	iterations := 100
	alpha, eta := float32(.9), float32(.1)
	points := make(plotter.XYs, 0, iterations)
	start := time.Now()
	for i := 0; i < iterations; i++ {
		for i := range verses {
			j := i + rng.Intn(len(verses)-i)
			verses[i], verses[j] = verses[j], verses[i]
		}

		total := float32(0)
		for i := 0; i < len(verses); i++ {
			verse := "^" + verses[i].Verse + "$"
			for i := range feedback.X {
				feedback.X[i] = 0
			}
			feedback.Zero()
			set.Zero()
			cost := float32(0)
			for l, symbol := range verse[:len(verses[i].Verse)-1] {
				for i := range input.X {
					input.X[i] = 0
				}
				input.Zero()
				input.X[int(symbol)] = 1
				for i := range output.X {
					output.X[i] = 0
				}
				output.Zero()
				output.X[int(verse[l+1])] = 1
				cost += tf32.Gradient(l3).X[0]

				l2(func(a *tf32.V) bool {
					copy(feedback.X, a.X)
					return true
				})
			}
			norm := float32(0)
			for _, p := range set.Weights {
				for _, d := range p.D {
					norm += d * d
				}
			}
			norm = float32(math.Sqrt(float64(norm)))
			if norm > 1 {
				scaling := 1 / norm
				for k, p := range set.Weights {
					if p.N == "w2" || p.N == "b2" || p.N == "w2d" || p.N == "b2d" {
						continue
					}
					for l, d := range p.D {
						deltas[k][l] = alpha*deltas[k][l] - eta*d*scaling
						p.X[l] += deltas[k][l]
					}
				}
			} else {
				for k, p := range set.Weights {
					if p.N == "w2" || p.N == "b2" || p.N == "w2d" || p.N == "b2d" {
						continue
					}
					for l, d := range p.D {
						deltas[k][l] = alpha*deltas[k][l] - eta*d
						p.X[l] += deltas[k][l]
					}
				}
			}
			cost /= float32(len(verses[i].Verse))
			total += cost
			fmt.Println(cost)
		}
		fmt.Printf("\n")

		err := set.Save(fmt.Sprintf("weights_%d.w", i), total, i)
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

// Inference inference r2n2 model
func Inference() {
	set := tf32.NewSet()
	cost, epoch, err := set.Open(*FlagInference)
	if err != nil {
		panic(err)
	}
	fmt.Println(cost, epoch)
	bestSum, best := float32(0.0), []rune{}
	var search func(depth int, most []rune, previous *tf32.V, sum float32)
	search = func(depth int, most []rune, previous *tf32.V, sum float32) {
		if depth > 2 {
			if sum > bestSum {
				best, bestSum = most, sum
				fmt.Println(best)
				fmt.Println(string(best))
			}
			return
		}

		input, feedback := tf32.NewV(Symbols, 1), tf32.NewV(Space, 1)
		input.X = input.X[:cap(input.X)]
		feedback.X = feedback.X[:cap(feedback.X)]
		copy(feedback.X, previous.X)
		l1 := tf32.Sigmoid(tf32.Add(tf32.Mul(set.Get("w1"), input.Meta()), set.Get("b1")))
		l2 := tf32.Sigmoid(tf32.Add(tf32.Mul(set.Get("w2"), tf32.Concat(l1, feedback.Meta())), set.Get("b2")))
		l3 := tf32.Sigmoid(tf32.Add(tf32.Mul(set.Get("w3"), l2), set.Get("b3")))
		setSymbol := func(s rune) {
			for i := range input.X {
				input.X[i] = 0
			}
			symbol := int(s)
			input.X[symbol] = 1
		}
		setSymbol(most[len(most)-1])
		next := tf32.NewV(Space, 1)
		next.X = next.X[:cap(next.X)]
		l2(func(a *tf32.V) bool {
			copy(next.X, a.X)
			return true
		})
		l3(func(a *tf32.V) bool {
			symbols := a.X
			for i, symbol := range symbols {
				cp := make([]rune, len(most))
				copy(cp, most)
				cp = append(cp, rune(i))
				search(depth+1, cp, &next, sum+symbol)
			}
			return true
		})
	}
	initial := tf32.NewV(Space, 1)
	initial.X = initial.X[:cap(initial.X)]
	search(0, []rune{'^'}, &initial, 0)
}

// Learn2X learns 2X the r2n2 model
func Learn2X() {
	rng := rand.New(rand.NewSource(1))
	bible, err := bible.Load()
	if err != nil {
		panic(err)
	}
	verses := bible.GetVerses()

	input := tf32.NewV(Symbols, 1)
	input.X = input.X[:cap(input.X)]
	output := tf32.NewV(Symbols, 1)
	output.X = output.X[:cap(output.X)]
	feedback := tf32.NewV(Space, 1)
	feedback.X = feedback.X[:cap(feedback.X)]
	set := tf32.NewSet()
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
		if strings.HasPrefix(w.N, "b") {
			w.X = w.X[:cap(w.X)]
			w.States = make([][]float32, StateTotal)
			for i := range w.States {
				w.States[i] = make([]float32, len(w.X))
			}
			continue
		}
		factor := float32(math.Sqrt(float64(w.S[0])))
		for i := 0; i < cap(w.X); i++ {
			w.X = append(w.X, Random32(rng, -1, 1)/factor)
		}
		w.States = make([][]float32, StateTotal)
		for i := range w.States {
			w.States[i] = make([]float32, len(w.X))
		}
	}
	{
		pow := func(x float32) float32 {
			y := math.Pow(float64(x), float64(1))
			if math.IsNaN(y) || math.IsInf(y, 0) {
				return 0
			}
			return float32(y)
		}
		for i := range feedback.X {
			feedback.X[i] = 0
		}
		feedback.Zero()
		for i := 0; i < 256; i++ {
			set.Zero()
			inputs := make([]*tf32.V, 0, 8)
			input := tf32.NewV(Symbols, 1)
			input.X = input.X[:cap(input.X)]
			for i := range input.X {
				e := math.Exp(rng.NormFloat64())
				input.X[i] = float32(e / (e + 1))
			}
			inputs = append(inputs, &input)
			l1 := tf32.Sigmoid(tf32.Add(tf32.Mul(set.Get("w2"),
				tf32.Concat(input.Meta(), feedback.Meta())), set.Get("b2")))
			length := rng.Intn(32) + 1
			for j := 0; j < length; j++ {
				input := tf32.NewV(Symbols, 1)
				input.X = input.X[:cap(input.X)]
				for i := range input.X {
					e := math.Exp(rng.NormFloat64())
					input.X[i] = float32(e / (e + 1))
				}
				inputs = append(inputs, &input)
				l1 = tf32.Sigmoid(tf32.Add(tf32.Mul(set.Get("w2"),
					tf32.Concat(input.Meta(), l1)), set.Get("b2")))
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
			l1d := tf32.Sigmoid(tf32.Add(tf32.Mul(set.Get("w2d"), l1), set.Get("b2d")))
			cost := tf32.Avg(tf32.Quadratic(tf32.Slice(l1d, options), inputs[0].Meta()))
			for j := 0; j < length; j++ {
				l1d = tf32.Sigmoid(tf32.Add(tf32.Mul(set.Get("w2d"), tf32.Slice(l1d, options1)), set.Get("b2d")))
				cost = tf32.Add(cost, tf32.Avg(tf32.Quadratic(tf32.Slice(l1d, options), inputs[j+1].Meta())))
			}
			total := tf32.Gradient(cost).X[0]
			norm := float32(0)
			for _, p := range set.Weights {
				for _, d := range p.D {
					norm += d * d
				}
			}
			norm = float32(math.Sqrt(float64(norm)))
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
						w.X[l] -= Eta * mhat / (float32(math.Sqrt(float64(vhat))) + 1e-8)
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
						w.X[l] -= Eta * mhat / (float32(math.Sqrt(float64(vhat))) + 1e-8)
					}
				}
			}
			fmt.Println("pre", length, total/float32(length))
		}
	}

	l1 := tf32.Sigmoid(tf32.Add(tf32.Mul(set.Get("w1"), input.Meta()), set.Get("b1")))
	l1a := tf32.Sigmoid(tf32.Add(tf32.Mul(set.Get("w1a"), l1), set.Get("b1a")))
	l2 := tf32.Sigmoid(tf32.Add(tf32.Mul(set.Get("w2"), tf32.Concat(l1a, feedback.Meta())), set.Get("b2")))
	l3 := tf32.Sigmoid(tf32.Add(tf32.Mul(set.Get("w3"), l2), set.Get("b3")))
	l3a := tf32.CrossEntropy(tf32.Softmax(tf32.Add(tf32.Mul(set.Get("w3a"), l3), set.Get("b3a"))), output.Meta())

	iterations := 100
	points := make(plotter.XYs, 0, iterations)
	start := time.Now()
	for i := 0; i < iterations; i++ {
		pow := func(x float32) float32 {
			y := math.Pow(float64(x), float64(i+1))
			if math.IsNaN(y) || math.IsInf(y, 0) {
				return 0
			}
			return float32(y)
		}

		for i := range verses {
			j := i + rng.Intn(len(verses)-i)
			verses[i], verses[j] = verses[j], verses[i]
		}

		total := float32(0)
		for i := 0; i < len(verses); i++ {
			verse := "^" + verses[i].Verse + "$"
			for i := range feedback.X {
				feedback.X[i] = 0
			}
			feedback.Zero()
			set.Zero()
			cost := float32(0)
			for l, symbol := range verse[:len(verses[i].Verse)-1] {
				for i := range input.X {
					input.X[i] = 0
				}
				input.Zero()
				input.X[int(symbol)] = 1
				for i := range output.X {
					output.X[i] = 0
				}
				output.Zero()
				output.X[int(verse[l+1])] = 1
				cost += tf32.Gradient(l3a).X[0]

				l2(func(a *tf32.V) bool {
					copy(feedback.X, a.X)
					return true
				})
			}
			norm := float32(0)
			for _, p := range set.Weights {
				for _, d := range p.D {
					norm += d * d
				}
			}
			norm = float32(math.Sqrt(float64(norm)))
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
						w.X[l] -= Eta * mhat / (float32(math.Sqrt(float64(vhat))) + 1e-8)
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
						w.X[l] -= Eta * mhat / (float32(math.Sqrt(float64(vhat))) + 1e-8)
					}
				}
			}
			cost /= float32(len(verses[i].Verse))
			total += cost
			fmt.Println(cost)
		}
		fmt.Printf("\n")

		err := set.Save(fmt.Sprintf("weights_%d.w", i), total, i)
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

// Inference2X inference 2X r2n2 model
func Inference2X() {
	set := tf32.NewSet()
	cost, epoch, err := set.Open(*FlagInference)
	if err != nil {
		panic(err)
	}
	fmt.Println(cost, epoch)
	bestSum, best := float32(0.0), []rune{}
	var search func(depth int, most []rune, previous *tf32.V, sum float32)
	search = func(depth int, most []rune, previous *tf32.V, sum float32) {
		if depth > 2 {
			if sum > bestSum {
				best, bestSum = most, sum
				fmt.Println(best)
				fmt.Println(string(best))
			}
			return
		}

		input, feedback := tf32.NewV(Symbols, 1), tf32.NewV(Space, 1)
		input.X = input.X[:cap(input.X)]
		feedback.X = feedback.X[:cap(feedback.X)]
		copy(feedback.X, previous.X)
		l1 := tf32.Sigmoid(tf32.Add(tf32.Mul(set.Get("w1"), input.Meta()), set.Get("b1")))
		l1a := tf32.Sigmoid(tf32.Add(tf32.Mul(set.Get("w1a"), l1), set.Get("b1a")))
		l2 := tf32.Sigmoid(tf32.Add(tf32.Mul(set.Get("w2"), tf32.Concat(l1a, feedback.Meta())), set.Get("b2")))
		l3 := tf32.Sigmoid(tf32.Add(tf32.Mul(set.Get("w3"), l2), set.Get("b3")))
		l3a := tf32.Softmax(tf32.Add(tf32.Mul(set.Get("w3a"), l3), set.Get("b3a")))
		setSymbol := func(s rune) {
			for i := range input.X {
				input.X[i] = 0
			}
			symbol := int(s)
			input.X[symbol] = 1
		}
		setSymbol(most[len(most)-1])
		next := tf32.NewV(Space, 1)
		next.X = next.X[:cap(next.X)]
		l2(func(a *tf32.V) bool {
			copy(next.X, a.X)
			return true
		})
		l3a(func(a *tf32.V) bool {
			symbols := a.X
			for i, symbol := range symbols {
				cp := make([]rune, len(most))
				copy(cp, most)
				cp = append(cp, rune(i))
				search(depth+1, cp, &next, sum+symbol)
			}
			return true
		})
	}
	in := []rune{'^', 'T'}
	input := tf32.NewV(Space, 1)
	input.X = input.X[:cap(input.X)]
	initial := tf32.NewV(Space, 1)
	initial.X = initial.X[:cap(initial.X)]

	l1 := tf32.Sigmoid(tf32.Add(tf32.Mul(set.Get("w1"), input.Meta()), set.Get("b1")))
	l1a := tf32.Sigmoid(tf32.Add(tf32.Mul(set.Get("w1a"), l1), set.Get("b1a")))
	l2 := tf32.Sigmoid(tf32.Add(tf32.Mul(set.Get("w2"), tf32.Concat(l1a, initial.Meta())), set.Get("b2")))
	//l3 := tf32.Sigmoid(tf32.Add(tf32.Mul(set.Get("w3"), l2), set.Get("b3")))
	//l3a := tf32.Softmax(tf32.Add(tf32.Mul(set.Get("w3a"), l3), set.Get("b3a")))
	for i := range in[:len(in)-1] {
		for j := range input.X {
			input.X[j] = 0
		}
		input.X[int(in[i])] = 1
		l2(func(a *tf32.V) bool {
			copy(initial.X, a.X)
			return true
		})
	}
	search(0, in[len(in)-1:], &initial, 0)
}

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
func Learn2X64() {
	rng := rand.New(rand.NewSource(1))
	bible, err := bible.Load()
	if err != nil {
		panic(err)
	}
	verses := bible.GetVerses()

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
			for l, symbol := range verse[:len(verses[i].Verse)-1] {
				for i := range input.X {
					input.X[i] = 0
				}
				input.Zero()
				input.X[int(symbol)] = 1
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

		err := set.Save(fmt.Sprintf("weights_%d.w", i), total, i)
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
	set := tf64.NewSet()
	cost, epoch, err := set.Open(*FlagInference)
	if err != nil {
		panic(err)
	}
	fmt.Println(cost, epoch)
	bestSum, best := 0.0, []rune{}
	var search func(depth int, most []rune, previous *tf64.V, sum float64)
	search = func(depth int, most []rune, previous *tf64.V, sum float64) {
		if depth > 2 {
			if sum > bestSum {
				best, bestSum = most, sum
				fmt.Println(best)
				fmt.Println(string(best))
			}
			return
		}

		input, feedback := tf64.NewV(Symbols, 1), tf64.NewV(Space, 1)
		input.X = input.X[:cap(input.X)]
		feedback.X = feedback.X[:cap(feedback.X)]
		copy(feedback.X, previous.X)
		l1 := tf64.Sigmoid(tf64.Add(tf64.Mul(set.Get("w1"), input.Meta()), set.Get("b1")))
		l1a := tf64.Add(tf64.Mul(set.Get("w1a"), l1), set.Get("b1a"))
		l2 := tf64.Sigmoid(tf64.Add(tf64.Mul(set.Get("w2"), tf64.Concat(l1a, feedback.Meta())), set.Get("b2")))
		l3 := tf64.Sigmoid(tf64.Add(tf64.Mul(set.Get("w3"), l2), set.Get("b3")))
		l3a := tf64.Softmax(tf64.Add(tf64.Mul(set.Get("w3a"), l3), set.Get("b3a")))
		setSymbol := func(s rune) {
			for i := range input.X {
				input.X[i] = 0
			}
			symbol := int(s)
			input.X[symbol] = 1
		}
		setSymbol(most[len(most)-1])
		next := tf64.NewV(Space, 1)
		next.X = next.X[:cap(next.X)]
		l2(func(a *tf64.V) bool {
			copy(next.X, a.X)
			return true
		})
		l3a(func(a *tf64.V) bool {
			symbols := a.X
			for i, symbol := range symbols {
				cp := make([]rune, len(most))
				copy(cp, most)
				cp = append(cp, rune(i))
				search(depth+1, cp, &next, sum+symbol)
			}
			return true
		})
	}
	in := []rune{'^', 'T'}
	input := tf64.NewV(Space, 1)
	input.X = input.X[:cap(input.X)]
	initial := tf64.NewV(Space, 1)
	initial.X = initial.X[:cap(initial.X)]

	l1 := tf64.Sigmoid(tf64.Add(tf64.Mul(set.Get("w1"), input.Meta()), set.Get("b1")))
	l1a := tf64.Add(tf64.Mul(set.Get("w1a"), l1), set.Get("b1a"))
	l2 := tf64.Sigmoid(tf64.Add(tf64.Mul(set.Get("w2"), tf64.Concat(l1a, initial.Meta())), set.Get("b2")))
	//l3 := tf64.Sigmoid(tf64.Add(tf64.Mul(set.Get("w3"), l2), set.Get("b3")))
	//l3a := tf64.Softmax(tf64.Add(tf64.Mul(set.Get("w3a"), l3), set.Get("b3a")))
	for i := range in[:len(in)-1] {
		for j := range input.X {
			input.X[j] = 0
		}
		input.X[int(in[i])] = 1
		l2(func(a *tf64.V) bool {
			copy(initial.X, a.X)
			return true
		})
	}
	search(0, in[len(in)-1:], &initial, 0)
}

// Random32 return a random float32
func Random32(rng *rand.Rand, a, b float32) float32 {
	return (b-a)*rand.Float32() + a
}
