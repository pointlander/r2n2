// Copyright 2024 The R2N2 Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"math"
	"math/rand"
	"strconv"
	"strings"
	"time"

	"github.com/pointlander/datum/bible"
	"github.com/pointlander/gradient/tf64"
	"github.com/pointlander/matrix"
)

// LearnX64SA learns 64 bit self attention r2n2 model
func LearnX64SA(name string) {
	const (
		B1  = 0.7
		B2  = 0.79
		Eta = .0001
	)

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

	fmt.Println("find x1")
	x1 := GenerateMatrix(1, Space, Space)
	fmt.Println("found x1")
	fmt.Println("find x2")
	x2 := GenerateMatrix(2, Space, Space)
	fmt.Println("found x2")
	fmt.Println("find x3")
	x3 := GenerateMatrix(3, Symbols, Symbols)
	fmt.Println("found x3")

	in := tf64.NewV(Space, 3)
	in.X = in.X[:cap(in.X)]
	out := tf64.NewV(Symbols, 3)
	out.X = out.X[:cap(out.X)]

	set := tf64.NewSet()
	set.Add("w1", Space, Space)
	set.Add("b1", Space)
	set.Add("w2", Space, Symbols)
	set.Add("b2", Symbols)

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
		fmt.Printf("find %s\n", w.N)
		m := GenerateMatrix(int64(4+i), w.S[0], w.S[1])
		for i := 0; i < size; i++ {
			w.X = append(w.X, float64(m.Data[i]))
		}
		fmt.Printf("found %s\n", w.N)
		w.States = make([][]float64, StateTotal)
		for i := range w.States {
			w.States[i] = make([]float64, len(w.X))
		}
	}

	l1 := tf64.Sigmoid(tf64.Add(tf64.Mul(set.Get("w1"), in.Meta()), set.Get("b1")))
	l2 := tf64.Avg(tf64.CrossEntropy(tf64.Softmax(tf64.Add(tf64.Mul(set.Get("w2"), l1), set.Get("b2"))), out.Meta()))

	iterations := 100
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
		start := time.Now()
		for i := 0; i < len(verses); i++ {
			in.Zero()
			out.Zero()
			set.Zero()
			verse := "^" + verses[i].Verse + "$"
			input := matrix.NewZeroMatrix(Symbols, 3)
			feedback := matrix.NewZeroMatrix(Space, 3)
			for i := 0; i < 3; i++ {
				feedback.Data[i*Space] = 1
			}
			cost := 0.0
			index := 0
			for l, symbol := range verse[:len(verses[i].Verse)-1] {
				for i := 0; i < Symbols; i++ {
					input.Data[index*Symbols+i] = 0
					out.X[index*Symbols+i] = 0
				}
				input.Data[index*Symbols+int(symbol)] = 1
				out.X[index*Symbols+int(verse[l+1])] = 1
				index = (index + 1) % 3
				q := matrix.MulT(x1, feedback)
				k := matrix.MulT(x2, feedback)
				v := matrix.MulT(x3, input)
				output := matrix.Sigmoid(matrix.SelfAttention(q, k, v))
				copy(feedback.Data, output.Data)

				for j := range in.X {
					in.X[j] = float64(output.Data[j])
				}
				cost += tf64.Gradient(l2).X[0]
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
		err := set.Save(fmt.Sprintf("weights_%d_%d.w", seed, i), total, i)
		if err != nil {
			panic(err)
		}

		fmt.Println(i, total, time.Now().Sub(start))
	}
}

// InferenceX64SA inference 64 bit self attention r2n2 model
func InferenceX64SA() {
	rng := rand.New(rand.NewSource(int64(0) + 1))
	generated := []rune{'^'}

	factor := math.Sqrt(2.0 / float64(Width))
	x1 := matrix.NewMatrix(Width, Space)
	x2 := matrix.NewMatrix(Width, Space)
	x3 := matrix.NewMatrix(Width, Space)
	for i := 0; i < x1.Size(); i++ {
		x1.Data = append(x1.Data, float32(rng.NormFloat64()*factor))
		x2.Data = append(x2.Data, float32(rng.NormFloat64()*factor))
		x3.Data = append(x3.Data, float32(rng.NormFloat64()*factor))
	}

	in := tf64.NewV(Space, 3)
	in.X = in.X[:cap(in.X)]
	out := tf64.NewV(Symbols, 3)
	out.X = out.X[:cap(out.X)]

	name := *FlagInference
	set := tf64.NewSet()
	cost, epoch, err := set.Open(name)
	if err != nil {
		panic(err)
	}
	fmt.Println(name, cost, epoch)

	t := 2.0
	temp := tf64.NewV(Symbols, 1)
	for i := 0; i < Symbols; i++ {
		temp.X = append(temp.X, 1/t)
	}

	l1 := tf64.Sigmoid(tf64.Add(tf64.Mul(set.Get("w1"), in.Meta()), set.Get("b1")))
	l2 := tf64.Softmax(tf64.Hadamard(tf64.Add(tf64.Mul(set.Get("w2"), l1), set.Get("b2")), temp.Meta()))

	input := matrix.NewZeroMatrix(Width, 3)
	for i := 0; i < 256; i++ {
		symbol := generated[len(generated)-1]
		for j := 0; j < 3; j++ {
			for k := 0; k < Symbols; k++ {
				input.Data[j*Width+k] = 0
			}
		}
		input.Data[symbol] = 1
		input.Data[Width+symbol] = 1
		input.Data[2*Width+symbol] = 1
		q := matrix.MulT(x1, input)
		k := matrix.MulT(x2, input)
		v := matrix.MulT(x3, input)
		output := matrix.Sigmoid(matrix.SelfAttention(q, k, v))
		for j := 0; j < 3; j++ {
			copy(input.Data[j*Width+Symbols:], output.Data[j*Space:(j+1)*Space])
		}

		for j := range in.X {
			in.X[j] = float64(output.Data[j])
		}

		l2(func(a *tf64.V) bool {
			selected, sum := rng.Float64(), 0.0
			for j := 0; j < Symbols; j++ {
				for k := 0; k < 3; k++ {
					sum += a.X[k*Symbols+j] / 3.0
				}
				if selected < sum {
					generated = append(generated, rune(j))
					break
				}
			}
			return true
		})
		fmt.Println(string(generated))
	}
}
