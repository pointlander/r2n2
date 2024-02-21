// Copyright 2024 The R2N2 Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"math"
	"math/rand"
	"strconv"

	"github.com/pointlander/datum/bible"
	"github.com/pointlander/matrix"
)

// LearnX64SA learns 64 bit self attention r2n2 model
func LearnX64SA(name string) {
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

	factor := math.Sqrt(2.0 / float64(Width))
	x1 := matrix.NewMatrix(Width, Space)
	x2 := matrix.NewMatrix(Width, Space)
	x3 := matrix.NewMatrix(Width, Space)
	for i := 0; i < x1.Size(); i++ {
		x1.Data = append(x1.Data, float32(rng.NormFloat64()*factor))
		x2.Data = append(x2.Data, float32(rng.NormFloat64()*factor))
		x3.Data = append(x3.Data, float32(rng.NormFloat64()*factor))
	}
	iterations := 100
	for i := 0; i < iterations; i++ {
		for i := range verses {
			j := i + rng.Intn(len(verses)-i)
			verses[i], verses[j] = verses[j], verses[i]
		}

		for i := 0; i < len(verses); i++ {
			verse := "^" + verses[i].Verse + "$"
			input := matrix.NewZeroMatrix(Width, 3)
			for _, symbol := range verse[:len(verses[i].Verse)-1] {
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
				output := matrix.SelfAttention(q, k, v)
				for j := 0; j < 3; j++ {
					copy(input.Data[j*Width+Symbols:], output.Data[j*Space:(j+1)*Space])
				}
			}
		}
		fmt.Println(i)
	}
}
