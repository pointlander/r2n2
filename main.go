// Copyright 2024 The R2N2 Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"flag"
	"fmt"
	"math"
	"math/rand"
	"runtime"

	"github.com/pointlander/datum/bible"
	"github.com/pointlander/matrix"
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
	// Flag2X64SE use the 64 bit self entropy 2X model
	Flag2X64SE = flag.Bool("2X64SE", false, "learn the 64 bit self entropy 2X model")
	// FlagX64SA learn 64 bit self attention model
	FlagX64SA = flag.Bool("X64SA", false, "learn 64 bit self attention model")
	// FlagGraph graphs the model files
	FlagGraph = flag.String("graph", "", "graph mode")
	// FlagInference load weights and generate probable strings
	FlagInference = flag.String("inference", "", "inference mode")
)

func Markov(verses []bible.Verse) *[256][256][256]float32 {
	markov := [256][256][256]float32{}
	for _, verse := range verses {
		a, b := 0, 0
		for _, v := range verse.Verse {
			markov[a][b][v]++
			a, b = int(v), a
		}
	}
	for i := range markov {
		for j := range markov[i] {
			sum := float32(0.0)
			for _, v := range markov[i][j] {
				sum += v
			}
			if sum > 0 {
				for k := range markov[i][j] {
					markov[i][j][k] /= sum
				}
			}
		}
	}
	return &markov
}

func GenerateMatrix(seed int64, cols, rows int) matrix.Matrix {
	rng := rand.New(rand.NewSource(seed))
	output := matrix.NewMatrix(cols, rows)
	factor := math.Sqrt(2.0 / float64(cols))
	for i := 0; i < output.Size(); i++ {
		output.Data = append(output.Data, float32(factor*rng.NormFloat64()))
	}
	/*optimizer := matrix.NewOptimizer(rng, 8, .1, 1, func(samples []matrix.Sample, x ...matrix.Matrix) {
		done := make(chan bool, 8)
		process := func(index int) {
			x := samples[index].Vars[0][0]
			y := samples[index].Vars[0][1]
			z := samples[index].Vars[0][2]
			sample := matrix.Add(x, matrix.H(y, z))
			rows, cols := sample.Rows, sample.Cols
			sum := 0.0
			for i := 0; i < rows; i++ {
				for j := i + 1; j < rows; j++ {
					for k := 0; k < cols; k++ {
						sum += float64(sample.Data[i*cols+k] * sample.Data[j*cols+k])
					}
				}
			}
			sum = math.Abs(sum)
			samples[index].Cost = sum
			done <- true
		}
		for j := range samples {
			go process(j)
		}
		for range samples {
			<-done
		}
	}, matrix.NewCoord(cols, rows))
	s := optimizer.Optimize(1e-3)
	output := matrix.Add(s.Vars[0][0], matrix.H(s.Vars[0][1], s.Vars[0][2]))
	factor := math.Sqrt(2.0 / float64(cols))
	for i := range output.Data {
		output.Data[i] *= float32(factor)
	}*/
	return output
}

func main() {
	flag.Parse()

	if *FlagLearn != "" {
		if *Flag2X {
			Learn2X()
			return
		} else if *Flag2X64 {
			Learn2X64(*FlagLearn)
			return
		} else if *FlagX64SA {
			LearnX64SA(*FlagLearn)
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
		} else if *Flag2X64SE {
			Inference2X64SE()
			return
		} else if *FlagX64SA {
			InferenceX64SA()
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

	markov := Markov(verses)
	_ = markov

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
