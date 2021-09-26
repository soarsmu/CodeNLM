package main

import (
	"fmt"
	"github.com/eroatta/token/conserv"
	"github.com/eroatta/token/expansion"
	"github.com/eroatta/token/gentest"
	"github.com/eroatta/token/greedy"
	"github.com/eroatta/token/lists"
)
func run_gentest() {
	var simCalculator gentest.SimilarityCalculator
	context := lists.NewBuilder().Add(lists.Dictionary.Elements()...).
		Add(lists.KnownAbbreviations.Elements()...).
		Add(lists.Stop.Elements()...).
		Build()
	possibleExpansions := expansion.NewSetBuilder().AddList(lists.Dictionary).Build()

	splitted := gentest.Split("httpResponse", simCalculator, context, possibleExpansions)

	fmt.Println(splitted) // [http response]
}

func run_greedy() {
	list := lists.NewBuilder().Add(lists.Dictionary.Elements()...).
		Add(lists.KnownAbbreviations.Elements()...).
		Add(lists.Stop.Elements()...).
		Build()
	splitted := greedy.Split("httpResponse", list)

	fmt.Println(splitted) // "http response"
}

func run_conserv() {
	splitted := conserv.Split("httpResponse")

	fmt.Println(splitted) // "http response"
}

func main() {
	fmt.Println("gentest")
	run_gentest()
	fmt.Println("greedy")
	run_greedy()
	fmt.Println("conserv")
	run_conserv()
}