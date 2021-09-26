package main

import (
	"C"
	"fmt"
	"github.com/eroatta/token/conserv"
	"github.com/eroatta/token/expansion"
	"github.com/eroatta/token/gentest"
	"github.com/eroatta/token/greedy"
	"github.com/eroatta/token/lists"
)
//export Run_gentest
func Run_gentest(s string) []string {
	var simCalculator gentest.SimilarityCalculator
	context := lists.NewBuilder().Add(lists.Dictionary.Elements()...).
		Add(lists.KnownAbbreviations.Elements()...).
		Add(lists.Stop.Elements()...).
		Build()
	possibleExpansions := expansion.NewSetBuilder().AddList(lists.Dictionary).Build()

	splitted := gentest.Split(s, simCalculator, context, possibleExpansions)

	fmt.Println(splitted)
	return splitted

}
//export Run_greedy
func Run_greedy(s string) string {
	list := lists.NewBuilder().Add(lists.Dictionary.Elements()...).
		Add(lists.KnownAbbreviations.Elements()...).
		Add(lists.Stop.Elements()...).
		Build()
	splitted := greedy.Split(s, list)

	fmt.Println(splitted)
	return splitted
}
//export Run_conserv
func Run_conserv(s string) string {
	splitted := conserv.Split(s)

	fmt.Println(splitted)
	return splitted
}

func main() {
}