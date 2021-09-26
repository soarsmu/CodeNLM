package main

import (
	"C"
	"fmt"
	"github.com/eroatta/token/conserv"
	"github.com/eroatta/token/expansion"
	"github.com/eroatta/token/gentest"
	"github.com/eroatta/token/greedy"
	"github.com/eroatta/token/lists"
	"strings"
)
//export Run_gentest
func Run_gentest(val *C.char ) *C.char  {
	var s = C.GoString(val)
	var simCalculator gentest.SimilarityCalculator
	context := lists.NewBuilder().Add(lists.Dictionary.Elements()...).
		Add(lists.KnownAbbreviations.Elements()...).
		Add(lists.Stop.Elements()...).
		Build()
	possibleExpansions := expansion.NewSetBuilder().AddList(lists.Dictionary).Build()

	splitted := gentest.Split(s, simCalculator, context, possibleExpansions)
	ret := strings.Join(splitted, " ")
	fmt.Println(ret)
	return C.CString(ret)
}
//export Run_greedy
func Run_greedy(val *C.char ) *C.char  {
	var s = C.GoString(val)
	list := lists.NewBuilder().Add(lists.Dictionary.Elements()...).
		Add(lists.KnownAbbreviations.Elements()...).
		Add(lists.Stop.Elements()...).
		Build()
	splitted := greedy.Split(s, list)

	fmt.Println(splitted)
	return C.CString(splitted)
}
//export Run_conserv
func Run_conserv(val *C.char ) *C.char  {
	var s = C.GoString(val)
	splitted := conserv.Split(s)

	fmt.Println(splitted)
	return C.CString(splitted)
}

func main() {
	Run_greedy(C.CString("httpResponse"))
	Run_conserv(C.CString("httpResponse"))
	Run_gentest(C.CString("httpResponse"))
}
