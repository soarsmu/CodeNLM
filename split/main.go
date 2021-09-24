package main

import (
	"fmt"
	"github.com/eroatta/token/lists"
	"github.com/eroatta/token/samurai"
)

func call_samurai() {
	localFreqTable := samurai.NewFrequencyTable()
	localFreqTable.SetOccurrences("http", 100)
	localFreqTable.SetOccurrences("response", 100)

	globalFreqTable := samurai.NewFrequencyTable()
	globalFreqTable.SetOccurrences("http", 120)
	globalFreqTable.SetOccurrences("response", 120)

	tokenContext := samurai.NewTokenContext(localFreqTable, globalFreqTable)

	splitted := samurai.Split("httpresponse", tokenContext, lists.Prefixes, lists.Suffixes)

	fmt.Println(splitted) // "http response"

}
func main() {
	call_samurai()
}