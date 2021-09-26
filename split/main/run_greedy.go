package main

import (
	"fmt"

	"github.com/eroatta/token/greedy"
	"github.com/eroatta/token/lists"
)

func main() {
	list := lists.NewBuilder().Add(lists.Dictionary.Elements()...).
		Add(lists.KnownAbbreviations.Elements()...).
		Add(lists.Stop.Elements()...).
		Build()
	splitted := greedy.Split("httpResponse", list)

	fmt.Println(splitted) // "http response"
}