package main

import (
	"fmt"

	"github.com/eroatta/token/conserv"
)

func main() {
	splitted := conserv.Split("httpResponse")

	fmt.Println(splitted) // "http response"
}