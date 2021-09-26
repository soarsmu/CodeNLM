# token

[![GoDoc](https://godoc.org/github.com/eroatta/token?status.svg)](https://godoc.org/github.com/eroatta/token)
[![Go Report Card](https://goreportcard.com/badge/github.com/eroatta/token)](https://goreportcard.com/report/github.com/eroatta/token)

Collection of token splitting algorithms.
The following lists show the supported algorithms.

## Splitting algorithms

* **Conserv**: This is the reference algorithm.
Each token is split using separation markers as underscores, numbers and regular camel case.
* **Greedy**: This algorithm is based on a greedy approach and uses several lists to find the best split, analyzing the token looking for preffixes and suffixes.
*Related paper:* [Identifier splitting: A study of two techniques (Feild, Binkley and Lawrie)]([https://link](https://www.academia.edu/2852176/Identifier_splitting_A_study_of_two_techniques))
* **Samurai**: This algorithm splits identifiers into sequences of words by mining word frequencies in source code.
This is a technique to split identifiers into their component terms by mining frequencies in large source code bases, and relies on two assumptions:
  1. A substring composed an identifier is also likely to be used in other parts of the program or in other programs alone or as part of other identifiers.
  2. Given two possible splits of a given identifier, the split that most likely represents the developer's intent partitions the identifier into terms occurring more often in the program.
* **GenTest**: This is a splitting algorithm that consists of two parts: generation and test. The generation part of GenTest generates all possible splittings; the test part, however, evaluates a scoring function against each proposed splitting.
GenTest uses a set of metrics to characterize the quality of the split.

## Usage

### Conserv

A token can be splitted just by calling the splitting function: `conserv.Split(token)`.

```python
import ctypes
from ctypes import c_char_p

so = ctypes.CDLL("split.so")
so.Run_conserv.restype = ctypes.c_char_p
result = so.Run_conservy(c_char_p("httpResponse".encode("utf-8")))

```
Alternatively use the spiral library
```python
from spiral import elementary_split
for s in [ 'mStartCData', 'nonnegativedecimaltype', 'getUtf8Octets', 'GPSmodule', 'savefileas', 'nbrOfbugs']:
    print(ronin.split(s))

```


### Greedy

```python
import ctypes
from ctypes import c_char_p

so = ctypes.CDLL("split.so")
so.Run_greedy.restype = ctypes.c_char_p
result = so.Run_greedy(c_char_p("httpResponse".encode("utf-8")))

```

### Samurai
For Samurai, it is easier to use the Spiral Python Function
```python
from spiral import samurai
for s in [ 'mStartCData', 'nonnegativedecimaltype', 'getUtf8Octets', 'GPSmodule', 'savefileas', 'nbrOfbugs']:
    print(samurai.split(s))
```

### GenTest
```python
import ctypes
from ctypes import c_char_p

so = ctypes.CDLL("split.so")
so.Run_gentest.restype = ctypes.c_char_p
result = so.Run_gentest(c_char_p("httpResponse".encode("utf-8")))

```


## License

See the [LICENSE](LICENSE.md) file for license rights and limitations (MIT).
