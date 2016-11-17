// neuro project neuro.go
package neuro

import (
	"encoding/json"
	"fmt"
	"testing"
)

func TestNetwork(t *testing.T) {
	n := Generate(2, 2, 1)
	n.Layers[0].Nodes[0].Weights[0] = 10
	n.Layers[0].Nodes[0].Weights[1] = 10
	n.Layers[0].Nodes[0].Bias = -5
	n.Layers[0].Nodes[1].Weights[0] = 10
	n.Layers[0].Nodes[1].Weights[1] = 10
	n.Layers[0].Nodes[1].Bias = -10
	n.Layers[1].Nodes[0].Weights[0] = 20
	n.Layers[1].Nodes[0].Weights[1] = -20
	n.Layers[1].Nodes[0].Bias = -5
	fmt.Println(n.Calculate(1, 1))
	fmt.Println(n.Calculate(1, 0))
	fmt.Println(n.Calculate(0, 1))
	fmt.Println(n.Calculate(0, 0))
}

func TestTrain(t *testing.T) {
	n := Generate(2, 2, 1)
	fmt.Println(n.Calculate(1, 1))
	fmt.Println(n.Calculate(1, 0))
	fmt.Println(n.Calculate(0, 1))
	fmt.Println(n.Calculate(0, 0))
	for l1 := 0; l1 < 100000; l1++ {
		n.Train(1, []float64{1, 1}, []float64{0})
		n.Train(1, []float64{0, 1}, []float64{1})
		n.Train(1, []float64{1, 0}, []float64{1})
		n.Train(1, []float64{0, 0}, []float64{0})
	}
	fmt.Println(n.Calculate(1, 1))
	fmt.Println(n.Calculate(1, 0))
	fmt.Println(n.Calculate(0, 1))
	fmt.Println(n.Calculate(0, 0))

}

func TestMarshal(t *testing.T) {
	n := Generate(2, 2, 1)

	for l1 := 0; l1 < 100000; l1++ {
		n.Train(1, []float64{1, 1}, []float64{0})
		n.Train(1, []float64{0, 1}, []float64{1})
		n.Train(1, []float64{1, 0}, []float64{1})
		n.Train(1, []float64{0, 0}, []float64{0})
	}
	buf, _ := json.Marshal(n)
	fmt.Println(string(buf))
}

func TestUnmarshal(t *testing.T) {
	var n Network

	json.Unmarshal([]byte(`
	{"Layers":[{"Nodes":[{"Weights":[6.25373756439419,-7.206804422407824],"Bias":-2.716267781016341},{"Weights":[6.841351898317155,-7.0797008659018426],"Bias":3.6330141937000073}]},{"Nodes":[{"Weights":[12.681299378656709,-12.546108235191902],"Bias":5.9031746170960995}]}]}
	`), &n)

	fmt.Println(n.Calculate(1, 1))
	fmt.Println(n.Calculate(1, 0))
	fmt.Println(n.Calculate(0, 1))
	fmt.Println(n.Calculate(0, 0))
}
