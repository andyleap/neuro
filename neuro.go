// neuro project neuro.go
package neuro

import (
	"math"
	"math/rand"
)

type Node struct {
	Weights []float64
	Bias    float64
}

type Layer struct {
	Nodes []Node
}

type Network struct {
	Layers []Layer
}

func Sigmoid(t float64) float64 {
	return (1 / (1 + math.Exp(-t)))
}

func (n *Node) Calculate(inputs ...float64) float64 {
	newVal := n.Bias
	for i, val := range inputs {
		if i >= len(n.Weights) {
			break
		}
		newVal = newVal + val*n.Weights[i]
	}
	newVal = Sigmoid(newVal)
	return newVal
}

func (l *Layer) Calculate(inputs ...float64) []float64 {
	outputs := make([]float64, len(l.Nodes))
	for i, n := range l.Nodes {
		outputs[i] = n.Calculate(inputs...)
	}
	return outputs
}

func (l *Layer) BackProp(train float64, input []float64, errors []float64) []float64 {
	newErrors := make([]float64, len(input))
	for l1 := range l.Nodes {
		nodeNet := l.Nodes[l1].Bias
		for i, val := range input {
			if i >= len(l.Nodes[l1].Weights) {
				break
			}
			nodeNet = nodeNet + val*l.Nodes[l1].Weights[i]
		}
		nodeError := Sigmoid(nodeNet) * (1 - Sigmoid(nodeNet)) * errors[l1]
		for l2 := range l.Nodes[l1].Weights {
			newErrors[l2] += nodeError * l.Nodes[l1].Weights[l2]
			l.Nodes[l1].Weights[l2] += train * input[l2] * nodeError
		}
		l.Nodes[l1].Bias += train * Sigmoid(l.Nodes[l1].Bias) * nodeError
	}
	return newErrors
}

func (n *Network) Calculate(input ...float64) []float64 {
	output := input
	for _, l := range n.Layers {
		output = l.Calculate(output...)
	}
	return output
}

func (n *Network) Train(train float64, input []float64, target []float64) float64 {
	ins := [][]float64{}
	curval := input

	for l1 := range n.Layers {
		ins = append(ins, curval)
		curval = n.Layers[l1].Calculate(curval...)
	}

	errors := make([]float64, len(target))
	for l1 := range target {
		errors[l1] = target[l1] - curval[l1]
	}

	for l1 := range n.Layers {
		l1 = len(n.Layers) - 1 - l1
		errors = n.Layers[l1].BackProp(train, ins[l1], errors)
	}
	return 0
}

func Generate(input int, layers ...int) *Network {
	n := &Network{Layers: make([]Layer, len(layers))}
	priorCount := input
	for l1, c := range layers {
		n.Layers[l1].Nodes = make([]Node, c)
		for l2 := range n.Layers[l1].Nodes {
			n.Layers[l1].Nodes[l2].Weights = make([]float64, priorCount)
			for l3 := range n.Layers[l1].Nodes[l2].Weights {
				n.Layers[l1].Nodes[l2].Weights[l3] = rand.Float64()*2 - 1
			}
			n.Layers[l1].Nodes[l2].Bias = rand.Float64()*2 - 1
		}
		priorCount = c
	}
	return n
}
