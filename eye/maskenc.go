package main

import (
	"encoding/csv"
	"fmt"
	"image"
	"image/png"
	"math"
	"os"
)

func isBlocked(mask image.Image, x int, y int) bool {
	r, g, b, a := mask.At(x, y).RGBA()
	intensity := ((r + g + b) / 3) * (a / 255)
	return intensity < (255 / 2)
}

func main() {
	// Read image from file
	mask, err := os.Open("mask.png")
	if err != nil {
		fmt.Fprintf(os.Stderr, "%s\n", err)
		return
	}
	defer mask.Close()

	maskData, err := png.Decode(mask)
	if err != nil {
		fmt.Fprintf(os.Stderr, "%s\n", err)
		return
	}
	maskSize := maskData.Bounds().Size()

	// Create occupancy matrix
	occupancy := make([][]float64, maskSize.X)
	for i := 0; i < maskSize.X; i++ {
		occupancy[i] = make([]float64, maskSize.Y)
		fmt.Printf("Encoding col: [%d/%d]\n", i+1, maskSize.X)
		for j := 0; j < maskSize.Y; j++ {
			var mindist float64 = float64(maskSize.X) * float64(maskSize.Y)
			if isBlocked(maskData, i, j) {
				occupancy[i][j] = 0
				continue
			}

			// Find available occupancy
			for u := 0; u < maskSize.X; u++ {
				for v := 0; v < maskSize.Y; v++ {
					if !isBlocked(maskData, u, v) {
						continue
					}

					var distance float64 = math.Sqrt(
						math.Pow(float64(u-i), 2) + math.Pow(float64(v-j), 2))
					if distance < mindist {
						mindist = distance
					}
				}
			}

			// Store
			occupancy[i][j] = mindist
		}
	}

	// Save out
	result, err := os.Create("maskOccupancy.csv")
	if err != nil {
		fmt.Fprintf(os.Stderr, "%s\n", err)
		return
	}
	defer result.Close()
	writer := csv.NewWriter(result)
	defer writer.Flush()

	for _, col := range occupancy {
		colStrings := make([]string, len(col))
		for i := range col {
			colStrings[i] = fmt.Sprintf("%f", col[i])
		}
		err := writer.Write(colStrings)
		if err != nil {
			fmt.Fprintf(os.Stderr, "%s\n", err)
			return
		}
	}
}
