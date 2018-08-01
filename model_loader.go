package main

import (
	"archive/zip"
	"bufio"
	"fmt"
	"image"
	"image/jpeg"
	"io"
	"io/ioutil"
	"log"
	"math"
	"net/http"
	"os"
	"path/filepath"

	"github.com/nfnt/resize"

	tf "github.com/tensorflow/tensorflow/tensorflow/go"
)

var (
	graph  *tf.Graph
	labels []string
)

func errReporter(err error) {
	if err != nil {
		panic(err)
	}
}

// Unzip unzips an archive given by the path src path and extract it to the dst path
func Unzip(src string, dest string) ([]string, error) {

	var filenames []string

	r, err := zip.OpenReader(src)
	if err != nil {
		return filenames, err
	}
	defer r.Close()

	for _, f := range r.File {

		rc, err := f.Open()
		if err != nil {
			return filenames, err
		}
		defer rc.Close()

		// Store filename/path for returning and using later on
		fpath := filepath.Join(dest, f.Name)
		filenames = append(filenames, fpath)

		if f.FileInfo().IsDir() {

			// Make Folder
			os.MkdirAll(fpath, os.ModePerm)

		} else {

			// Make File
			if err = os.MkdirAll(filepath.Dir(fpath), os.ModePerm); err != nil {
				return filenames, err
			}

			outFile, err := os.OpenFile(fpath, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, f.Mode())
			if err != nil {
				return filenames, err
			}

			_, err = io.Copy(outFile, rc)

			// Close the file without defer to close before next iteration of loop
			outFile.Close()

			if err != nil {
				return filenames, err
			}

		}
	}
	return filenames, nil
}

// MaybeDownloadModel downloads a model from the link provided and save it to the path given
func MaybeDownloadModel(filepath string, url string) error {
	if _, err := os.Stat(filepath); os.IsNotExist(err) {
		// Create the file
		out, err := os.Create(filepath)
		if err != nil {
			return err
		}
		defer out.Close()
		// Get the data
		resp, err := http.Get(url)
		if err != nil {
			return err
		}
		defer resp.Body.Close()
		// Write the body to file
		_, err = io.Copy(out, resp.Body)
		if err != nil {
			return err
		}
	}

	if _, err := os.Stat(filepath + "tensorflow_inception_graph.pb"); os.IsNotExist(err) {

		if _, err := os.Stat("inception.zip"); os.IsNotExist(err) {
			// out, err := os.OpenFile(filepath, os.O_RDWR|os.O_CREATE, 0755)
			// if err != nil {
			// 	log.Fatal(err)

			// }

			// Create the file
			out, err := os.Create(filepath + "tensorflow_inception_graph.pb")
			if err != nil {
				return err
			}
			defer out.Close()

			// Get the data
			resp, err := http.Get(url)
			if err != nil {
				return err
			}
			defer resp.Body.Close()

			// Write the body to file
			_, err = io.Copy(out, resp.Body)
			if err != nil {
				return err
			}
		}
		// fmt.Println("seven")
		// files, err := Unzip(filepath+"inception.zip", filepath)
		// if err != nil {
		// 	print(err)
		// 	panic("error unzipping the model file")

		// }

		// fmt.Println(" Model file " + strings.Join(files, "\n") + "unzipped correctly\n")
	}

	return nil
}

func init() {
	// damn important or else At(), Bounds() functions will
	// caused memory pointer error!!
	image.RegisterFormat("jpeg", "jpeg", jpeg.Decode, jpeg.DecodeConfig)
}

func rgbaToPixel(r uint32, g uint32, b uint32, a uint32) Pixel {
	return Pixel{float32(r / 257), float32(g / 257), float32(b / 257)}
}

// Pixel is a container of image type ready to be converted into a image tensor
type Pixel struct {
	R float32
	G float32
	B float32
}

// AddValue is the overloading operator add for the Pixel struct
func (a *Pixel) AddValue(v *Pixel) {
	a.R += v.R
	a.G += v.G
	a.B += v.B
}

// ScalDivide is the overloading operator divide by scalar for the Pixel struct
func (a *Pixel) ScalDivide(v *float32) {
	a.R = a.R / *v
	a.G = a.G / *v
	a.B = a.B / *v
}

// PixelSubstruct is the overloading operator substruction by pixel for the Pixel struct
func (A *Pixels) PixelSubstruct(v *Pixel) {
	for i, _ := range *A {
		for j, _ := range (*A)[i] {
			(*A)[i][j].R -= v.R
			(*A)[i][j].G -= v.G
			(*A)[i][j].B -= v.B
		}
	}
}

// PixelDivide is the overloading operator divide by pixel for the Pixel struct
func (A *Pixels) PixelDivide(v *int) {
	for i, _ := range *A {
		for j, _ := range (*A)[i] {
			(*A)[i][j].R = (*A)[i][j].R / float32(*v)
			(*A)[i][j].G = (*A)[i][j].G / float32(*v)
			(*A)[i][j].B = (*A)[i][j].B / float32(*v)
		}
	}

}

// Round is the overloading operator round by scalar for the Pixel struct
func (a *Pixel) Round() {
	a.R = float32(math.Round(float64(a.R)))
	a.G = float32(math.Round(float64(a.G)))
	a.B = float32(math.Round(float64(a.B)))
}

// Pixels is a container of image type ready to be converted into a image tensor
type Pixels [][]Pixel

// transformStructToSlice is the overloading operator round by scalar for the Pixel struct
func transformStructToSlice(batch []Pixels) [5][224][224][3]float32 {

	arrayBatch := [5][224][224][3]float32{}

	for i, a := range batch {
		for j, b := range a {
			for k, c := range b {
				arrayBatch[i][j][k][0] = float32(c.R)
				arrayBatch[i][j][k][1] = float32(c.G)
				arrayBatch[i][j][k][2] = float32(c.B)
			}
		}
	}

	return arrayBatch
}

func main() {

	// Data Collection Model, Label, Images
	// ///////////////////////////////////////////////////////////////////////////////////////////////
	inceptionV3Link := "https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip"
	imagePath := "/data/images/"
	modelPath := ""

	// check if the model path is set and if not download model Inception v3
	if len(os.Args) >= 3 {
		modelPath = os.Args[1]

	} else {

		modelPath = "/data/models/"
		err := MaybeDownloadModel(modelPath, inceptionV3Link)
		if err != nil {
			panic(err)
		}
		// out, _ := exec.Command("ls").Output()
		// fmt.Println("this is the ls out function : \n", string(out))
	}
	// //////////////////////////////////////////////////////////////////////////////////////////////

	// Load Image batch into tensor
	// //////////////////////////////////////////////////////////////////////////////////////////////
	// path to images folder for inference (should be given as a first argument when running)
	// imagesDirectoryPath := os.Args[2]
	// get files names in the inference directory
	files, err := ioutil.ReadDir(imagePath)
	if err != nil {
		log.Fatal(err)
	}
	var imageBatch []Pixels
	for _, file := range files {

		//  Load image and decode it
		imgIncoded, err := os.Open(imagePath + file.Name())
		if err != nil {
			panic(err)
		}
		defer imgIncoded.Close()

		bigimg, _, err := image.Decode(imgIncoded)
		if err != nil {
			panic(err)
		}

		img := resize.Resize(224, 224, bigimg, resize.Bilinear)

		// fmt.Println("Image format after decoding", reflect.TypeOf(img), img.At(0, 0))
		/////////////////////////////
		// fmt.Println("file name", file.Name(), "image size", img.Bounds())
		// normalize image and resize it
		bounds := img.Bounds()
		width, height := bounds.Max.X, bounds.Max.Y
		pixel := Pixel{0, 0, 0}
		// fmt.Println("average pixel value before computations: ", pixel)
		var pixels Pixels
		counter := 0
		for y := 0; y < height; y++ {
			var row []Pixel
			for x := 0; x < width; x++ {
				row = append(row, rgbaToPixel(img.At(x, y).RGBA()))
				aux := rgbaToPixel(img.At(x, y).RGBA())
				pixel.AddValue(&aux)
				counter++
			}
			pixels = append(pixels, row)
		}
		// fmt.Println("average pixel value after computations: ", pixel)
		// fmt.Println("number of pixels counted :", counter, "theoretical pixel values : ", width*height)
		divider := float32(counter)
		pixel.ScalDivide(&divider)
		// fmt.Println("average pixel value after normalisation: ", pixel)
		pixel.Round()
		// fmt.Println("average pixel value after rounding: ", pixel)
		scale := 255
		// fmt.Println("pixels before:", pixels[0], len(pixels))
		// fmt.Println("pixel values: ", pixel)
		// fmt.Println("pixels slice type : ", reflect.TypeOf(pixels), "pixels values before substruction", pixels[0][0])
		pixels.PixelSubstruct(&pixel)
		// fmt.Println("pixels after substruction ", pixels[0][0])
		pixels.PixelDivide(&scale)
		// fmt.Println("pixels after division: ", pixels[0][0])

		// fmt.Println(pixel.Divide(&divider), counter)
		// meanValue := stat.Mean(pixels, nil)

		//////////////////////////////////

		// byteimage := [][]byte(fmt.Sprintf("%v", pixels))
		// var imgBuf, tensorBuf bytes.Buffer

		imageBatch = append(imageBatch, pixels)
	}

	fmt.Println("1", len(imageBatch), "2", len(imageBatch[0]), "3", len(imageBatch[0][0]))

	imBatchSlice := transformStructToSlice(imageBatch)
	// aa := []byte(fmt.Sprintf("%v", imageBatch))

	// m1 := [2][224][224][3]float32{}
	// for p := 0; p < 2; p++ {
	// 	for i := 0; i < 128; i++ {
	// 		for j := 0; j < 128; j++ {
	// 			for q := 0; q < 3; q++ {
	// 				intVAr := rand.Intn(255)
	// 				floatVar := float32(intVAr)
	// 				m1[p][i][j][q] = floatVar
	// 			}
	// 		}
	// 	}
	// }

	// fmt.Println(m1)
	tensor, err := tf.NewTensor(imBatchSlice)
	// tensor, err := tf.NewTensor(fmt.Sprintf("%v", imageBatch))
	if err != nil {
		fmt.Print("Tensor Conversion Error : ")
		panic(err)
	}

	// //////////////////////////////////////////////////////////////////////////////////////////////////

	// // load model file
	///////////////////////////////////////////////////////////////////////////////////////////////////////
	// modelFile, err := os.Open("/model/tensorflow_inception_graph.pb")
	// if err != nil {
	// 	panic("Model not loaded correctly")
	// }
	// defer modelFile.Close()
	// load model file
	// fmt.Println(modelPath + "/tensorflow_inception_graph.pb")
	model, err := ioutil.ReadFile(modelPath + "tensorflow_inception_graph.pb")
	if err != nil {
		log.Fatal(err)
	}
	// Construct the graph from the model
	graph = tf.NewGraph()
	err = graph.Import(model, "")
	if err != nil {
		panic(err)
	}
	///////////////////////////////////////////////////////////////////////////////////////////////////////

	// Load labels file
	///////////////////////////////////////////////////////////////////////////////////////////////////////
	labelsFile, err := os.Open(modelPath + "imagenet_comp_graph_label_strings.txt")
	if err != nil {
		panic(err)
	}
	defer labelsFile.Close()

	scanner := bufio.NewScanner(labelsFile)
	// Labels are separated by newlines
	for scanner.Scan() {
		labels = append(labels, scanner.Text())
	}
	if err := scanner.Err(); err != nil {
		panic(err)
	}
	////////////////////////////////////////////////////////////////////////////////////////////////////////

	// Run inference
	////////////////////////////////////////////////////////////////////////////////////////////////////////
	// Construct a graph with an operation that produces a string constant.

	session, err := tf.NewSession(graph, nil)
	if err != nil {
		log.Fatal(err)
	}
	defer session.Close()
	// fmt.Println(map[tf.Output]*tf.Tensor{graph.Operation("input").Output(0): tensor})

	output, err := session.Run(
		map[tf.Output]*tf.Tensor{
			graph.Operation("input").Output(0): tensor,
		},
		[]tf.Output{
			graph.Operation("output").Output(0),
		},
		nil)
	if err != nil {
		log.Fatal(err)
	}
	// output[0].Value() is a vector containing probabilities of
	// labels for each image in the "batch". The batch size was 1.
	// Find the most probably label index.

	probabilities := output[0].Value().([][]float32)[0]
	printBestLabel(probabilities, modelPath+"imagenet_comp_graph_label_strings.txt")

	// fmt.Println(probabilities)
	// printBestLabel(probabilities, labelsfile)
	////////////////////////////////////////////////////////////////////////////////////////////////////////
	// fmt.Print(map[tf.Output]*tf.Tensor{graph.Operation("input").Output(0): tensor})
	// output, err := session.Run( map[tf.Output]*tf.Tensor{ graph.Operation("input").Output(0): tensor,}, []tf.Output{ graph.Operation("output").Output(0),},nil)
}

func printBestLabel(probabilities []float32, labelsFile string) {
	bestIdx := 0
	for i, p := range probabilities {
		if p > probabilities[bestIdx] {
			bestIdx = i
		}
	}
	// Found the best match. Read the string from labelsFile, which
	// contains one line per label.
	file, err := os.Open(labelsFile)
	if err != nil {
		log.Fatal(err)
	}
	defer file.Close()
	scanner := bufio.NewScanner(file)
	var labels []string
	for scanner.Scan() {
		labels = append(labels, scanner.Text())
	}
	if err := scanner.Err(); err != nil {
		log.Printf("ERROR: failed to read %s: %v", labelsFile, err)
	}
	fmt.Printf("BEST MATCH: (%2.0f%% likely) %s\n", probabilities[bestIdx]*100.0, labels[bestIdx])
}
