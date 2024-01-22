package main

import (
	"encoding/binary"
	"flag"
	"fmt"
	"image"
	"image/color"
	"image/jpeg"
	"os"
	"path/filepath"
)

func read4bytes(offset int64, file *os.File) uint32 {
	_, err := file.Seek(offset, 0)
	if err != nil {
		panic(err)
	}
	buffer := make([]byte, 4)
	_, err = file.Read(buffer)
	if err != nil {
		panic(err)
	}
	result := binary.LittleEndian.Uint32(buffer)
	return result
}

func image_search(file *os.File, offset int64) (width uint32, height uint32, bitPerPix uint32, x int64) {

	for {
		teg := read4bytes(offset, file)
		lenght := read4bytes(offset+4, file)

		//fmt.Println("teg: ", teg)
		//fmt.Println("lenght: ", lenght)

		if teg == 7 {
			offset = offset + 20
			width = read4bytes(offset, file)
			height = read4bytes(offset+4, file)
			bitPerPix = read4bytes(offset+8, file)
			//AttributePresencelnformation := read4bytes(offset-4, file)
			//fmt.Println("Quality of an image: ", AttributePresencelnformation)
			//fmt.Println("IMAGE: ", width, height, bitPerPix)

			offset = offset + 12
			return width, height, bitPerPix, offset
		}
		offset = offset + int64(lenght)
	}
}

func main() {

	source := flag.String("source", "L15", "Specify the source")
	save := flag.String("save", "images", "Specify the images folder")
	flag.Parse()

	dir_path := *source
	outputImageFolder := *save

	_, err := os.Stat(outputImageFolder)
	if os.IsNotExist(err) {
		// The directory does not exist
		if err := os.Mkdir(outputImageFolder, os.ModePerm); err != nil {
			panic(err)
		}
	} else if err != nil {
		// An error occurred (other than the directory not existing)
		fmt.Println("Error:", err)
	}

	dir, err := os.Open(dir_path)
	if err != nil {
		fmt.Println("Error opening directory:", err)
		return
	}
	files, err := dir.ReadDir(0)
	if err != nil {
		fmt.Println("Error reading directory:", err)
		return
	}
	defer dir.Close()

	for _, file_info := range files {
		filePath := filepath.Join(dir_path, file_info.Name())
		file, err := os.Open(filePath)
		if err != nil {
			panic(err)
		}
		defer file.Close()
		//header
		offset := int64(20)
		buffer := read4bytes(offset, file)
		offset = int64(buffer)

		width, height, bitPerPix, offset := image_search(file, offset)
		bitPerPix = bitPerPix / 8

		if width > 0 && height > 0 {
			_, err = file.Seek(offset, 0)
			if err != nil {
				panic(err)
			}
			img_buffer := make([]byte, width*height*bitPerPix)
			_, err = file.Read(img_buffer)
			if err != nil {
				panic(err)
			}

			img := image.NewGray(image.Rect(0, 0, int(width), int(height)))

			for y := 0; y < int(height); y++ {
				index := y * int(height) * int(bitPerPix)
				for x := 0; x < int(width); x++ {

					pixelValue := binary.LittleEndian.Uint16(img_buffer[index : index+int(bitPerPix)])
					index = index + int(bitPerPix)

					img.SetGray(x, y, color.Gray{Y: uint8(pixelValue)})
				}
			}
			outputImagePath := filepath.Join(outputImageFolder, file_info.Name()[0:len(file_info.Name())-3]+"jpg")
			outputFile, err := os.Create(outputImagePath)
			if err != nil {
				fmt.Println("Error creating output file:", err)
				return
			}
			defer outputFile.Close()

			// Encode the image as JPEG and write to the file
			err = jpeg.Encode(outputFile, img, nil)
			if err != nil {
				fmt.Println("Error encoding and writing image:", err)
				return
			}

			fmt.Println("Image successfully saved to", outputImagePath)

		}

	}
}
