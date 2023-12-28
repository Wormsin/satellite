package main

import (
	//"bufio"
	"encoding/binary"
	
	"fmt"
	"image"
	"image/color"
	"image/jpeg"
	"os"
	//"golang.org/x/text/encoding/charmap"
)



func main() {
	file, err := os.Open("../file.L15")
    if err != nil {
        panic(err)
        }
    defer file.Close()

	offset := int64(1568+72+292+676+17152+35024+205864+1600+176+4+4+4+4+4+4+4+4)
	_, err = file.Seek(offset, 0)
	if err != nil {
        panic(err)
        }
	buffer := make([]byte, 11136*11136*2)
	_, err = file.Read(buffer)
	if err != nil{
		panic(err)
	}
	
	//teg := binary.LittleEndian.Uint32(buffer)
	//fmt.Println("teg: ", teg)
	outputImagePath := "output_image2.jpg"

	// Create a grayscale image with the specified dimensions
	img := image.NewGray(image.Rect(0, 0, 11136, 11136))
   
	
	for y := 0; y < 11136; y++ {
	 index := y*11136*2
	 for x := 0; x < 11136; x++ {
	  
	//index := y*11136+x
	  //pixelValue := binaryToUint16(buffer[index : index+2])
	  //pixelValue := binary.LittleEndian.Uint16(buffer[index : index+2])
	  pixelValue := binary.LittleEndian.Uint16(buffer[index : index+2])
	  //pixelValue := buffer[index]
		index = index+2
	
   	//img.Set(x, y, color.Gray16{Y: pixelValue})
   img.SetGray(x, y, color.Gray{Y: uint8(pixelValue)})
	 }
	}
   
	// Create a new JPEG file
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
func binaryToUint16(bytes []byte) uint16 {
	return uint16(bytes[0]) | (uint16(bytes[1]) << 8)
   }

