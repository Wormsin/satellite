package main

import (
	"bufio"
	"encoding/binary"
	"fmt"
	"os"

	"golang.org/x/text/encoding/charmap"
)

func main() {
    file, err := os.Open("seek/L15/file1.L15")
    if err != nil {
        panic(err)
        }
    defer file.Close()


    decoder := charmap.Windows1251.NewDecoder()
    reader := bufio.NewReader(file)

    line, _, err := reader.ReadLine()
    if err != nil {
        panic(err)
        }
    var length_bytes [4]byte
    var teg_bytes [4]byte
    var step int
    copy(length_bytes[:], line[20:20+4])
    length := binary.LittleEndian.Uint32(length_bytes[:])
    size := len(line)
    fmt.Println(length)

    decodeLine, err := decoder.String(string(line))
    fmt.Println(decodeLine)
    fmt.Println(107%10)
    for {

        size = size-int(length)

        for size <= 0{
            line, _, err = reader.ReadLine()
            if err != nil {
                panic(err)
            }
            length = uint32(-size)
            size = len(line)-int(length)
        }
        step =len(line)-size

        copy(teg_bytes[:], line[step:step+4])
        copy(length_bytes[:], line[step+4:step+8])

        teg := binary.LittleEndian.Uint32(teg_bytes[:])
        length = binary.LittleEndian.Uint32(length_bytes[:])

        
        
        //decodeLine, err := decoder.String(string(line))
        //if err != nil {
        //    panic(err)
        //    }

        if teg > 6{
            fmt.Println(line[:])

        }

        fmt.Println("teg:", teg)
        fmt.Println("lenght", length)
        //fmt.Println("First line:", decodeLine)
        var c int
        fmt.Println("Exit? ")
        fmt.Scanln(&c)
        if c == 1{
        break
        }
        //line, _, err := reader.ReadLine()
        //if err != nil {
        //    panic(err)
         //   }
    // if decodeLine == "UFDMSU-GSFILE" {
        //    fmt.Println("image teg")
        //   break
        //}
}
    //decodeLine, err := decoder.String(string(line[:4]))
    //if err != nil {
     //   panic(err)
      //  }
    //fmt.Println("First line:", decodeLine)
}