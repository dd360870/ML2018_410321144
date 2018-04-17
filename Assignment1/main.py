from PIL import Image
import numpy
import random

I = Image.open("C:\\Users\\Ruzy\\Downloads\\Image_and_ImageData\\I.png")
K1 = Image.open("C:\\Users\\Ruzy\\Downloads\\Image_and_ImageData\\key1.png")
K2 = Image.open("C:\\Users\\Ruzy\\Downloads\\Image_and_ImageData\\key2.png")
E = Image.open("C:\\Users\\Ruzy\\Downloads\\Image_and_ImageData\\E.png")
Eprime = Image.open("C:\\Users\\Ruzy\\Downloads\\Image_and_ImageData\\Eprime.png")
width = E.size[0]
height = E.size[1]

output = Image.new("L", (width, height), 0)


def run():
    w = [random.random(), random.random(), random.random()]
    print(w)
    learning_rate = 0.00000001
    epoch = 1
    while epoch == 1 or (epoch < 20):
        for k in range(0, width * height):
            hypothesis = w[0] * K1.getpixel((k / height, k % height)) +\
                         w[1] * K2.getpixel((k / height, k % height)) +\
                         w[2] * I.getpixel((k / height, k % height))
            error = E.getpixel((k / height, k % height)) - hypothesis
            w[0] += learning_rate * error * K1.getpixel((k / height, k % height))
            w[1] += learning_rate * error * K2.getpixel((k / height, k % height))
            w[2] += learning_rate * error * E.getpixel((k / height, k % height))
        epoch += 1
        hypothesis = w[0] * K1.getpixel((k / height, k % height)) +\
                     w[1] * K2.getpixel((k / height, k % height)) +\
                     w[2] * I.getpixel((k / height, k % height))
        error = E.getpixel((k / height, k % height)) - hypothesis
        print("error"+str(error))
        print(epoch)
    print(w)
    for i in range(0, width):
        for j in range(0, height):
            output.putpixel((i, j), int(round((Eprime.getpixel((i, j)) - w[0] * K1.getpixel((i, j)) - w[1] * K2.getpixel((i, j))) / w[2])))
    output.show()


run()
