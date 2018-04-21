from PIL import Image
import matplotlib.pyplot as plt
import random

I = Image.open("C:\\Users\\Ruzy\\Downloads\\Image_and_ImageData\\I.png")
K1 = Image.open("C:\\Users\\Ruzy\\Downloads\\Image_and_ImageData\\key1.png")
K2 = Image.open("C:\\Users\\Ruzy\\Downloads\\Image_and_ImageData\\key2.png")
E = Image.open("C:\\Users\\Ruzy\\Downloads\\Image_and_ImageData\\E.png")
Eprime = Image.open("C:\\Users\\Ruzy\\Downloads\\Image_and_ImageData\\Eprime.png")
width = E.size[0]
height = E.size[1]

output = Image.new("L", (width, height), 0)


class AdalineGD(object):

    def __init__(self, eta=1e-8, epochs=10):
        self.eta = eta
        self.epochs = epochs
        # self.w = [random.random(), random.random(), random.random()]
        self.w = [1e3, 1e3, 1e3]
        self.cost = [self.error_average()]

    def train(self):
        print('train(eta='+str(self.eta)+', epochs='+str(self.epochs)+')', flush=True)
        epoch = 0
        while epoch < self.epochs or self.cost[epoch-1] > 0.1:
            for i in range(0, width):
                for j in range(0, height):
                    hypothesis = self.w[0]*K1.getpixel((i, j)) + \
                                  self.w[1]*K2.getpixel((i, j)) + \
                                  self.w[2]*I.getpixel((i, j))
                    error = (E.getpixel((i, j)) - hypothesis)
                    self.w[0] += self.eta * error * K1.getpixel((i, j))
                    self.w[1] += self.eta * error * K2.getpixel((i, j))
                    self.w[2] += self.eta * error * E.getpixel((i, j))
            self.cost.append(self.error_average())
            epoch += 1
            print('.', end='', flush=True)
        print('done', flush=True)
        print('Iterations='+str(epoch), flush=True)
        return self

    def error_average(self):
        total = 0
        for i in range(0, width):
            for j in range(0, height):
                hypothesis = self.w[0] * K1.getpixel((i, j)) + \
                             self.w[1] * K2.getpixel((i, j)) + \
                             self.w[2] * I.getpixel((i, j))
                error = (E.getpixel((i, j)) - hypothesis)
                total += ((error**2) / 2.0)
        return total/(400.0*300.0)


ada = AdalineGD(eta=1e-5).train()
plt.plot(range(1, len(ada.cost)), ada.cost[1:], marker='o')
plt.xlabel('Iterations')
plt.ylabel('Sum-squared-error')
plt.title('Learning rate 1e-5')
plt.show()
print(ada.w)
for i in range(0, width):
    for j in range(0, height):
        output.putpixel((i, j), int(round((Eprime.getpixel((i, j))-ada.w[0]*K1.getpixel((i, j))-ada.w[1]*K2.getpixel((i, j)))/ada.w[2])))
output.save("Eprime-decrypted.png")
