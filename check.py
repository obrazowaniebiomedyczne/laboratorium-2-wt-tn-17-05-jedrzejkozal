from solution import *
from obpng import read_png, write_png


print("- Ocena dostateczna")
renew_pictures()

print("- Ocena dobra")
#image = read_png("figures/crushed.png")
image = cv2.imread('figures/crushed.png')
own_eros = own_simple_erosion(image)
write_png(own_eros, "results/own_simple_erosion.png")
cv2.imwrite("results/own_simple_erosion1.png", own_eros)


print("- Ocena bardzo dobra")
#image = read_png("figures/crushed.png")
image = cv2.imread('figures/crushed.png')
kernel = np.array([[0, 1, 1, 1, 0],
                   [0, 1, 1, 1, 0],
                   [1, 1, 1, 1, 1],
                   [0, 1, 1, 1, 0],
                   [0, 1, 1, 1, 0]])
erosion = own_erosion(image, kernel)
write_png(erosion, "results/own_erosion.png")

#image = read_png("figures/crushed.png")
image = cv2.imread('figures/crushed.png')
kernel = np.zeros((3,3),np.uint8)
kernel[:,1] = 1
kernel[1,:] = 1
erosion = cv2.erode(image, kernel, iterations = 1)
write_png(erosion, "results/erosion_comparison.png")
cv2.imwrite("results/erosion_comparison1.png", erosion)
