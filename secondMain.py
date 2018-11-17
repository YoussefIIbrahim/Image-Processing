import cv2
import numpy as np
import math
import copy
from matplotlib import pyplot as plt


def grayscale(image):
    new_image = np.zeros((len(image), len(image[0])), dtype=np.uint8)
    for i in range(len(image)):
        for j in range(len(image[i])):
            red = image[i][j][0]
            green = image[i][j][1]
            blue = image[i][j][2]
            new_image[i][j] = red*0.3 + green*0.59 + blue*0.11

    return new_image


def inversion(image):
    for x in range(len(image)):
        for y in range(len(image[x])):
            image[x][y][0] = 255 - image[x][y][0]
            image[x][y][1] = 255 - image[x][y][1]
            image[x][y][2] = 255 - image[x][y][2]
    return image


def truncate(value):

    if value < 0:
        value = 0

    if value > 255:
        value = 255

    return value


def brightness(image, const):
    for x in range(len(image)):
        for y in range(len(image[x])):
            image[x][y][0] = truncate(image[x][y][0] + const)
            image[x][y][1] = truncate(image[x][y][1] + const)
            image[x][y][2] = truncate(image[x][y][2] + const)
    return image


def contrast(image, const):
    new_image = np.zeros((len(image), len(image[0]), 3), dtype=np.uint8)
    fact = (259 * (const + 255)) / (255 * (259 - const))
    print(fact)
    for x in range(len(image)):
        for y in range(len(image[x])):
            new_image[x][y][0] = truncate((fact * (image[x][y][0] - 128)) + 128)
            new_image[x][y][1] = truncate((fact * (image[x][y][1] - 128)) + 128)
            new_image[x][y][2] = truncate((fact * (image[x][y][2] - 128)) + 128)
    return new_image


def binarization1(image, thresholding):
    new_image = np.zeros((len(image), len(image[0])), dtype=np.uint8)
    for x in range(len(image)):
        for y in range(len(image[x])):
            if image[x][y] <= thresholding:
                new_image[x][y] = 0
            else:
                new_image[x][y] = 255
    return new_image


def binarization2(image, thresholding):
    new_image = np.zeros((len(image), len(image[0])), dtype=np.uint8)
    for x in range(len(image)):
        for y in range(len(image[x])):
            if image[x][y] <= thresholding:
                new_image[x][y] = 0
            else:
                new_image[x][y] = image[x][y]
    return new_image


def bernsen(image, radius):
    blank_image = np.zeros((len(image), len(image[0])))
    for i in range(len(image)):
        for j in range(len(image[i])):
            left = max(0, j - radius)
            right = min(len(image[i]), j + radius)
            top = max(0, i - radius)
            bottom = min(len(image), i + radius)

            min_val = 255
            max_val = 0

            for x in range(top, bottom):
                for y in range(left, right):
                    min_val = min(min_val, image[x][y])
                    max_val = max(max_val, image[x][y])

            rad = (int(max_val) + int(min_val)) / 2
            if image[i][j] <= rad:
                blank_image[i][j] = 0
            else:
                blank_image[i][j] = 255
    return blank_image


def graphs(image):
    plot_horizontal = np.zeros(len(image))
    plot_vertical = np.zeros(len(image[0]))
    for i in range(len(image)):
        for j in range(len(image[i])):
            if image[i][j] == 0:
                plot_horizontal[i] += 1
                plot_vertical[j] += 1
    plt.subplot(2, 1, 1), plt.plot(plot_horizontal)
    plt.title('Horizontal projection'), plt.xticks([]), plt.yticks([])
    plt.subplot(2, 1, 2), plt.plot(plot_vertical)
    plt.title('Vertical projection'), plt.xticks([]), plt.yticks([])
    plt.show()


def getmin(image):
    min_value = 255;
    for i in range(len(image)):
        for j in range(len(image[i])):
            if image[i][j]<min_value:
                min_value = image[i][j]
    return min_value


def getmax(image):
    max_value = 0;
    for i in range(len(image)):
        for j in range(len(image[i])):
            if image[i][j]>max_value:
                max_value = image[i][j]
    return max_value


def normalize(image):
    orig = np.zeros(256)
    modified = np.zeros(256)
    new_image = np.zeros((len(image), len(image[0])), dtype=np.uint8)
    min = getmin(image)
    max = getmax(image)

    # histogram for original image
    for i in range(len(image)):
        for j in range(len(image[i])):
            val = image[i][j]
            orig[val] = orig[val]+1

    # normalization of image
    for i in range(len(image)):
        for j in range(len(image[i])):
            new_image[i][j] = ((image[i][j] - min)/(max - min)) * 255

    # histogram for normalized image
    for i in range(len(new_image)):
        for j in range(len(new_image[i])):
            val = new_image[i][j]
            modified[val] = modified[val]+1

    plt.subplot(2, 1, 1), plt.plot(orig)
    plt.title('Original histogram'), plt.xticks([]), plt.yticks([])
    plt.subplot(2, 1, 2), plt.plot(modified)
    plt.title('Equalized histogram'), plt.xticks([]), plt.yticks([])
    plt.show()
    return new_image


def gaussianfilter(image):
    new_image = np.zeros((len(image), len(image[0])), dtype=np.uint8)
    filtering = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]])
    for x in range(len(image)):
        for y in range(len(image[x])):

            if x-1 < 0:

                left = image[x + 1][y]
                right = image[x + 1][y]

                if y-1 < 0:
                    left_down = image[x+1][y+1]
                    right_up = image[x+1][y+1]
                    up = image[x][y+1]
                    down = image[x][y+1]
                    right_down = image[x+1][y+1]
                    left_up = image[x+1][y+1]

                elif y+1 == len(image[x]):
                    left_up = image[x+1][y-1]
                    right_down = image[x+1][y-1]
                    down = image[x][y-1]
                    up = image[x][y-1]
                    left_down = image[x+1][y-1]
                    right_up = image[x+1][y-1]

                else:
                    left_up = image[x+1][y+1]
                    left_down = image[x+1][y-1]
                    up = image[x][y-1]
                    down = image[x][y+1]
                    right_up = image[x+1][y-1]
                    right_down = image[x+1][y+1]

            elif x+1 == len(image):

                right = image[x-1][y]
                left = image[x-1][y]

                if y-1 < 0:
                    right_down = image[x-1][y+1]
                    left_up = image[x-1][y+1]
                    up = image[x][y+1]
                    down = image[x][y+1]
                    right_up = image[x-1][y+1]
                    left_down = image[x-1][y+1]

                elif y+1 == len(image[x]):
                    left_down = image[x-1][y-1]
                    right_up = image[x-1][y-1]
                    down = image[x][y-1]
                    up = image[x][y-1]
                    left_up = image[x-1][y-1]
                    right_down = image[x-1][y-1]

                else:
                    up = image[x][y - 1]
                    down = image[x][y + 1]
                    left_up = image[x-1][y-1]
                    left_down = image[x-1][y+1]
                    right_up = image[x-1][y+1]
                    right_down = image[x-1][y-1]

            else:
                left = image[x - 1][y]
                right = image[x+1][y]

                if y-1 < 0:
                    up = image[x][y+1]
                    down = image[x][y+1]
                    left_up = image[x+1][y+1] # =right_down
                    right_up = image[x-1][y+1] # =left_down
                    left_down = image[x-1][y+1]
                    right_down = image[x+1][y+1]

                elif y+1 == len(image[x]):
                    down = image[x][y-1]
                    up = image[x][y-1]
                    left_down = image[x+1][y-1] # =right_up
                    right_down = image[x-1][y-1] # =left_up
                    left_up = image[x-1][y-1]
                    right_up = image[x+1][y-1]
                else:
                    up = image[x][y-1]
                    down = image[x][y+1]
                    left_up = image[x-1][y-1]
                    left_down = image[x-1][y+1]
                    right_up = image[x+1][y-1]
                    right_down = image[x+1][y+1]

            elems = np.array([[left_up, up, right_up], [left, image[x][y], right], [left_down, down, right_down]])
            new_image[x][y] = (sum(sum(filtering * elems))/16)
    return new_image


def sharpeningfilter(image):
    new_image = np.zeros((len(image), len(image[0])), dtype=np.uint8)
    filtering = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
    for x in range(len(image)):
        for y in range(len(image[x])):

            if x-1 < 0:

                left = image[x + 1][y]
                right = image[x + 1][y]

                if y-1 < 0:
                    left_down = image[x+1][y+1]
                    right_up = image[x+1][y+1]
                    up = image[x][y+1]
                    down = image[x][y+1]
                    right_down = image[x+1][y+1]
                    left_up = image[x+1][y+1]

                elif y+1 == len(image[x]):
                    left_up = image[x+1][y-1]
                    right_down = image[x+1][y-1]
                    down = image[x][y-1]
                    up = image[x][y-1]
                    left_down = image[x+1][y-1]
                    right_up = image[x+1][y-1]

                else:
                    left_up = image[x+1][y+1]
                    left_down = image[x+1][y-1]
                    up = image[x][y-1]
                    down = image[x][y+1]
                    right_up = image[x+1][y-1]
                    right_down = image[x+1][y+1]

            elif x+1 == len(image):

                right = image[x-1][y]
                left = image[x-1][y]

                if y-1 < 0:
                    right_down = image[x-1][y+1]
                    left_up = image[x-1][y+1]
                    up = image[x][y+1]
                    down = image[x][y+1]
                    right_up = image[x-1][y+1]
                    left_down = image[x-1][y+1]

                elif y+1 == len(image[x]):
                    left_down = image[x-1][y-1]
                    right_up = image[x-1][y-1]
                    down = image[x][y-1]
                    up = image[x][y-1]
                    left_up = image[x-1][y-1]
                    right_down = image[x-1][y-1]

                else:
                    up = image[x][y - 1]
                    down = image[x][y + 1]
                    left_up = image[x-1][y-1]
                    left_down = image[x-1][y+1]
                    right_up = image[x-1][y+1]
                    right_down = image[x-1][y-1]

            else:
                left = image[x - 1][y]
                right = image[x+1][y]

                if y-1 < 0:
                    up = image[x][y+1]
                    down = image[x][y+1]
                    left_up = image[x+1][y+1] # =right_down
                    right_up = image[x-1][y+1] # =left_down
                    left_down = image[x-1][y+1]
                    right_down = image[x+1][y+1]

                elif y+1 == len(image[x]):
                    down = image[x][y-1]
                    up = image[x][y-1]
                    left_down = image[x+1][y-1] # =right_up
                    right_down = image[x-1][y-1] # =left_up
                    left_up = image[x-1][y-1]
                    right_up = image[x+1][y-1]
                else:
                    up = image[x][y-1]
                    down = image[x][y+1]
                    left_up = image[x-1][y-1]
                    left_down = image[x-1][y+1]
                    right_up = image[x+1][y-1]
                    right_down = image[x+1][y+1]

            elems = np.array([[left_up, up, right_up], [left, image[x][y], right], [left_down, down, right_down]])
            new_image[x][y] = image[x][y] + (sum(sum(filtering * elems))/16)
    return new_image


def sobelfilter(image):

    gx_filtering = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    gy_filtering = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    blank_image = np.zeros((len(image), len(image[0])), dtype=np.uint8)

    for x in range(len(image)):
        for y in range(len(image[x])):

            if x-1 < 0:

                left = image[x + 1][y]
                right = image[x + 1][y]

                if y-1 < 0:
                    left_down = image[x+1][y+1]
                    right_up = image[x+1][y+1]
                    up = image[x][y+1]
                    down = image[x][y+1]
                    right_down = image[x+1][y+1]
                    left_up = image[x+1][y+1]

                elif y+1 == len(image[x]):
                    left_up = image[x+1][y-1]
                    right_down = image[x+1][y-1]
                    down = image[x][y-1]
                    up = image[x][y-1]
                    left_down = image[x+1][y-1]
                    right_up = image[x+1][y-1]

                else:
                    left_up = image[x+1][y+1]
                    left_down = image[x+1][y-1]
                    up = image[x][y-1]
                    down = image[x][y+1]
                    right_up = image[x+1][y-1]
                    right_down = image[x+1][y+1]

            elif x+1 == len(image):

                right = image[x-1][y]
                left = image[x-1][y]

                if y-1 < 0:
                    right_down = image[x-1][y+1]
                    left_up = image[x-1][y+1]
                    up = image[x][y+1]
                    down = image[x][y+1]
                    right_up = image[x-1][y+1]
                    left_down = image[x-1][y+1]

                elif y+1 == len(image[x]):
                    left_down = image[x-1][y-1]
                    right_up = image[x-1][y-1]
                    down = image[x][y-1]
                    up = image[x][y-1]
                    left_up = image[x-1][y-1]
                    right_down = image[x-1][y-1]

                else:
                    up = image[x][y - 1]
                    down = image[x][y + 1]
                    left_up = image[x-1][y-1]
                    left_down = image[x-1][y+1]
                    right_up = image[x-1][y+1]
                    right_down = image[x-1][y-1]

            else:
                left = image[x - 1][y]
                right = image[x+1][y]

                if y-1 < 0:
                    up = image[x][y+1]
                    down = image[x][y+1]
                    left_up = image[x+1][y+1] # =right_down
                    right_up = image[x-1][y+1] # =left_down
                    left_down = image[x-1][y+1]
                    right_down = image[x+1][y+1]

                elif y+1 == len(image[x]):
                    down = image[x][y-1]
                    up = image[x][y-1]
                    left_down = image[x+1][y-1] # =right_up
                    right_down = image[x-1][y-1] # =left_up
                    left_up = image[x-1][y-1]
                    right_up = image[x+1][y-1]
                else:
                    up = image[x][y-1]
                    down = image[x][y+1]
                    left_up = image[x-1][y-1]
                    left_down = image[x-1][y+1]
                    right_up = image[x+1][y-1]
                    right_down = image[x+1][y+1]

            elems = np.array([[left_up, up, right_up], [left, image[x][y], right], [left_down, down, right_down]])
            blank_image[x][y] = math.sqrt(
                int(np.sum(elems * gx_filtering)) ** 2 + int(np.sum(elems * gy_filtering)) ** 2)

    return blank_image


def robertscrossfilter(image):
    gx_filtering = np.array([[1, 0], [0, -1]])
    gy_filtering = np.array([[0, 1], [-1, 0]])
    new_image = np.zeros((len(image), len(image[0])), dtype=np.uint8)
    for x in range(len(image)):
        for y in range(len(image[x])):

            if x + 1 == len(image):

                right = image[x - 1][y]

                if y - 1 < 0:
                    right_down = image[x][y]
                    down = image[x][y + 1]

                elif y + 1 == len(image[x]):
                    down = image[x][y - 1]
                    right_down = image[x - 1][y - 1]

                else:
                    down = image[x][y + 1]
                    right_down = image[x - 1][y - 1]

            else:

                right = image[x + 1][y]

                if y + 1 == len(image[x]):

                    down = image[x][y-1]

                    if x + 1 == len(image):
                        right_down = image[x][y]

                    else:
                        right_down = image[x-1][y-1]

                else:
                    down = image[x][y + 1]
                    right_down = image[x + 1][y + 1]

            elems = np.array([[image[x][y], right], [down, right_down]])

            gx = (sum(sum(gx_filtering * elems)))
            gy = (sum(sum(gy_filtering * elems)))
            new_image[x][y] = math.sqrt(gx ** 2 + gy ** 2)

    return new_image


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# REPORT 2 MATERIAL
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def dilate(image, radius):
    radius = int(radius/2)
    blank_image = np.zeros((len(image), len(image[0])), dtype=np.uint8)
    for i in range(len(image)):
        for j in range(len(image[i])):
            left = max(0, j - radius)
            right = min(len(image[i]), j + radius+1)
            top = max(0, i - radius)
            bottom = min(len(image), i + radius+1)

            max_val = 255
            for x in range(top, bottom):
                for y in range(left, right):
                    # print(image[x][y])
                    max_val = min(max_val, image[x][y])
            # print("max", max_val)

            blank_image[i][j] = max_val

    return blank_image

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Find below the desired operation and uncomment it. DO NOT uncomment all at the same time as
# it will take FOREVER to compile. They are in the same order as the above methods.
# And I'm using the copy.deepcopy() to not change the input image (easier to achieve)
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


original_img = cv2.imread("eye.jpg")
#
grayscale_img = grayscale(copy.deepcopy(original_img))
#
# inversion_img = inversion(copy.deepcopy(original_img))
#
# brightness_imgp = brightness(copy.deepcopy(original_img), 50)
# brightness_imgn = brightness(copy.deepcopy(original_img), -50)
#
# contrast_imgp = contrast(copy.deepcopy(original_img), 100)
# contrast_imgn = contrast(copy.deepcopy(original_img), -100)
#
# binarization1_img = binarization1(copy.deepcopy(grayscale_img), 100)
#
binarization2_img = binarization2(copy.deepcopy(grayscale_img), 50)
#
# bernsen_img = bernsen(copy.deepcopy(grayscale_img), 10)
#
# graphs(copy.deepcopy(bernsen_img))
#
# normalized_img = normalize(copy.deepcopy(grayscale_img))
#
gaussian_img = gaussianfilter(copy.deepcopy(binarization2_img))
#
# sharpened_img = sharpeningfilter(copy.deepcopy(grayscale_img))
#
# sobel_img = sobelfilter(copy.deepcopy(grayscale_img))
#
# roberts_img = robertscrossfilter(copy.deepcopy(grayscale_img))
#
dilated_img = dilate(copy.deepcopy(grayscale_img), 3)
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Here is to show the output images, you may change the first image and the second as desired
# For example I left the following code to show the original image androbert cross filter image
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# plt.subplot(2, 1, 1), plt.imshow(grayscale_img, cmap="gray")
# plt.title('Original Image'), plt.xticks([]), plt.yticks([])
# plt.subplot(2, 1, 2), plt.imshow(roberts_img, cmap="gray")
# plt.title('Roberts cross filter'), plt.xticks([]), plt.yticks([])
# plt.show()


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Here you can also check the output image for each algorithm by just uncommenting the actual
# call of the method above and uncommenting the corresponding line below here
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# cv2.imshow("ORIGINAL", original_img)
cv2.imshow("GRAYSCALE", grayscale_img)
# cv2.imshow("INVERSION", inversion_img)
# cv2.imshow("BRIGHTNESS", brightness_img)
# cv2.imshow("CONTRAST", contrast_img)
cv2.imshow("BINARIZATION 1", binarization2_img)
# cv2.imshow("BINARIZATION 2", binarization2_img)
# cv2.imshow("BRENSEN", bernsen_img)
# cv2.imshow("NORMALIZED", normalized_img)
cv2.imshow("GAUSSIAN", gaussian_img)
# cv2.imshow("SHARPENED", sharpened_img)
# cv2.imshow("SOBEL", sobel_img)
# cv2.imshow("ROBERTS CROSS", roberts_img)
cv2.imshow("DILATED", dilated_img)
# # # # # # # # # # # # # # # # # # # # # # # # # # #
# YOU HAVE TO UNCOMMENT THE FOLLOWING TWO LINES IN
# ORDER FOR THE ABOVE LINES TO PRODUCE A WINDOW WITH
# THE OUTPUT IMAGE
# # # # # # # # # # # # # # # # # # # # # # # # # # #
cv2.waitKey(0)
cv2.destroyAllWindows()
