import cv2
import numpy


def image_filtering(img, kernel) -> numpy.array:
    length = len(img.shape)
    if length == 2:
        h, w = img.shape
        image_color = numpy.array([[[0]] * w] * h)
        image_color[:, :, 0] = img[:, :]
        return image_filtering_color(image_color, kernel)[:, :, 0]
    elif length == 3:
        return image_filtering_color(img, kernel)


def image_filtering_color(img, kernel) -> numpy.array:
    kernel_length = kernel.shape[0]
    kernel_center = kernel_length // 2

    matrix_filing = filing(img, kernel_center)
    height, width, color = matrix_filing.shape
    image_result = matrix_filing.copy()         # sigma

    for h in range(kernel_center, height - kernel_center):
        for w in range(kernel_center, width - kernel_center):
            for c in range(0, color):
                start_height = h - kernel_center
                finish_height = start_height + kernel_length
                start_width = w - kernel_center
                finish_width = start_width + kernel_length

                mult_matrix = matrix_filing[start_height:finish_height, start_width:finish_width, c] * kernel
                kernel_sum = numpy.sum(kernel)
                if kernel_sum != 0:
                    result = int(numpy.sum(mult_matrix) / numpy.sum(kernel))
                else:
                    result = numpy.sum(mult_matrix)

                if result < 0:
                    image_result[h, w, c] = 0
                elif result > 255:
                    image_result[h, w, c] = 255
                else:
                    image_result[h, w, c] = result
    return emarginate(image_result, kernel_center).astype(numpy.uint8)


def filing(img, number) -> numpy.array:
    height, width, color = img.shape
    matrix_w = width + number * 2
    matrix_h = height + number * 2
    matrix = numpy.array([[[0] * color] * matrix_w] * matrix_h)

    for c in range(0, color):
        for h in range(0, height):
            for w in range(0, width):
                matrix[h + number, w + number, c] = img[h, w, c]

    # filing angel
    for c in range(0, color):
        for h in range(0, number):
            for w in range(0, number):
                matrix[h, w, c] = img[number - h, number - w, c]

    for c in range(0, color):
        for h in range(0, number):
            for w in range(0, number):
                matrix[h, matrix_w - w - 1, c] = img[number - h, width - number + w - 1, c]

    for c in range(0, color):
        for h in range(0, number):
            for w in range(0, number):
                matrix[matrix_h - h - 1, w, c] = img[height - number + h - 1, number - w, c]

    for c in range(0, color):
        for h in range(0, number):
            for w in range(0, number):
                matrix[matrix_h - h - 1, matrix_w - w - 1, c] = img[height - number + h - 1, width - number + w - 1, c]

    # filling edge
    for c in range(0, color):
        for h in range(0, number):
            for w in range(0, width):
                matrix[h, number + w, c] = img[number - h, w, c]

    for c in range(0, color):
        for h in range(0, number):
            for w in range(0, width):
                matrix[matrix_h - h - 1, number + w, c] = img[height - number + h - 1, w, c]

    for c in range(0, color):
        for h in range(0, height):
            for w in range(0, number):
                matrix[number + h, w, c] = img[h, number - w, c]

    for c in range(0, color):
        for h in range(0, height):
            for w in range(0, number):
                matrix[number + h, matrix_w - w - 1, c] = img[h, width - number + w - 1, c]
    return matrix


def emarginate(img, number) -> numpy.array:
    return img[number:-number, number: -number, :]


if __name__ == '__main__':
    # kernel = numpy.array([[0, -1, 0],
    #                       [-1, 5, -1],
    #                       [0, -1, 0]])
    # kernel = numpy.array([[0, -1, 0],
    #                       [-1, 4, -1],
    #                       [0, -1, 0]])
    # kernel = numpy.array([[1, 1, 1],
    #                       [1, 1, 1],
    #                       [1, 1, 1]]) / 9
    # kernel = numpy.array([[-1, -1, -1],
    #                       [-1, 8, -1],
    #                       [-1, -1, -1]])
    # kernel = numpy.array([[-2, -1, 0],
    #                       [-1, 1, 1],
    #                       [0, 1, 2]])
    # kernel = numpy.array([[-1, 0, 1],
    #                       [-2, 0, 2],
    #                       [-1, 0, 1]])
    kernel = numpy.array([[0, 0, -1, 0, 0],
                          [0, 0, -1, 0, 0],
                          [-1, -1, 8, -1, -1],
                          [0, 0, -1, 0, 0],
                          [0, 0, -1, 0, 0]])

    # image_color = cv2.imread("../resources/1.png")
    image_color = cv2.imread("../resources/poly.jpg")
    image_gray = cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY)

    center_kernel = len(kernel)//2

    image_filtering = image_filtering(image_gray, kernel)
    image_openCV = cv2.filter2D(src=image_gray, ddepth=-1, kernel=kernel)

    # cv2.imshow("Color: image", image_color)
    cv2.imshow("Color: image openCV", image_openCV)
    cv2.imshow("Color: image filtering", image_filtering)

    cv2.waitKey()



