import cv2 as cv
import numpy as np
import sys
import matplotlib.pyplot as plt
from skimage.segmentation import clear_border

def order_points(corners):
    """
    gives the 4 corners a fixed order from top left to bottom left
    input: 4 corners
    output: 4 corners
    """
    top_left = corners.sum(1).argmin()
    bottom_right = corners.sum(1).argmax()
    top_right = np.diff(corners).argmin()
    bottom_left = np.diff(corners).argmax()

    ordered = np.array([corners[top_left], corners[top_right], corners[bottom_left], corners[bottom_right]], dtype = "float32")

    return ordered


def transform(image, corners, squared=False):
    """
    returns only the highlighted part of an image (the corners)
    input:
    image: np.array,
    corners: 4 corners,
    squared: Boolean default=False
    
    If squared == True ignores proportions
    """
    crns = order_points(corners)
    tl, tr, bl, br = crns

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    width = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    height = max(int(heightA), int(heightB))

    if squared:
        size = max(height, width)
        new_crns = np.array([
            [0,0],
            [size-1, 0],
            [0, size-1],
            [size-1, size-1]
        ], dtype = "float32")
        M = cv.getPerspectiveTransform(crns, new_crns)
        warped = cv.warpPerspective(image, M, (size, size))

    else:
        new_crns = np.array([
            [0,0],
            [width-1, 0],
            [0, height-1],
            [width-1, height-1]
        ], dtype = "float32")
        M = cv.getPerspectiveTransform(crns, new_crns)
        warped = cv.warpPerspective(image, M, (width, height))

    return warped


def find_sudoku(img, kernel_size=7, canny_threshold=100, printer="nothing", squared=False):
    """
    finds the sudoku from a photo.
    input:
    img: np.array,
    kernel_size(for gaussian blur): int, default = 7,
    canny_threshold(for edge detection): int, default=100,
    printer(for visuals): string. options are:
    'img', 'gray', 'blurred', 'thresh', 'edges', 'cnts', 'outline', 'warped'
    squared: Boolean, return square or not. default= False
    """
    if img is None:
        sys.exit("Could not read the image.")
        pass
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    blurred = cv.GaussianBlur(gray, (kernel_size, kernel_size), cv.BORDER_DEFAULT)
    thresh = cv.adaptiveThreshold(blurred, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)
    thresh = cv.bitwise_not(thresh)
    edges = cv.Canny(img,canny_threshold,canny_threshold * 2)

    cnts, hierarchy = cv.findContours(thresh.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv.contourArea, reverse=True)

    puzzle = None
    for c in cnts:
    # approximate the contour
        peri = cv.arcLength(c, True)
        approx = cv.approxPolyDP(c, 0.02 * peri, True)
        # if our approximated contour has four points, then we can
        # assume we have found the outline of the puzzle
        
        if len(approx) == 4:
            puzzle = approx
            break
            
    if type(puzzle) == None:
        print("that didn't work")
        
    outline = img.copy()
    cv.drawContours(outline, [puzzle], -1, (0,255,0), 3)
    #### if we put [puzzle] we get the whole grid. Without it we only get the corners
    
    corners = puzzle.sum(1)
    warped = transform(img, corners, squared)

    if printer != "nothing":
        if printer in ["gray", "blurred", "thresh", "edges"]:
            plt.imshow(eval(printer), cmap="gray")
            plt.title(printer, size=20)
        else:
            plt.imshow(eval(printer))
            plt.title(printer, size=20)
        
    return warped
    


def get_digit(cell, border_size=5):
    """
    recognized digit from within a cell.
    input:
    cell: np.array of shape (28,28,1),
    border_size: int, default=5. Defines minimum border around the digit contours
    """
    gray = cv.cvtColor(cell,cv.COLOR_BGR2GRAY)
    thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)[1]
    thresh = clear_border(thresh)

    cnts, hierarchy = cv.findContours(thresh.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv.contourArea, reverse=True)

    if len(cnts) > 0:
        cnt = cnts[0]
        outline = cell.copy()
        digit = np.array([[[cnt[:,:,0].min()-border_size, cnt[:,:,1].min()-border_size]], [[cnt[:,:,0].max()+border_size, cnt[:,:,1].min()-border_size]], [[cnt[:,:,0].min()-border_size, cnt[:,:,1].max()+border_size]], [[cnt[:,:,0].max()+border_size, cnt[:,:,1].max()+border_size]]])
        cv.drawContours(outline, digit, -1, (0,255,0), 3)
        corners = digit.sum(1)
        zoom = transform(thresh, corners)
        height, width = zoom.shape
        border_top, border_bottom, border_left, border_right = 0,0,0,0
        if height > width:
            border_right = int(np.round(((height - width) / 2) - 0.1))
            border_left = int(np.round(((height - width) / 2) + 0.1))
        elif width > height:
            border_top = int(np.round(((width - height) / 2) - 0.1))
            border_bottom = int(np.round(((width - height) / 2) + 0.1))
        final = cv.copyMakeBorder(zoom, border_top, border_bottom, border_left, border_right, borderType=cv.BORDER_CONSTANT, value=0)
        
    else:
        final = thresh
        
    return final


def fill_grid(img, model, visuals=False, border_size=5):
    """
    input
    img: np.array
    model: eg convnet,
    visuals: default=False,
    border_size: default=5

    returns
    grid: np.array,
    new_digits: only for testing
    predictions: only for testing
    """

    ### ignore these
    new_digits = []
    predictions = []

    grid = np.zeros((9,9), dtype=int)
    cell_height = img.shape[0] / 9
    cell_width = img.shape[1] / 9
    if visuals:
        fig, axs = plt.subplots(nrows=9, ncols=9,figsize=(8, 8))

    for i in range(9):
        for j in range(9):
            cell_top = int(np.round(i * cell_height))
            cell_bottom = int(np.round((i+1) * cell_height))
            cell_left = int(np.round(j * cell_width))
            cell_right = int(np.round((j+1) * cell_width))
            if i == 8:
                cell_bottom = -1
            if j == 8:
                cell_right = -1
                
            cell = get_digit(img[cell_top:cell_bottom, cell_left:cell_right, :], border_size=border_size)
            if visuals:
                axs[i][j].imshow(cell, cmap="gray")
            if (cell==255).sum() / (cell==0).sum() < 0.03:
                grid[i,j] = 0
            else:
                cell = cv.resize(cell,(28, 28))
                cell = cell.astype("float") / 255.0
                cell = np.expand_dims(cell, axis=-1)
                cell = np.expand_dims(cell, axis=0)
                prediction = np.argmax(model.predict(cell))
                grid[i,j] = prediction
                
                ### Ignore this
                if prediction in [1, 7]:
                    predictions.append(prediction)
                    new_digits.append(cell)
                
    return grid, new_digits, predictions
