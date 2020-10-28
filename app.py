import cv2 as cv
import numpy as np
import sys
import matplotlib.pyplot as plt
from skimage.segmentation import clear_border
from tensorflow import keras
import functions.sudoku as sk
import functions.solver as solver
import streamlit as st
from PIL import Image

st.write("""
# Automatic Sudoku Solver
""")

file = st.file_uploader("Upload a sudoku to scan", type="jpg")
if file is not None:
    img = Image.open(file)
    st.image(img, caption='Uploaded Image.', use_column_width=True)
    img = np.array(img)
else:
    img = cv.imread("figures/1.jpg") # alternatively add flags = 0, which turns image into grayscale
    st.image(img, caption="here's an example Image.", use_column_width=True)
#plt.imshow(img)
_=plt.title("original image", size=20)

warped = sk.find_sudoku(img, kernel_size=7, canny_threshold=100, printer="outline")

grid = warped.copy()
cell_height = warped.shape[0] / 9
cell_width = warped.shape[1] / 9
for i in range(10):
    cv.line(grid,(int(i * cell_width),-1000),(int(i * cell_width),1000),(0,255,0),1)
for i in range(10):
    cv.line(grid,(-1000, int(i * cell_height)),(1000, int(i * cell_height)),(0,255,0),1)
_=plt.imshow(grid)

model = keras.models.load_model("models/third_model")
grid, _, _ =sk.fill_grid(warped, model)

#solver.visualize(grid)

solution, worked = solver.solve(grid)


st.write(f"""
{solver.markdown_grid(grid)}
""")

if worked == False:
    solver.visualize(solution)
    st.write("something went wrong. This sudoku is unsolvable")

if worked == True:
    if st.button("solve"):
        st.write(f"""
        {solver.markdown_grid(solution)}
        """)
