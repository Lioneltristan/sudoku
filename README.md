# sudoku identifier and solver
I have built a program to identify and solve a sudoku grid from almost any background. 

You can try it out here: https://lionelssudoku.herokuapp.com/
It still sometimes confuses the numbers, particularly the 1s and 7s, because the algorithm has been trained on north-american handwritten digits. 
I will fix this when I have time to collect a seperate dataset and retrain the algorithm.

Here is an example of a raw image:

![](/figures/raw.png)

After some preprocessing using opencv we first find the outline of the sudoku grid with edge detection which gives us this:

![](/figures/outline.png)

We can then easily transform it and zoom in

![](/figures/grid.png)

to display the grid only.

We then divide the grid in its 81 cells and look at each cell to find out if it has a number in it or not. 
If it does, we use a conv-net model that we trained on the mnist dataset as well as some generated digits from given sudokus to identify the numbers inside the grid. We had to generate a few more digits, because the mnist dataset is mainly trained on "american ones", being just a straight line, whereas the typed digits have a 1 with a small "roof". This results in a lot of confusion for the algorithm regarding 1s and 7s from the grid.
We can now convert the grid into a numpy array.

![](/figures/digital.png)

In the end we use backtracking to solve the sudoku

Alternatively you can use py-sudoku to solve it



In future updates I will use a dataset different to mnist to train the algorithm. It still has some issues with a few of the numbers
