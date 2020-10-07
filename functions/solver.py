import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import copy

def visualize(sudoku):
    """
    interprets an np.array as a sudoku grid
    """
    for i, row in enumerate(sudoku):
        if i in [0,3,6]:
            print("- "*13)
        print("|", row[0], row[1], row[2], "|", row[3], row[4], row[5], "|", row[6], row[7], row[8], "|")
    print("- "*13)
    
    
def box(row_number, column_number, sudoku):
    """
    helper function to identify the numbers in the small 3x3 grid

    """
    entries = []
    for i in range(3):
        for j in range(3):
            entry = sudoku[(row_number // 3)*3 + i][(column_number // 3)*3 + j]
            if entry != 0:
                entries.append(sudoku[(row_number // 3)*3 + i][(column_number // 3)*3 + j])
    return set(entries)

def find_empty(sudoku):
    """
    finds the next empty field in a sudoku
    """
    for row_number, row in enumerate(sudoku):
        for column_number, field in enumerate(row):
            if field == 0:
                return [row_number, column_number]
    return True


def backtrack(sudoku):
    """
    the backtracking algorithm.
    input: np.array
    """
    indices = find_empty(sudoku)
    if type(indices) == bool:
        return True
    else:
        [row_number, column_number] = indices

    row = sudoku[row_number]
    column = sudoku[column_number]

    box_entries = box(row_number, column_number, sudoku)
    row_entries = set([entry for entry in row if entry > 0])
    column_entries = set([row[column_number] for row in sudoku if row[column_number] > 0])
            
    excluded = row_entries|box_entries|column_entries

    for num in range(1, 10):
        if num not in excluded:
            sudoku[row_number][column_number] = num
            if backtrack(sudoku) == True:
                return True
            sudoku[row_number][column_number] = 0
    return False

def solve(sudoku):
    """
    solves the sudoku
    input: np.array
    """
    solved = copy.deepcopy(sudoku)
    if backtrack(solved) == True:
        print("It worked:")
        return solved
    else:
        print("didn't work :(")
        return sudoku
