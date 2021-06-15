class EntryData:
    def __init__(self, r, c, n):
        self.row = r
        self.col = c
        self.choices = n

    def set_data(self, r, c, n):
        self.row = r
        self.col = c
        self.choices = n
        
#solving sudoku using best-first search method
def solve_sudoku(matrix):
    cont = [True]
    #finding the possilbilty for a solution
    for i in range(9):
        for j in range(9):
            if not can_be_correct(matrix, i, j):                       #stop if there is no possibility for a solution
                return
    sudoku_helper(matrix, cont)                                      # otherwise try to solve the sudoku puzzle

# helper function 
def sudoku_helper(matrix, cont):
    if not cont[0]:                                         # stopping point 1
        return

    #finding the one with the least possibilities
    best_candidate = EntryData(-1, -1, 100)
    for i in range(9):
        for j in range(9):
            if matrix[i][j] == 0:                                      # if it is unfilled
                num_choices = count_choices(matrix, i, j)
                if best_candidate.choices > num_choices:
                    best_candidate.set_data(i, j, num_choices)

   
    #if didn't find any choices means it has filled all board and the best-first search is done.
    if best_candidate.choices == 100: 
        cont[0] = False                         # setting the flag for stop 
        return

    row = best_candidate.row
    col = best_candidate.col

    #if best candidate found, try to fill 1-9
    for j in range(1, 10):
        if not cont[0]: # Stopping point 2
            return

        matrix[row][col] = j

        if can_be_correct(matrix, row, col):
            sudoku_helper(matrix, cont)

    if not cont[0]: # Stopping point 3
        return
    matrix[row][col] = 0 # backtracking and marking the current cell empty again
            

# counting the number of choices haven't been used
def count_choices(matrix, i, j):
    can_pick = [True,True,True,True,True,True,True,True,True,True];            
    
    # checking row
    for k in range(9):
        can_pick[matrix[i][k]] = False

    # checking col
    for k in range(9):
        can_pick[matrix[k][j]] = False;

    # checking 3x3 square
    r = i // 3
    c = j // 3
    for row in range(r*3, r*3+3):
        for col in range(c*3, c*3+3):
            can_pick[matrix[row][col]] = False

    #counting
    count = 0
    for k in range(1, 10):  # 1 to 9
        if can_pick[k]:
            count += 1

    return count

#return true if the current cell doesn't create any violation
def can_be_correct(matrix, row, col):
    
    #checking row
    for c in range(9):
        if matrix[row][col] != 0 and col != c and matrix[row][col] == matrix[row][c]:
            return False

    #checking column
    for r in range(9):
        if matrix[row][col] != 0 and row != r and matrix[row][col] == matrix[r][col]:
            return False

    #checking 3x3 square
    r = row // 3
    c = col // 3
    for i in range(r*3, r*3+3):
        for j in range(c*3, c*3+3):
            if row != i and col != j and matrix[i][j] != 0 and matrix[i][j] == matrix[row][col]:
                return False
    
    return True

#returning true if the whole board has been occupied by non-zero numbers as this is the solution to the original sudoku
def all_board_non_zero(matrix):
    for i in range(9):
        for j in range(9):
            if matrix[i][j] == 0:
                return False
    return True