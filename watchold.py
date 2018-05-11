import random
from collections import defaultdict

"""
Classes for representing a Watch Your Back! game board and the Black and White
pieces that move around it.

Created for the tasks of finding legal moves and applying them to the board.
Also allows moves to be 'undone'.

Does not contain any logic to check for errors: all the functions and methods
assume that they will be used correctly (with the correct order and with the
correct input values). Otherwise the behaviour is not defined.


Author: Matt Farrugia <matt.farrugia@unimelb.edu.au>
April 2018
"""

# HELPERS

WHITE, BLACK, CORNER, EMPTY, CLEAR = ['O','@','X','-', " "]
ENEMIES = {WHITE: {BLACK, CORNER}, BLACK: {WHITE, CORNER}, CORNER: {}, EMPTY: {}, CLEAR: {}}
FRIENDS = {WHITE: {WHITE, CORNER}, BLACK: {BLACK, CORNER}, CORNER: {}, EMPTY: {}, CLEAR: {}}
PLAY_ENEMY = {WHITE: BLACK, BLACK: WHITE}

FULL_EMPTY = \
[['X','-','-','-','-','-','-','X'], \
['-','-','-','-','-','-','-','-'], \
['-','-','-','-','-','-','-','-'], \
['-','-','-','-','-','-','-','-'], \
['-','-','-','-','-','-','-','-'], \
['-','-','-','-','-','-','-','-'], \
['-','-','-','-','-','-','-','-'], \
['X','-','-','-','-','-','-','X']]




# ************************ temp placement stuff ************************** #

# all possible placements for white and black players
white_poss_placements = []
for i in range(8):
    for j in range(6):
        pos = (i, j)
        if (pos != (0, 0)) and (pos != (7, 0)):
            white_poss_placements.append(pos)

black_poss_placements = []
for i in range(8):
    for j in range(2,8):
        pos = (i, j)
        if (pos != (0, 7)) and (pos != (7, 7)):
            black_poss_placements.append(pos)


# hard-coded testing
white_places = []
# for i in range(3,7):
#     for j in range(1,4):
for i in range(1, 7):
    for j in range(0, 2):
        white_places.append((i,j))


black_places = []
for i in range(1,7):
    for j in range(6,8):
        black_places.append((i,j))
# black_places.append((6,6))
# black_places.append((7,3))



# evaluation array adapted from chess wiki of Queen evaluation array
EVAL_WHITE  = [ \
    [ 0.0001, 2, 2, 2, 2, 2, 2, 0.0001], \
    [ 2,  5,  5,  15,  15,  5,  5, 2], \
    [ 2,  5,  10,  12,  12,  10,  5, 2], \
    [ 2,  5,  7,  7,  7,  7,  5, 2], \
    [ 2,  5,  5,  5, 5,  5,  5, 2], \
    [ 2,  5,  5,  5,  5,  5,  5, 2], \
    [ 0.0001,  0.0001, 0.0001,  0.0001,  0.0001,  0.0001,  0.0001, 0.0001], \
    [ 0.0001,  0.0001, 0.0001,  0.0001,  0.0001,  0.0001,  0.0001, 0.0001], \
]

EVAL_BLACK = [ \

    [ 0.0001,  0.0001, 0.0001,  0.0001,  0.0001,  0.0001,  0.0001, 0.0001], \
    [ 0.0001,  0.0001, 0.0001,  0.0001,  0.0001,  0.0001,  0.0001, 0.0001], \
    [ 2,  5,  5,  5,  5,  5,  5, 2], \
    [ 2,  5,  5,  5,  5,  5,  5, 2], \
    [ 2,  5,  7,  7,  7,  7,  5, 2], \
    [ 2,  5,  7,  10,  10,  7,  5, 2], \
    [ 2,  5,  10,  15,  15,  10,  5, 2], \
    [ 0.0001, 2, 2, 2, 2, 2, 2, 0.0001], \
]


# ************************ temp placement stuff ************************** #


DIRECTIONS = UP, DOWN, LEFT, RIGHT = (0, -1), (0, 1), (-1, 0), (1, 0)
def step(position, direction):
    """
    Take an (x, y) tuple `position` and a `direction` (UP, DOWN, LEFT or RIGHT)
    and combine to produce a new tuple representing a position one 'step' in
    that direction from the original position.
    """
    px, py = position
    dx, dy = direction
    return (px+dx, py+dy)

# CLASSES

class Player:
    def __init__(self, colour):
        self.type = WHITE if colour == "white" else BLACK
        self.enemy = BLACK if colour == "white" else WHITE
        self.board = Board(FULL_EMPTY, self.type)
        self.placeMode = True

        self.playerEval = EVAL_WHITE if colour == "white" else EVAL_BLACK
        self.enemyEval = EVAL_BLACK if colour == "white" else EVAL_WHITE

    def action(self, turns):

        # Shrink the board after each player makes 64, then another 32 moves
        if turns == 127 or turns == 128:
            self.board.shrink((6, 1), (1, 1), (6, 6), (1, 6))

        if turns == 191 or turns == 192:
            self.board.shrink((5, 2), (2, 2), (5, 5), (2, 5))

        # Not in play mode while in the placing stage
        if self.placeMode:
            if turns < 24:

                value = self.miniMaxPlacement(self.board.grid, 2, True, True, 0, 1000000000)

                if (value[1][0][1] == self.type):
                    action = value[1][0][0]
                else:
                    action = value[1][1][0]

                self.board.place_piece(action, self.type)

            # Switch to move mode if its going first or second
            if turns == 22 or turns == 23:
                self.placeMode = False

        # Now in the movement (playing) stage of the game
        else:
            value, pos, boardState = self.miniMaxPlacement(self.board.grid, 2, True, False, -1e500, 1e500)
            action = self.findmove(boardState)

            self.board.makemove(action[0], action[1])

        # check if any playing pieces should be deleted (due to shrinking)
        for loc in self.board.grid.keys():
            if self.board.grid[loc] == WHITE or self.board.grid[loc] == BLACK:
                for forward, backward in [(UP, DOWN), (LEFT, RIGHT)]:
                    front_square = step(loc, forward)
                    back_square  = step(loc, backward)
                    if self.board.find_piece(front_square) in ENEMIES[self.board.grid[loc]] \
                    and self.board.find_piece(back_square) in ENEMIES[self.board.grid[loc]]:
                        self.board.grid[loc] = EMPTY
                        break
        print(self.type, action)
        return action

    # gets all branches of the subtree
    # returns an array of all the possible branches
    # and the position of the new piece
    def getBranches(self, boardState, maximizingPlayer):

        branch = []
        i = 0
        num = 0

        # create empty dictionaries for all possible boards
        for count in (boardState):
            if boardState[count] == EMPTY:
                num += 1

        # create (max amount of empty positions available) new dictionaries
        sectionsDict = []
        for var in range(num):
           sectionsDict.append(dict())

        # add to new dictionary for all possible new spots
        for position in boardState:
            if boardState[position] == EMPTY:
                sectionsDict[i] = boardState.copy()
                if maximizingPlayer:
                    sectionsDict[i][position] = self.type
                else:
                    sectionsDict[i][position] = self.enemy
                branch.append(sectionsDict[i])
                i += 1
        return branch

    # may not work because using a 2nd move not in actual board
    def getBranchesMoves(self, boardState, maximizingPlayer):

        positions = []
        allMoves = []
        branch = []
        sectionsDict = []

        # find all the possible moves
        for ogPos in boardState:

            if (maximizingPlayer and boardState[ogPos] == self.type):
                    moves = self.board.moves(ogPos)
                    if moves:
                        positions.append((maximizingPlayer, ogPos, moves))

            if (not maximizingPlayer and boardState[ogPos] == self.enemy):
                    moves = self.board.moves(ogPos)
                    if moves:
                        positions.append((not maximizingPlayer, ogPos, moves))

        # pos is array of arrays
        for i in range(len(positions)):
            for j in range(len(positions[i][2])):
                allMoves.append((positions[i][1],(positions[i][2][j]))) # amount of moves available

        # create (max amount of empty positions available) new dictionaries
        for var in range(len(allMoves)):
           sectionsDict.append(dict())

        for i in range(len(allMoves)):
            sectionsDict[i] = boardState.copy()
            # need to make original move empty
            if maximizingPlayer:
                #print(sectionsDict[i][allMoves[i][1]])
                sectionsDict[i][allMoves[i][1]] = self.type
            else:
                sectionsDict[i][allMoves[i][1]] = self.enemy
            sectionsDict[i][allMoves[i][0]] = EMPTY
            branch.append(sectionsDict[i])

        return branch
    # want to return branches

    # determines which is the current piece in question
    def findmove(self, boardState):

        action = None
        new = None
        old = None

        for pos in boardState:
            if boardState[pos] != self.board.grid[pos]:
                if boardState[pos] == EMPTY and self.board.grid[pos] == self.type:
                    old = pos
                elif boardState[pos] == self.type and self.board.grid[pos] == EMPTY:
                    new = pos

        if not new: # both players want to move to the same spot
            for pos in boardState:
                if boardState[pos] != self.board.grid[pos]:
                    if boardState[pos] == self.enemy and self.board.grid[pos] == EMPTY:
                        new = pos

        action = (old, new)
        return action

    def compareBoards(self, boardState):

        newPos = []
        for pos in boardState:
            if boardState[pos] != self.board.grid[pos]:
                newPos.append((pos, boardState[pos])) # e.g. ((2,3), '@')
        return newPos

    # returns how many good and enemy pieces are close enough to kill you
    def findNeighbours(self, x, y):

        # want to check if piece will die
        avoid = False
        kill = False
        player = 0
        enemy = 0
        locs = [(x+1,y+1), (x-1,y-1), (x,y+1), (x,y-1), \
                (x+1,y), (x-1,y), (x-1,y+1), (x+1,y-1), \
                (x+2,y), (x-2,y), (x, y+2), (x,y-2)]

        for pos in self.board.grid:
            if (pos in locs):
                if self.board.grid[pos] == self.type:
                    player += 1
                else:
                    enemy += 1

        #check if you will die in both directions
        if ((x,y+1) in self.board.grid and (x,y-1) in self.board.grid):
            if (self.board.grid[(x,y+1)] == self.enemy and self.board.grid[(x,y-1)] == self.enemy):
                avoid = True
            elif (self.board.grid[(x,y+1)] == self.enemy and self.board.grid[(x,y-1)] == self.enemy):
                kill = True
        elif ((x+1,y) in self.board.grid and (x-1,y) in self.board.grid):
            if (self.board.grid[(x+1, y)] == self.enemy and self.board.grid[(x-1, y)] == self.enemy):
                avoid = True
            elif (self.board.grid[(x+1, y)] == self.enemy and self.board.grid[(x-1, y)] == self.enemy):
                kill = True

        return player, enemy, avoid, kill

    # finds heuristic value of position in board
    def getHeuristic(self, boardState, maximizingPlayer):

        DEF = 2 # the players risk aversion

        newPos = self.compareBoards(boardState)

        copyBoard = boardState.copy()

        if (newPos[0][1] == self.type):
            x = newPos[0][0][0]
            y = newPos[0][0][1]
        else:
            x = newPos[1][0][0]
            y = newPos[1][0][1]

        player, enemy, avoid, kill = self.findNeighbours(x, y)
        num = (float(self.playerEval[y][x]))+(DEF*(player-enemy))

        if avoid: # avoid getting killed
            num *= 1000

        elif kill: # kill the other player if you can
            num /= 1000

        return (num, newPos, copyBoard)

    # finds heuristic value of position in board
    def getHeuristic2(self, boardState, maximizingPlayer):

            DEF = -0.5 # the players risk aversion

            newPos = self.compareBoards(boardState)

            copyBoard = boardState.copy()

            if (newPos[0][1] == self.type):
                x = newPos[0][0][0]
                y = newPos[0][0][1]
            else:
                x = newPos[1][0][0]
                y = newPos[1][0][1]

            player, enemy, avoid, kill = self.findNeighbours(x, y)
            num = (float(self.playerEval[y][x]))+(DEF*(player-enemy))

            if avoid: # avoid getting killed
                num *= 1000

            elif kill: # kill the other player if you can
                num /= 1000

            return (num, newPos, copyBoard)

    def miniMaxPlacement(self, boardState, depth, maximizingPlayer, placing, alpha, beta):

        if depth == 0: #cannot have winning node since only replacing
            if placing:
                return self.getHeuristic(boardState, maximizingPlayer)
            else:
                return self.getHeuristic2(boardState, maximizingPlayer)

        if maximizingPlayer:  # self.player - alpha
            bestValue = (-1e500, None)
            if placing:
                branch = self.getBranches(boardState, maximizingPlayer)
            elif not placing:
                branch = self.getBranchesMoves(boardState, maximizingPlayer)

            i = 0
            for child in branch:
                currValue = self.miniMaxPlacement(child, depth-1, False, placing, alpha, beta)
                if currValue[0] > bestValue[0]:
                    bestValue = currValue

                if bestValue[0] > alpha:
                    alpha = bestValue[0]

                if beta <= alpha:
                    break

            return bestValue

        else:   # self.enemy - minimizing player - beta
            bestValue = (1e500, None)
            if placing:
                branch = self.getBranches(boardState, maximizingPlayer)
            elif not placing:
                branch = self.getBranchesMoves(boardState, maximizingPlayer)

            i = 0
            for child in branch:
                currValue = self.miniMaxPlacement(child, depth-1, True, placing, alpha, beta)

                if currValue[0] < bestValue[0]:
                    bestValue = currValue
                i += 1

                if bestValue[0] < beta:
                    beta = bestValue[0]

                if beta <= alpha:
                    break

            return bestValue

    # Heuristic currently defined as being the difference in the number of
    # pieces for each player + average euclidean distances
    def heuristic(self, type):

        players = []
        enemies = []
        for pos in self.board.grid:
            if self.board.grid[pos] == type:
                players.append(pos)
            elif self.board.grid[pos] in ENEMIES[type]:
                enemies.append(pos)

        score = 0
        for i in players:
            for j in enemies:
                score += self.euclidean_distance(i, j)

        return (len(players) - len(enemies))*5 #+ score

    def euclid_states(self):
        best_score = float('inf')       # THIS IS AIDS
        best_move = ()
        for loc in self.board.grid.keys():
            if self.board.grid[loc] == self.type:
                if self.board.moves(loc):
                    move, score = self.euclidean(loc)
                    if score < best_score:
                        best_score = score
                        best_move = move
        print(best_score)
        return best_move

    def euclidean(self, loc):
        min_score = float('inf')           # THIS IS AIDS
        for newpos in self.board.moves(loc):
            oldtype = self.board.grid[loc]
            eliminated_pieces = self.board.makemove(loc, newpos)

            # calculate euclidean score of current board
            score = 0

            players = []
            enemies = []
            for pos in self.board.grid:
                if self.board.grid[pos] == self.type:
                    players.append(pos)
                elif self.board.grid[pos] == self.enemy:
                    enemies.append(pos)

            for i in players:
                for j in enemies:
                    score += self.euclidean_distance(i, j)

            self.board.undomove(loc, oldtype, newpos, eliminated_pieces)

            if score < min_score:
                min_score = score
                best_move = newpos

        if self.board.moves(loc):
            return ((loc, best_move), min_score)
        else:
            return None

    def euclidean_distance(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def update(self, action):
        if type(action[0]) is int:
            self.board.place_piece(action, self.enemy)
        else:
            self.board.makemove(action[0], action[1])


class Board:
    """
    A class to represent a Watch Your Back! board, keeping track of the pieces
    on the board (in `whites` and `blacks` lists) and the state of each board
    square (in `grid` dictionary, indexed by `(x, y)` tuples).
    """
    def __init__(self, data, player):
        """
        Create a new board based on a nested list of characters representing
        an initial board configuration (`data`).
        """
        self.size = len(data)
        self.player = player
        self.grid = {}

        for y, row in enumerate(data):
            for x, char in enumerate(row):
                self.grid[x, y] = char

    def __str__(self):
        """Compute a string representation of the board's current state."""
        ran = range(self.size)
        return 'local board\n' + '\n'.join(' '.join(self.grid[x,y] for x in ran) for y in ran) +'\n\n'

    def find_piece(self, square):
        """
        An O(n) operation (n = number of pieces) to find the piece object
        for the piece occupying a given position on the board. This method
        could be improved by separately keeping track of which piece is at
        each position.
        """
        if square in self.grid:
            return self.grid[square]

    def place_piece(self, pos, type):
        #new_piece = Piece(type, pos, self)
        #self.white_pieces.append(new_piece) if type == WHITE else \
        #self.black_pieces.append(new_piece)
        self.grid[pos] = type

    # NEED TO COMPLETE
    def shrink(self, tr_corner, tl_corner, br_corner, bl_corner):
        toDel = []
        for pos in self.grid.keys():
            if (pos[0] < tl_corner[0]) or (pos[0] > tr_corner[0]) or \
            (pos[1] > br_corner[1]) or (pos[1] < tr_corner[1]):
                toDel.append(pos)

        for pos in toDel:
            self.grid[pos] = CLEAR

        print(toDel)

        self.grid[tr_corner] = CORNER
        self.grid[tl_corner] = CORNER
        self.grid[br_corner] = CORNER
        self.grid[bl_corner] = CORNER


    def moves(self, pos):
        """
        Compute and return a list of the available moves for this piece based
        on the current board state.

        Do not call with method on pieces with `alive = False`.
        """

        possible_moves = []
        for direction in DIRECTIONS:
            # a normal move to an adjacent square?
            adjacent_square = step(pos, direction)
            # if adjacent_square in self.board.grid:
            #     print((adjacent_square, self.board.grid[adjacent_square].type))

            if self.find_piece(adjacent_square) == EMPTY:

                possible_moves.append(adjacent_square)
                continue # a jump move is not possible in this direction

            # if not, how about a jump move to the opposite square?
            opposite_square = step(adjacent_square, direction)
            if self.find_piece(opposite_square) == EMPTY:
                possible_moves.append(opposite_square)
        #print(possible_moves)
        return possible_moves

    def makemove(self, oldpos, newpos):
        """
        Carry out a move from this piece's current position to the position
        `newpos` (a position from the list returned from the `moves()` method)
        Update the board including eliminating any nearby pieces surrounded as
        a result of this move.

        Return a list of pieces eliminated by this move (to be passed back to
        the `undomove()` method if you want to reverse this move).

        Do not call with method on pieces with `alive = False`.
        """

        # eliminate any newly surrounded pieces
        eliminated_pieces = []

        # make the move
        my_type = self.grid[oldpos]
        self.grid[oldpos] = EMPTY
        self.grid[newpos] = my_type

        # check adjacent squares: did this move eliminate anyone?
        for direction in DIRECTIONS:
            adjacent_square = step(newpos, direction)
            opposite_square = step(adjacent_square, direction)
            if self.find_piece(adjacent_square) in ENEMIES[my_type] \
            and self.find_piece(opposite_square) in FRIENDS[my_type]:

                eliminated_piece = (self.find_piece(adjacent_square), adjacent_square)
                self.grid[adjacent_square] = EMPTY
                eliminated_pieces.append(eliminated_piece)
                #eliminated_pieces.append(eliminated_piece)

        # check horizontally and vertically: does the piece itself get
        # eliminated?
        for forward, backward in [(UP, DOWN), (LEFT, RIGHT)]:
            front_square = step(newpos, forward)
            back_square  = step(newpos, backward)
            if self.find_piece(front_square) in ENEMIES[my_type] \
            and self.find_piece(back_square) in ENEMIES[my_type]:
                eliminated_piece = (my_type, newpos)
                self.grid[newpos] = EMPTY
                eliminated_pieces.append(eliminated_piece)
                break

        #eliminated_pieces.append((my_type, oldpos))

        return eliminated_pieces

    def undomove(self, oldpos, oldtype, newpos, eliminated_pieces):

        """
        Roll back a move for this piece to its previous position `oldpos`,
        restoring the pieces it had eliminated `eliminated_pieces` (a list as
        returned from the `makemove()` method).

        A move should only be 'undone' if no other moves have been made since
        (unless they have already been 'undone' also).

        Do not call with method on pieces with `alive = False` unless you are
        undoing the move that eliminated this piece.
        """
        # put back the pieces that were eliminated
        for data in eliminated_pieces:
            self.grid[data[1]] = data[0]

        # undo the move itself
        self.grid[newpos] = EMPTY
        self.grid[oldpos] = oldtype
