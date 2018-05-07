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

WHITE, BLACK, CORNER, EMPTY = ['O','@','X','-']
ENEMIES = {WHITE: {BLACK, CORNER}, BLACK: {WHITE, CORNER}, CORNER: {}, EMPTY: {}}
FRIENDS = {WHITE: {WHITE, CORNER}, BLACK: {BLACK, CORNER}, CORNER: {}, EMPTY: {}}

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
for i in range(3,7):
    for j in range(1,4):
# for i in range(1, 7):
#     for j in range(0, 2):
        white_places.append((i,j))


black_places = []
for i in range(1,7):
    for j in range(6,8):
        black_places.append((i,j))
# black_places.append((6,6))
# black_places.append((7,3))



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

        # self.my_pieces = self.board.white_pieces if self.type == WHITE \
        # else self.board.black_pieces
        # self.enemy_pieces = self.board.white_pieces if self.type == BLACK \
        # else self.board.black_pieces

    def action(self, turns):

        print(self.board)

        # Not in play mode while in the placing stage
        if self.placeMode:
            if turns < 24:
                if self.type == WHITE:
                    action = white_places[turns//2]
                    self.board.place_piece(action, self.type)
                else:
                    action = black_places[turns//2]
                    self.board.place_piece(action, self.type)
            # Switch to move mode if its going first or second
            if turns == 22 or turns == 23:
                self.placeMode = False

        # Now in the movement (playing) stage of the game
        else:
            action = self.bestmove(self.type, turns)

            self.board.makemove(action[0], action[1])

        print(self.type, action)
        print(self.board)
        return action

    def bestmove(self, type, turns):
        if type == WHITE:
            if turns == 0:
                move = ((6,1),(7,1))
            elif turns == 2:
                move = ((6,3), (7,3))
            else:
                move = ((7,3), (7,2))
        else:
            if turns == 1:
                move = ((7,3),(7,2))
            else:
                move = ((7,2),(7,1))

        return self.evalstate()

    def evalstate(self):
        best_score = float('inf')       # THIS IS AIDS
        #for piece in self.my_pieces:
        best_move = ()
        for loc in self.board.grid:
            if self.board.grid[loc] == self.type:
                if self.board.moves(loc):
                    move, score = self.euclidean(loc)
                    if score < best_score:
                        best_score = score
                        best_move = move

        return best_move


    def euclidean(self, loc):
        min_score = float('inf')           # THIS IS AIDS
        # print(loc, self.board.moves(loc))
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
    def shrink(self, turns):
        if turns == 128:
            toDels = [0, 7]
            newCorners = [(1, 1), (6, 1), (1, 6), (6, 6)]

        if turns == 192:
            toDels = [1, 6]
            newCorners = [(2, 2), (5, 2), (2, 5), (5, 5)]

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

        #eliminated_pieces.append((oldtype, oldpos))

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
