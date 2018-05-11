"""
A Python program that plays a complete game of Watch Your Back!, for Part B
of the Artificial Intelligence (COMP30024) subject.

Contains a class that represents the board of a game of Watch Your Back!, and a
class that represents each players' actions and handles updates to their local
game states.

Determines subsequent moves using the Minimax algorithm alongside Alpha-beta
pruning. Calculates state values using a heuristic.

Board class code (methods for making moves and undoing moves) was adapted from
the sample solution to Project: Part A, provided by Matt Farrugia.

Authors: Max Philip (836472), Myles Adams (761125)
May 2018
"""

# HELPERS
WHITE, BLACK, CORNER, EMPTY, CLEAR = ['O','@','X','-', " "]
ENEMIES = {WHITE: {BLACK, CORNER}, BLACK: {WHITE, CORNER}, CORNER: {}, EMPTY: {}, CLEAR: {}}
FRIENDS = {WHITE: {WHITE, CORNER}, BLACK: {BLACK, CORNER}, CORNER: {}, EMPTY: {}, CLEAR: {}}
PLAY_ENEMY = {WHITE: BLACK, BLACK: WHITE}
INFINITY = float('inf')

# For initializing the local game state of the player at turn 0.
FULL_EMPTY = \
[['X','-','-','-','-','-','-','X'], \
['-','-','-','-','-','-','-','-'], \
['-','-','-','-','-','-','-','-'], \
['-','-','-','-','-','-','-','-'], \
['-','-','-','-','-','-','-','-'], \
['-','-','-','-','-','-','-','-'], \
['-','-','-','-','-','-','-','-'], \
['X','-','-','-','-','-','-','X']]

# Store the possible coordinates of white and black pieces' starting zones
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

START_ZONE = {WHITE: white_poss_placements, BLACK: black_poss_placements}


# evaluation array adapted from chess wiki of Queen evaluation array
EVAL_WHITE  = [ \
    [ 0.0001, 1, 1, 1, 1, 1, 1, 0.0001], \
    [ 1,  8,  12,  15,  15,  12,  8, 1], \
    [ 1,  7,  10,  12,  12,  10,  7, 1], \
    [ 1,  6,  8,  10,  10,  8,  6, 1], \
    [ 1,  5,  7,  8,  8,  7,  5, 1], \
    [ 1,  4,  5,  5,  5,  5,  5, 4], \
    [ 0.0001,  0.0001, 0.0001,  0.0001,  0.0001,  0.0001,  0.0001, 0.0001], \
    [ 0.0001,  0.0001, 0.0001,  0.0001,  0.0001,  0.0001,  0.0001, 0.0001], \
]

EVAL_BLACK = [ \
    [ 0.0001,  0.0001, 0.0001,  0.0001,  0.0001,  0.0001,  0.0001, 0.0001], \
    [ 0.0001,  0.0001, 0.0001,  0.0001,  0.0001,  0.0001,  0.0001, 0.0001], \
    [ 1,  4,  5,  5,  5,  5,  4, 1], \
    [ 1,  5,  7,  8,  8,  7,  5, 1], \
    [ 1,  6,  8,  10,  10,  8,  6, 1], \
    [ 1,  7,  10,  12,  12,  10,  7, 1], \
    [ 1,  8,  12,  15,  15,  12,  8, 1], \
    [ 0.0001, 1, 1, 1, 1, 1, 1, 0.0001], \
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
        self.depthMax = 2

        self.playerEval = EVAL_WHITE if colour == "white" else EVAL_BLACK
        self.enemyEval = EVAL_BLACK if colour == "white" else EVAL_WHITE

    def action(self, turns):
        """
        Calculate the best action for a given game state, using the defined
        heuristic. Returns a single tuple (e.g. (4,4)) if the game is in
        the placing phase. Returns a tuple of two tuples (e.g. ((4,4),(4,5)))
        if the game is in the moving phase.
        """
        # Shrink the board after each player makes 64, then another 32 moves
        if turns == 127 or turns == 128:
            self.board.shrink((6, 1), (1, 1), (6, 6), (1, 6))

        if turns == 191 or turns == 192:
            self.board.shrink((5, 2), (2, 2), (5, 5), (2, 5))

        # NOTe: CAN MAKE MOVE OUT OF SHRINK ZONE??? BUT NOT INTO THE NEW CORNER??



        # Not in play mode while in the placing stage
        if self.placeMode:
            if turns < 24:
                if self.type == WHITE and (turns == 0 or turns == 1):
                    action = (3,2) if self.board.grid[(3,2)] == EMPTY else (4,2)
                elif self.type == BLACK and (turns == 0 or turns == 1):
                    action = (4,5) if self.board.grid[(4,5)] == EMPTY else (3,5)

                else:

                    action = self.bestmove(self.type, turns, True)
                self.board.place_piece(action, self.type)

            # Switch to move mode if its going first or second
            if turns == 22 or turns == 23:
                self.placeMode = False
                self.depthMax = 2

        # Now in the movement (playing) stage of the game
        else:
            action = self.bestmove(self.type, turns, False)
            if action != (None, None):
                self.board.makemove(action[0], action[1])
            else:
                return None

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

        return action

    def heuristic(self, state):
        """
        Calculate the value of a board state.
        """
        play_eval = 0
        euc_score = 0
        too_back = 0
        players = []
        enemies = []

        for loc in state:
            if state[loc] == self.type:
                if self.placeMode:
                    if (self.type == WHITE):
                        play_eval += EVAL_WHITE[loc[1]][loc[0]]
                    else:
                        play_eval += EVAL_BLACK[loc[1]][loc[0]]
                else:
                    play_eval += EVALPLAY_WHITE2[loc[1]][loc[0]]
                players.append(loc)
                euc_score += self.euclidean_distance(loc, (3.5, 3.5))

            if state[loc] == self.enemy:
                enemies.append(loc)

        kill_diff = (len(players) - len(enemies))*1000
        # play_eval = random.randint(1, 10000)
        return kill_diff - euc_score #+ 100000000*play_eval#+ weak_points#+ euc_score #+ play_eval

    def minimax(self, node, depth):
        """
        Return the best move a player can make using the minimax algorithm,
        traversing the game tree to the specified depth. Alpha-beta pruning is
        used to decrease the total number of nodes traversed, significantly
        improving time performance without compromising the final output.
        """
        alpha = -INFINITY
        beta = INFINITY
        best_val = self.max_value(node, alpha, beta, depth)

        successors = self.getSuccPlacement(node) if self.placeMode else \
        self.getSuccessors(node)

        # Return (None, None) to represent the forfeiture of a move if there is
        # no available move to make.
        if successors:
            best_move = successors[0][0]
            move = successors[0][1]
        else:
            return None, None

        for elem in successors:
            if self.heuristic(elem[0]) == best_val:
                best_move = elem[0]
                move = elem[1]
                break

        return best_move, move

    def max_value(self, node, alpha, beta, depth):
        if depth > self.depthMax:
            return self.heuristic(node)

        max_value = -INFINITY
        successors_states = self.getSuccPlacement(node) if self.placeMode else \
        self.getSuccessors(node)
        for state in successors_states:
            max_value = max(max_value, self.min_value(state[0], alpha, beta, depth+1))
            alpha = max(alpha, max_value)
            if (beta <= alpha):
                break
        return max_value

    def min_value(self, node, alpha, beta, depth):
        if depth > self.depthMax:
            return self.heuristic(node)

        min_value = INFINITY
        successors_states = self.getSuccPlacement(node) if self.placeMode \
        else self.getSuccessors(node)
        for state in successors_states:
            min_value = min(min_value, self.max_value(state[0], alpha, beta, depth+1))
            beta = min(beta, min_value)
            if (beta <= alpha):
                break
        return min_value

    def getSuccessors(self, node):
        assert node is not None
        states = []
        for loc in self.board.grid.keys():
            if self.board.grid[loc] == self.type:
                for move in self.board.moves(loc):
                    elims = self.board.makemove(loc, move)
                    states.append((self.board.grid.copy(), move))
                    self.board.undomove(loc, self.type, move, elims)
        return states

    def getSuccPlacement(self, node):
        assert node is not None
        states = []
        for loc in self.board.grid.keys():
            # if self.board.grid[loc] == self.enemy:
            #     for move in self.board.moves(loc):
            #         if move in START_ZONE[self.type]:
            #             elims = self.board.place_piece(move, self.type)
            #             states.append((self.board.grid.copy(), move))
            #             self.board.undomove(loc, self.enemy, move, elims)

            if self.board.grid[loc] == EMPTY and loc in START_ZONE[self.type] \
            and not self.findNeighbours(loc[1], loc[0]):

                elims = self.board.place_piece(loc, self.type)
                states.append((self.board.grid.copy(), loc))
                #self.board.undomove(loc, self.enemy, loc, elims)
                for data in elims:
                    self.board.grid[data[1]] = data[0]

                # undo the move itself
                self.board.grid[loc] = EMPTY
                #self.grid[oldpos] = oldtype

        return states

    def findNeighbours(self, x, y):

        # want to check if piece will die
        avoid = False


        left = (x-1, y)
        right = (x+1, y)
        up = (x, y-1)
        down = (x, y+1)

        #check if you will die in both directions
        if (down in self.board.grid and up in self.board.grid):
            if (self.board.grid[up] == self.enemy and self.board.grid[down] == EMPTY \
            or self.board.grid[down] == self.enemy and self.board.grid[up] == EMPTY):
                avoid = True
        if (right in self.board.grid and left in self.board.grid):
            if (self.board.grid[right] == self.enemy and self.board.grid[left] == EMPTY \
            or self.board.grid[left] == self.enemy and self.board.grid[right] == EMPTY):
                avoid = True

        return avoid

    def bestmove(self, type, turns, placeMode):

        if placeMode:
            return self.minimax(self.board.grid, 0)[1]
        else:

            new_state, move = self.minimax(self.board.grid, 0)
            if (new_state, move) == (None, None):
                return None, None

            for loc in self.board.grid.keys():
                if self.board.grid[loc] == self.type:
                    if new_state[loc] == EMPTY:
                        start = loc

            return start, move

    def euclid_states(self):
        best_score = INFINITY
        best_move = ()
        for loc in self.board.grid.keys():
            if self.board.grid[loc] == self.type:
                if self.board.moves(loc):
                    move, score = self.euclidean(loc)
                    if score < best_score:
                        best_score = score
                        best_move = move
        return best_move

    def euclidean(self, loc):
        min_score = INFINITY
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
        if action:
            if type(action[0]) is int:
                self.board.place_piece(action, self.enemy)
            else:
                self.board.makemove(action[0], action[1])


class Board:
    """
    A class to represent a Watch Your Back! board, keeping track of the pieces
    on the board in the `grid` dictionary. (x, y) coordinates of the pieces are
    used as the `grid` keys.
    """
    def __init__(self, data, player):
        """
        Create a new board based on a nested list of characters representing
        an initial board configuration, which is empty.
        """
        self.size = len(data)
        self.player = player
        self.grid = {}

        for y, row in enumerate(data):
            for x, char in enumerate(row):
                self.grid[x, y] = char

    def __str__(self):
        """Return a string representation of the board's current state."""
        size = range(self.size)
        return 'local board\n' + '\n'.join(' '.join(self.grid[x,y] for x in \
        size) for y in size) +'\n\n'

    def find_piece(self, square):
        """
        An O(n) operation (n = number of pieces) to find the type of piece
        located at the input coordinate, if it exists on the board.
        """
        if square in self.grid:
            return self.grid[square]

    def shrink(self, tr_corner, tl_corner, br_corner, bl_corner):
        """
        Reduce the size of the local board representation. Remove all of the
        pieces aren't on the new board - setting them to the CLEAR (" ") state.
        """
        # "Delete" each piece that is outside of the new boundaries
        for pos in self.grid.keys():
            if (pos[0] < tl_corner[0]) or (pos[0] > tr_corner[0]) or \
            (pos[1] > br_corner[1]) or (pos[1] < tr_corner[1]):
                self.grid[pos] = CLEAR

        # Set the new corner locations
        self.grid[tr_corner] = CORNER
        self.grid[tl_corner] = CORNER
        self.grid[br_corner] = CORNER
        self.grid[bl_corner] = CORNER

    def place_piece(self, newpos, my_type):
        """
        Place a piece with the input type and position.
        """


        self.grid[newpos] = my_type

        return self.remove_pieces_after_move(newpos, my_type)

    def moves(self, pos):
        """
        Compute and return a list of the available moves for this piece based
        on the current board state.
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
        `newpos` Update the board including eliminating any nearby pieces
        surrounded as a result of this move.

        Return a list of pieces eliminated by this move (to be passed back to
        `undomove()` if the move is to be reversed).
        """

        # make the move
        my_type = self.grid[oldpos]
        self.grid[oldpos] = EMPTY
        self.grid[newpos] = my_type

        return self.remove_pieces_after_move(newpos, my_type)

    def remove_pieces_after_move(self, newpos, my_type):

        # eliminate any newly surrounded pieces
        eliminated_pieces = []
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

        return eliminated_pieces

    def undomove(self, oldpos, oldtype, newpos, eliminated_pieces):

        """
        Roll back a move for this piece to its previous position `oldpos`,
        restoring the pieces it had eliminated `eliminated_pieces` (a list as
        returned from the `makemove()` method).

        A move is only be 'undone' if no other moves have been made since.
        """
        # put back the pieces that were eliminated
        for data in eliminated_pieces:
            self.grid[data[1]] = data[0]

        # undo the move itself
        self.grid[newpos] = EMPTY
        self.grid[oldpos] = oldtype
