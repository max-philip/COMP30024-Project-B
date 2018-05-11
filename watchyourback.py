"""
A Python program that plays a complete game of Watch Your Back!, for Part B
of the final Artificial Intelligence (COMP30024) project.

Contains a class that represents the board of a game of Watch Your Back!, and a
class that represents each players' actions and handles updates to their local
game states.

Determines subsequent moves using the Minimax algorithm alongside Alpha-beta
pruning and calculates state values using a heuristic.

Board class code (`find_piece`), Piece class code, which was incorporated into
the Board class (`moves`, `makemove`, `undomove`) and the`step` method were
adapted from the sample solution to Project: Part A, provided by Matt Farrugia.

Our Minimax implementation was loosely based on the implementation found at
<https://tonypoer.io/2016/10/28/implementing-minimax-and-alpha-beta-pruning-
using-python/>. Our Alpha-beta pruning adaptation was not based on the one
found provided by this site.

Authors: Max Philip (836472), Myles Adams (761125)
May 2018
"""

# HELPERS
WHITE, BLACK, CORNER, EMPTY, CLEAR = ['O','@','X','-', " "]
ENEMIES = {WHITE: {BLACK, CORNER}, BLACK: {WHITE, CORNER}, CORNER: {}, \
            EMPTY: {}, CLEAR: {}}
FRIENDS = {WHITE: {WHITE, CORNER}, BLACK: {BLACK, CORNER}, CORNER: {}, \
            EMPTY: {}, CLEAR: {}}
PLAY_ENEMY = {WHITE: BLACK, BLACK: WHITE}

INFINITY = float('inf')
FIRST_SHRINK = 128
SECOND_SHRINK = 192
MOVE_PHASE_TURNS = 24
BOARD_LEN = 8
CENTRE = 3.5


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

# Possible coordinates of the white and black pieces' starting zones
WHITE_POSS_PLACEMENTS = []
for i in range(BOARD_LEN):
    for j in range(BOARD_LEN-2):
        pos = (i, j)
        if (pos != (0, 0)) and (pos != (7, 0)):
            WHITE_POSS_PLACEMENTS.append(pos)

BLACK_POSS_PLACEMENTS = []
for i in range(BOARD_LEN):
    for j in range(2, BOARD_LEN):
        pos = (i, j)
        if (pos != (0, 7)) and (pos != (7, 7)):
            BLACK_POSS_PLACEMENTS.append(pos)

START_ZONE = {WHITE: WHITE_POSS_PLACEMENTS, BLACK: BLACK_POSS_PLACEMENTS}


# EVALUATION ARRAYS - adapted from chess wiki of Queen evaluation array
EVAL_WHITE  = [ \
    [ 0, 1, 1, 1, 1, 1, 1, 0], \
    [ 1,  6,  10,  12,  12,  10,  6, 1], \
    [ 1,  8,  12,  15,  15,  12,  8, 1], \
    [ 1,  6,  10,  12,  12,  10,  6, 1], \
    [ 1,  5,  7,  10,  10,  7,  5, 1], \
    [ 1,  4,  5,  5,  5,  5,  5, 4], \
    [ 0,  0, 0,  0,  0,  0,  0, 0], \
    [ 0,  0, 0,  0,  0,  0,  0, 0], \
]

EVAL_BLACK = [ \
    [ 0,  0, 0,  0,  0,  0,  0, 0], \
    [ 0,  0, 0,  0,  0,  0,  0, 0], \
    [ 1,  4,  5,  5,  5,  5,  4, 1], \
    [ 1,  5,  7,  10,  10,  7,  5, 1], \
    [ 1,  6,  10,  12,  12,  10,  6, 1], \
    [ 1,  8,  12,  15,  15,  12,  8, 1], \
    [ 1,  6,  10,  12,  12,  10,  6, 1], \
    [ 0, 1, 1, 1, 1, 1, 1, 0], \
]

# Step directions by one square
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
    """
    A class to represent the players of a Watch Your Back! game. Determines the
    next action of the player using a Minimax with Alpha-beta pruning
    algorithm, using the `action` method. Updates the internal game board with
    the `update` method. Stores its internal board representation within its
    own board object.
    """
    def __init__(self, colour):
        self.type = WHITE if colour == "white" else BLACK
        self.enemy = BLACK if colour == "white" else WHITE
        self.board = Board(FULL_EMPTY, self.type)
        self.placeMode = True
        self.depthMax = 2

    def action(self, turns):
        """
        Calculate the best action for a given game state, using the defined
        heuristic. Returns a single tuple (e.g. (4,4)) if the game is in
        the placing phase. Returns a tuple of two tuples (e.g. ((4,4),(4,5)))
        if the game is in the moving phase.
        """
        # Shrink the board after each player makes 64, then another 32 moves
        if turns == FIRST_SHRINK-1 or turns == FIRST_SHRINK:
            self.board.shrink((6, 1), (1, 1), (6, 6), (1, 6))

        if turns == SECOND_SHRINK-1 or turns == SECOND_SHRINK:
            self.board.shrink((5, 2), (2, 2), (5, 5), (2, 5))

        # Not in play mode while in the placing stage
        if self.placeMode:
            if turns < MOVE_PHASE_TURNS:
                if self.type == WHITE and (turns == 0 or turns == 1):
                    action = (3,2) if self.board.grid[(3,2)] == EMPTY else (4,2)
                elif self.type == BLACK and (turns == 0 or turns == 1):
                    action = (4,5) if self.board.grid[(4,5)] == EMPTY else (3,5)

                else:

                    action = self.bestmove(self.type, turns)
                self.board.place_piece(action, self.type)

            # Switch to move mode if its going first or second
            if turns == MOVE_PHASE_TURNS-2 or turns == MOVE_PHASE_TURNS-1:
                self.placeMode = False

        # Now in the movement (playing) stage of the game
        else:
            action = self.bestmove(self.type, turns)

            # Only make the move locally if a move is possible and is being
            # returned.
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
                    if self.board.find_piece(front_square) in \
                    ENEMIES[self.board.grid[loc]] \
                    and self.board.find_piece(back_square) in \
                    ENEMIES[self.board.grid[loc]]:

                        self.board.grid[loc] = EMPTY
                        break

        return action

    def heuristic(self, state):
        """
        Calculate the value of a board state with the heuristic. Heuristic
        includes a high weighting towards the difference in the number of
        player pieces vs. opponent pieces, a tendency for pieces to move
        towards the centre of the board as the game progresses, and an
        evaluation of placements via the predetermined evaluation arrays.
        """
        play_eval = 0
        euc_score = 0
        players = []
        enemies = []

        # Calculate a heuristic value for the entire board - post move
        for loc in state:
            if state[loc] == self.type:

                # Evaluate the board based on the evaluation arrays.
                if self.placeMode:
                    if (self.type == WHITE):
                        play_eval += EVAL_WHITE[loc[1]][loc[0]]
                    else:
                        play_eval += EVAL_BLACK[loc[1]][loc[0]]

                # Euclidean distance of player pieces to the centre of the
                # board - want to minimise this.
                euc_score += self.euclidean_distance(loc, (CENTRE, CENTRE))
                players.append(loc)

            if state[loc] == self.enemy:
                enemies.append(loc)

        # Very high weighting on the difference in player pieces vs. opponent
        # pieces. This has the double effect of making our player prioritise
        # both eliminating enemies and self preservation
        kill_diff = (len(players) - len(enemies))*1000

        # Return the composite heuristic value
        return kill_diff - euc_score + play_eval

    def minimax(self, node, depth):
        """
        Return the best move a player can make using the minimax algorithm,
        traversing the game tree to the specified depth. Alpha-beta pruning is
        used to decrease the total number of nodes traversed, significantly
        improving time performance without compromising the final output.
        """

        # Initilise alpha and beta values
        alpha = -INFINITY
        beta = INFINITY

        # Start the recursive step of the algorithm by maximising the player's
        # first move
        best_val = self.maximise(node, alpha, beta, depth)

        # Different state calculating strategies for placement and movement
        children = self.getChildPlacement(node) if self.placeMode else \
        self.getChildren(node)

        # Return (None, None) to represent the forfeiture of a move if there is
        # no available move to make.
        if children:
            best_move = children[0][0]
            move = children[0][1]
        else:
            return None, None

        # Iterate through the initial state's children until the already
        # calculated best move is found, then return it
        for elem in children:
            if self.heuristic(elem[0]) == best_val:
                best_move = elem[0]
                move = elem[1]
                break

        return best_move, move

    def maximise(self, node, alpha, beta, depth):
        """
        The maximising part of the minimax algorithm. It iterates through each
        child state of a given game state, recursively finding the maximum of
        the minimised heuristic values. Alpha-beta pruning is used to avoid
        checking unnecessary heuristic values that would never be used.
        """
        # End recursion when the specified minimax depth is reached
        if depth > self.depthMax:
            return self.heuristic(node)

        max_val = -INFINITY

        # Different state calculating strategies for placement and movement
        children_states = self.getChildPlacement(node) if self.placeMode else \
        self.getChildren(node)
        for state in children_states:

            # Find the highest minimum state value at this minimax tree level
            # The tree is pruned in cases where alpha is too high.
            max_val = \
                max(max_val, self.minimise(state[0], alpha, beta, depth+1))
            alpha = max(alpha, max_val)
            if (beta <= alpha):
                break
        return max_val

    def minimise(self, node, alpha, beta, depth):
        """
        The minimising part of the minimax algorithm. It iterates through each
        child state of a given game state, recursively finding the minimum of
        the maximsed heuristic values. Alpha-beta pruning is used to avoid
        checking unnecessary heuristic values that would never be used.
        """
        # End recursion when the specified minimax depth is reached
        if depth > self.depthMax:
            return self.heuristic(node)

        min_val = INFINITY

        # Different state calculating strategies for placement and movement
        children_states = self.getChildPlacement(node) if self.placeMode \
        else self.getChildren(node)
        for state in children_states:

            # Find the lowest maximum state value at this minimax tree level.
            # The tree is pruned in cases where beta is too low.
            min_val = \
                min(min_val, self.maximise(state[0], alpha, beta, depth+1))
            beta = min(beta, min_val)
            if (beta <= alpha):
                break

        return min_val

    def getChildren(self, node):
        """
        Returns all of a game state's child nodes (states) in the game tree.
        Does this by going through each move available to each player's pieces.
        """
        assert node is not None
        states = []
        for loc in self.board.grid.keys():
            if self.board.grid[loc] == self.type:
                for move in self.board.moves(loc):
                    elims = self.board.makemove(loc, move)
                    states.append((self.board.grid.copy(), move))
                    self.board.undomove(loc, self.type, move, elims)
        return states

    def getChildPlacement(self, node):
        """
        Returns all of an input game state's child nodes (states). Placement is
        different because it also uses `findNeighbours` and undoes piece
        placements differently to piece movements.
        """
        assert node is not None
        states = []
        for loc in self.board.grid.keys():

            # Potential positions are empty, within the player's starting zone,
            # and are not in danger of getting eliminated by the opponent.
            if self.board.grid[loc] == EMPTY and loc in START_ZONE[self.type] \
            and not self.findNeighbours(loc[1], loc[0]):
                elims = self.board.place_piece(loc, self.type)
                states.append((self.board.grid.copy(), loc))

                # Undo the placement
                for data in elims:
                    self.board.grid[data[1]] = data[0]
                self.board.grid[loc] = EMPTY

        return states

    def findNeighbours(self, x, y):
        """
        Determines if a position should be avoided. A placement should be
        avoided if it has the potential for the opponent to eliminate the
        placed piece in the next turn. This is only used in the placing phase
        of the game.
        """
        avoid = False

        # Positions one step in each direction from the starting pos
        left = (x-1, y)
        right = (x+1, y)
        up = (x, y-1)
        down = (x, y+1)

        # Check if the position is susceptible to elimination in the opponent's
        # next turn
        if (down in self.board.grid and up in self.board.grid):
            up_pos = self.board.grid[up]
            down_pos = self.board.grid[down]

            if (up_pos == self.enemy and down_pos == EMPTY \
            or down_pos == self.enemy and up_pos == EMPTY):
                avoid = True

        if (right in self.board.grid and left in self.board.grid):
            right_pos = self.board.grid[right]
            left_pos = self.board.grid[left]

            if (right_pos == self.enemy and left_pos == EMPTY \
            or left_pos == self.enemy and right_pos == EMPTY):
                avoid = True

        return avoid

    def euclidean_distance(self, pos1, pos2):
        """Returns the eucliean distance between two positions."""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def bestmove(self, type, turns):
        """
        Returns the best move as determined through the Minimax and Alpha-beta
        pruning algorithms, and the strategy defined by the heuristic.

        Has a case for both the placing and moving phases.
        """
        if self.placeMode:
            return self.minimax(self.board.grid, 0)[1]
        else:
            new_state, move = self.minimax(self.board.grid, 0)

            # Forfeits the move if there are no options.
            if (new_state, move) == (None, None):
                return None, None

            # Recreates the move through checking the best board state.
            for loc in self.board.grid.keys():
                if self.board.grid[loc] == self.type:
                    if new_state[loc] == EMPTY:
                        start = loc

            return start, move

    def update(self, action):
        """
        Receives the opponent's action and updates the player's internal game
        configuration. Handles both placements (single position tuple) and
        movements (tuple of two position tuples).
        """
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
        an initial board configuration, which will be empty at the start of
        the game.
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
        Takes as input, then sets the new corner pieces.
        """
        # "Delete" each piece that is outside of the new boundaries.
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
        Place a piece with the input type and position. Update the board
        including eliminating any nearby pieces that are newly surrounded.

        Return a list of pieces eliminated by this move (to be passed back to
        `undomove()` if the move is to be reversed).
        """
        self.grid[newpos] = my_type

        return self.remove_pieces_after_move(newpos, my_type)

    def moves(self, pos):
        """
        Compute and return a list of the available moves for the piece at pos,
        based on the current board state.
        """
        possible_moves = []
        for direction in DIRECTIONS:
            adjacent_square = step(pos, direction)

            # Attempt a move to an adjacent position
            if self.find_piece(adjacent_square) == EMPTY:
                possible_moves.append(adjacent_square)
                continue

            # If adjacent position is occupied, attempt a jump move to an
            # opposite position.
            opposite_square = step(adjacent_square, direction)
            if self.find_piece(opposite_square) == EMPTY:
                possible_moves.append(opposite_square)
        return possible_moves

    def makemove(self, oldpos, newpos):
        """
        Carry out a move from this piece's current position to the position
        `newpos`. Update the board including eliminating any nearby pieces
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

                eliminated_piece = \
                    (self.find_piece(adjacent_square), adjacent_square)
                self.grid[adjacent_square] = EMPTY
                eliminated_pieces.append(eliminated_piece)
                #eliminated_pieces.append(eliminated_piece)

        # check horizontally and vertically to see if the piece itself gets
        # eliminated
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
