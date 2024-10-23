import itertools
import random
import string
import sys

ALL_TILES = tuple((x, y) for x, y in itertools.product(range(4), range(4)))
ALL_INDICES = tuple(range(16))


def read_words():
    with open('words.txt') as f:
        return set(w.lower()
                   for w in f.read().replace('"', '').splitlines()
                   if all(c in string.ascii_letters for c in w) and w.lower() == w and len(w) > 2 and len(set(w)) <= 16)


def simple_board(words_list, all_words_list=None):
    if all_words_list is not None and len(words_list) < 6:
        words_list = words_list + random.sample(all_words_list, 6 - len(words_list))
    letters = list(''.join(random.sample(words_list, 6))[:16])
    random.shuffle(letters)
    return letters


def print_board(letters):
    assert len(letters) == 16
    print('\n'.join(' '.join(c or '.' for c in letters[i:i+4]).upper() for i in range(0, 16, 4)))


def words_in_board_dupes(board, word_prefixes, words, word_so_far='', x=None, y=None):
    if x is None:
        assert y is None
        for x in range(4):
            for y in range(4):
                yield from words_in_board_dupes(board, word_prefixes, words, word_so_far, x, y)
        return
    if x < 0 or x >= 4 or y < 0 or y >= 4 or board[y * 4 + x] is None:
        return
    word_so_far += board[y * 4 + x]
    if word_so_far not in word_prefixes:
        return
    if word_so_far in words:
        yield word_so_far
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            if not dx and not dy:
                continue
            yield from words_in_board_dupes(board, word_prefixes, words, word_so_far, x + dx, y + dy)


def all_filled_boards(partial_board):
    none_slots = list(i for i, cell in enumerate(partial_board) if cell is None)
    for filled_vals in itertools.product(*itertools.repeat(string.ascii_lowercase, len(none_slots))):
        for i, letter in zip(none_slots, filled_vals):
            partial_board[i] = letter
        yield partial_board
    for i in none_slots:
        partial_board[i] = None


def anneal_board(board, board_score):
    # if None in board:
    #     original_board_score = board_score
    #     board_score = lambda b: max(original_board_score(b) for b in all_filled_boards(b))
    while True:
        best_swap = None, None
        original_score = best_score = board_score(board)
        for i, j in itertools.combinations(range(16), 2):
            board[i], board[j] = board[j], board[i]
            score = board_score(board)
            if score > best_score:
                best_swap = i, j
                best_score = score
            else:
                board[i], board[j] = board[j], board[i]
        i, j = best_swap
        if i is None:
            assert j is None
            return
        print(f'ANNEALING {original_score} -> {best_score}')


def words_in_board(board, word_prefixes, words):
    return set(words_in_board_dupes(board, word_prefixes, words))


NON_OVERLAPPING_INDICES_CACHE = []


def non_overlapping_indices(n):
    while n >= len(NON_OVERLAPPING_INDICES_CACHE):
        cl = len(NON_OVERLAPPING_INDICES_CACHE)
        NON_OVERLAPPING_INDICES_CACHE.append(tuple(order
                                                   for order in itertools.permutations(range(cl))
                                                   if all(i != o for i, o in zip(range(cl), order))))
    return NON_OVERLAPPING_INDICES_CACHE[n]


def _precompute_indices(last_index):
    last_y, last_x = divmod(last_index, 4)
    indices = [
        (4 * y + x)
        for x, y in itertools.product(range(max(0, last_x - 1), min(4, last_x + 2)),
                                      range(max(0, last_y - 1), min(4, last_y + 2)))
        if x != last_x or y != last_y
    ]
    # Note: We precompute values for speed *BUT* also shuffle so re-running the program gives varied boards.
    random.shuffle(indices)
    return tuple(indices)


_PRECOMPUTED_INDICES = [_precompute_indices(index) for index in range(16)]
_PRECOMPUTED_INDICES.append(tuple(sorted(ALL_INDICES, key=lambda i: random.random())))


def _partial_board_helper_fast_unsafe(word_remaining, tile_so_far, last_index):
    if not word_remaining:
        yield tile_so_far
        return

    next_letter = word_remaining.pop()
    for index in _PRECOMPUTED_INDICES[last_index]:
        cur_space = tile_so_far[index]
        if cur_space is None or cur_space == next_letter:
            tile_so_far[index] = next_letter
            yield from _partial_board_helper_fast_unsafe(word_remaining, tile_so_far, index)
            tile_so_far[index] = cur_space
    word_remaining.append(next_letter)


def partial_boards_fast_unsafe(word, partial_board):
    word_remaining = list(word)
    word_remaining.reverse()
    for partial_board in _partial_board_helper_fast_unsafe(word_remaining, partial_board, -1):
        yield partial_board


def partial_boards_multi(desired_words, partial_board, words_in_board, min_words):
    if len(desired_words) + len(words_in_board) < min_words:
        return
    if not desired_words:
        yield partial_board, words_in_board
        return
    for i, next_word in enumerate(desired_words):
        words_in_board.append(next_word)
        for b in partial_boards_fast_unsafe(next_word, partial_board):
            yield from partial_boards_multi(desired_words[i+1:], b, words_in_board, min_words)
        words_in_board.pop()


def optimize_for_specific_words_demo(desired_words=None, min_words=1):
    if desired_words is None:
        desired_words = ['red', 'orange', 'yellow', 'green', 'blue', 'purple', 'violet', 'black', 'white', 'brown', 'pink']

    # Note: The sorting isn't necessary but does speed things up.
    desired_words = sorted(desired_words, key=len, reverse=True)
    best_len_so_far = 0
    words_in_board_seen = set()
    for board, words_in_board in partial_boards_multi(list(desired_words), [None] * 16, [], min_words):
        words_in_board = frozenset(words_in_board)
        if len(words_in_board) >= best_len_so_far and words_in_board not in words_in_board_seen:
            words_in_board_seen.add(words_in_board)
            best_len_so_far = len(words_in_board)
            print(f'number of unique letters: {len(set(c for w in desired_words for c in w))}')
            print_board(board)
            print(f'{len(words_in_board)=}  {sorted(words_in_board)}')
            print()


def optimize_for_generic_metric_demo():
    words = read_words()

    desired_words = ['red', 'orange', 'yellow', 'green', 'blue', 'purple', 'violet', 'black', 'white', 'brown', 'pink']
    word_prefixes = set(w[:i] for w in itertools.chain(words, desired_words) for i in range(len(w) + 1))

    # for _ in range(1):
    for score_name, score_function in (
        ('number of words', lambda board: sum(1 for _ in words_in_board(board, word_prefixes, words))),
        ('more long words', lambda board: sum(len(w)**3 for w in words_in_board(board, word_prefixes, words))),
        ('desired words and long words', lambda board: sum((1_000_000_000 if w in desired_words else len(w)**3)
                                                           for w in words_in_board(board, word_prefixes, words)))
    ):
        board = simple_board(list(desired_words), list(words))
        print('Original board:')
        print_board(board)
        print(f'Annealing to optimize for "{score_name}"')
        anneal_board(board, score_function)
        print_board(board)
        print('Words in final board:', sorted(words_in_board(board, word_prefixes, words)))
        print('Desired words in final board:', sorted(
            w for w in words_in_board(board, word_prefixes, words) if w in desired_words))
        print()


def num_desired_words_in_board(board, desired_words, desired_words_prefixes):
    return sum(1 for w in words_in_board(board, desired_words_prefixes, desired_words))


def optimize_for_specific_words_then_generic_metric_demo(desired_words=None, min_words=1, score_function=None):
    if desired_words is None:
        desired_words = ['red', 'orange', 'yellow', 'green', 'blue', 'purple', 'violet', 'black', 'white', 'brown', 'pink']
    if score_function is None:
        # Strong bias towards long words
        score_function = lambda board: sum(len(w)**3 for w in words_in_board(board, word_prefixes, words))

    words = read_words()
    words_list = list(words)
    word_prefixes = set(w[:i] for w in itertools.chain(words, desired_words) for i in range(len(w) + 1))
    desired_words_prefixes = set(w[:i] for w in desired_words for i in range(len(w) + 1))

    # Note: The sorting isn't necessary but does speed things up.
    best_len_so_far = 0
    desired_words = sorted(desired_words, key=len, reverse=True)
    for board, wib in partial_boards_multi(list(desired_words), [None] * 16, [], min_words):
        if len(wib) <= best_len_so_far:
            continue
        board = list(board)
        if None in board:  # If this is a partial board, then fill it in!
            filled_board = simple_board(words_list)
            for i, (original_cell, backup_cell) in enumerate(zip(board, filled_board)):
                if original_cell is None:
                    board[i] = backup_cell
        best_len_so_far = len(wib)
        print('Original board:')
        print_board(board)
        anneal_board(board, lambda b: (num_desired_words_in_board(b, desired_words, desired_words_prefixes), score_function(b)))
        print_board(board)
        print('Words in final board:', sorted(words_in_board(board, word_prefixes, words)))
        print('Desired words in final board:', sorted(
            w for w in words_in_board(board, word_prefixes, words) if w in desired_words))
        print()
        print()


def main(argv):
    desired_words = argv[1:] if len(argv) > 1 else None
    # optimize_for_generic_metric_demo()
    # optimize_for_specific_words_demo(desired_words)
    optimize_for_specific_words_then_generic_metric_demo(desired_words)


if __name__ == '__main__':
    main(sys.argv)
