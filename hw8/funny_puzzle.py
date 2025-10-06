import heapq

def state_check(state):
    """check the format of state, and return corresponding goal state.
       Do NOT edit this function."""
    non_zero_numbers = [n for n in state if n != 0]
    num_tiles = len(non_zero_numbers)
    if num_tiles == 0:
        raise ValueError('At least one number is not zero.')
    elif num_tiles > 9:
        raise ValueError('At most nine numbers in the state.')
    matched_seq = list(range(1, num_tiles + 1))
    if len(state) != 9 or not all(isinstance(n, int) for n in state):
        raise ValueError('State must be a list contain 9 integers.')
    elif not all(0 <= n <= 9 for n in state):
        raise ValueError('The number in state must be within [0,9].')
    elif len(set(non_zero_numbers)) != len(non_zero_numbers):
        raise ValueError('State can not have repeated numbers, except 0.')
    elif sorted(non_zero_numbers) != matched_seq:
        raise ValueError('For puzzles with X tiles, the non-zero numbers must be within [1,X], '
                          'and there will be 9-X grids labeled as 0.')
    goal_state = matched_seq
    for _ in range(9 - num_tiles):
        goal_state.append(0)
    return tuple(goal_state)

def get_manhattan_distance(from_state, to_state):
    distance = 0
    for i, tile in enumerate(from_state):
        if tile != 0:
            current_x, current_y = divmod(i, 3)
            goal_index = to_state.index(tile)
            goal_x, goal_y = divmod(goal_index, 3)
            distance += abs(current_x - goal_x) + abs(current_y - goal_y)
    return distance

def get_succ(state):
    succ_states = []
    empty_indices = [i for i, x in enumerate(state) if x == 0]
    
    for empty_index in empty_indices:
        row, col = divmod(empty_index, 3)
        moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right
        
        for dx, dy in moves:
            new_row, new_col = row + dx, col + dy
            if 0 <= new_row < 3 and 0 <= new_col < 3:
                new_index = new_row * 3 + new_col
                # Skip if we're trying to swap with another empty position
                if state[new_index] == 0:
                    continue
                new_state = list(state)  # Convert to list to ensure mutability
                new_state[empty_index], new_state[new_index] = new_state[new_index], new_state[empty_index]
                succ_states.append(new_state)
    
    return sorted(succ_states)

def print_succ(state):
    goal_state = state_check(state)
    succ_states = get_succ(state)
    for succ_state in succ_states:
        h_value = get_manhattan_distance(succ_state, goal_state)
        print(succ_state, "h={}".format(h_value))

def solve(state):
    goal_state = state_check(state)
    
    def is_solvable(s):
        non_zero_state = [n for n in s if n != 0]
        inversions = sum(
            1 for i in range(len(non_zero_state)) for j in range(i + 1, len(non_zero_state)) 
            if non_zero_state[i] > non_zero_state[j]
        )
        return inversions % 2 == 0

    if not is_solvable(state):
        print(False)
        return

    pq = []
    visited = set()
    parent_map = {}
    heapq.heappush(pq, (0, state, (0, get_manhattan_distance(state, goal_state), -1)))
    visited.add(tuple(state))
    parent_map[tuple(state)] = None
    max_queue_length = 0

    while pq:
        max_queue_length = max(max_queue_length, len(pq))
        _, current_state, (g, _, _) = heapq.heappop(pq)

        if tuple(current_state) == goal_state:
            print(True)
            path = []
            while current_state is not None:
                path.append((current_state, g))
                current_state = parent_map[tuple(current_state)]
            path.reverse()
            for p_state, moves in path:
                print(p_state, "h={}".format(get_manhattan_distance(p_state, goal_state)), "moves: {}".format(moves))
            print("Max queue length: {}".format(max_queue_length))
            return

        for succ_state in get_succ(current_state):
            succ_tuple = tuple(succ_state)
            if succ_tuple not in visited:
                visited.add(succ_tuple)
                new_g = g + 1
                new_h = get_manhattan_distance(succ_state, goal_state)
                heapq.heappush(pq, (new_g + new_h, succ_state, (new_g, new_h, g)))
                parent_map[succ_tuple] = current_state

if __name__ == "__main__":
    # You can test the functions here
    print_succ([2,5,1,4,0,6,7,0,3])
    print()
    solve([2,5,1,4,0,6,7,0,3])
