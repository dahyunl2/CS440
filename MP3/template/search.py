import heapq
# You do not need any other imports

def best_first_search(starting_state):
    '''
    Implementation of best first search algorithm

    Input:
        starting_state: an AbstractState object

    Return:
        A path consisting of a list of AbstractState states
        The first state should be starting_state
        The last state should have state.is_goal() == True
    '''
    # we will use this visited_states dictionary to serve multiple purposes
    # - visited_states[state] = (parent_state, distance_of_state_from_start)
    #   - keep track of which states have been visited by the search algorithm
    #   - keep track of the parent of each state, so we can call backtrack(visited_states, goal_state) and obtain the path
    #   - keep track of the distance of each state from start node
    #       - if we find a shorter path to the same state we can update with the new state 
    # NOTE: we can hash states because the __hash__/__eq__ method of AbstractState is implemented
    visited_states = {starting_state: (None, 0)}

    # The frontier is a priority queue
    # You can pop from the queue using "heapq.heappop(frontier)"
    # You can push onto the queue using "heapq.heappush(frontier, state)"
    # NOTE: states are ordered because the __lt__ method of AbstractState is implemented
    frontier = [starting_state]
    
    # # TODO(III): implement the rest of the best first search algorithm
    # # HINTS:
    # #   - add new states to the frontier by calling state.get_neighbors()
    # #   - check whether you've finished the search by calling state.is_goal()
    # #       - then call backtrack(visited_states, state)...
    # # Your code here ---------------
    # check is it a goal
    # if it is not, look at neighbors
    #call neighbors and go over all neighbor in the neighbors
    #for each one, sometimes need to be pused into forntier
    #if it is not in visited, push it into frontier and update visited
    #when I push it in to frontier, I can update visited
    #if it is visited, compare cost
    # if you do not find the goal return an empty list

    while len(frontier)>0:
        
        state=heapq.heappop(frontier)
        
        if state.is_goal():
            res=backtrack(visited_states,state)
            res.reverse()
            return res
        else:
            neighbors=state.get_neighbors()
            for n in neighbors:
                if n in visited_states:
                    if n.dist_from_start<visited_states[n][1]:
                        visited_states[n] = (state, n.dist_from_start)
                        heapq.heappush(frontier, n)
                else:
                    heapq.heappush(frontier, n)
                    visited_states[n]=(state, n.dist_from_start)
    return []



# TODO(III): implement backtrack method, to be called by best_first_search upon reaching goal_state
# Go backwards through the pointers in visited_states until you reach the starting state
# NOTE: the parent of the starting state is None
def backtrack(visited_states, goal_state):
    path = []
    # Your code here ---------------
    state=goal_state
    path.append(state)
    while visited_states[state][1]!=0:
        path.append(visited_states[state][0])
        state=visited_states[state][0]
    # ------------------------------
    return path