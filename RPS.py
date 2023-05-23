import random

def player(prev_play, opponent_history=[]):
    moves = ['R', 'P', 'S']

    # Counter strategy against specific bot moves
    if 'R' in opponent_history:
        return 'P'
    elif 'P' in opponent_history:
        return 'S'
    elif 'S' in opponent_history:
        return 'R'

    # Random move if no previous play
    return random.choice(moves)
  
  
from RPS_game import play, player
# Play 1000 games against a bot and see the results of each game
play(player, bot_name, 1000, verbose=True)
