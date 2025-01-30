# K Map Solver
from enum import Enum
import random
from manim import *
import numpy as np
from itertools import combinations

class grouped_terms:
  def __init__(self, start_corner: tuple[int, int], end_corner: tuple[int, int], delta_coords: tuple[int, int]):
    self.start = start_corner
    self.final = end_corner
    self.delta = (0, 0) if delta_coords is None else delta_coords


class tileStatus(Enum):
  INVALID = 0
  UNEXPLORED = 1
  VISITED = 2
  
def create_super_k_matrix(matrix):
  matrix = np.array(matrix)
  # Get the dimensions of the matrix
  m, n = matrix.shape

  # Split indices
  mid_m, mid_n = m // 2, n // 2

  top_left, top_right       = matrix[:mid_m, :mid_n], matrix[:mid_m, mid_n:]
  bottom_left, bottom_right = matrix[mid_m:, :mid_n], matrix[mid_m:, mid_n:]
  
  top = np.hstack((bottom_right, bottom_left))
  bottom = np.hstack((top_right, top_left))
  recurssion_square = lambda x, y : x((y, y))
  
  super_matrix = recurssion_square(np.hstack, recurssion_square(np.vstack, np.vstack((top, bottom))))
  return super_matrix

def print_bool_array(array):
  print(np.matrix([[int(value) for value in row] for row in array]))
  
def color_selector(predefined_colors):
    selected_colors = set()
    
    def select_color():
        # If all predefined colors are selected, generate a new one
        if len(selected_colors) == len(predefined_colors):
            new_color = "#{:02x}{:02x}{:02x}".format(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            return new_color
        else:
            # Select a unique color from the predefined list
            color = random.choice([color for color in predefined_colors if (color.to_hex() if type(color) is ManimColor else str(color)) not in selected_colors])
            selected_colors.add(color.to_hex() if type(color) is ManimColor else str(color))
            return color
    
    return select_color

def create_min_groups(matrix: list[list[int]]):
  matrix = np.array(matrix)
  
  super_matrix = create_super_k_matrix(matrix)
  t_width, t_length = super_matrix.shape
  table_size = (t_width, t_length)
  visited = [[False for _ in range(t_length)] for _ in range(t_width)]
  
  min_terms = []
  coord_shift = lambda x, k, shift: x % (k//2) + shift//2
  
  def is_valid(x, y):
    return super_matrix[y][x] == 1 and 0 <= x < t_width and 0 <= y < t_length 

  def is_filled(matrix, top_left, bottom_right, signature = 1):
    start_row, start_col = top_left
    end_row, end_col = bottom_right
    for x in range(start_col, end_col + 1):
      for y in range(start_row, end_row + 1):
        if matrix[x][y] != signature:
            return False
    return True


  def find_rectangle(start_x, start_y):

    max_x, max_y = start_x, start_y
    delta_x, delta_y = 0, 0
    is_delx_possible, is_dely_possible = tileStatus.UNEXPLORED, tileStatus.UNEXPLORED
    use_remaining = False
    
    def satisfy(axis: int, max_x: int, max_y: int, extension_x: int, extension_y:int ):
      """Check if a given direction of a corner satisfy rules to be grouped

      Args:
          axis (int): _description_
          max_x (int): _description_
          max_y (int): _description_
          extension_x (int): _description_
          extension_y (int): _description_

      Returns:
          _type_: _description_
      """
      g_terms = grouped_terms((start_x, start_y), (max_x, max_y), (extension_x, extension_y))

      start = g_terms.start[axis]
      delta = g_terms.delta[axis]
      final = [max_x, max_y]
      
      table_dim = table_size[axis] 
      final[axis] = start + 1 if delta == 0 else start + delta * 2 + 1
      
      g_terms.final = tuple(final)
      
      if g_terms.final[axis] >= table_dim:
        return tileStatus.INVALID
      
      unexplored = not is_filled(visited, g_terms.start, g_terms.final, True)
      filled = is_filled(super_matrix, g_terms.start, g_terms.final)
      limit = table_dim // 2

      if unexplored and filled and (delta <= limit):
          return tileStatus.UNEXPLORED
      elif filled and (delta <= limit):
          return tileStatus.VISITED
      else:
          return tileStatus.INVALID
    
    add_delta = lambda dim: 1 if dim == 0 else dim * 2
    
    while is_dely_possible != tileStatus.INVALID or is_delx_possible != tileStatus.INVALID:
      possible_expansion = lambda tile: tile == tileStatus.UNEXPLORED or tile == tileStatus.VISITED and use_remaining

      is_delx_possible = satisfy(0, max_x, max_y, delta_x, delta_y)
      if possible_expansion(is_delx_possible):
        delta_x = add_delta(delta_x)
        max_x = start_x + delta_x

      is_dely_possible = satisfy(1, max_x, max_y, delta_x, delta_y)
      if possible_expansion(is_dely_possible):
        delta_y = add_delta(delta_y)
        max_y = start_y + delta_y

      if is_dely_possible != tileStatus.UNEXPLORED and is_delx_possible != tileStatus.UNEXPLORED:
        use_remaining = True

    return grouped_terms((start_x, start_y), (max_x, max_y))

  for y in range(t_width):
    for x in range(t_length):
      if not is_valid(x, y):
        continue
      
      new_rect = find_rectangle(x, y)
      (start_x, start_y, max_x, max_y) = new_rect.start, new_rect.final

      submatrix = [row[new_rect[0]:new_rect[2] + 1] for row in visited[new_rect[1]:new_rect[3] + 1]]
      
      if not any(not all(row) for row in submatrix):
        continue
      
      # Mark cells as visited
      for ky in range(start_y, max_y + 1):
        for kx in range(start_x, max_x + 1):
          xs = lambda shift: coord_shift(kx, t_length, shift) # xshifted
          ys = lambda shift: coord_shift(ky, t_width, shift) # yshifted
          
          visited[ys(0)][xs(t_length)] = True
          visited[ys(t_width)][xs(0)] = True
          visited[ys(t_width)][xs(t_length)] = True
          visited[ys(0)][xs(0)] = True
      coords = lambda shift_x, shift_y: [(start_x - t_length// 4 + shift_x, start_y - t_width//4 + shift_y), (max_x - t_length// 4 + shift_x, max_y - t_width//4 + shift_y)]
      min_terms.append([coords(0, 0), coords(t_length//2, 0), coords(0, t_width//2), coords(t_length//2, t_width//2)])


  return min_terms
