# K Map Solver
from enum import Enum
import random
import re
from manim import *
from pyeda.inter import *
import numpy as np
from itertools import combinations

class term_group:
  def __init__(self, start_corner: tuple[int, int], end_corner: tuple[int, int], delta_coords: tuple[int, int] = None):
    self.start = list(start_corner)
    self.final = list(end_corner)
    self.delta = list((0, 0)) if delta_coords is None else list(delta_coords)
    self.size = 1
    
  def update_end_corner(self, dim, new_delta):
    self.delta[dim] = new_delta
    self.final[dim] = self.start[dim] + self.delta[dim] - 1
    
  def update_size(self):
    self.size *= 2

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

def iterate_2d(x_codomain, y_range, func):
  for y in y_range:
    for x in x_codomain:
      func(x, y)

def create_min_groups(matrix: list[list[int]]):
  matrix = np.array(matrix)
  
  super_matrix = create_super_k_matrix(matrix)
  t_width, t_length = super_matrix.shape
  table_size = (t_width, t_length)
  visited = [[False for _ in range(t_length)] for _ in range(t_width)]
  
  min_terms = []
  coord_shift = lambda x, k, shift: x % (k//2) + shift//2
  
  def is_valid(x, y):
    nonlocal super_matrix
    return super_matrix[y][x] == 1 and 0 <= x < t_width and 0 <= y < t_length 

  def is_filled(matrix, top_left, bottom_right, signature = 1):
    start_row, start_col = top_left
    end_row, end_col = bottom_right
    try:
      for x in range(start_col, end_col + 1):
        for y in range(start_row, end_row + 1):
          if matrix[x][y] != signature:
              return False
    except:
      return False
    return True


  add_delta = lambda dim: 2 ** dim
  
  def find_rectangle(start_x, start_y):
    rect_group = term_group((start_x, start_y), (start_x, start_y))
    is_delx_possible, is_dely_possible = tileStatus.UNEXPLORED, tileStatus.UNEXPLORED
    use_remaining = False
    
    def satisfy(axis: int, max_coord: tuple[int, int], extensnion_vect: tuple[int, int]):
      """Check if a given direction of a corner satisfy rules to be grouped

      Args:
          axis (int): dimension to check satisfaction
          max_x (int): end x-dimension of the corner that fulfills 'satisfaction' requirements
          max_y (int): end y-dimension of the corner that fulfills 'satisfaction' requirements
          extension_x (int): width of expansion
          extension_y (int): length of expansion

      Returns:
          tileStatus: Enum denoting expansion status.
      """
      nonlocal start_x, start_y
      g_terms = term_group((start_x, start_y), max_coord, extensnion_vect)

      start = (g_terms.start)[axis]
      delta = (g_terms.delta)[axis]
      final = list(max_coord)
      
      table_dim = table_size[1-axis]
      
      size = g_terms.size
      final[axis] = start + add_delta(delta) - 1
      g_terms.final = tuple(final)
      
      if g_terms.final[axis] >= table_dim:
        return tileStatus.INVALID
      
      unexplored  = not is_filled(visited, g_terms.start, g_terms.final, True)
      filled      = is_filled(super_matrix, g_terms.start, g_terms.final)
      limit       = table_dim // 2

      if unexplored and filled and (delta < limit):
          return tileStatus.UNEXPLORED
      elif filled and (delta < limit):
          return tileStatus.VISITED
      else:
          return tileStatus.INVALID
    
    
    possible_expansion = lambda tile: tile == tileStatus.UNEXPLORED or tile == tileStatus.VISITED and use_remaining
    
    def check_updates(dim):
      is_deldim_possible = satisfy(dim, rect_group.final, rect_group.delta)
      if possible_expansion(is_deldim_possible):
        rect_group.update_end_corner(dim, add_delta(rect_group.delta[dim]))
      return is_deldim_possible
    
    while is_dely_possible != tileStatus.INVALID or is_delx_possible != tileStatus.INVALID:
      

      is_delx_possible = check_updates(0)
      is_dely_possible = check_updates(1)

      if is_dely_possible != tileStatus.UNEXPLORED and is_delx_possible != tileStatus.UNEXPLORED:
        use_remaining = True

    return rect_group
  
  def update_visited(kx, ky):
    nonlocal visited
    xs = lambda shift: coord_shift(kx, t_length, shift) # xshifted
    ys = lambda shift: coord_shift(ky, t_width, shift) # yshifted
    
    visited[ys(0)][xs(0)] = True
    visited[ys(0)][xs(t_length)] = True
    visited[ys(t_width)][xs(t_length)] = True
    visited[ys(t_width)][xs(0)] = True
    
  
  def check_matrix_cell(cell_x, cell_y):
    if not is_valid(cell_x, cell_y):
      return
      
    nonlocal t_length, t_width, min_terms
    new_rect = find_rectangle(cell_x, cell_y)
    (start_x, start_y), (max_x, max_y) = new_rect.start, new_rect.final

    submatrix = [row[new_rect.start[0]:new_rect.final[0] + 1] for row in visited[new_rect.start[1]:new_rect.final[1] + 1]]
    
    if not any(not all(row) for row in submatrix):
      return
    
    # Mark cells as visited
    iterate_2d(range(start_x, max_x + 1), range(start_y, max_y + 1), update_visited)
    coords = lambda shift_x, shift_y: [\
      (start_x - t_length// 4 + shift_x, start_y - t_width//4 + shift_y), \
      (max_x - t_length// 4 + shift_x, max_y - t_width//4 + shift_y)
      ]
    min_terms.append([coords(0, 0), 
                      coords(t_length//2, 0), 
                      coords(0, t_width//2), 
                      coords(t_length//2, t_width//2),
                      coords(-t_length//2, t_width//2),
                      coords(0, -t_width//2), 
                      coords(t_length//2, -t_width//2),
                      coords(-t_length//2, 0), 
                      ])
  
  iterate_2d(range(t_length), range(t_width), check_matrix_cell)
      
  return min_terms

def graycode_gen(n: int):
    """Generate Gray code ordering for n bits."""
    result = [i ^ (i >> 1) for i in range(2**n)]
    return result

def truth_table_to_kmap(f_in):
    # Size can be odd like for a 3 table.
    f = espresso_exprs(f_in.simplify().to_dnf())[0]
    simplified_input = str(f)[3:][:-1]
    comma_regex = r',\s*(?![^()]*\))'
    simplified_input = re.split(comma_regex, simplified_input)
    simplified_input

    simplified_input  

    truth_table = expr2truthtable(f)
    num_vars = len(truth_table.support)
    half_vars = num_vars // 2
    num_rows = 2**half_vars
    num_cols = 2**(num_vars - half_vars)
    
    row_graycode, col_graycode = graycode_gen(half_vars), graycode_gen(num_vars - half_vars)
    
    kmap = np.zeros((num_rows, num_cols), dtype=int)
    
    row_vars = list(truth_table.support)[:half_vars]
    col_vars = list(truth_table.support)[half_vars:]

    for position, value in truth_table.iter_relation():

      row, col = 0, 0
      position = tuple(position.values())

      match len(position):
        case 4:
          row = position[:half_vars][0] * 2 + position[1]
          col = position[half_vars:][0] * 2 + position[3]
        case 3:
          row = position[:half_vars][0]
          col = position[half_vars:][0] * 2 + position[half_vars:][1]
        case 2:
          row = position[:half_vars][0] 
          col = position[half_vars:][0] 
        case 1:
          row = 0
          col = position[half_vars:][0] 
      row_index = row_graycode[row]
      col_index = col_graycode[col]
      kmap[row_index, col_index] = value
    return ((row_vars, col_vars), kmap)