import numpy as np

def kmap_wrap(kmap):
    rows, cols = kmap.shape
    groups = []

    def is_valid_cell(r, c, visited):
        return (0 <= r < rows and 0 <= c < cols and 
                kmap[r][c] == 1 and (r, c) not in visited)

    def find_group(r, c, visited):
        stack = [(r, c)]
        group = []
        while stack:
            cr, cc = stack.pop()
            if (cr, cc) not in visited:
                visited.add((cr, cc))
                group.append((cr, cc))

                # Check adjacent cells with wrap-around
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr = (cr + dr) % rows
                    nc = (cc + dc) % cols
                    if is_valid_cell(nr, nc, visited):
                        stack.append((nr, nc))

        return group

    visited = set()
    for r in range(rows):
        for c in range(cols):
            if is_valid_cell(r, c, visited):
                group = find_group(r, c, visited)
                groups.append(group)

    return groups

# Example usage
kmap = np.array([
    [1, 1, 0, 0],
    [0, 1, 1, 0],
    [0, 0, 1, 1],
    [1, 0, 0, 1]
])

groups = kmap_wrap(kmap)
print("Groups:")
for group in groups:
    print(group)