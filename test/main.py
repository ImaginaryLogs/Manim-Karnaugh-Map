from manim import *

class KarnaughMap(Scene):
    def construct(self):
        # Example matrix (you can replace this with any n x m matrix)
        matrix = [[0, 1, 1, 0],
                  [1, 0, 0, 1],
                  [0, 1, 0, 1],
                  [1, 0, 1, 0]]

        # Call the function to generate the K-map
        kmap, labels = self.create_kmap(matrix)

        # Add a title
        title = Text("Karnaugh Map", font_size=36).to_edge(UP)
        self.play(Write(title))

        # Add the K-map labels and grid to the scene
        self.play(Create(labels), Create(kmap))

        # Find and visualize the minterm rectangles
        rectangles = self.find_minterm_rectangles(matrix)
        rectangle_objects = self.create_rectangle_objects(rectangles, kmap)
        self.play(AnimationGroup(*[Create(rect) for rect in rectangle_objects]))

        self.wait(2)

    def create_kmap(self, matrix):
        """
        Create a flat Karnaugh Map representation for an n x m matrix with Gray code labels.
        """
        n = len(matrix)      # Number of rows
        m = len(matrix[0])   # Number of columns

        # Constants for layout
        cell_width = 1.0
        cell_height = 1.0
        cell_margin = 0

        # Group to hold all K-map elements
        kmap_group = VGroup()
        labels_group = VGroup()

        # Gray code generation for labels
        def gray_code(num_bits):
            return [bin(i ^ (i >> 1))[2:].zfill(num_bits) for i in range(2 ** num_bits)]

        row_labels = gray_code(int.bit_length(max(1, n - 1)))
        col_labels = gray_code(int.bit_length(max(1, m - 1)))

        # Generate the grid of the K-map
        for i, row in enumerate(matrix):
            for j, value in enumerate(row):
                # Create the rectangle for the cell
                rect = Rectangle(width=cell_width, height=cell_height, color=WHITE)
                rect.move_to(np.array([(j - m / 2) * (cell_width + cell_margin) + cell_width / 2,
                                       -(i - n / 2) * (cell_height + cell_margin) - cell_height / 2,
                                       0]))

                # Add the value as a text in the center of the cell
                text = Text(str(value), font_size=24)
                text.move_to(rect.get_center())

                # Group the rectangle and text together
                cell_group = VGroup(rect, text)

                # Add to the overall K-map group
                kmap_group.add(cell_group)

        # Add row labels
        for i, label in enumerate(row_labels):
            label_text = Text(label, font_size=24)
            label_text.next_to(kmap_group[0].get_left(), LEFT, buff=0.5)
            label_text.shift(DOWN * i * (cell_height + cell_margin))
            labels_group.add(label_text)

        # Add column labels
        for j, label in enumerate(col_labels):
            label_text = Text(label, font_size=24)
            label_text.next_to(kmap_group[0].get_top(), UP, buff=0.5)
            label_text.shift(RIGHT * j * (cell_width + cell_margin))
            labels_group.add(label_text)

        return kmap_group, labels_group

    def find_minterm_rectangles(self, matrix):
        """
        Find and return rectangles that select minterms in the K-map.
        """
        n = len(matrix)      # Number of rows
        m = len(matrix[0])   # Number of columns

        visited = [[False for _ in range(m)] for _ in range(n)]
        rectangles = []

        def is_valid(i, j):
            return 0 <= i < n and 0 <= j < m and matrix[i][j] == 1 and not visited[i][j]

        def find_rectangle(start_i, start_j):
            """Find the largest rectangle starting from (start_i, start_j)."""
            max_i, max_j = start_i, start_j

            # Expand downward
            while max_i + 1 < n and all(matrix[max_i + 1][k] == 1 and not visited[max_i + 1][k] for k in range(start_j, max_j + 1)):
                max_i += 1

            # Expand rightward
            while max_j + 1 < m and all(matrix[k][max_j + 1] == 1 and not visited[k][max_j + 1] for k in range(start_i, max_i + 1)):
                max_j += 1

            # Mark cells as visited
            for i in range(start_i, max_i + 1):
                for j in range(start_j, max_j + 1):
                    visited[i][j] = True

            return (start_i, start_j, max_i, max_j)

        for i in range(n):
            for j in range(m):
                if is_valid(i, j):
                    rectangles.append(find_rectangle(i, j))

        return rectangles

    def create_rectangle_objects(self, rectangles, kmap):
        """
        Create Rectangle objects for the identified minterm rectangles.
        """
        cell_width = 1.0
        cell_height = 1.0
        cell_margin = 0.1

        rectangle_objects = []
        for start_i, start_j, end_i, end_j in rectangles:
            # Calculate the dimensions and position of the rectangle
            width = (end_j - start_j + 1) * (cell_width + cell_margin) - cell_margin
            height = (end_i - start_i + 1) * (cell_height + cell_margin) - cell_margin
            x_center = ((start_j + end_j) / 2 - (len(kmap) / 2) + 0.5) * (cell_width + cell_margin)
            y_center = -((start_i + end_i) / 2 - (len(kmap[0]) / 2) + 0.5) * (cell_height + cell_margin)

            # Create the rectangle
            rect = Rectangle(width=width, height=height, color=YELLOW).move_to([x_center, y_center, 0])
            rectangle_objects.append(rect)

        return rectangle_objects
