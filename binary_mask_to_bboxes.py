def get_bounding_boxes(binary_mask):
    """
    Given a binary mask, returns a list of bounding boxes in PyTorch standard format [x_min, y_min, x_max, y_max]
    for object detection.

    Args:
        binary_mask (np.array or list[list[int]] or torch.Tensor): A binary mask represented as a list of lists of integers.

    Returns:
        list[list[int]]: A list of bounding boxes in PyTorch format [x_min, y_min, x_max, y_max].
    """
    def update_box(i, j):
        """
        Update the current bounding box coordinates based on the current pixel coordinates (i, j).
        
        Args:
        - i (int): The row index of the current pixel.
        - j (int): The column index of the current pixel.
        
        Returns:
        None
        """
        nonlocal current_box
        current_box[0] = min(current_box[0], j)
        current_box[1] = min(current_box[1], i)
        current_box[2] = max(current_box[2], j)
        current_box[3] = max(current_box[3], i)

    rows, cols = len(binary_mask), len(binary_mask[0])
    visited = [[False] * cols for _ in range(rows)]
    bounding_boxes = []
    stack = []

    for i in range(rows):
        for j in range(cols):
            if binary_mask[i][j] == 1 and not visited[i][j]:
                current_box = [j, i, j, i]  # PyTorch format: [x_min, y_min, x_max, y_max] 
                stack.append((i, j))

                while stack:
                    ni, nj = stack.pop()
                    if 0 <= ni < rows and 0 <= nj < cols and binary_mask[ni][nj] == 1 and not visited[ni][nj]:
                        visited[ni][nj] = True
                        update_box(ni, nj)
                        stack.extend([(ni - 1, nj), (ni + 1, nj), (ni, nj - 1), (ni, nj + 1)])

                bounding_boxes.append(current_box)

    return bounding_boxes