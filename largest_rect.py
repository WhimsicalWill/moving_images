import numpy as np

# takes in 2d nparray
def get_largest_rect(grid):
    histogram = np.zeros(grid.shape[1], dtype=int)
    max_rect = 0
    for rownum, row in enumerate(grid):
        for colnum, elem in enumerate(row):
            if elem:
                histogram[colnum] += 1
            else:
                histogram[colnum] = 0
        x0temp, x1temp, height, rect_size = histogram_rect(histogram)
        if rect_size > max_rect:
            max_rect = rect_size
            y0, y1 = (rownum - height + 1, rownum)
            x0, x1 = (x0temp, x1temp)
    return x0, x1, y0, y1, max_rect

# returns largest rectangle given a histogram
def histogram_rect(hist):
    # can improve this by ignoring 0-height bars
    stack = []
    x0 = x1 = height = max_area = i = 0
    while i < len(hist):
        if len(stack) == 0 or hist[stack[-1]] <= hist[i]:
            stack.append(i)
            i += 1
        else:
            curr_max = stack.pop()
            if len(stack) == 0:
                area = hist[curr_max] * i
            else:
                area = hist[curr_max] * (i - 1 - stack[-1])
            # enforce a h:w ratio of at least 3:5
            ratio = hist[curr_max] / (i - 1 - stack[-1])
            ratio = min(ratio, 1/ratio)
            if area > max_area and ratio >= .6:
                max_area = area
                if len(stack) == 0:
                    x0 = 0
                else:
                    x0 = stack[-1] + 1
                x1 = i - 1
                height = hist[curr_max]
    while len(stack) > 0:
        curr_max = stack.pop()
        if len(stack) == 0:
            area = hist[curr_max] * (i - 1)
        else:
            area = hist[curr_max] * (i - 1 - stack[-1])
        if area > max_area:
            max_area = area
            if len(stack) == 0:
                x0 = 0
            else:
                x0 = stack[-1] + 1
            x1 = i - 1
            height = hist[curr_max]
    return x0, x1, height, max_area
