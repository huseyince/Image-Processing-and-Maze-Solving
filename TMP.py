import MazeSolver
import cv2

img = cv2.imread("50x50-rg.png")

# MazeSolver import denemesi
solver = MazeSolver.Solver(img)
img, path = solver.solve()

for i in range(len(path)):
    print(i, "->", path[i].x, path[i].y)

waylist = []
for i in range(1, len(path)):
    if path[i].x-path[i-1].x > 0 and path[i].y == path[i-1].y:
        waylist.append(('right', path[i-1].x, path[i-1].y))

    elif path[i].x-path[i-1].x < 0 and path[i].y == path[i-1].y:
        waylist.append(('left', path[i - 1].x, path[i - 1].y))

    elif path[i].y-path[i-1].y > 0 and path[i].x == path[i-1].x:
        waylist.append(('up', path[i - 1].x, path[i - 1].y))

    elif path[i].y-path[i-1].y < 0 and path[i].x == path[i-1].x:
        waylist.append(('down', path[i - 1].x, path[i - 1].y))

print(waylist)

waylen = [[None, None]]
for i in waylist:
    if i[0] == 'right':
        if waylen[-1][0] == 'right':
            waylen[-1][1] += 1
        elif waylen[-1][0] != 'right':
            waylen.append(['right', 1])
    elif i[0] == 'left':
        if waylen[-1][0] == 'left':
            waylen[-1][1] += 1
        elif waylen[-1][0] != 'left':
            waylen.append(['left', 1])
    elif i[0] == 'up':
        if waylen[-1][0] == 'up':
            waylen[-1][1] += 1
        elif waylen[-1][0] != 'up':
            waylen.append(['up', 1])
    elif i[0] == 'down':
        if waylen[-1][0] == 'down':
            waylen[-1][1] += 1
        elif waylen[-1][0] != 'down':
            waylen.append(['down', 1])

print(waylen)
