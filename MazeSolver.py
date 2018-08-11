#!/usr/bin/env python3.6
# coding=utf-8

""" MAZE SOLVER
Labirent çözümü yapan modül
"""

__author__ = "Süleyman ERGEN"

import cv2
import numpy as np
import imutils
import threading


IMAGE_PATH = "50x50-rg.png"
SAVE_PATH = "solved_maze.png"


class Detector:
    """
    Başlangıç ve bitiş renklerini belirlemek için kullanılan sınıf.
    """
    # RED = ( [0, 0, 100], [150, 150, 255] )
    # GREEN = ( [0, 100, 0], [150, 255, 150] )
    # BLUE = ( [100, 0, 0], [255, 150, 150] )

    RED_RANGE = ([0, 0, 100], [150, 150, 255])
    GREEN_RANGE = ([0, 100, 0], [150, 255, 150])

    def __init__(self):
        pass

    @staticmethod
    def detect_color(img, color_range):
        """ kırmızı yada yeşil alanı tespit eden fonksiyon.

        :param img: Kaynak resim.
        :param color_range: renk aralığı.
        :return: Tespit edilen renk alanını.
        """
        lower_range = np.array(color_range[0], dtype=np.uint8)
        upper_range = np.array(color_range[1], dtype=np.uint8)

        mask = cv2.inRange(img, lower_range, upper_range)

        result = cv2.bitwise_and(img, img, mask=mask)
        return result

    def detect_point(self, img, color):
        """
        başlangıç yada bitiş noktasını tespit eden ve tespit edilen alanın merkezini döndüren fonksiyon.
        color: RED_RANGE verilirse başlangıç noktasını tespit eder.
        color: GREEN_RANGE verilirse başlangıç noktasını tespit eder.

        :param img: Resim nesnesi.
        :param color: Tespit edilecek renk aralığı..
        :return: Tespit edilen rengin merkez koordinatı. (x, y) gibi.
        """

        cX, cY = None, None
        img = self.detect_color(img, color)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]

        # find contours in the thresholded image
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if imutils.is_cv2() else cnts[1]

        for c in cnts:
            # compute the center of the contour
            M = cv2.moments(c)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            # print("points:", cX, cY)
        return cX, cY


class Dimension:
    """
    Gezilen noktaları tutar
    """
    def __init__(self, point=(0, 0)):
        self.x, self.y = point

    def __add__(self, other):
        return Dimension((self.x + other.x, self.y + other.y))

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y


class Solver:
    """
    Labirentin çözümünü yapan sınıftır.
    """
    def __init__(self, img):
        self.p = 0
        self.directions = [Dimension((0, -1)), Dimension((0, 1)), Dimension((1, 0)), Dimension((-1, 0))]
        self.img = img
        self.height, self.width = img.shape[:2]

        # çözüm aşamasını görüntülemek için thread oluşturma ve başlatma
        t = threading.Thread(target=self.disp, args=())
        t.daemon = True
        # t.start()

    def bfs(self, start_point, end_point):
        """Labirentin çözümünü yapan ve kaydeden fonksiyon.
        Breath first search algotitması kullanıldı.

        :param start_point: Başlangıç koordinat noktası.
        :param end_point: Bitiş koordinat noktası.
        """
        found = False
        queue = []
        visited = [[0 for _ in range(self.width)] for _ in range(self.height)]
        parent = [[Dimension() for _ in range(self.width)] for _ in range(self.height)]

        queue.append(start_point)
        visited[start_point.y][start_point.x] = 1
        while len(queue) > 0:
            p = queue.pop(0)
            for d in self.directions:
                cell = p + d
                if (0 <= cell.x < self.width and 0 <= cell.y < self.height and visited[cell.y][cell.x] == 0 and
                        (self.img[cell.y][cell.x][0] != 0 or self.img[cell.y][cell.x][1] != 0 or self.img[cell.y][cell.x][2] != 0)):
                    queue.append(cell)
                    visited[cell.y][cell.x] = visited[p.y][p.x] + 1

                    self.img[cell.y][cell.x] = [127, 127, 127]  # ziyaret edilen noktaları gri işaretler.
                    parent[cell.y][cell.x] = p
                    if cell == end_point:
                        found = True
                        del queue[:]
                        break

        # çözüm yolunu kırmızı renkle işaretler. ve resmi kaydeder.
        path = []
        if found:
            p = end_point
            while p != start_point:
                path.append(p)
                p = parent[p.y][p.x]
            path.append(p)
            path.reverse()

            for p in path:
                self.img[p.y][p.x] = [250, 0, 0]
            cv2.imwrite(SAVE_PATH, self.img)
        return self.img, path

    def solve(self):
        """Labirentin çözümünü yapar ve çözümüş labirenti resim ve liste olarak geri döndürür.

        :return img: çözülmüz labirentin resim hali
        :return path: çözülmüş labirentin liste hali
        """
        image = self.img

        # Başlangıç ve bitiş noktalarının belirlenmesi
        # kırmızı başlangıç noktası, yeşil bitiş noktası
        detector = Detector()
        start_p = detector.detect_point(image, Detector.RED_RANGE)
        end_p = detector.detect_point(image, Detector.GREEN_RANGE)
        start_point = Dimension(start_p)
        end_point = Dimension(end_p)

        # resmi önce gray, binary ve bgr formatına dönüştürür.
        # yani labirent çözmeden önce resmi hazırlar.
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, img_binary = cv2.threshold(img_gray, 60, 255, cv2.THRESH_BINARY)
        img_bgr = cv2.cvtColor(img_binary, cv2.COLOR_GRAY2BGR)
        self.img = img_bgr

        image, path = self.bfs(start_point, end_point)
        return image, path

    def disp(self):
        """
        labirenti çözüm aşamasında görüntüler.
        çözüm aşamasını adım adım görmek için.
        geliştirme amaçlı oluşturuldu.
        """
        cv2.imshow("Image", self.img)
        while True:
            cv2.imshow("Image", self.img)
            cv2.waitKey(1)


class Path:
    @classmethod
    def way_point_list(cls, path):
        """Labirentin çözümünden elde edilen yoldan her noktada(pixel) hangi yöne gidilmesi gerektiğini gösteren liste
        döndüren fonksiyon.

        :param path: Labirentin çözümünden elde edilen yol.
        :return: Her noktada hangi yöne gitmesi gerektiğini gösteren liste [('up',x,y)] gibi.
        """
        waylist = []
        for i in range(1, len(path)):
            if path[i].x - path[i - 1].x > 0 and path[i].y == path[i - 1].y:
                waylist.append(('right', path[i - 1].x, path[i - 1].y))

            elif path[i].x - path[i - 1].x < 0 and path[i].y == path[i - 1].y:
                waylist.append(('left', path[i - 1].x, path[i - 1].y))

            elif path[i].y - path[i - 1].y > 0 and path[i].x == path[i - 1].x:
                waylist.append(('up', path[i - 1].x, path[i - 1].y))

            elif path[i].y - path[i - 1].y < 0 and path[i].x == path[i - 1].x:
                waylist.append(('down', path[i - 1].x, path[i - 1].y))
        return waylist

    @classmethod
    def way_len_list(cls, path):
        """Labirentin çözümünden elde edilen yoldan hangi yöne ne kadar mesafe gidilmesi gerektiğini hesaplayan ve bunu
        liste olarak döndüren fonksiyon.

        :param path: Labirentin çözümünden elde edilen yol. path.
        :return: Hangi yöne ne kadar uzunlukta gitmesi gerektiğini gösteren liste. [['up', 12]] gibi.
        """
        waylist = cls.way_point_list(path)

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
        return waylen


if __name__ == "__main__":
    src_image = cv2.imread(IMAGE_PATH)

    # çözme işlemini başlat
    solver = Solver(src_image)
    solved_img, solved_path = solver.solve()

    # çözülmüş labirenti göster.
    cv2.imshow("title", solved_img)
    print(solved_path)

    # Labirentte hangi noktada(pixel) hangi yöne gidilmesi gerektiğini gösteren listeyi bul.
    waypointlist = Path.way_point_list(solved_path)
    print(waypointlist)

    # Labirentte hangi yöne ne kadar mesafe gidilmesi gerektiğini hesapla.
    waylenlist = Path.way_len_list(solved_path)
    print(waylenlist)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
