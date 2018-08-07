#!/usr/bin/env python3
import socket
import sys

TCP_IP = "10.42.0.1"
FILE_PORT = 5005
DATA_PORT = 5006
timeout = 3
buf = 1024

sock_f = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock_f.bind((TCP_IP, FILE_PORT))
sock_f.listen(1)

sock_d = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock_d.bind((TCP_IP, DATA_PORT))
sock_d.listen(1)


while True:
    conn, addr = sock_f.accept()
    data = conn.recv(buf)
    if data:
        print "File name:", data
        file_name = data.strip()

    f = open(file_name, 'wb')

    conn, addr = sock_d.accept()
    while True:
        data = conn.recv(buf)
        if not data:
            break
        f.write(data)

    print "%s Finish!" % file_name
    f.close()
    sys.exit(1)
