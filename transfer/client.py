#!/usr/bin/env python
import socket
import sys

TCP_IP = "10.42.0.1"
FILE_PORT = 5005
DATA_PORT = 5006
buf = 1024
file_name = "max.jpg"


try:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((TCP_IP, FILE_PORT))
    sock.sendall(file_name.encode('utf-8'))
    sock.close()

    print("Sending %s ..." % file_name)

    f = open(file_name, "rb")
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((TCP_IP, DATA_PORT))
    data = f.read()
    sock.send(data)

finally:
    sock.close()
    f.close()
