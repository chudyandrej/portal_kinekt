import random
import thread
import time
import socket
import sys
import binascii


POLL_INTERVAL = 0.01
TIME_TO_KEEP_TAGS = 20
MAX_TAG_WINDOW_LENGTH = 2
SIGNAL_THRESHOLD = 0.3
TIME_TO_BAN = 1
class AntennaReader():

    def read_forever(self):

        # Create a TCP/IP socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        # Bind the socket to the address given on the command line
        server_address = ('', 8888)
        sock.bind(server_address)
        print >>sys.stderr, 'starting up on %s port %s' % sock.getsockname()
        sock.listen(1)

        while True:
            print >>sys.stderr, 'waiting for a connection'
            self.connection, client_address = sock.accept()
            try:
                print >>sys.stderr, 'client connected:', client_address
                while True:
                    data = self.connection.recv(1000)
                    if not data:
                        break
                    data = data.replace("\x00", "")
                    data = data.replace("\r", "")
                    print data
                    
                    tags = data.split('\n')
                    self.banned_tags_with_timestamp = filter (lambda tag: tag[1] > time.time() - TIME_TO_BAN, self.banned_tags_with_timestamp)
                    banned_tags = map(lambda tag: tag[0], self.banned_tags_with_timestamp)
                    print "BAnned tags: " + str(banned_tags)
                    for tag_string in tags:
                        tag = tag_string.split(',')
                        if len(tag) == 4:
                            tag_ID = int(tag[1], 16) & 0xffffff
                            print tag_ID
                            if tag_ID not in banned_tags:
                                self.tag_list.append((int(tag[1], 16) & 0xffffff,float(tag[3]), time.time()))
                self.tag_list = filter(lambda tag: tag[2] > time.time() - TIME_TO_KEEP_TAGS , self.tag_list)

                    
            finally:
                print "connection closed"
                self.connection.close()

    def __init__(self):
        self.tag_list = []
        self.banned_tags_with_timestamp = []
        thread.start_new_thread(self.read_forever, ())

    def __del__(self):
        self.connection.close()
   
    
    def get_object_tag_id(self, center_time):
       

        valid_tags = []
        tag_window_length = 0.5

        while not valid_tags and tag_window_length < MAX_TAG_WINDOW_LENGTH:

            min_time = center_time - tag_window_length
            max_time = center_time + tag_window_length
            valid_tags = filter(lambda tag: tag[2] > min_time and tag[2] < max_time, self.tag_list)
            tag_window_length += 0.2

        print valid_tags
        if not valid_tags:
            return -1, True
        max_tuple = max(valid_tags, key=lambda tag: tag[1])
        max_signal = max_tuple[1]
        certainity = True
        for tag in valid_tags:
            if tag[0] != max_tuple[0] and tag[1] > max_signal - SIGNAL_THRESHOLD:
                certainity = False
        self.tag_list = filter(lambda tag: tag[0] != max_tuple[0], self.tag_list)
        self.banned_tags_with_timestamp.append((max_tuple[0], time.time()))
        scale = 16 ## equals to hexadecimal
        tag_ID = max_tuple[0]
        
        return tag_ID, certainity

        
