import websocket
import requests
import json
import time
import os
import ast
from threading import Lock
import sys
import thread
transaction_lock = Lock()
tags_list = []

####### Configuring Settings #######
server_url = "192.168.1.23:3000"
portalID = "1"
####################################

def on_message(ws, message):
	if message == "ping":
		message_to_server = "pong"
		ws.send(message_to_server)
	if message == "change":
	    print "Settings changed on server - downloading new values"
	    get_list_tag()

def on_error(ws, error):
	  print "websocket error " + error

def on_close(ws):
	 print "### websocket closed ###"

def on_open(ws):
	print "### websocket open ###"

def websockets():
	websocket.enableTrace(False)
	ws = websocket.WebSocketApp("ws://"+server_url+"/",
	on_message = on_message,
	on_error = on_error,
	on_close = on_close)
	ws.on_open = on_open
	ws.run_forever()

def send_transaction(tag_id , direction, time,certainity, alarm):           
    url = "http://"+server_url+"/api/portal_endpoint/transaction/"+portalID+""
    data = '{"tagId":'+str(tag_id)+',"direction":"'+direction+'","time":'+str(time)+',"certainity":"'+str(certainity)+'","alarm":"'+str(alarm)+'"}'
    print "transaction_data : " + data
    headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}
    transaction_lock.acquire(True)

    try:
        r = requests.post(url, data=data, headers=headers)  #try send transaction
        if r.status_code != 201:
            raise ValueError('Server failure')
        thread.start_new_thread(play_sound,(direction, r.text))
        if os.stat("transaction_backup.txt").st_size != 0:  #if existing some old transaction
            trans_backup = open("transaction_backup.txt")   
            for line in trans_backup:           #send all old transaction
                r = requests.post(url, data=line, headers=headers)  
            trans_backup.close()
            open('transaction_backup.txt', 'w').close()
    except:
        thread.start_new_thread(play_sound,(direction, "Chyba"))
        target = open("transaction_backup.txt", 'a')    
        target.write(data+"\n")
        target.close()  
    
    transaction_lock.release()

def get_request(url):
	try: 
		r = requests.get(url)	#try get request
	except:
		print '\033[1;31m Server not reachable \033[1;m'
		return None ,None
	json_request = json.loads(r.text)	#convert to json
	return json_request , r.text	

def get_json_settings():
	settings, text = get_request("http://"+server_url+"/api/portal_endpoint/settings/"+portalID+"")
	if settings != None:					#if server is ok
		file_settings = open("settings.txt", 'w')
		file_settings.write(text)
		file_settings.close()
	else :									#if server not reachable
		file_set = open("settings.txt", 'r')
		settings_file = file_set.read()
		settings = json.loads(settings_file)
	return settings

def get_list_tag():
	_ , tags = get_request("http://"+server_url+"/api/portal_endpoint/permissions/"+portalID+"")
	if tags != None:    #if server is ok
		file_tags = open("tags.txt", 'w')
		file_tags.write(tags)
		file_tags.close()
	else :                  #if server not reachable
	    file_tags = open("tags.txt", 'r')
	    tags = file_tags.read()
	global tags_list
	tags_list = ast.literal_eval(tags)  
	return tags_list

def get_tag_permission(tag):
    if tag in tags_list:
        return False
    return True


def play_sound(direction, name):
        if direction == "in":
            direction = "dnu"
        else:
            direction = "von"
        
        command = "espeak -v sk --stdout '%s %s' | aplay" % (name, direction)
        os.system(command.encode('UTF-8'))







