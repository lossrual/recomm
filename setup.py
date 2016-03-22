import csv
import numpy as np
import pandas as pd
from datetime import datetime
from pymongo import Connection, MongoClient

class load_file(object):
    def __init__(self):
        self.db = MongoClient().EventRecommender
        
    def read_csv(self, path):
        with open(path, 'r') as fcsv:
            reader = csv.reader(fcsv)
            reader.next()
            lines = [int(line[0]) for line in reader]
        return lines
    
    def load_users_info(self, path):
        lines = self.read_csv(path)
        users_info = {}
        for line in lines:
            rows = line.strip().split(',')
            user = {}
            user['id'] = rows[0]
            user['language'] = rows[1]
            user['gender'] = 1 if rows[3] == 'male' else 0
            user['location'] = rows[5] if rows[5] != '' else None
            user['timeZone'] = rows[6].strip()
            user['age'] = 2015 - int(rows[2]) if user['age'] else None
            user['joinTime'] = datetime.strptime(rows[4], '%Y-%m-%dT%H:%M:%S.%fZ') if user['joinTime'] else None
            users_info[user['id']] = user
        return users_info
    
    def load_friends_info(self, path, users_info):
        lines = self.read_csv(path)
        for line in lines:
            rows = line.strip().split(',')
            users_info[rows[0]]['friends'] = [friendIndex for friendIndex in rows[1].split()]
        return users_info
    
    def insert_users(self, users_path, friends_path):
        users = self.load_users_info(users_path)
        users = self.load_friends_info(friends_path, users)
        for line in users.values():
            self.db.user.insert(line)
        
    def load_events_info(self, path):
        lines = self.read_csv(path)
        events = {}
        for line in lines:
            rows = line.strip().split(',')
            event = {}
            event['id'] = rows[0]
            event['hostUserId'] = rows[1]
            event['startTime'] = datetime.strptime(rows[2], '%Y-%m-%dT%H:%M:%S.%fZ') if rows[2] else None
            event['city'] = rows[3] if rows[3] else None
            event['state'] = rows[4] if rows[4] else None
            event['zip'] = rows[5] if rows[5] else None
            event['country'] = rows[6] if rows[6] else None
            event['latitude'] = float(rows[7]) if rows[7] else None
            event['longitude'] = float(rows[8]) if rows[8] else None
            event['keywords'] = [int(row) for row in rows[9:]]
            events[event['id']] = event
        return events
    
    def load_eventattendees_info(self, path, events):
        lines = self.read_csv(path)
        for line in lines:
            rows = line.strip().split(',')
            events[rows[0]]['yesattend'] = [it for it in rows[1].split()] if rows[1] else []
            events[rows[0]]['mayattend'] = [it for it in rows[2].split()] if rows[2] else []
            events[rows[0]]['invitedattend'] = [it for items in rows[3].split()] if rows[3] else []
            events[rows[0]]['noattend'] = [it for it in rows[4].split()] if rows[4] else []
        return events
    
    def insert_events(self, path, attendees_csv):
        events = self.load_events(path)
        events = self.load_eventattendees_info(attendees_csv, events) 
        for event in events.values():
            self.db.event.insert(event)
      
if __name__ == '__main__':
    dl = load_file()
    dl.insert_users('../data/users.csv', '../data/user_friends.csv')
    dl.insert_events('../data/events_valid.csv', '../data/event_attendees.csv')
