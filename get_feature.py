import sys
import numpy as np 
from pymongo import MongoClient
from setup import load_file
from sklearn.cluster import KMeans
from geopy import geocoders, distance
from dateutil import parser

class generate_feature(object):
    def __init__(self):
        self.attendeeType = ['yesattend', 'noattend', 'mayattend', 'invitedattend']
        self.clusterer = None
        try:
            self.db = MongoClient().EventRecommender
        except Exception as e:
            print e
            sys.exit()
    
    #load 2 dict
    def load_data(self, path):
        lines = load_file().read_csv(path)
        train_sample = []
        for line in lines:
            reader = line.strip().split(',')
            feature = {}
            feature['user_index'] = reader[0] 
            feature['event_index'] = reader[1]
            feature['invited'] = int(reader[2])
            #bu kao lv shi qu
            feature['first_time'] = parser.parse(reader[3]).replace(tzinfo=None)
            feature['interested'] = int(reader[4])
            feature['nointerested'] = int(reader[5])
            train_sample.append(feature)
        return train_sample
    
    def get_feature_vector(self, data):
        feature_vector = []
        user_index, event_index, first_time, invited = data['user_index'], data['event_index'], data['first_time'], data['invited']
        event = self.db.event.find_one({'id' : event_index})
        user = self.db.user.find_one({'id' : user_index})
        feature_vector.extend(self.get_attendees(event))
        feature_vector.extend(self.get_friend_attendees(user, event))
        feature_vector.extend(self.get_attendees_ratio(event))
        feature_vector.extend(self.get_friend_attendees_ratio(user, event))
        feature_vector.extend(self.get_num_similar_users(user, event, type=1))
        feature_vector.extend(self.get_event_similarity(user, event, num_clusters = 10,type=1))
        feature_vector.append(self.get_user_event_distance(user, event, is_search_eachtime=True))
        feature_vector.append(self.get_time_info(user, event))
        feature_vector.append(self.get_age(user, event,type = 1))
        feature_vector.append(self.get_gender(user, event, type = 1))
        feature_vector.append(self.is_friend_creator(user, event))
        feature_vector.append(invited)
        return feature_vector
       
    def get_attendees(self, event):
        num_yesattend = len(event[self.attendeeType[0]])
        num_mayattend = len(event[self.attendeeType[1]])
        num_invited = len(event[self.attendeeType[2]])
        num_noattend = len(event[self.attendeeType[3]])
        return num_yesattend, num_mayattend, num_invited, num_noattend

    def get_attendees_ratio(self, event):
        num_yesattend = len(event[self.attendeeType[0]])
        num_mayattend = len(event[self.attendeeType[1]])
        num_invited = len(event[self.attendeeType[2]])
        num_noattend = len(event[self.attendeeType[3]])
        num_total = 1.0 * (num_yesattend + num_mayattend + num_invited + num_noattend)
        return num_yesattend/num_total, num_mayattend/num_total, num_invited/num_total, num_noattend/num_total


    
    def get_friend_attendees(self, user, event):
        num_yesattend, num_mayattend, num_invited, num_noattend = 0, 0, 0, 0
        friends = user['friends']
        num_yesattend = len(set(friends).intersection(event[self.attendeeType[0]]))
        num_mayattend = len(set(friends).intersection(event[self.attendeeType[1]]))
        num_invited = len(set(friends).intersection(event[self.attendeeType[2]]))
        num_noattend = len(set(friends).intersection(event[self.attendeeType[3]]))
        return num_yesattend, num_mayattend, num_invited, num_noattend

       
    def get_friend_attendees_ratio(self, user, event):
        num_yesattend, num_mayattend, num_invited, num_noattend = 0, 0, 0, 0
        friends = user['friends']
        num_yesattend = len(set(friends).intersection(event[self.attendeeType[0]]))
        num_mayattend = len(set(friends).intersection(event[self.attendeeType[1]]))
        num_invited = len(set(friends).intersection(event[self.attendeeType[2]]))
        num_noattend = len(set(friends).intersection(event[self.attendeeType[3]]))
        num_total = num_yesattend + num_mayattend + num_invited + num_noattend
        return num_yesattend/num_total, num_mayattend/num_total, num_invited/num_total, num_noattend/num_total
 

    #调用谷歌API,还有用yahoo的，好高端，get it
    #if don't have user location info, return 0
    def get_user_event_distance(self, user, event, is_search_eachtime=False):
        g = geocoders.GeoNames()
        user_event_distance = 0
        if user['location'] is not None and event['latitude'] is not None and event['longitude'] is not None:
                #不超过400  
            user_event_distance = 400
            event_coordinate = (event['latitude'], event['longitude'])
            if 'latitude' in user and is_search_eachtime == False:
                user_coordinate = (user['latitude'], user['longitude'])
                d = distance.distance(user_coordinate, event_coordinate).miles
                if d < user_event_distance:
                    user_event_distance = d
                else:
                    user_event_distance = user_event_distance
            else:
                    #多个距离选最近的
                results = g.geocode(user['location'], exactly_one = False)
                closest_index = 0
                for i, (_, user_coordinate) in enumerate(results):
                    d = distance.distance(user_coordinate, event_coordinate).miles
                    if d < user_event_distance:
                        user_event_distance = d
                        closest_index = i
                        self.db.user.update({'id' : user['id']}, {'$set' : {'latitude' : results[closest_index][1][0], 'longitude': results[closest_index][1][1]}})
        return user_event_distance
   
   #找到以前有意向这四种情况现在还是同样的选择的
    def get_num_similar_users(self, user, event, type = 4):
        user_index = user['id']
        num_similar_users = [0] * type
        for i, attendee_type in enumerate(self.attendee_type[0 : type]):
            for f in self.db.event.find({attendee_type : {'$in' : [user_index]}}):
                for userId in event[attendee_type]:
                    if user_index in f[attendee_type]:
                        num_similar_users[i] += 1
                    else:
                        num_similar_users[i] += 0
        return num_similar_users
    
    #当前的与之前attend的event的相似度 
    def get_event_similarity(self, user, event, num_clusters = 10, type = 4):
        user_index = user['id']
        event_similarity = [0] * type
        predicted_cluster = self.clusterer.predict(np.array(event['keywords']).astype(float))[0]
        for i, attendee_type in enumerate(self.attendee_type[0 : type]):
                cluster_record = [0] * num_clusters
                #聚类events的个数比上总体的
                events = list(self.db.event.find({attendee_type : {'$in' : [user_index]}}))
                for f in events:
                    #聚成k类
                    k = self.clusterer.predict(np.array(f['keywords']).astype(float))[0]
                    cluster_record[k] += 1
                if sum(cluster_record) is not None:
                    event_similarity[i] = float(cluster_record[predicted_cluster]) / sum(cluster_record)
                else:
                    event_similarity[i] = 0
        return event_similarity
   

    def get_time_info(self, user, event, first_time):
        return  (event['startTime'] - first_time).days * 24 + (event['startTime'] - first_time).seconds / 3600
    
    def get_age(self, user, event, type = 0):
        user_index = user['id']
        attendee_type = self.attendee_type[type]
        event_avg = self.get_age_avg(event, attendee_type)
        total_avg = 0
        events = list(self.db.event.find({attendee_type : {'$in' : [user_index]}}))
        if len(events) == 0:
            return event_avg
        else:
            for event in events:
                total_avg += self.get_age_avg(event, attendee_type)
            diff = event_avg - (total_avg / len(events))
            return diff

    def get_gender(self, user, event, type = 0):
        user_index = user['id']
        attendee_type = self.attendee_type[type]
        event_avg = self.get_gender_avg(event, attendee_type)
        total_avg = 0
        events = list(self.db.event.find({attendee_type : {'$in' : [user_index]}}))
        if len(events) == 0:
            return event_avg
        else:
            for event in events:
                total_avg += self.get_gender_avg(event, attendee_type)
            diff = event_avg - (total_avg / len(events))
            return diff

    def get_age_avg(self, event, attendee_type):
        if len(event[attendee_type]) == 0:
            return 0
        total = 0
        num_valid_user = 0
        for user_index in event[attendee_type]:
            user = self.db.user.find_one({'id' : user_index})
            value = 0
            #有缺失值
            if user is not None and user['age'] is not None:
                num_valid_user += 1
                value = user['age']
            total += value
        if num_valid_user != 0:
            avg = (float(total) / num_valid_user) 
        else:
            avg = 0.0
        return avg


    def get_gender_avg(self, event, attendee_type):
        if len(event[attendee_type]) == 0:
            return 0
        total = 0
        num_valid_user = 0
        for user_index in event[attendee_type]:
            user = self.db.user.find_one({'id' : user_index})
            value = 0
            #有缺失值
            if user is not None and user['gender'] is not None:
                num_valid_user += 1
                value = user['gender']
            total += value
        if num_valid_user != 0:
            avg = (float(total) / num_valid_user) 
        else:
            avg = 0.0
        return avg
   
    def is_friend_creator(self, user, event):
        if event['friend_creator_index'] in user['friends']: 
            return 1
        else:
            return 0


    def get_feature_matrix(self, path):
        train_data = self.load_data(path)
        self.load_cluster_model()
        feature_matrix = []
        for data in enumerate(train_data):
            feature_vector = self.get_feature_vector(data)
            feature_vector.append(self.do_classification(data))
            feature_matrix.append(feature_vector)
        return feature_matrix 
    
    
    def do_classification(self, feature):
        if feature['interested'] == 1:
            return 1
        else:
            return 0
    
            
    def load_cluster_model(self, num_clusters = 10):
        model = self.db.cluster.find_one({'k' : num_clusters})
        if model is not None:
            centers = np.array(model['centers'])
            self.clusterer = KMeans(n_clusters= num_clusters, n_init = 1, init = centers).fit(centers)
        else:
            print 'sth wrong'
      
                    
    #一次没法读很多，pandas？hdf5?
    def train_cluster_model(self, path = None, max_num = 10000, num_clusters = 10):
        data = np.loadtxt(path)
        events = self.db.event.find().limit(max_num)
        keywords = [event['keywords'] for event in events]
        data = np.array(keywords)
        kMeans_model = KMeans(n_clusters = num_clusters).fit(data)
        centers = kMeans_model.cluster_centers_.tolist()
        #remove first, then insert
        self.db.cluster.remove({'k' : num_clusters})
        self.db.cluster.insert({'k' : num_clusters, 'centers' : centers})
        
   
    
if __name__ == '__main__':
    fg = generate_feature()
    feature_matrix = fg.get_feature_matrix('train.csv')
    with open('feature.csv', 'w') as f:
        for feature_vector in feature_matrix:
            f.write(' '.join([str(it) for it in feature_vector]) + '\n')

