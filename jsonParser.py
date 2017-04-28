import json
import random
import re
import sys

with open('imdb_output.json') as data_file:
	data = json.load(data_file)

# All keys: ['color', 'images', 'num_voted_users', 'duration', 'gross', 'genres', 'cast_info', 'movie_title', 
# 'num_critic_for_reviews', 'plot_keywords', 'image_urls', 'movie_imdb_link', 'num_user_for_reviews', 'language', 'country', 'director_info', 
# 'content_rating', 'budget', 'title_year', 'num_facebook_like', 'storyline', 'imdb_score', 'aspect_ratio']

# Use: 'gross', 'budget', 'num_facebook_like' --> 'imdb_score'

keys_of_interest = ['gross', 'budget', 'num_voted_users', 'num_facebook_like','imdb_score']

training = []
crossVal = []
testing = []
initN = len(data)

minGross = sys.maxsize
maxGross = 0
minBudget = sys.maxsize
maxBudget = 0
minNumVoted = sys.maxsize
maxNumVoted = 0
minFBLikes = sys.maxsize
maxFBLikes = 0
minDirFBLikes = sys.maxsize
maxDirFBLikes = 0

# Filter out unnecessary keys
count = 0
for i in range(initN):
	if (data[i]['director_info'] != None and data[i]['director_info']['director_facebook_likes'] != None and data[i]['director_info']['director_facebook_likes'] != "One"):
		if (len(data[i]['gross']) > 0 and len(data[i]['budget']) > 0 and data[i]['num_facebook_like'] != None):
			gross = int(re.sub("\D", "", data[i]['gross'][0]))
			budget = int(re.sub("\D", "", data[i]['budget'][0]))
			num_voted = data[i]['num_voted_users']
			fb_likes = int(re.sub("\D", "", data[i]['num_facebook_like'])) * 1000
			imdb_score = float(data[i]['imdb_score'][0])
			dir_fb_likes = int(re.sub("\D", "", data[i]['director_info']['director_facebook_likes'])) * 1000
			training = training + [{'gross' : gross, 'budget' : budget, 'num_voted_users' : num_voted, 'num_facebook_like' : fb_likes,
									'director_facebook_likes' : dir_fb_likes, 'imdb_score' : imdb_score}]
			if (gross < minGross):
				minGross = gross
			if (gross > maxGross):
				maxGross = gross
			if (budget < minBudget):
				minBudget = budget
			if (budget > maxBudget):
				maxBudget = budget
			if (num_voted < minNumVoted):
				minNumVoted = num_voted
			if (num_voted > maxNumVoted):
				maxNumVoted = num_voted
			if (fb_likes < minFBLikes):
				minFBLikes = fb_likes
			if (fb_likes > maxFBLikes):
				maxFBLikes = fb_likes
			if (dir_fb_likes < minDirFBLikes):
				minDirFBLikes = dir_fb_likes
			if (dir_fb_likes > maxDirFBLikes):
				maxDirFBLikes = dir_fb_likes
			count += 1

# Normalize Data
for i in range(len(training)):
	gross = (training[i]['gross'] - minGross) / (maxGross - minGross)
	budget = (training[i]['budget'] - minBudget) / (maxBudget - minBudget)
	num_voted = (training[i]['num_voted_users'] - minNumVoted) / (maxNumVoted - minNumVoted)
	fb_likes = (training[i]['num_facebook_like'] - minFBLikes) / (maxFBLikes - minFBLikes)
	dir_fb_likes = (training[i]['director_facebook_likes'] - minDirFBLikes) / (maxDirFBLikes - minDirFBLikes)
	training[i]['gross'] = gross
	training[i]['budget'] = budget
	training[i]['num_voted_users'] = num_voted
	training[i]['num_facebook_like'] = fb_likes
	training[i]['director_facebook_likes'] = dir_fb_likes

# Split data into 500 testing set and remaining training set
for i in range(500):
	r = random.randint(0, len(training) - 1)
	testing = testing + [training.pop(r)]

# Split data into 500 cross validation set and remaining training set
for i in range(500):
	r = random.randint(0, len(training) - 1)
	crossVal = crossVal + [training.pop(r)]

# print(len(training))
# print(training[0])
# print(len(crossVal))
# print(crossVal[0])
# print(len(testing))
# print(testing[0])