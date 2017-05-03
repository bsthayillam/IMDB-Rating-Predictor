import json
import random
import re
import sys

class Parser():

	def parse(self, filename):
		with open(filename) as data_file:
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

		self.minGross = sys.maxsize
		self.maxGross = 0
		self.minBudget = sys.maxsize
		self.maxBudget = 0
		self.minNumVoted = sys.maxsize
		self.maxNumVoted = 0
		self.minFBLikes = sys.maxsize
		self.maxFBLikes = 0
		self.minDirFBLikes = sys.maxsize
		self.maxDirFBLikes = 0

		# Filter out unnecessary keys
		count = 0
		for i in range(initN):
			if (data[i]['director_info'] != None and data[i]['director_info']['director_facebook_likes'] != None and data[i]['director_info']['director_facebook_likes'] != "One"):
				if (len(data[i]['gross']) > 0 and len(data[i]['budget']) > 0 and data[i]['num_facebook_like'] != None):
					gross = int(re.sub("\D", "", data[i]['gross'][0]))
					budget = int(re.sub("\D", "", data[i]['budget'][0]))
					num_voted = data[i]['num_voted_users']
					fb_likes = int(re.sub("\D", "", data[i]['num_facebook_like']))
					if ('K' in data[i]['num_facebook_like']):
						fb_likes = fb_likes * 1000
					imdb_score = float(data[i]['imdb_score'][0])
					dir_fb_likes = int(re.sub("\D", "", data[i]['director_info']['director_facebook_likes']))
					if ('K' in data[i]['director_info']['director_facebook_likes']):
						dir_fb_likes = dir_fb_likes * 1000
					training = training + [{'gross' : gross, 'budget' : budget, 'num_voted_users' : num_voted, 'num_facebook_like' : fb_likes,
											'director_facebook_likes' : dir_fb_likes, 'imdb_score' : imdb_score}]
					if (gross < self.minGross):
						self.minGross = gross
					if (gross > self.maxGross):
						self.maxGross = gross
					if (budget < self.minBudget):
						self.minBudget = budget
					if (budget > self.maxBudget):
						self.maxBudget = budget
					if (num_voted < self.minNumVoted):
						self.minNumVoted = num_voted
					if (num_voted > self.maxNumVoted):
						self.maxNumVoted = num_voted
					if (fb_likes < self.minFBLikes):
						self.minFBLikes = fb_likes
					if (fb_likes > self.maxFBLikes):
						self.maxFBLikes = fb_likes
					if (dir_fb_likes < self.minDirFBLikes):
						self.minDirFBLikes = dir_fb_likes
					if (dir_fb_likes > self.maxDirFBLikes):
						self.maxDirFBLikes = dir_fb_likes
					count += 1

		# Normalize Data
		for i in range(len(training)):
			gross = (training[i]['gross'] - self.minGross) / (self.maxGross - self.minGross)
			budget = (training[i]['budget'] - self.minBudget) / (self.maxBudget - self.minBudget)
			num_voted = (training[i]['num_voted_users'] - self.minNumVoted) / (self.maxNumVoted - self.minNumVoted)
			fb_likes = (training[i]['num_facebook_like'] - self.minFBLikes) / (self.maxFBLikes - self.minFBLikes)
			dir_fb_likes = (training[i]['director_facebook_likes'] - self.minDirFBLikes) / (self.maxDirFBLikes - self.minDirFBLikes)
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

		# Normalize output for just training data
		for i in range(len(training)):
			imdbNorm = training[i]['imdb_score'] / 10.0
			training[i]['imdb_score'] = imdbNorm

		return (training, crossVal, testing)

	def getMinGross(self):
		return self.minGross

	def getMaxGross(self):
		return self.maxGross

	def getMinBudget(self):
		return self.minBudget

	def getMaxBudget(self):
		return self.maxBudget

	def getMinNumVoted(self):
		return self.minNumVoted

	def getMaxNumVoted(self):
		return self.maxNumVoted

	def getMinFBLikes(self):
		return self.minFBLikes

	def getMaxFBLikes(self):
		return self.maxFBLikes

	def getMinDirFBLikes(self):
		return self.minDirFBLikes

	def getMaxDirFBLikes(self):
		return self.maxDirFBLikes