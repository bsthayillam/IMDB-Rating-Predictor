import json
import random
import pprint

with open('imdb_output.json') as data_file:
	data = json.load(data_file)

# All keys: ['color', 'images', 'num_voted_users', 'duration', 'gross', 'genres', 'cast_info', 'movie_title', 
# 'num_critic_for_reviews', 'plot_keywords', 'image_urls', 'movie_imdb_link', 'num_user_for_reviews', 'language', 'country', 'director_info', 
# 'content_rating', 'budget', 'title_year', 'num_facebook_like', 'storyline', 'imdb_score', 'aspect_ratio']

# Use: 'gross', 'budget', 'num_facebook_like' --> 'imdb_score'

keys_of_interest = ['gross', 'budget', 'num_facebook_like', 'imdb_score']

training = data
testing = []

# Filter out unnecessary keys
for i in range(len(training)):
	training[i] = {key: value for (key, value) in training[i].items() if (key in keys_of_interest)}

for i in range(500):
	r = random.randint(0, len(training) - 1)
	testing = testing + [training.pop(r)]

print(len(training))
print(training[0].keys())
print(len(testing))
print(testing[0].keys())