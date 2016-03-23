"""

author: Ivan
title : Random Forest				
description: Property of Madata Team. This is the random forest implementation of kaggle expedia project.

"""
import pandas as pd
import numpy as np
import csv as csv
import pylab as P
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split

""" TEATURE EXTRACTION """
def extract_features(frame):
	print 'Extracing features...'
	# feature 1, property price difference again its historical mean
	# need to take care of missing data of prop_log_historical_price, which is 0
	# use median number to replace that data before extracing this feature
	frame.loc[ (frame.prop_log_historical_price==0), 'prop_log_historical_price'] = np.nan
	median_plhp = frame.prop_log_historical_price.dropna().median() 
	frame.loc[ (frame.prop_log_historical_price.isnull()), 'prop_log_historical_price'] = median_plhp
	frame['f_prop_price_diff'] = frame['prop_log_historical_price'].map(np.exp) - frame['price_usd']

	# feature 2, the price difference of user historical mean value and the property price
	# need to tackle null value of visitor_hist_adr_usd
	# assign median value for this case
	median_vhau = frame.visitor_hist_adr_usd.dropna().median()
	frame.loc[(frame.visitor_hist_adr_usd.isnull()),'visitor_hist_adr_usd'] = median_vhau
	frame['f_user_price_diff'] = frame['visitor_hist_adr_usd'] - frame['price_usd']

	# feature 3, starrating difference between user historical mean starrating and property historical mean starrating
	# need to tackle null value of visitor_hist_starrating
	# assign mean value for this case
	mean_vhsr = frame['visitor_hist_starrating'].dropna().mean()
	frame.loc[(frame.visitor_hist_starrating.isnull()),'visitor_hist_starrating'] = mean_vhsr
	frame['f_starrating_diff'] = frame['visitor_hist_starrating'] - frame['prop_starrating']

	# feature 4, fee per person
	frame['f_per_fee'] = frame.price_usd * frame.srch_room_count / (frame.srch_adults_count + frame.srch_children_count)

	# feature 5, total fees
	frame['f_total_fee'] = frame.price_usd * frame.srch_room_count * frame.srch_length_of_stay

	# feature 6, overall price advantage indicator between expedia and competitors
	# the larger the more advantages against competitors
	frame['f_comp_rate'] = (frame.comp1_rate.fillna(0) + frame.comp2_rate.fillna(0) + \
						 frame.comp3_rate.fillna(0) + frame.comp4_rate.fillna(0) + \
						 frame.comp5_rate.fillna(0) + frame.comp6_rate.fillna(0) + \
						 frame.comp7_rate.fillna(0) + frame.comp8_rate.fillna(0)).astype(int)

	# feature 7, overall availability advantage indicator between expedia and competitors
	# the larger the more advantages against competitors
	frame['f_comp_inv'] = (frame.comp1_inv.fillna(0) + frame.comp2_inv.fillna(0) + \
						frame.comp3_inv.fillna(0) + frame.comp4_inv.fillna(0) + \
						frame.comp5_inv.fillna(0) + frame.comp6_inv.fillna(0) + \
						frame.comp7_inv.fillna(0) + frame.comp8_inv.fillna(0)).astype(int)

	# feature 8, prop_location_score2 * srch_query_affinity_score
	median_pls2 = frame.prop_location_score2.dropna().median()
	mean_sqas = frame.srch_query_affinity_score.mean()
	frame.loc[(frame.prop_location_score2.isnull()),'prop_location_score2'] = median_pls2
	frame.loc[(frame.srch_query_affinity_score.isnull()),'srch_query_affinity_score'] = mean_sqas
	frame['f_score2ma'] = frame.prop_location_score2 * frame.srch_query_affinity_score

	# feature 9, score1 devide score2
	frame['f_score1d2'] = frame.prop_location_score2.map(lambda x : x + 0.0001) / frame.prop_location_score1.map(lambda x : x + 0.0001)

	# Other features we can use
	#  
	# --int64 typed features--use directly--
	# prop_id, prop_starrating, prop_brand_bool, promotion_flag, 
	# srch_booking_window, srch_saturday_night_bool, random_bool, srch_destination_id
	#  //generated features: f_comp_rate, f_comp_inv
	#
	# --float typed features--need normalization
	# price_usd, prop_location_score1, prop_location_score2, prop_review_score, orig_destination_distance
	# //generated features: f_prop_price_diff, f_user_price_diff, f_starrating_diff, f_per_fee, f_total_fee, f_score2ma, f_score1d2
	# 
	#normalize data in rage (-1,1)
	fnormalize = lambda x : (x - x.mean()) / (x.max() - x.min())

	frame['f_prop_location_score1'] = (frame[['prop_location_score1']].apply(fnormalize)*10//1.0).astype(int)

	frame['f_prop_location_score2'] = (frame[['prop_location_score2']].apply(fnormalize)*10//1.0).astype(int)

	median_prs = frame.prop_review_score.dropna().median()
	frame.loc[(frame.prop_review_score.isnull()), 'prop_review_score'] = median_prs
	frame['f_prop_review_score'] = (frame[['prop_review_score']].apply(fnormalize)*10//1.0).astype(int)

	median_odd = frame.orig_destination_distance.dropna().median()
	frame.loc[(frame.orig_destination_distance.isnull()),'orig_destination_distance'] = median_odd
	frame['f_orig_destination_distance'] = (frame[['orig_destination_distance']].apply(fnormalize)*10//1.0).astype(int)

	frame['ff_starrating_diff'] = (frame[['f_starrating_diff']].apply(fnormalize)*10//1.0).astype(int)

	frame['ff_score2ma'] = (frame[['f_score2ma']].apply(fnormalize)*10//1.0).astype(int)

	frame['ff_score1d2'] = (frame[['f_score1d2']].apply(fnormalize)*10//1.0).astype(int)

	# for any of the price-related feature, we have to pay spacial attention to the outliers before we normalize the data
	# that is, some prop has too high price which would dis-form our normalization
	# a way to handle those feature is to bin them with a uppper limit other than to normalize them

	CEILING = 1000
	BRACKET_SIZE = 50
	NUM_BRACKET = CEILING // BRACKET_SIZE

	frame['f_price_usd'] = (frame.price_usd//BRACKET_SIZE).clip_upper(NUM_BRACKET-1).astype(np.int)

	frame['ff_prop_price_diff'] = (frame.f_prop_price_diff//BRACKET_SIZE).clip_upper(NUM_BRACKET-1).clip_lower(-NUM_BRACKET).astype(np.int)

	frame['ff_user_price_diff'] = (frame.f_user_price_diff//BRACKET_SIZE).clip_upper(NUM_BRACKET-1).clip_lower(-NUM_BRACKET).astype(np.int)

	frame['ff_total_fee'] = (frame.f_total_fee//BRACKET_SIZE).clip_upper(NUM_BRACKET-1).astype(np.int)

	CEILING = 500
	BRACKET_SIZE = 25
	frame['ff_per_fee'] = (frame.f_per_fee//BRACKET_SIZE).clip_upper(NUM_BRACKET-1).astype(np.int)
	print 'Finished extracting features'
	return frame



""" TARGET EXTRACTION """
def extract_target(frame):
	# Note, all property with booking_bool = 1 must have click_bool = 1
	# 2 means a booking, 1 means a click, 0 means neither booked or clicked
	print 'Extracting target...'
	frame['target'] = frame.click_bool + frame.booking_bool; 
	print 'Finish extracting target'
	return frame


""" DATASET FINALIZATION """
def finalize_traindata(frame):
	print 'Final training data set generating'
	data_final = frame[['srch_id', 'prop_country_id', 'prop_id', 'prop_starrating', 'prop_brand_bool', \
						'promotion_flag', 'srch_booking_window', 'srch_saturday_night_bool',\
						'random_bool', 'srch_destination_id', 'f_comp_rate', 'f_comp_inv', \
						'f_prop_location_score1', 'f_prop_location_score2', 'f_prop_review_score', \
						'f_orig_destination_distance', 'ff_starrating_diff', 'ff_score2ma', \
						'ff_score1d2', 'f_price_usd', 'ff_prop_price_diff', 'ff_user_price_diff',
						'ff_total_fee','target']]
	print 'final training data set generated'
	return data_final

def finalize_testdata(frame):
	print 'Final testing data set generating'
	data_final = frame[['srch_id', 'prop_country_id', 'prop_id', 'prop_starrating', 'prop_brand_bool', \
						'promotion_flag', 'srch_booking_window', 'srch_saturday_night_bool',\
						'random_bool', 'srch_destination_id', 'f_comp_rate', 'f_comp_inv', \
						'f_prop_location_score1', 'f_prop_location_score2', 'f_prop_review_score', \
						'f_orig_destination_distance', 'ff_starrating_diff', 'ff_score2ma', \
						'ff_score1d2', 'f_price_usd', 'ff_prop_price_diff', 'ff_user_price_diff',
						'ff_total_fee']]
	print 'final testing data set generated'
	return data_final





# ================================
# formal start of program
# ================================

""" DATA LOADING """
print 'Loading data...'
train = pd.read_csv('train.csv', header=0)
test = pd.read_csv('test.csv', header=0)
print 'Finished loading data'

train_f = extract_features(train)
test_f = extract_features(test)

train_ft = extract_target(train_f)

train_data = finalize_traindata(train_ft)
test_data = finalize_testdata(test_f)

""" SPLIT VALIDATION SET """
print 'spliting training set to 0.9 and 0.1'
train_local, test_local = train_test_split(train_data, test_size = 0.1)

""" SAVE VALIDATION TAGE """
benchmark = test_local[['srch_id', 'target']]
test_local = test_local.drop(['target'], axis=1)

""" SPLIT DATA BY COUNTRY """
print 'spliting data by country id'
country_list = list(train_data.prop_country_id.unique())
country_forest_dict = {country : RandomForestClassifier(n_estimators=10, n_jobs=-1, min_samples_split=10, verbose=1, random_state=1) for country in country_list}
country_train_dict = {country : train_local[train_local.prop_country_id == country] for country in country_list}
country_testlocal_dict = {country : test_local[test_local.prop_country_id == country] for country in country_list}
country_test_dict = {country : test_data[test_data.prop_country_id == country] for country in country_list}

""" BALANCE NEGATIVE DATA """
print 'balancing negative data'
balanced_country_train_dict = {country : (data[data.target==0].sample(len(data[data.target!=0])+1)).append(data[data.target!=0]) for country, data in country_train_dict.iteritems()} 

""" TRAINING """
print 'Training ...';
for country, data in balanced_country_train_dict.iteritems():
	print ('training country %d'% country);
	country_forest_dict[country] = country_forest_dict[country].fit(data.values[0::,0:-1], data.values[0::,-1]);
print 'Finished training!'

# """ LOCAL TEST """
# print 'Local validation test...'
# country_outputlocal_dict = {}
# for country, data in country_testlocal_dict.iteritems():
# 	output = pd.Series(country_forest_dict[country].predict(data))
# 	df = pd.DataFrame()
# 	df['id'] = data.srch_id
# 	df['output'] = output
# 	country_outputlocal_dict[country] = df 
# print 'Finished local validation test'

# """ FINAL TEST """
print 'Predicting...'
country_output_dict = {}
for country, data in country_test_dict.iteritems():
	output = pd.Series(country_forest_dict[country].predict(data))
	df = pd.DataFrame()
	df['srch_id'] = data.srch_id
	df['prop_id'] = data.prop_id
	df['output'] = output
	country_output_dict[country] = df 
print 'Finished local validation test'

# ONE FOREST TEST
train_oneforest = (train_data[train_data.target==0].sample(len(train_data[train_data.target!=0])+1)).append(train_data[train_data.target!=0])
forest = RandomForestClassifier(n_estimators=200, n_jobs=-1, min_samples_split=10, verbose=2, random_state=1)
forest = forest.fit(train_oneforest.values[0::,0:-1],train_oneforest.values[0::,-1])
output = pd.Series(forest.predict(test_data.values))
df2 = pd.DataFrame()
df2['output'] = output
df2['srch_id'] = test_data.srch_id
df2['prop_id'] = test_data.prop_id

# SORT
# g = df2.groupby('srch_id')
# result = pd.DataFrame()
# for sid in g.groups.keys():
# 	temp = g.get_group(sid)
# 	temp = temp.sort('output', ascending=0)
# 	result = result.append(temp)

# SORT AND WRITE
recommendations = zip(df2.srch_id.values, df2.prop_id.values, df2.output.values*(-1))
from operator import itemgetter
rows = [(srch_id, prop_id) for srch_id, prop_id, output in sorted(recommendations, key=itemgetter(0,2))]
print 'write to csv'
writer = csv.writer(open('result.csv', "w"), lineterminator="\n")
writer.writerow(("SearchId", "PropertyId"))
writer.writerows(rows)
print 'Finish writing'





