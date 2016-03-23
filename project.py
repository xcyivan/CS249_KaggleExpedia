#load data from csv files 
import pandas as pd
import numpy as np
import csv as csv
training_data = pd.DataFrame.from_csv('data/train.csv').reset_index()
testing_data = pd.DataFrame.from_csv('data/test.csv').reset_index()
print(training_data.columns.values)
training_data.shape
print(testing_data.columns.values)
# create target : click_score [0,1,2]
def generate_target(training_data):
    training_data['click_score'] = pd.Series(0, index=training_data.index)
    training_data.loc[training_data.click_bool == 1, 'click_score'] = 1
    training_data.loc[training_data.booking_bool == 1, 'click_score'] = 2
    #pd.Series.unique(training_data['click_score'])
    return training_data
    # generate new features
def generate_features(frame):
    print('Generating features...')
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
    
    
    ## try
    #frame.loc[(frame.prop_location_score2.isnull()),'prop_location_score2'] = median_pls2
    frame.loc[(frame.prop_location_score2.isnull()),'prop_location_score2'] = 0

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
    #try
    frame.f_hotel_quality_1.fillna(0, inplace=True)
    frame.f_hotel_quality_2.fillna(0, inplace=True)
    frame['f_hotel_quality_1'] = (frame[['f_hotel_quality_1']].apply(fnormalize)*10/1.0)
    frame['f_hotel_quality_2'] = (frame[['f_hotel_quality_2']].apply(fnormalize)*10/1.0)
    
    
    median_prs = frame.prop_review_score.dropna().median()
    #try
    #frame.loc[(frame.prop_review_score.isnull()), 'prop_review_score'] = median_prs
    frame.loc[(frame.prop_review_score.isnull()), 'prop_review_score'] = 3.0
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
    
    print('Finished generating features')
    return frame
#feature_names = list(training_data_local.columns)
# feature_names.remove("click_bool")
# feature_names.remove("booking_bool")
# feature_names.remove("gross_bookings_usd")
# feature_names.remove("date_time")
# feature_names.remove("position")

feature_names = ['srch_id', 'prop_country_id', 'prop_id', 'prop_starrating', 'prop_brand_bool', \
                 'promotion_flag', 'srch_booking_window', 'srch_saturday_night_bool',\
                'random_bool', 'srch_destination_id', 'f_comp_rate', 'f_comp_inv', \
                'f_prop_location_score1', 'f_prop_location_score2', 'f_prop_review_score', \
                'f_orig_destination_distance', 'ff_starrating_diff', 'ff_score2ma', \
                'ff_score1d2', 'f_price_usd', 'ff_prop_price_diff', 'ff_user_price_diff',
                'ff_total_fee']
# feature_names = ['prop_country_id', 'prop_id', 'prop_starrating', 'prop_brand_bool', \
#                  'promotion_flag', 'srch_booking_window', 'srch_saturday_night_bool',\
#                 'random_bool', 'srch_destination_id', 'f_comp_rate', 'f_comp_inv', \
#                 'f_prop_location_score1', 'f_prop_location_score2', 'f_prop_review_score', \
#                 'f_orig_destination_distance', 'ff_starrating_diff', 'ff_score2ma', \
#                 'ff_score1d2', 'f_price_usd', 'ff_prop_price_diff', 'ff_user_price_diff',
#                 'ff_total_fee']

#feature_names_ori = ['srch_length_of_stay','srch_adults_count','srch_children_count', 'srch_room_count']
feature_names_ori = list(training_data.columns[:27])
feature_names_hotel_quality = ['f_hotel_quality_1','f_hotel_quality_2'] 
feature_names = feature_names + feature_names_ori + feature_names_hotel_quality
feature_names = list(set(feature_names))
feature_names.remove("date_time")
feature_names.remove("position")
print(len(feature_names))
# # hotel quality feature : f_hotel_quality_1 and f_hotel_quality_2
# counter = 0
# for p_id in pd.Series.unique(training_data.prop_id):
#     hotel_quality_1 = training_data.loc[training_data.prop_id == p_id, 'click_bool'].sum()/training_data.loc[training_data.prop_id == p_id, 'srch_id'].count()
#     hotel_quality_2 = training_data.loc[training_data.prop_id == p_id, 'booking_bool'].sum()/training_data.loc[training_data.prop_id == p_id, 'srch_id'].count()
#     training_data.loc[training_data.prop_id == p_id,'f_hotel_quality_1'] = hotel_quality_1
#     training_data.loc[training_data.prop_id == p_id,'f_hotel_quality_2'] = hotel_quality_2
#     testing_data.loc[testing_data.prop_id == p_id,'f_hotel_quality_1'] = hotel_quality_1
#     testing_data.loc[testing_data.prop_id == p_id,'f_hotel_quality_2'] = hotel_quality_2
#     counter = counter + 1
#     if(counter % 10000 == 0):
#         print("counter: ",counter ,"pid: ",p_id)
# mean_quality_1 = testing_data.f_hotel_quality_1.dropna().mean()
# testing_data.loc[testing_data.f_hotel_quality_1.isnull(), 'f_hotel_quality_1'] = mean_quality_1
# mean_quality_2 = testing_data.f_hotel_quality_2.dropna().mean()
# testing_data.loc[testing_data.f_hotel_quality_2.isnull(), 'f_hotel_quality_2'] = mean_quality_2
count_map_book = {}
count_map_click = {}
count_map = {}
training_data['f_hotel_quality_1'] = np.nan
training_data['f_hotel_quality_2'] = np.nan
testing_data['f_hotel_quality_1'] = np.nan
testing_data['f_hotel_quality_2'] = np.nan
#first scan
counter = 0
for row in training_data.itertuples():
    key = row.prop_id
    if(key not in count_map_book and row.booking_bool == 1):
        count_map_book[key] = 1
    elif(row.booking_bool == 1):
        count_map_book[key] += 1
    if(key not in count_map_click and row.click_bool == 1):
        count_map_click[key] = 1
    elif(row.click_bool == 1):
        count_map_click[key] += 1  
    if(key not in count_map):
        count_map[key] = 1
    else:
        count_map[key] += 1  
    counter += 1
    if(counter % 1000000 == 0):
        print("counter = ",counter)
print('map constructed')            
# for i in range(training_data.shape[0]):
#     key = training_data.iloc[i].prop_id
#     if(key not in count_map_book):
#         count_map_book[key] = 1
#     elif(training_data.iloc[i].booking_bool == 1):
#         count_map_book[key] += 1
#     if(key not in count_map_click):
#         count_map_click[key] = 1
#     elif(training_data.iloc[i].click_bool == 1):
#         count_map_click[key] += 1  
#     if(key not in count_map):
#         count_map[key] = 1
#     else:
#         count_map[key] += 1  
#     counter += 1
#     if(counter % 100000 == 0):
#         print("counter = ",counter)
counter = 0
for row in training_data.itertuples():
    key = row.prop_id
    index = row[0]
    if(key in count_map_click and key in count_map):
        training_data.set_value(index, 'f_hotel_quality_1' , count_map_click[key]/count_map[key])
    if(key in count_map_book and key in count_map):   
        training_data.set_value(index, 'f_hotel_quality_2' , count_map_book[key]/count_map[key])
    counter += 1
    if(counter % 1000000 == 0):
        print("counter = ",counter)
print('training data hotel quality feature created')
# for i in range(training_data.shape[0]):
#     key = training_data.iloc[i].prop_id
#     training_data.set_value(i, 'f_hotel_quality_1' , count_map_click[key]/count_map[key])
#     training_data.set_value(i, 'f_hotel_quality_2' , count_map_book[key]/count_map[key])
#     counter += 1
#     if(counter % 100000 == 0):
#         print("counter = ",counter)
# print('training data hotel quality feature created')
counter = 0
for row in testing_data.itertuples():
    key = row.prop_id
    index = row[0]
    if(key in count_map_click and key in count_map):
        testing_data.set_value(index, 'f_hotel_quality_1' , count_map_click[key]/count_map[key])
    if(key in count_map_book and key in count_map):   
        testing_data.set_value(index, 'f_hotel_quality_2' , count_map_book[key]/count_map[key])
    counter += 1
    if(counter % 1000000 == 0):
        print("counter = ",counter)
print('testing data hotel quality feature created')
# for i in range(testing_data.shape[0]):
#     key = testing_data.iloc[i].prop_id
#     testing_data.set_value(i, 'f_hotel_quality_1' , count_map_click[key]/count_map[key])
#     testing_data.set_value(i, 'f_hotel_quality_2' , count_map_book[key]/count_map[key])
#     counter += 1
#     if(counter % 100000 == 0):
#         print("counter = ",counter)
# print('testing data hotel quality feature created')
# # local testing starts here
# from sklearn.cross_validation import train_test_split
# training_data_local, testing_data_local = train_test_split(training_data, test_size = 0.1)
# training_data_local = generate_features(training_data_local)
# training_data_local = generate_target(training_data_local)
# testing_data_local = generate_features(testing_data_local)
# # only available for local dataset
# testing_data_local = generate_target(testing_data_local)
# For unknown testing dataset
testing_data = generate_features(testing_data)
training_data = generate_features(training_data)
training_data = generate_target(training_data)
# country data balancing 
# balanced_country_train_dict = {country : (data[data.click_score==0].sample(len(data[data.click_score!=0])+1)).append(data[data.click_score!=0]) for country, data in country_train_dict.items()} 
# whole data balancing
# balanced_training_data_local = training_data_local[training_data_local.click_score==0].sample(len(training_data_local[training_data_local.click_score!=0])+1).append(training_data_local[training_data_local.click_score!=0]) 
# whole global data balancing
balanced_training_data = training_data[training_data.click_score==0].sample(len(training_data[training_data.click_score!=0])+1).append(training_data[training_data.click_score!=0]) 
# # local dataset feature and target extraction
# training_features_local = balanced_training_data_local.get(feature_names)
# testing_features_local = testing_data_local.get(feature_names)
# training_target_local = balanced_training_data_local.get(['click_score'])
# testing_target_local = testing_data_local.get(['click_score'])
# whole dataset feature and target extraction
training_features = balanced_training_data.get(feature_names)
testing_features = testing_data.get(feature_names)
training_target = balanced_training_data.get(['click_score'])
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.grid_search import GridSearchCV
clf = RandomForestClassifier(n_estimators=800, 
                                        verbose=1,
                                        n_jobs=4,
                                        min_samples_split=2,
                                        random_state=1)
# get lower score 0.39299 =.= gao mao
# params = {
#     'min_samples_split': [2,5,10],
#     'max_depth': [3,4,5]
# }
# clf = GridSearchCV(clf, params)

# can not use bulit-in cv method for this large dataset(entire dataset)
#scores = cross_val_score(clf, training_features, training_target.values, scoring = 'accuracy', cv=10)
#print(scores.min(), scores.mean(), scores.max())
#clf.fit(training_features_local, training_target_local.values.ravel())
clf.fit(training_features, training_target.values.ravel())
predict_values = clf.predict(testing_features)
print(feature_names)
predict_values_proba = clf.predict_proba(testing_features)
# sort and write to file
recommendations = zip(testing_features.srch_id.values, testing_features.prop_id.values, predict_final*(-1))
from operator import itemgetter
rows = [(srch_id, prop_id) for srch_id, prop_id, output in sorted(recommendations, key=itemgetter(0,2))]
print('write to csv')
writer = csv.writer(open('result_RF_proba.csv', "w"), lineterminator="\n")
writer.writerow(("SearchId", "PropertyId"))
writer.writerows(rows)
print('Finish writing')
print(feature_names)
clf.feature_importances_

