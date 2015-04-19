import sys
import numpy as np
from pyspark import SparkContext, SparkConf

# define the arguement routine
NUM_FACTORS = int(sys.argv[1])
NUM_WORKERS = int(sys.argv[2])
NUM_ITERATIONS = int(sys.argv[3])
BETA_VALUE = float(sys.argv[4])
LAMBDA_VALUE = float(sys.argv[5])
INPUTV_FILEPATH = sys.argv[6]
OUTPUTW_FILEPATH = sys.argv[7]
OUTPUTH_FILEPATH = sys.argv[8]

# read rating csv file  
def readRatingFromTestFile(str):
    line = str.split(",") # split by comma
    return (int(line[1])-1, int(line[0])-1, int(line[2]))  # return a triple with order (movie id, use id, score)

# set up sc
conf = SparkConf().setAppName("SGD")
sc = SparkContext(conf=conf)
# read data from file and transform as rdd for further manipulation
rdd = sc.textFile(INPUTV_FILEPATH).map(readRatingFromTestFile)
# select the distinct movie ids from rdd
MOVIEIDS = rdd.map(lambda line: line[0]).distinct()
# initialize factor array with random number for each movie id
H = MOVIEIDS.map(lambda key: (key, np.random.rand(NUM_FACTORS)))
# select the distinct user ids from rdd
USERIDS = rdd.map(lambda line: line[1]).distinct()
# initialize factor array with random number for each user id
W = USERIDS.map(lambda key: (key, np.random.rand(NUM_FACTORS)))
# convert rdd W to a dictionary called DICT_W
DICT_W = dict(W.collect())
# convert rdd H to a dictionary called DICT_H
DICT_H = dict(H.collect())
# select the max movie id from the movie ids for block segmentation
MOVIEID_MAX = rdd.map(lambda line: line[0]).max()
# select the max user id from the user ids for block segmentation
USERID_MAX = rdd.map(lambda line: line[1]).max()
# compute the length of user blocks according to the number of workers
USER_BLOCK_LENGTH = (USERID_MAX/NUM_WORKERS) + 1
# compute the length of movie blocks according to the number of workers
MOVIE_BLOCK_LENGTH = (MOVIEID_MAX/NUM_WORKERS) + 1
# assign the data in W to different blocks with block_id
W_BLOCKS = W.map(lambda (key, value): ((key/USER_BLOCK_LENGTH), (key, value)))
# assign the data in H to different blocks with block_id
H_BLOCKS = H.map(lambda (key, value): ((key/MOVIE_BLOCK_LENGTH), (key, value)))

# a fucntion for updating W and H
def updatewh(matrix_v, block_w, block_h, prev_iter):
    n_movie_dict = dict()
    n_user_dict = dict()
    for point in matrix_v:
        movie_id = point[0]
        user_id = point[1]
        n_movie_dict[movie_id] = n_movie_dict.get(movie_id, 0) + 1.0
        n_user_dict[user_id] = n_user_dict.get(user_id, 0) + 1.0
    cur_iter = 0
    result = []
    w_dic = dict(block_w)
    h_dic = dict(block_h)
    for point in matrix_v:
        moive_id = point[0]
        user_id = point[1]
        rating = point[2]
        w_array = w_dic[user_id]
        h_array = h_dic[moive_id]
        distance = rating - np.sum(w_array * h_array)
        stepsize = (100 + prev_iter + cur_iter)**(-BETA_VALUE)
        slip_w = (-2)*distance * h_array + 2*LAMBDA_VALUE/n_user_dict.get(user_id)*w_array
        w_vector = w_array - slip_w * stepsize
        slip_h = (-2)*distance*w_array+2*LAMBDA_VALUE/n_movie_dict.get(moive_id)*h_array
        h_vector = h_array - slip_h * stepsize
        w_dic[user_id] = np.asarray(w_vector)
        h_dic[moive_id] = np.asarray(h_vector)
        cur_iter += 1
    result.append(w_dic.items())
    result.append(h_dic.items())
    result.append(cur_iter)
    return result


if __name__ == "__main__": 
    EPSILON_ITERATION = 0
    for itera in range(NUM_ITERATIONS):
        V_stratum_data = rdd.filter(lambda triple: 
            ((triple[1]/USER_BLOCK_LENGTH+itera)%NUM_WORKERS) == (triple[0]/MOVIE_BLOCK_LENGTH))
        V_users_stratum = V_stratum_data.map(lambda triple: 
                                             ((triple[1]/USER_BLOCK_LENGTH), triple))
        V_user_block_stratum = V_users_stratum.groupByKey().map(
            lambda (key, value): (key, list(value)))
        H_shift = H_BLOCKS.groupByKey().map(
            lambda (key, value): (((key+NUM_WORKERS-itera)%NUM_WORKERS), list(value)))
        W_H_union = W_BLOCKS.groupByKey().map(
            lambda (key, value): (key, list(value))).join(H_shift)
        parallel_stratum = V_user_block_stratum.join(W_H_union).partitionBy(NUM_WORKERS)
        workers_update = parallel_stratum.map(
            lambda (key, value): updatewh(value[0], value[1][0], value[1][1], EPSILON_ITERATION)).collect()
        for worker in workers_update:
            worker_W = dict(worker[0])
            worker_H = dict(worker[1])
            EPSILON_ITERATION += worker[2]
            DICT_W.update(worker_W)
            DICT_H.update(worker_H)
        W_BLOCKS = sc.parallelize(DICT_W.items()).map(
            lambda (key, value): ((key/USER_BLOCK_LENGTH), (key, value)))
        H_BLOCKS = sc.parallelize(DICT_H.items()).map(
            lambda (key, value): ((key/MOVIE_BLOCK_LENGTH), (key, value)))
    W = np.array(W_BLOCKS.map(lambda (key, value): value).
                 sortByKey().map(lambda (key, value): value).collect())
    H = np.array(H_BLOCKS.map(lambda (key, value): value).
                 sortByKey().map(lambda (key, value): value).collect())
    np.savetxt(OUTPUTW_FILEPATH, W, delimiter=',', fmt="%f")
    np.savetxt(OUTPUTH_FILEPATH, H.T, delimiter=',', fmt="%f")





