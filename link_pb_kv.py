
import pandas as pd
import numpy as np
from pyjarowinkler import distance
from fuzzywuzzy import fuzz
import datetime
import time
import re
import multiprocessing
from multiprocessing import Pool
from joblib import Parallel, delayed
import os.path

"""For each year, create a folder with the name of the year under dataset
	change the path name to the correspond year folder in the function 'compare_name'
	change the query in function 'get_all_pb' and 'get_all_registed' to the corresponds year
	change the query1 to the correct year in the __main__function
"""

# reference: https://pythonadventures.wordpress.com/tag/import-from-parent-directory/


def load_src(name, fpath):
    import os
    import imp
    # get the directory of this python file, which is under other needed files
    # prefer the absolute path
    p = fpath if os.path.isabs(fpath) else os.path.join(
        os.path.dirname(__file__), fpath)
    return imp.load_source(name, p)


def deduplicate_words(s):
    # Take a string s and split it into words and return a string with unique words
    words = re.sub('[^\wöäåÖÄÅ]+', " ", s)
    ulist = [x for x in words.split(" ") if x != 'box']
    return "".join(set(ulist))


def de_sortedwords(s):
    """Delete the duplicated words in s and return a sorted words string"""
    words = re.findall('[\wöäåÖÄÅ]+', s)
    ulist = []
    [ulist.append(x) for x in words if x not in ulist]
    return "".join(sorted(ulist))


def get_jw(x, y):
    return distance.get_jaro_distance(x, y)


def get_method(l):
    # l is a list of position of the max score(s), return the methd(s) of the highest score
    method = ""
    if sorted(l) == [0]:
        method = 'jw'
    elif sorted(l) == [1]:
        method = 'jw_sorted'
    elif sorted(l) == [2]:
        method = 'fw'
    elif sorted(l) == [3]:
        method = 'fw_sorted'
    elif sorted(l) == [0, 1]:  # two methods
        method = 'jw/jw_sorted'
    elif sorted(l) == [0, 2]:
        method = 'jw/fw'
    elif sorted(l) == [0, 3]:
        method = 'jw/fw_sorted'
    elif sorted(l) == [2, 3]:
        method = 'jw_sorted/fw'
    elif sorted(l) == [2, 4]:
        method = 'jw_sorted/fw_sorted'
    elif sorted(l) == [3, 4]:
        method = 'fw/fw_sorted'
    elif sorted(l) == [1, 2, 3]:
        method = 'jw/jw_sorted/fw'
    elif sorted(l) == [1, 2, 4]:
        method = 'jw/jw_sorted/fw_sorted'
    elif sorted(l) == [2, 3, 4]:
        method = 'jw_sorted/fw/fw_sorted'
    elif sorted(l) == [1, 2, 3, 4]:
        method = 'All mehtods'
    else:
        method = 'error'
    return method


def merge_2adresses(pbdf):
    # the function go through pbdf and check if besoksadress and utdelningsadress are the same
    # some rows in pb have two addresses, some have one
    # check if besoksadress is missing, fill it with utdelningsadress
    # try to use both columns for compare names
    # @Return a new df without column utdelningsadress
    changed = 0
    index = 0
    for r in pbdf.itertuples():
        # compare the string without spaces
        if not r.BESOKSADRESS.replace(" ", "") and r.UTDELNINGSADRESS.replace(" ", "")!="":
            s = " ".join([r.BESOKSADRESS, r.UTDELNINGSADRESS])
            # delete digits, duplicted words and word-box
            s = deduplicate_words(s)
            pbdf.at[index, 'BESOKSADRESS'] = s
            changed += 1
        index += 1

    pbdf_new = pbdf[['PLATSNUMMER', 'AG_NAMN',
                     'NAMN', 'BESOKSADRESS', 'BESOKSORT', 'Orgnr']]
    print("The number of adresses changed is %d (%f)" %
          (changed, changed/pbdf.shape[0]))
    return pbdf_new


def get_all_pb(en):
    # change besoksort to besoksort_standard
    query1 = (
        "select distinct PLATSNUMMER, trim(p.AG_NAMN) AG_NAMN, trim(j.Namn) NAMN, trim(p.BESOKSADRESS) BESOKSADRESS, "
        "trim(p.UTDELNINGSADRESS) UTDELNINGSADRESS, trim(p.BESOKSORT) BESOKSORT, substring(j.PeOrgNr, 3, 12) Orgnr from"
        " (select *  from pb_1 where YEAR=2018) p join [fdb_prod].[FDB2018].[dbo].[JE] j on "
        "p.ORGNRNUM = substring(j.PeOrgNr, 3, 12)"
            )
    query = "select count(*) cnt from pb_1 where Year = 2019"
    pb = pd.read_sql(query1, en)
    pb.fillna("")
    pb.astype(str)
    # transform strings to lowercase
    pb = pb.applymap(lambda x: str(x).lower())
    pb = merge_2adresses(pb)
    total_year = pd.read_sql(query, en)['cnt'].values[0]
    print("year 2019 find %f matches in FDB" % (pb.shape[0]/total_year))

    return pb


def get_all_registed(en):
   # change the fdb year in the query to retrieve the correct year-data
    query2 = (
        "select a.CfarNr, substring(a.PeOrgNr, 3, 12) OrgNr, trim(a.Ben) Ben, trim(j.Namn) Namn, trim(a.BGata) BGata, "
        "trim(a.BGatuNr) BGatuNr, trim(a.BGatuRest) BGatuRest, trim(a.BPostOrt) BPostOrt from (select CfarNr, PeOrgNr, "
        "Ben, BGata, BGatuNr, BGatuRest, BPostOrt from [fdb_prod].[FDB2018].[dbo].[AE]) a join (select PeOrgNr, Namn from "
        "[fdb_prod].[FDB2018].[dbo].[JE]) j on a.PeOrgNr = j.PeOrgNr"
    )
    registed_names = pd.read_sql(query2, en)
    registed_names.fillna("")
    registed_names.astype(str)
    registed_names = registed_names.applymap(lambda x: str(x).lower())
    return registed_names


def get_one_pb(p, orgnr):
    # this is combined with the whole table if the table is selected
    pb_o = p.loc[p['Orgnr'] == orgnr]
    pb_o.fillna("")
    # delete all the duplicates in df
    pb_o = pb_o.drop_duplicates(inplace=True)
    return pb_o


def get_one_localunit(r, orgnr):
    # this is used when the whole table is choosen
    localunit = r.loc[r['OrgNr'] == orgnr]
    localunit.fillna("")
    localunit = localunit.drop_duplicates(inplace=True)
    return localunit


def prepare_data(p, r, orgnr):
    # Extract the dataframe matching the orgnr
    # Input: @p is the dataframe of pb data
    # @localunit- is the dataframe of Business registed data
    # @orgnrlist: all the unique orgnr in pb of the year
    # Return:a list of list e.g. inner_list: [pb_orgnr, localunit_orgnr, orgnr]
    # returned outerlist: [[pb_orgnr1, localunit_orgnr1, orgnr1], inner_list, ......[]]
    pb_o = p.loc[p['Orgnr'] == orgnr]
    pb_o.fillna("")
    pb_o.astype(str)
    # this is used when the whole table is choosen
    localunit = r.loc[r['OrgNr'] == orgnr]
    localunit.fillna("")
    localunit.astype(str)
    print("----print from prepare_data%s-----" % orgnr)
    print("-----found pb shape----(%d, %d)" % (pb_o.shape))
    print("------found localunit shape ---------(%d, %d)" % (localunit.shape))
    return (orgnr, pb_o, localunit)

    # print("-----------in the function prepare_data print the type of orgnr--------------------")
    # print(type(orgnr))
    # compare_name(orgnr, pb_o, localunit)
    # print("finish on %s" % orgnr)


def prepare_data_list(p, r, orglist):
    # combine this function with the slice_data

    return [(o, get_one_pb(p, o), get_one_localunit(r, o)) for o in orglist]


def compare_location_name(name, sorted_name, df):
    # compare the name n with the df and return a subset of df with the max values of df
    # only allow one matching result, the best result returned
    df.loc[:, 'jw'] = df['conc_name'].apply(
        lambda x: distance.get_jaro_distance(x, name))
    df.loc[:, 'fw'] = df['conc_name'].apply(
        lambda x: fuzz.token_set_ratio(x, name))
    df.loc[:, 'jw_sorted'] = df['sorted_name'].apply(
        lambda x: distance.get_jaro_distance(x, sorted_name))
    df.loc[:, 'fw_sorted'] = df['sorted_name'].apply(
        lambda x: fuzz.token_set_ratio(x, sorted_name))
    # choose the max values from the jw values

    jw_max = df['jw'].max()
    fw_max = df['fw'].max()
    jw_sorted_max = df['jw_sorted'].max()
    fw_sorted_max = df['fw_sorted'].max()
    scores = [jw_max, jw_sorted_max, fw_max/100, fw_sorted_max/100]
    # return the position of the max scores in the list
    pos_max_score = [scores.index(max(scores))]
    result = pd.DataFrame()
    # compare mv_jw and mv_fw and take the max value of them tow
    if(get_method(pos_max_score) == 'jw'):
        result = df[df['jw'] == jw_max]
        result['method'] = "jw"
    elif(get_method(pos_max_score) == 'fw'):
        result = df[df['fw'] == fw_max]
        result['method'] = "fw"
    elif(get_method(pos_max_score) == 'jw/fw'):
        result = df[df['fw'] == fw_max]
        result['method'] = "jw/fw"
    elif(get_method(pos_max_score) == 'jw_sorted'):
        result = df[df['jw_sorted'] == jw_sorted_max]
        result['method'] = "jw_sorted"
    elif(get_method(pos_max_score) == 'fw_sorted'):
        result = df[df['fw_sorted'] == fw_sorted_max]
        result['method'] = "fw_sorted"
    elif(get_method(pos_max_score) == 'jw_sorted/fw_sorted'):
        result = df[df['jw_sorted'] == jw_sorted_max]
        result['method'] = "jw_sorted/fw_sorted"
    elif (get_method(pos_max_score) == 'jw/jw_sorted'):
        result = df[df['jw_sorted'] == jw_sorted_max]
        result['method'] = "jw/jw_sorted"
    elif (get_method(pos_max_score) == 'jw/fw_sorted'):
        result = df[df['jw'] == jw_max]
        result['method'] = "jw/fw_sorted"
    elif (get_method(pos_max_score) == 'fw/fw_sorted_max'):
        result = df[df['fw_sorted'] == fw_sorted_max]
        result['method'] = "fw/fw_sorted"
    elif (get_method(pos_max_score) == 'jw_sorted/fw'):
        result = df[df['jw_sorted'] == jw_sorted_max]
        result['method'] = "jw_sorted/fw"
    elif (get_method(pos_max_score) == 'fw/fw_sorted'):
        result = df[df['fw_sorted'] == fw_sorted_max]
        result['method'] = "fw/fw_sorted"
    elif (get_method(pos_max_score) == "jw/jw_sorted/fw"):
        result = df[df['jw_sorted'] == jw_sorted_max]
        result['method'] = "jw/jw_sorted/fw"
    elif (get_method(pos_max_score) == "jw/jw_sorted/fw_sorted"):
        result = df[df['jw_sorted'] == jw_sorted_max]
        result['method'] = "jw/jw_sorted/fw_sorted"
    elif (get_method(pos_max_score) == "jw_sorted/fw/fw_sorted"):
        result = df[df['jw_sorted'] == jw_sorted_max]
        result['method'] = "jw_sorted/fw/fw_sorted"
    elif (get_method(pos_max_score) == "All methods"):
        result = df[df['jw_sorted'] == jw_sorted_max]
        result['method'] = "jw/jw_sorted/fw/fw_sorted"
    else:
        print("Error")
    # if the result has more than one matches, we choose the highest total score or the first row of df
    if result.shape[0] > 1:
        result['total'] = result['jw'] + result['fw'] + \
            result['jw_sorted'] + result['fw_sorted']
        result = result.sort_values(by=['total'], ascending=False)
        # return the first row of the sorted dataframe
        result = result.iloc[[0]]
    return result


# with the given orgnr compare it with ae_ben and write the result into a file
def compare_name(orgnr, pb_one, registed_one):
    # @input: orgnr
    # @pb_one: all the pb.AG_NAMN+Je.NAMN+pb.BESOKADRESS+pb.BESOKORT under the orgnr
    # @registed_one: all the ae.BEN+je.NAMN+ae.ADRESS+ae.POSTORT under the orgnr
    # Return: for each orgnr a df is created and the file contained cfarnr is copied to a file

    df = pd.DataFrame()
    localunit = registed_one
    pb_o = pb_one
    print("-------------in the compare_name print the type of orgnr----------------")
    print(type(orgnr))
    # do not check the illegual orgnr e.g. 0
    if (len(str(orgnr)) == 10):

        # add new columns combining Ben and Namn and address and deduplicate and sort the string
        # comments from https://stackoverflow.com/questions/12555323/adding-new-column-to-existing-dataframe-in-python-pandas, avoid warning message
        localunit.loc[:, 'name'] = localunit['Ben'] + " " + localunit['Namn'] + " " + localunit['BGata'] + \
            " " + localunit['BGatuNr'] + " " + \
            localunit['BGatuRest'] + " " + localunit['BPostOrt']
        localunit['name'] = localunit['name'].apply(lambda x: str(x).upper())
        localunit.loc[:, 'conc_name'] = localunit['name'].apply(
            lambda x: deduplicate_words(x))
        localunit.loc[:, 'sorted_name'] = localunit['name'].apply(
            lambda x: de_sortedwords(x))
        # print("----------localunit has (%i) rows----------" %localunit.shape[0])
        # When the names are found
        if(localunit.empty == False):
            # Go through names in PB, AG_NAMNs are different, but orgnr is the same
            for row in pb_o.itertuples():
                result = pd.DataFrame()

                # # #take the items from the localunit of the same ort
                merge_pb_localunit = localunit[localunit['BPostOrt']
                                               == row.BESOKSORT]
                merge_pb_localunit.fillna("")
                # print(merge_pb_localunit.shape)
                pb_name = row.AG_NAMN+" "+row.NAMN+" "+row.BESOKSADRESS+" "+row.BESOKSORT
                pb_de_name = deduplicate_words(pb_name.upper())
                pb_sorted_name = de_sortedwords(pb_name.upper())
                # # # print("the found items in je_as of the same ort are %i" % merge_pb_localunit.shape[0])
                if (merge_pb_localunit.shape[0] != 0):
                    # compare within the same portort
                    result = compare_location_name(
                        pb_de_name, pb_sorted_name, merge_pb_localunit)
                    result['AG_NAMN'] = pb_name
                    result['PLATSNUMMER'] = row.PLATSNUMMER
                # 	#compare the name n with the df and return a subset of df with the max values of df
                else:  # if the location cannot be found compare this row with the whole dataframe localunit
                    result = compare_location_name(
                        pb_de_name, pb_sorted_name, localunit)
                    result['AG_NAMN'] = pb_name
                    result['PLATSNUMMER'] = row.PLATSNUMMER
                print("add one ag_name (%i, %i)" % result.shape)
                df = df.append(result)

            print("write one file to the folder (%i, %i)" % df.shape)
            # encoding should be specified, otherwise this server save the file with windows encoding, e.g. iso-8859-1
            # df.to_csv("C:/Users/TSTDAWU/Documents/datasets/2012/%s.csv"%orgnr)
            df.to_csv(
                "C:/Users/TSTDAWU/Documents/datasets/2018/%s.csv" % orgnr)
        else:  # write the orgnr that do not find match in je/ae to a file
            try:  # a+ create a file if it does not exist or append text to it
                # with open("c:/Users/TSTDAWU/Documents/results/nonorgnr_2012.csv", "a+") as f:
                with open("C:/Users/TSTDAWU/Documents/datasets/cfarTest208.csv", "a+") as f:
                    f.write(
                        "%s   did not found matches in the Business register\n" % orgnr)
            except:
                print("Error with writing to the file")
    else:  # if the orgnr is not 10-figures
        try:
            # with open("c:/Users/TSTDAWU/Documents/results/nonorgnr_2012.csv", "a+") as f:
            with open("C:/Users/TSTDAWU/Documents/datasets/nonorgnr_cfarTest2018.csv", "a+") as f:
                f.write("%s   is ill-formed orgnr\n" % orgnr)
        except:
            print("Error with writing to the file")


def multi_compare_names(args):
    return compare_name(*args)


def slice_data(data, nprocs):
    # https://www.kth.se/blogs/pdc/2019/02/parallel-programming-in-python-multiprocessing-part-1/
    # this function divide data into smaller pieces before the parallel process for speed increasing
    # the function in the multiporcesses need to change for handling a list
    aver, res = divmod(len(data), nprocs)
    nums = []
    for proc in range(nprocs):
        # if the process is less than the rest of the dataset
        if proc < res:
            nums.append(aver + 1)
        else:
            nums.append(aver)
    count = 0
    slices = []
    for proc in range(nprocs):
        slices.append(data[(count):(count + nums[proc])])
        count += nums[proc]
    return slices

"""
if __name__ == '__main__':
    inlab = load_src(
        "connectinlab", "C:/Users/TSTDAWU/Documents/Bitbucket/Bitbucket/repos/wp1-main-repository/python/importpb_inlab.py")
    en = inlab.connectinlab()

    # choose all the distinct orgnization numbers from pb_1
    # choose from the pb_prv private sector, do not do the unnecessary work!!!!! Or?
    query1 = "select distinct ORGNRNUM from pb_1 where Year=2018 and len(ORGNRNUM)=10 order by ORGNRNUM"
    orgnrsdf = pd.read_sql(query1, en)
    orgnrs = orgnrsdf['ORGNRNUM'].astype(str)
    start_time = time.process_time()
    pb = get_all_pb(en)
    registed = get_all_registed(en)
    print("get all the data from database")
    print("-----------Execution time with for in list is %f seconds -----------" %
          (time.process_time()-start_time))

    start_time_1 = time.process_time()
    # #Method 1: with the following 3 rows, the execution tiime is 16581 secondes, which is much faster than the
    # 	#the execution time after the second cpu updating is 12002.67 seconds

    # orgargs = [(o, get_one_pb(pb, o), get_one_localunit(registed, o))
    #           for o in orgnrs]
    # The code blow jobs=-1 only 4% cpu are used, it takes longer time than pool
    # orgargs = Parallel(n_jobs=-1, backend='threading')(delayed(prepare_data)
    #                                                 (pb, registed, o) for o in orgnrs)

    pool = Pool(46)
    orgargs = pool.starmap(prepare_data, [(pb, registed, o) for o in orgnrs])
    pool.close()
    print("multiprocess prepartion of data used %f" %
          (time.process_time()-start_time_1))
    # cpu and memory usage 17%, --used 740 seconds
    # data = prepare_data(pb, registed, orgnrs)
    # org_slices = slice_data(orgnrs, 40)
    print(len(orgargs))
    print(orgargs[0])
    print("execution time on preparing data is %f" %
          (time.time() - start_time_1))

    # divide the list of orgargs into smaller slices
    start_time_2 = time.time()
    with Pool(46) as p:
        p.map(multi_compare_names, [(o, pb_one, r_one)
                                    for (o, pb_one, r_one) in orgargs])
    # # #end of the function
    print("-----------Execution time with for in list is %f seconds -----------" %
          (time.time()-start_time_2))

    # print(pb.shape)
    # print(registed.shape)
    # print(len(orgnrs))
    # pool_size = multiprocessing.cpu_count() *7
    # print("pool size is %i"%pool_size)
    # #This Pool is very inefficient, cpu and memory are used up to 90% but it took 24087.377840727935 seconds to execut the preåare data of pb_o
    # args = [(pb, registed, o) for o in orgnrs]
    # with Pool(pool_size, ) as p:
    # 	print(p.map(multi_prepare_data, args))

    # execution time is 2560.8915324442839 seconds/3130.6129055680944/2217.0647
    # after cpu upgrading 1041.51331270128 seconds/1036.2024048180142 seconds
    # orgnrs = ['5565595450', '5567941173', '5564472677', '5566370341', '2120000142', '2120001355', '2321000131', '5565479630', '2321000016', '5562421718']
    # #this function used 539 (ten orgnrs) seconds on 10 orgnrs

    # cpu usage and memory has increased to  90%, but the total execution time is long, cannot finish the Pool execution
    # end_time_1 = time.clock()
    # print("---------time used with multipleprocessing on orgnrs is  %f sectond---------"&end_time_1-start_time)
    # #cpu usage: with the get_one
    # start_time_2 = time.clock()
    # this line does not work anymore.???????????
    # print("pool_size is %i, and orgnrs are %i" %(pool_size, len(orgargs)))

    # start_time_1=time.clock()
    # Method 2: this parallel process is not effieient either, on the total 2017 data, the execution time is 170201.970180
    # this method combined the function of prepraring the data and process the data directly in the compare_name function
    # print("------------execution time with joblib is %f seconds ---------------"% (time.clock()-start_time))
    # add the result in INLAB
    # print columns in the datframe, deltete the unwanted column and ingect into INLAB
    #
"""
