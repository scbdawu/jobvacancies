import glob
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import datetime
import seaborn as sns
import time
import re
import af_api_json_processing as afjson
from pyjarowinkler import distance
from fuzzywuzzy import fuzz
from multiprocessing import Pool


def get_quarter(month):
    #by the input month (str)
    #return the quarter
    if month < 4:
        return 1
    elif month < 7:
        return 2
    elif month < 10:
        return 3
    elif month < 13:
        return 4
    else:
        return np.nan


def get_ngcategory(ng):
    #convert the detailed ng number to the category required in the database table
    category = ""
    if str(ng)[0:2] in ['01', '02', '03']:
        category = 'A'
    elif str(ng)[0:2] in [
            '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15',
            '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26',
            '27', '28', '29', '30', '31', '32', '33'
    ]:
        category = 'B+C'
    elif str(ng)[0:2] in ['35', '36', '37', '38', '39']:
        category = 'D+E'
    elif str(ng)[0:2] in ['41', '42', '43']:
        category = 'F'
    elif str(ng)[0:2] in ['45', '46', '47']:
        cdategory = 'G'
    elif str(ng)[0:2] in ['49', '50', '51', '52', '53']:
        category = 'H'
    elif str(ng)[0:2] in ['55', '56']:
        category = 'I'
    elif str(ng)[0:2] in ['58', '59', '60', '61', '62', '63']:
        category = 'J'
    elif str(ng)[0:2] in ['64', '65', '66', '68']:
        category = 'K+L'
    elif str(ng)[0:2] in ['69', '70', '71', '72', '73', '74', '75']:
        category = 'M'
    elif str(ng)[0:2] in ['77', '78', '79', '80', '81', '82']:
        category = 'N'
    elif str(ng)[0:2] in ['84']:
        category = 'O'
    elif str(ng)[0:2] in ['85', '86', '87', '88']:
        category = 'P+Q'
    elif str(ng)[0:2] in ['90', '91', '92', '93', '94', '95', '96']:
        category = 'R+S'
    else:
        category = np.nan
    return category


def read_json(path):
    file_dump = []
    #print(type(file_dump))
    #the file is saved in json format
    for f in glob.iglob(path + "*.json"):
        with open(f, "r") as file:
            data = json.load(file)
            date = datetime.date.fromisoformat(
                f.split('/')[-1].replace(".json", "")[0:10])
            number_ads = len(data)
            summary = {
                "date": date,
                "number_ads": number_ads,
                "file": f,
                "data": data
            }
            file_dump.append(summary)

    return file_dump


def aggregate_writeresult(file_name):
    #input file_name is the name of the file for the generated result
    #the function process json files day by day
    #both the simple keys and the employer information
    #aggregating the information into one total_df
    #save the total_df into one file
    #with the information, can the comparison be done to the fdb to seperate public/private sectors
    total_df = pd.DataFrame()
    for i in range(len(file_dump)):
        #keep only json_records have all keys, delete those have removed
        clean_json = afjson.divide_json(file_dump[i].get('data'))
        #read the simple keys, only the online ads
        df = afjson.read_one_day_dump(clean_json)
        #read in the complex keys
        #address_df = get_work_addresses(clean_json)
        #description_df = get_ads_description(clean_json)
        employer_df = afjson.get_employer_values(clean_json)
        oneday_df = pd.merge(df, employer_df, how="inner", on='ads_id')  ##
        #aggregate day by day data
        total_df = afjson.add_more_data_dump(total_df, oneday_df)
        print(f"add {i} day: {df.shape[0]} rows")
        #handle the nan value in vacancies
        total_df['number_of_vacancies'] = total_df[
            'number_of_vacancies'].fillna(1)
        print(total_df.loc[total_df['number_of_vacancies'].isna()])
        print(f"total rows: {total_df.shape}")

    total_df.to_csv("/home/inlab4/Documents/Dan_datasets/AF_results/%s" %
                    file_name,
                    index=False)
    print(f"saved {file_name}, total rows {total_df.shape[0]}")


def public_sector_table(input_file, output_file):
    #input file is generated from the function above
    #here the pb_data is compaired with fdb to generate the public sector table
    fdb_dtypes = {
        'AEAnt': np.int32,
        'Sektor': str,
        'COAdress': str,
        'GatuNr': str,
        'Ng1': str,
        'PeOrgNr': str,
        'JE_orgnr': str,
        'CfarNr': str,
        'BGatuNr': str,
        'BpostNr': str,
        'BNg1': str
    }
    fdb = pd.read_csv(r"/media/inlab4/My Passport/Dan/fdb2018/JeAe2018.csv",
                      sep=";",
                      dtype=fdb_dtypes)

    public_sectors = [
        '131110', '131120', '131130', '131311', '131312', '131313', '131321',
        '131322', '131323', '131400'
    ]
    public_df = fdb.loc[fdb.Sektor.isin(public_sectors)]

    public_df.drop_duplicates(subset=['JE_orgnr'], inplace=True)
    #print(public_df['Sektor'].value_counts())
    public_df.loc[:, 'JE_org'] = public_df['JE_orgnr'].apply(
        lambda x: str(x)[2:])

    result_public = pd.DataFrame()
    employer_dtypes = {'employer_organization_number': str, 'ads_id': str}
    employer = pd.read_csv(
        f"/home/inlab4/Documents/Dan_datasets/AF_results/{input_file}",
        dtype=employer_dtypes)
    employer = employer[[
        'ads_id', 'number_of_vacancies', 'publication_date',
        'employer_organization_number', 'last_publication_date', 'removed',
        'year', 'month'
    ]]
    employer[
        'employer_organization_number'] = employer.employer_organization_number.astype(
            str)

    data_df = pd.merge(employer,
                       public_df,
                       how="left",
                       left_on='employer_organization_number',
                       right_on='JE_org')

    result_public['orgnr'] = data_df['employer_organization_number'].values
    result_public['Sektor'] = data_df['Sektor'].values
    result_public['Sektor2'] = data_df['Sektor'].apply(
        lambda x: str(x)[:2]).values
    result_public['Platsnummer'] = data_df['ads_id']
    result_public['Antal_platser'] = data_df['number_of_vacancies']
    result_public['Publiceringsdatum'] = data_df['publication_date']
    result_public['Year'] = data_df['year']
    result_public['Month'] = data_df['month']
    result_public['Quarter'] = data_df['publication_date'].apply(
        lambda x: get_quarter(
            datetime.datetime.strptime(str(x), '%Y-%m-%dT%H:%M:%S').month))
    result_public['JE_org'] = data_df['JE_orgnr']
    result_public[~result_public['Sektor'].isna()].to_csv(
        f'/home/inlab4/Documents/Dan_datasets/AF_results/pb_off/{output_file}',
        index=False)
    print(f"write the {output_file}")


def aggregate_employer_address(file_name):
    total_df = pd.DataFrame()
    for i in range(len(file_dump)):
        #keep only json_records have all keys, delete those have removed
        clean_json = afjson.divide_json(file_dump[i].get('data'))
        #read the simple keys, only the online ads
        df = afjson.read_one_day_dump(clean_json)
        #read in the complex keys
        #address_df = get_work_addresses(clean_json)
        #description_df = get_ads_description(clean_json)
        employer_df = afjson.get_employer_values(clean_json)
        #print(employer_df.shape)
        address_df = afjson.get_work_addresses(clean_json)
        #print(address_df.shape)
        em_ad_df = pd.merge(employer_df,
                            address_df,
                            how='outer',
                            on='ads_id',
                            validate='one_to_one')
        print(em_ad_df.shape)
        print("-----------------")
        #add simple key + employer + address
        oneday_df = pd.merge(df, em_ad_df, how="inner", on='ads_id')
        #aggregate day by day data
        total_df = afjson.add_more_data_dump(total_df, oneday_df)
        print(f"add {i} day: {df.shape[0]} rows")
        #handle the nan value in vacancies
        total_df['number_of_vacancies'] = total_df[
            'number_of_vacancies'].fillna(1)
        print(total_df.loc[total_df['number_of_vacancies'].isna()])
        if df.shape[0] != employer_df.shape[0]:
            print(
                "diff between simple key df and employer df is {df.shape[0]-employer_df.shape[0]}"
            )
        if address_df.shape[0] != employer_df.shape[0]:
            print(
                "address between simple key df and employer df is {address_df.shape[0]-employer_df.shape[0]}"
            )

    total_df.to_csv("/home/inlab4/Documents/Dan_datasets/AF_results/%s" %
                    file_name,
                    index=False)
    print(f"saved {file_name}, total rows {total_df.shape[0]}")


#generate two extra columns standard municipality code and nuts2 code
def lookup_standard_municipality_code(name, code):
    #the function look up the standard code and nuts2 in kommun_lan table
    kommun_lan = pd.read_csv(
        "/home/inlab4/Documents/Dan_datasets/kommun_lan.csv",
        sep=";",
        dtype={'Code': str})

    result = {}
    #match only characters
    pattern = re.compile('\D+', re.IGNORECASE)
    #if the name is digits and code is characters, which are wrong
    if name.isdigit():
        match = kommun_lan.loc[kommun_lan.Code == name]
        if match.shape[0] == 1:
            result = {'code': name, "nuts2": match.nuts2.values.tolist()[0]}
    elif code.isdigit():
        match = kommun_lan.loc[kommun_lan.Code == code]
        if match.shape[0] == 1:
            result = {'code': code, "nuts2": match.nuts2.values.tolist()[0]}
    else:
        result = {'code': np.nan, 'nuts2': np.nan}
    return result


def deduplicate_words(s):
    # Take a string s and split it into words and return a string with unique words

    pattern = re.compile('[^\wöäå]+', re.IGNORECASE)
    #words is string
    words = re.sub(pattern, " ", s)
    ulist = set([str(x).upper() for x in words.split()])
    return "".join(ulist)


def de_sortedwords(s):
    """Delete the duplicated words in s and return a sorted words string"""

    words = re.findall('[\wöäåÖÄÅ]+', s)
    ulist = set([str(x).upper() for x in words])
    return "".join(sorted(ulist))


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


def compare_location_name(name, sortedname, df):
    # compare the name n with the df and return a subset of df with the max values of df
    # only allow one matching result, the best result returned

    df.loc[:, 'jw'] = df['concat_name'].apply(
        lambda x: 0
        if x is np.nan else distance.get_jaro_distance(str(x), name))
    df.loc[:, 'fw'] = df['concat_name'].apply(
        lambda x: 0 if x is np.nan else fuzz.token_set_ratio(str(x), name))
    df.loc[:, 'jw_sorted'] = df['sorted_name'].apply(
        lambda x: 0
        if x is np.nan else distance.get_jaro_distance(str(x), sortedname))
    df.loc[:, 'fw_sorted'] = df['sorted_name'].apply(
        lambda x: 0
        if x is np.nan else fuzz.token_set_ratio(str(x), sortedname))
    # choose the max values from the jw values

    jw_max = df['jw'].max()
    fw_max = df['fw'].max()
    jw_sorted_max = df['jw_sorted'].max()
    fw_sorted_max = df['fw_sorted'].max()
    scores = [jw_max, jw_sorted_max, fw_max / 100, fw_sorted_max / 100]
    # return the position of the max scores in the list
    pos_max_score = [scores.index(max(scores))]

    # compare mv_jw and mv_fw and take the max value of them tow
    if (get_method(pos_max_score) == 'jw'):
        result = df[df['jw'] == jw_max]
        result.loc[:, 'method'] = "jw"
    elif (get_method(pos_max_score) == 'fw'):
        result = df[df['fw'] == fw_max]
        result.loc[:, 'method'] = "fw"
    elif (get_method(pos_max_score) == 'jw/fw'):
        result = df[df['fw'] == fw_max]
        result.loc[:, 'method'] = "jw/fw"
    elif (get_method(pos_max_score) == 'jw_sorted'):
        result = df[df['jw_sorted'] == jw_sorted_max]
        result.loc[:, 'method'] = "jw_sorted"
    elif (get_method(pos_max_score) == 'fw_sorted'):
        result = df[df['fw_sorted'] == fw_sorted_max]
        result.loc[:, 'method'] = "fw_sorted"
    elif (get_method(pos_max_score) == 'jw_sorted/fw_sorted'):
        result = df[df['jw_sorted'] == jw_sorted_max]
        result.loc[:, 'method'] = "jw_sorted/fw_sorted"
    elif (get_method(pos_max_score) == 'jw/jw_sorted'):
        result = df[df['jw_sorted'] == jw_sorted_max]
        result.loc[:, 'method'] = "jw/jw_sorted"
    elif (get_method(pos_max_score) == 'jw/fw_sorted'):
        result = df[df['jw'] == jw_max]
        result.loc[:, 'method'] = "jw/fw_sorted"
    elif (get_method(pos_max_score) == 'fw/fw_sorted_max'):
        result = df[df['fw_sorted'] == fw_sorted_max]
        result.loc[:, 'method'] = "fw/fw_sorted"
    elif (get_method(pos_max_score) == 'jw_sorted/fw'):
        result = df[df['jw_sorted'] == jw_sorted_max]
        result.loc[:, 'method'] = "jw_sorted/fw"
    elif (get_method(pos_max_score) == 'fw/fw_sorted'):
        result = df[df['fw_sorted'] == fw_sorted_max]
        result.loc[:, 'method'] = "fw/fw_sorted"
    elif (get_method(pos_max_score) == "jw/jw_sorted/fw"):
        result = df[df['jw_sorted'] == jw_sorted_max]
        result.loc[:, 'method'] = "jw/jw_sorted/fw"
    elif (get_method(pos_max_score) == "jw/jw_sorted/fw_sorted"):
        result = df[df['jw_sorted'] == jw_sorted_max]
        result.loc[:, 'method'] = "jw/jw_sorted/fw_sorted"
    elif (get_method(pos_max_score) == "jw_sorted/fw/fw_sorted"):
        result = df[df['jw_sorted'] == jw_sorted_max]
        result.loc[:, 'method'] = "jw_sorted/fw/fw_sorted"
    elif (get_method(pos_max_score) == "All methods"):
        result = df[df['jw_sorted'] == jw_sorted_max]
        result.loc[:, 'method'] = "jw/jw_sorted/fw/fw_sorted"
    else:
        print("Error")
    # if the result has more than one matches, we choose the highest total score or the first row of df
    result.loc[:, 'total'] = result['jw'] + result['fw'] + \
            result['jw_sorted'] + result['fw_sorted']
    result = result.sort_values(by=['total'], ascending=False)
    # return the first row of the sorted dataframe
    result = result.iloc[[0]]
    return result


def get_cfarnr(indf, fdb_prv):
    #retrieve cfarnr by comparing to fdb_prv

    statistic_cfarnr = pd.DataFrame()
    cfarnr = []
    sector = []
    ng = []
    for row in indf.itertuples():

        theOrgNr = str(row.employer_organization_number)
        theKommun = str(row.standard_kommunkod)
        thePostcode = str(row.address_postcode)
        theName = " ".join([
            str(row.employer_name),
            str(row.employer_workplace),
            str(row.address_street_address)
        ])
        theName = deduplicate_words(theName)
        theSortedName = de_sortedwords(theName)
        #get the fdb data
        regdf = fdb_prv.loc[fdb_prv.JE_orgnr == theOrgNr]
        regdf.loc[:, 'name'] = regdf['Ben'].map(str) + " " + regdf['Namn'].map(
            str) + " " + regdf['BGata'].map(
                str) + " " + regdf['BGatuRest'].map(str)
        regdf.loc[:, 'concat_name'] = regdf['name'].apply(
            lambda x: deduplicate_words(x))
        regdf.loc[:, 'sorted_name'] = regdf['name'].apply(
            lambda x: de_sortedwords(x))
        if not regdf.empty:
            #find the sme postcode
            comparedf_postcode = regdf.loc[regdf.BPostNr == thePostcode]
            comparedf = regdf.loc[regdf.Kommun == theKommun]
            if comparedf_postcode.shape[0] == 1:

                cfarnr.append(comparedf_postcode['CfarNr'].values.tolist()[0])
                sector.append(comparedf_postcode['Sektor'].values.tolist()[0])
                ng_code = comparedf_postcode['BNg1'].values.tolist()[0]
                if ng_code == '00000':
                    ng_code = comparedf_postcode['Ng1'].values.tolist()[0]
                ng.append(ng_code)
            elif comparedf_postcode.empty and comparedf.shape[0] == 1:

                cfarnr.append(comparedf['CfarNr'].values.tolist()[0])
                sector.append(comparedf['Sektor'].values.tolist()[0])
                ng_code = comparedf['BNg1'].values.tolist()[0]
                if ng_code == '00000':
                    ng_code = comparedf['Ng1'].values.tolist()[0]
                ng.append(ng_code)

            else:
                #combine name and sorted name

                if comparedf_postcode.shape[0] > 1:
                    result = compare_location_name(theName, theSortedName,
                                                   comparedf_postcode)
                elif comparedf.shape[0] > 1 & comparedf_postcode.shape[0] < 1:
                    result = compare_location_name(theName, theSortedName,
                                                   comparedf)
                else:
                    result = compare_location_name(theName, theSortedName,
                                                   regdf)
                cfarnr.append(result['CfarNr'].values.tolist()[0])
                sector.append(result['Sektor'].values.tolist()[0])
                ng_code = result['BNg1'].values.tolist()[0]
                if ng_code == '00000':
                    ng_code == result['Ng1'].values.tolist()[0]
                ng.append(ng_code)

                statistic_cfarnr = pd.concat([statistic_cfarnr, result])
        else:
            #if the orgnr not match
            print("not found")
            cfarnr.append(np.nan)
            sector.append(np.nan)
            ng.append(np.nan)
    indf['Cfarnr'] = cfarnr
    indf['Sector'] = sector
    indf['Ng'] = ng

    print(
        statistic_cfarnr.to_csv(
            "/home/inlab4/Documents/Dan_datasets/AF_results/cfarnr_statistics.csv",
            mode="a+"))
    return indf


def generate_private_table(infile, outfile):
    #with the input file containing information of adress and employer name and orgnr
    #retrieving region and ng two variables for the private sector table
    #write outfile is files need Cfarnr deduction
    fdb_dtypes = {
        'AEAnt': np.int32,
        'COAdress': str,
        'GatuNr': str,
        'Ng1': str,
        'PeOrgNr': str,
        'JE_orgnr': str,
        'CfarNr': str,
        'BGatuNr': str,
        'BPostNr': str,
        'BNg1': str,
        'Kommun': str
    }
    fdb = pd.read_csv(r"/media/inlab4/My Passport/Dan/fdb2018/JeAe2018.csv",
                      sep=";",
                      dtype=fdb_dtypes)
    fdb['JE_orgnr'] = fdb['JE_orgnr'].apply(lambda x: x[2:12])
    fdb.dropna(subset=['CfarNr'], inplace=True)
    prv_sect = [
        '111000', '112000', '113000', '114000', '121000', '122100', '122200',
        '122400', '122500', '125200', '125300', '125900', '126100', '127000',
        '128100', '128200', '128300', '129100', '129300', '129400', '141000',
        '142000'
    ]
    # cfarnr >1 need to calculate the cfarnr
    fdb_prv = fdb.loc[fdb['Sektor'].isin(prv_sect)]
    #private sector orgnrs
    orgnrs = set(fdb_prv.JE_orgnr.values)
    indf = pd.read_csv(infile,
                       dtype={
                           'employer_organization_number': str,
                           'ads_id': str,
                           'address_municipality_code': str,
                           'address_municipality': str,
                           'address_postcode': str
                       })
    indf = indf.loc[indf['employer_organization_number'].isin(orgnrs)]

    standardkommun = []
    nuts = []
    #add standard kommun-kod to indf
    for row in indf.itertuples():
        code_result = lookup_standard_municipality_code(
            str(row.address_municipality), str(row.address_municipality_code))
        standardkommun.append(code_result.get('code'))
        nuts.append(code_result.get('nuts2'))

    indf['standard_kommunkod'] = standardkommun
    indf['nuts'] = nuts
    diff_kommun = indf[
        indf['standard_kommunkod'] != indf['address_municipality_code']]
    if diff_kommun.shape[0] != 0:
        print(f"need correction {diff_kommun.shape[0]}")
    #start to calculate cfarnr
    df_splits = np.array_split(indf, 6)
    pool = Pool(6)
    df = pd.concat(
        pool.starmap(get_cfarnr, [(df, fdb_prv) for df in df_splits]))
    pool.close()
    pool.join()
    df.to_csv(outfile, index=False)


if __name__ == "__main__":

    #step 1: read in json files, donot forget to change the saved files in step 2
    #path = "/home/inlab4/Documents/Dan_datasets/AF/april/"
    #files = read_json(path)
    #file_dump = sorted(files, key=lambda i: i['date'])
    #if scraping date presented in descending order, use range(len(file_dump), 0)
    #for item in file_dump:
    #    print("Scraping date: %s    [downloaded_number_of_ads:%d]" %
    #          (item['date'], item['number_ads']))
    #    print("-----------------------------------------")
    #step 2_public_sector: run together with step 1 use file_dump to generate the result file
    #aggregate_writeresult(file_name='Employer2020_04_20.csv')
    #step 2_private_sector: join number of vacancies, employer information and address information
    #aggregate_employer_address("Employer_address_april.csv")

    #step3: read in the result from step2 and compare FDB -> public sector file for loading in DB
    #when run step3, can comment step2 and step 1
    public_sector_table('Employer2020_04_20.csv', 'pb_off4.csv')
    #private sector
    #infile = "/home/inlab4/Documents/Dan_datasets/AF_results/Employer_address_feb.csv"
    #outfile = "/home/inlab4/Documents/Dan_datasets/AF_results/pb_prv/pb_prv_feb.csv"
    #generate_private_table(infile, outfile)
    #ads = pd.read_csv(infile)

    #step 4: write the variables needed in the csv file and use simple_table.py on server tst193 load into the table
    #total = pd.DataFrame()
    #for f in glob.glob(
    #        '/home/inlab4/Documents/Dan_datasets/AF_results/pb_prv/*'):
    #    df = pd.read_csv(f)
    #    total = pd.concat([total, df])
    #print(total.shape)
    #total.drop_duplicates(subset=['ads_id'], inplace=True)
    #print(total.columns)s
    #print(total.shape)
    #print(df.shape[0]/ads.shape[0])
    #print(df.Cfarnr.isna().sum())
    #print(df.Ng.isna().sum())
    #print(df.ads_id.value_counts())
    #table = total[[
    #    'ads_id', 'employer_organization_number', 'Cfarnr', 'Sector', 'Ng',
    #    'number_of_vacancies', 'publication_date', 'nuts', 'year', 'month'
    #]]
    #table.loc[:, 'Quarter'] = table['month'].apply(get_quarter)
    #table.loc[:, 'ngs1'] = table['Ng'].apply(get_ngcategory)
    #table.loc[:, 'Sector2'] = table['Sector'].apply(lambda x: str(x)[0:2])
    #table.to_csv(
    #    '/home/inlab4/Documents/Dan_datasets/AF_results/pb_prv/pb_prv_table_april.csv',
    #    index=False)
