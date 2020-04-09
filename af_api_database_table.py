import pandas as pd
import glob
import numpy as np
import json
import matplotlib.pyplot as plt
import datetime
import seaborn as sns
import time
import af_api_json_processing as afjson


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


def read_json(path="/home/inlab4/Documents/Dan_datasets/AF/april/"):
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


def public_sector_table(input_file="Employer2020_04_09.csv",
                        output_file="pb_off.csv"):
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
    public_df.loc[:'JE_org'] = public_df['JE_orgnr'].apply(
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
    print(f"write the {outputfile}")


if __name__ == "__main__":
    """
    #step 1: read in json files 
    files = read_json()
    file_dump = sorted(files, key=lambda i: i['date'])
    #if scraping date presented in descending order, use range(len(file_dump), 0)
    for item in file_dump:
        print("Scraping date: %s    [downloaded_number_of_ads:%d]" %
              (item['date'], item['number_ads']))
        print("-----------------------------------------")
    #step 2: run together with step 1 use file_dump to generate the result file
    aggregate_writeresult(file_name='Employer2020_04_09.csv')
    """
    #step3: read in the result from step2 and compare FDB -> public sector file for loading in DB
    #when run step3, can comment step2 and step 1
    #public_sector_table()
    df = pd.read_csv(
        r'/home/inlab4/Documents/Dan_datasets/AF_results/pb_off/pb_of_2020-04-07.csv'
    )
    print(df.head())
    quarter = df['Month'].apply(lambda x: get_quarter(x))
    print(quarter.tolist()[20:29])
    df['Quarter'].update = quarter
    print(quarter.value_counts())
    df.to_csv(
        '/home/inlab4/Documents/Dan_datasets/AF_results/pb_off/pb_of_2020-04-07.csv',
        index=False)
