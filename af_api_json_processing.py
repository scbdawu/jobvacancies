import pandas as pd
import json
import numpy as np
import datetime
from multiprocessing import Pool


def print_keys(data, ifprint=False):
    #data is a list of dictionaries, some key contain another dictionary
    #i is the index of the dictionary
    simple_keys = []
    complex_keys = []
    for k in (data[0].keys()):
        if isinstance(data[0][k], dict):
            complex_keys.append(k)
            if ifprint:
                print("---%s" % "---".join(data[0][k].keys()))
        else:
            simple_keys.append(k)
            if ifprint:
                print(k)

    result = {'simple_keys': simple_keys, "complex_keys": complex_keys}
    return result


def get_keyvalue(data, index, keyname):
    #return the value of the keyname at index in data
    #return type is a string
    if isinstance(data[index][keyname], dict):
        return None
    else:
        return data[index][keyname]


def get_keyvalues(data, keyname):
    #return a list values in data of the keyname
    #only the simple keys
    result = []
    for item in data:
        try:
            if item.get(keyname) != None:
                result.append(item[keyname])
            else:
                result.append(np.nan)
        except IndexError as error:
            #the key is not available
            result.append(np.nan)

    return result


def get_commonstructure_type(data, ckeyname):
    #with the given compley keyname
    #return values in df, with each subkey as a column
    #some complex keys share the same structure, i.e. the same subkeys
    #create lists of subkeys
    concept_id = []
    label = []
    legacy_ams_taxonomy_id = []
    annons_id = get_keyvalues(
        data, 'id')  #use this as key for matching back to other keys

    for i in range(len(data)):
        #access this complex key as dictionary???
        if data[i] is None:
            legacy_ams_taxonomy_id.append(np.nan)
            label.append(np.nan)
            concept_id.append(np.nan)
        else:
            node = data[i].get(ckeyname)
            if node != None:
                if node.get('concept_id') != None:
                    concept_id.append(node.get('concept_id'))
                else:
                    concept_id.append(np.nan)
                if node.get('label') != None:
                    label.append(node.get('label'))
                else:
                    label.append(np.nan)
                if node.get('legacy_ams_taxonomy_id') != None:
                    legacy_ams_taxonomy_id.append(
                        node.get('legacy_ams_taxonomy_id'))
                else:
                    legacy_ams_taxonomy_id.append(np.nan)

            else:
                legacy_ams_taxonomy_id.append(np.nan)
                label.append(np.nan)
                concept_id.append(np.nan)

    result = pd.DataFrame({
        '%s_concept_id' % ckeyname: concept_id,
        '%s_label' % ckeyname: label,
        '%s_legacy_ams_taxonomy_id' % ckeyname: legacy_ams_taxonomy_id,
        'ads_id': annons_id
    })
    return result


def get_ads_description(data, ckeyname='description'):
    #with the given compley keyname
    #return values in df, with each subkey as a column
    #create lists of subkeys
    text = []
    company_info = []
    needs = []
    requirements = []
    conditions = []
    annons_id = get_keyvalues(data,
                              'id')  #use ad key to match back to other keys

    #get subkeys
    subkeys = data[0].get(ckeyname).keys()
    for i in range(len(data)):
        if data[i] is None:
            text.append(np.nan)
            company_info.append(np.nan)
            requirements.append(np.nan)
            needs.append(np.nan)
            conditions.append(np.nan)
        else:
            description = data[i].get(ckeyname)
            if description != None:

                if description.get('text') != None:
                    text.append(description.get('text'))
                else:
                    text.append(np.nan)
                if description.get('company_info') != None:
                    company_info.append(description.get('company_info'))
                else:
                    company_info.append(np.nan)
                if description.get('needs') != None:
                    needs.append(description.get('needs'))
                else:
                    needs.append(np.nan)
                if description.get('requirements') != None:
                    requirements.append(description.get('requirements'))
                else:
                    requirements.append(np.nan)
                if description.get('conditions') != None:
                    conditions.append(description.get('conditions'))
                else:
                    conditions.append(np.nan)
            else:
                text.append(np.nan)
                company_info.append(np.nan)
                requirements.append(np.nan)
                needs.append(np.nan)
                conditions.append(np.nan)

    result = pd.DataFrame({
        'description_text': text,
        'description_company_info': company_info,
        'description_needs': needs,
        'description_requirements': requirements,
        'description_conditions': conditions,
        'ads_id': annons_id
    })
    return result


def get_employer_values(data, ckeyname='employer'):
    #with the given compley keyname
    #return values in df, with each subkey as a column
    #create lists of subkeys
    #create lists of subkeys
    phone = []
    email = []
    url = []
    orgnr = []
    name = []
    workplace = []  #similar to company name
    annons_id = get_keyvalues(data,
                              'id')  #use ad key to match back to other keys

    #get subkeys
    subkeys = data[0].get(ckeyname).keys()
    for i in range(len(data)):
        if data[i] is None:
            phone.append(np.nan)
            email.append(np.nan)
            url.append(np.nan)
            orgnr.append(np.nan)
            name.append(np.nan)
            workplace.append(np.nan)
        else:
            employer = data[i].get(ckeyname)
            if employer != None:

                if employer.get('phone_number') != None:
                    phone.append(employer.get('phone_number'))
                else:
                    phone.append(np.nan)
                if employer.get('email') != None:
                    email.append(employer.get('email'))
                else:
                    email.append(np.nan)
                if employer.get('url') != None:
                    url.append(employer.get('url'))
                else:
                    url.append(np.nan)
                if employer.get('organization_number') != None:
                    orgnr.append(employer.get('organization_number'))
                else:
                    orgnr.append(np.nan)
                if employer.get('name') != None:
                    name.append(employer.get('name'))
                else:
                    name.append(np.nan)
                if employer.get('workplace') != None:
                    workplace.append(employer.get('workplace'))
                else:
                    workplace.append(np.nan)
            else:
                phone.append(np.nan)
                email.append(np.nan)
                url.append(np.nan)
                orgnr.append(np.nan)
                name.append(np.nan)
                workplace.append(np.nan)
    result = pd.DataFrame({
        'employer_phone_number': phone,
        'employer_email': email,
        'employer_url': url,
        'employer_organization_number': orgnr,
        'employer_name': name,
        'employer_workplace': workplace,
        'ads_id': annons_id
    })
    return result


def get_work_addresses(jsondata, ckeyname="workplace_address"):
    #jsondata is a list of dictionaries

    municipality_code = []
    municipality = []
    region_code = []
    region = []
    country_code = []
    country = []
    street_address = []
    postcode = []
    city = []
    coordinates = []
    annons_id = get_keyvalues(jsondata, 'id')

    subkeys = jsondata[0].get(ckeyname).keys()
    for i in range(len(jsondata)):
        if jsondata[i] is None:
            municipality_code.append(np.nan)
            municipality.append(np.nan)
            region_code.append(np.nan)
            region.append(np.nan)
            country_code.append(np.nan)
            country.append(np.nan)
            street_address.append(np.nan)
            postcode.append(np.nan)
            city.append(np.nan)
            coordinates.append(np.nan)
        else:
            address = jsondata[i].get(ckeyname)
            if address != None:
                subkey_values = []
                for j in subkeys:
                    if address.get(j) != None:
                        subkey_values.append(address.get(j))
                    else:
                        subkey_values.append(np.nan)

                municipality_code.append(subkey_values[0])
                municipality.append(subkey_values[1])
                region_code.append(subkey_values[2])
                region.append(subkey_values[3])
                country_code.append(subkey_values[4])
                country.append(subkey_values[5])
                street_address.append(subkey_values[6])
                postcode.append(subkey_values[7])
                city.append(subkey_values[8])
                coordinates.append(subkey_values[9])
            else:
                municipality_code.append(np.nan)
                municipality.append(np.nan)
                region_code.append(np.nan)
                region.append(np.nan)
                country_code.append(np.nan)
                country.append(np.nan)
                street_address.append(np.nan)
                postcode.append(np.nan)
                city.append(np.nan)
                coordinates.append(np.nan)
    result = pd.DataFrame({
        'address_municipality_code': municipality_code,
        'address_municipality': municipality,
        'address_region_code': region_code,
        'address_region': region,
        'address_country_code': country_code,
        'address_country': country,
        'address_street_address': street_address,
        'address_postcode': postcode,
        'address_city': city,
        'address_coordinates': coordinates,
        'ads_id': annons_id
    })
    return result


def divide_json(json):
    #remove items with removed==True the json and return the cleaned one
    #re
    loopnr = len(json) - 1
    result = []
    for i in range(loopnr):
        if len(json[i].keys()) != 3:
            result.append(json[i])
    return result


def convert_json2df(jsondata):
    #api changed json data structure
    simple_keys = [
        'id', 'external_id', 'webpage_url', 'logo_url', 'headline',
        'application_deadline', 'number_of_vacancies', 'salary_description',
        'access', 'experience_required', 'access_to_own_car',
        'driving_license_required', 'driving_license', 'publication_date',
        'last_publication_date', 'removed', 'removed_date', 'source_type',
        'timestamp'
    ]
    #keys in the dataframe
    annons_id_removed = []
    annons_id = []
    external_id = []
    webpage_url = []
    logo_url = []
    headline = []
    application_deadline = []
    number_of_vacancies = []
    salary_description = []
    access = []
    experience_required = []
    access_to_own_car = []
    driving_license_required = []
    driving_license = []
    publication_date = []
    last_publication_date = []
    removed = []
    removed_date = []
    removed_removed = []
    removed_date_removed = []
    source_type = []
    timestamp = []

    annons_id = get_keyvalues(jsondata, simple_keys[0])
    #print(simple_keys[1])
    external_id = get_keyvalues(jsondata, simple_keys[1])
    #print(simple_keys[2])
    webpage_url = get_keyvalues(jsondata, simple_keys[2])
    #print(simple_keys[3])
    logo_url = get_keyvalues(jsondata, simple_keys[3])
    #print(simple_keys[4])
    headline = get_keyvalues(jsondata, simple_keys[4])
    #print(simple_keys[5])
    application_deadline = get_keyvalues(jsondata, simple_keys[5])
    #print(simple_keys[6])
    number_of_vacancies = get_keyvalues(jsondata, simple_keys[6])
    #print(simple_keys[7])
    salary_description = get_keyvalues(jsondata, simple_keys[7])
    #print(simple_keys[8])
    access = get_keyvalues(jsondata, simple_keys[8])
    #print(simple_keys[9])
    experience_required = get_keyvalues(jsondata, simple_keys[9])
    #print(print(simple_keys[10]))
    access_to_own_car = get_keyvalues(jsondata, simple_keys[10])
    #print(simple_keys[11])
    driving_license_required = get_keyvalues(jsondata, simple_keys[11])
    #print(simple_keys[12])
    driving_license = get_keyvalues(jsondata, simple_keys[12])
    #print(simple_keys[13])
    publication_date = get_keyvalues(jsondata, simple_keys[13])
    #print(simple_keys[14])
    last_publication_date = get_keyvalues(jsondata, simple_keys[14])
    #print(simple_keys[15])
    removed = get_keyvalues(jsondata, simple_keys[15])
    #print(simple_keys[16])
    removed_date = get_keyvalues(jsondata, simple_keys[16])
    #print(simple_keys[17])
    source_type = get_keyvalues(jsondata, simple_keys[17])
    #print(simple_keys[18])
    timestamp = get_keyvalues(jsondata, simple_keys[18])
    #convert data into df with a flat structure

    df = pd.DataFrame()
    df['ads_id'] = annons_id  #0
    df['external_id'] = external_id  #1
    df['webpage_url'] = webpage_url  #2
    df['logo_url'] = logo_url  #3
    df['headline'] = headline  #4
    df['application_deadline'] = application_deadline  #5
    df['number_of_vacancies'] = number_of_vacancies  #6
    df['number_of_vacancies'].astype(int, errors='ignore')
    df['salary_description'] = salary_description  #7
    df['access'] = access  #8
    df['experience_required'] = experience_required  #9
    df['access_to_own_car'] = access_to_own_car  #10
    df['driving_license_required'] = driving_license_required  #11
    df['driving_license_required'].astype(np.bool)
    df['driving_license'] = driving_license  #12
    df['publication_date'] = publication_date  #13
    df['last_publication_date'] = last_publication_date  #14
    df['removed'] = removed  #15
    df['removed'].astype(np.bool)
    df['removed_date'] = removed_date  #16
    df['source_type'] = source_type  #17
    df['timestamp'] = timestamp  #18
    return df


def str2datetime(s, format="%Y-%m-%dT%H:%M:%S"):
    return datetime.datetime.strptime(s, format)


def add_dates2df(df):
    #add year, month, week and day on according to publication date
    #handle NaN in publication date
    df['year'] = df['publication_date'].apply(
        lambda x: np.nan if isinstance(x, pd._libs.tslibs.nattype.NaTType) else
        (str2datetime(x).strftime("%Y")))  #4digits year
    df['month'] = df['publication_date'].apply(
        lambda x: np.nan if isinstance(x, pd._libs.tslibs.nattype.NaTType) else
        (str2datetime(x).strftime("%m")))

    return df


def parallelize_dataframe(df, func, n_cores=8):
    df_split = np.array_split(df, n_cores)
    pool = Pool(n_cores)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df


def read_one_day_dump(json_dump):
    #read one days data and add dates to the dataframe
    #return the df
    df = convert_json2df(json_dump)
    df.drop_duplicates(subset=['ads_id'], inplace=True)
    df['removed'] = df['removed'].astype(bool)
    df = df.loc[df.removed == False]
    #add dates to the df
    df = parallelize_dataframe(df, add_dates2df)
    return df


def add_more_data_dump(df1, df2):
    #combine two days data_dumps and remove ads with "True"
    #then drop duplicates
    totaldf = pd.concat([df1, df2])
    totaldf = totaldf.loc[totaldf['removed'] == False]
    totaldf.drop_duplicates(subset='ads_id', inplace=True)
    return totaldf


if __name__ == "__main__":
    pass