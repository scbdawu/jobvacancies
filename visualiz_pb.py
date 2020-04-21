import pandas as pd
import glob
import datetime
from matplotlib import pyplot

#write in private sector data for region and ng visualization
total_private = pd.read_csv(
    '/home/inlab4/Documents/Dan_datasets/AF_results/pb_prv/pb_prv_table_april.csv'
)
#read in occupation and public sector data
#path = /home/inlab4/Documents/Dan_datasets/AF_results/occupation2_*.csv
public_sector = '/home/inlab4/Documents/Dan_datasets/AF_results/pb_off/pb_off[1-4].csv'
#privat_sector = '/home/inlab4/Documents/Dan_datasets/AF_results/pb_prv/pb_prv_talbe_april.csv'
total_public = pd.DataFrame()
for f in glob.glob(public_sector):
    print(f)
    month = pd.read_csv(f)
    total_public = pd.concat([total_public, month])
#print(total.shape)
total_private.drop_duplicates(subset=['ads_id'], inplace=True)
print(total_private.columns)
total_public.drop_duplicates(subset=['Platsnummer'], inplace=True)

total_private['week'] = total_private['publication_date'].apply(
    lambda x: datetime.datetime.strptime(str(x), '%Y-%m-%dT%H:%M:%S').strftime(
        '%W'))
total_private['week'] = total_private['week'].apply(lambda x: int(x) + 1)
total_public['week'] = total_public['Publiceringsdatum'].apply(
    lambda x: datetime.datetime.strptime(x, '%Y-%m-%dT%H:%M:%S').strftime('%W'
                                                                          ))
total_public['week'] = total_public['week'].apply(lambda x: int(x) + 1)
#print(total.week.value_counts())
print(total_private.number_of_vacancies.isna().sum())
total_private['number_of_vacancies'] = total_private['number_of_vacancies'].fillna(1)
total2020_private = total_private[total_private['year'] == 2020]
total2020_public = total_public[total_public['Year'] == 2020]
private = total2020_private.groupby(['week'])['number_of_vacancies'].sum()
public = total2020_public.groupby(['week'])['Antal_platser'].sum()
print(private)
print(public)
total = pd.concat([private, public], axis=1)
total.columns = ['lediga_platser_privat', 'lediga_platser_offentlig']
total['total'] = total['lediga_platser_privat'] + total['lediga_platser_offentlig']
print(total)
total_fig = total.plot(marker='x').get_figure()
total_fig.savefig('/home/inlab4/Documents/Dan_datasets/AF_results/graph/total_private_public.jpeg')
#print public sector plot
#public_fig = public.plot(marker='x').get_figure()
#public_fig.savefig(
#    '/home/inlab4/Documents/Dan_datasets/AF_results/graph/public.jpeg')

#private sector plot
#private_fig = private.plot(marker='x'), get_figure()
#print(total2020[total2020['week'] == '15'].publication_date.value_counts())
#weekgroup = total2020.groupby(['week'])['number_of_vacancies'].sum()
#print(weekgroup)
##week_fig = weekgroup.plot(marker='X').get_figure()
#week_fig.savefig('/home/inlab4/Documents/Dan_datasets/AF_results/graph/week_total.jpeg')
#occupation2 = total2020.groupby([
#    'week', 'occupation_field_label'
#])['number_of_vacancies'].sum().reset_index(name='sum')
#print(occupation2)
#new_table = occupation2.pivot(index='week',
#                              columns='occupation_field_label',
#                              values='sum')
#print(new_table)
#occupation2_fig_1 = new_table.iloc[:, 0:7].plot(marker='x').get_figure()
#occupation2_fig_2 = new_table.iloc[:, 7:14].plot(marker='x',
#                                                 legend=False).get_figure()
#occupation2_fig_2.legend(frameon=False, loc='upper center', ncol=3)
#occupation2_fig_3 = new_table.iloc[:, 14:21].plot(marker='x',
#                                                  legend=False).get_figure()
#occupation2_fig_3.legend(frameon=False, loc='upper center', ncol=3)
#occupation2_fig_1.savefig(
#    '/home/inlab4/Documents/Dan_datasets/AF_results/graph/occupation2_1_total.jpeg'
#)
#occupation2_fig_2.savefig(
#    '/home/inlab4/Documents/Dan_datasets/AF_results/graph/occupation2_2_total.jpeg'
#)
#occupation2_fig_3.savefig(
#    '/home/inlab4/Documents/Dan_datasets/AF_results/graph/occupation2_3_total.jpeg'
#)
#
#private region plot
#region_group = total2020.groupby(['week', 'nuts'])['number_of_vacancies'].sum().reset_index(name='sum')
#new_table = region_group.pivot(index='week', columns='nuts', values='sum')
#print(new_table)
#region_fig = new_table.plot(marker='x').get_figure()
#region_fig.savefig('/home/inlab4/Documents/Dan_datasets/AF_results/graph/region.jpeg')
#private ng plots
#ng_group = total2020.groupby(['week', 'ngs1'])['number_of_vacancies'].sum().reset_index(name='sum')
#new_table = ng_group.pivot(index='week', columns='ngs1', values='sum')
#print(new_table)
#ng_fig = new_table.plot(marker='x').get_figure()
#ng_fig.savefig('/home/inlab4/Documents/Dan_datasets/AF_results/graph/ng.jpeg')