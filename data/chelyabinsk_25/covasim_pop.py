import pandas as pd
import json
import os
import synthpops as sps
import numpy as np
import covasim as cv
from tqdm import tqdm

school = json.load(
    open(os.path.expanduser('schools.json')))
people = pd.read_csv('people.txt', sep='\t')
households = pd.read_csv('households.txt', sep='\t')

d = {}
for i in tqdm(range(101)):
    d[i]=sorted(list(people[people.age==i]['sp_id']))

id_age = {}
for i in d:
    for j in d[i]:
        id_age[j]=i

households = []
for i in tqdm(list(people.groupby('sp_hh_id').size().index)):
    households.append(list(people[people.sp_hh_id==i]['sp_id']))

schools = []
for i in tqdm(list(people[(people.age<18)&(people.work_id!='X')].groupby('work_id').size().index)):
    schools.append(list(people[(people.work_id==i)&(people.age<18)]['sp_id']))

works = []
for i in tqdm(list(people[(people.age>=18)&(people.work_id!='X')].groupby('work_id').size().index)):
    works.append(list(people[(people.work_id==i)&(people.age>=18)]['sp_id']))

teachers = []
for i in range(len(schools)):
    teachers.append([])#schools[i][0]])
    #del schools[i][0]

popdict = sps.contact_networks.make_contacts(
    sps.pop.Pop, 
    age_by_uid=id_age, 
    homes_by_uids=households, 
    students_by_uid_lists=schools,
    workplace_by_uid_lists=works,
    teachers_by_uid_lists=teachers,
    average_class_size=100000,
)

def myconverter(obj):
        if isinstance(obj, np.int64):
            return int(obj)
        if isinstance(obj, set):
            return list(obj)
        raise TypeError

with open("./pop_covasim.json", "w", encoding="utf-8") as file:
    json.dump(popdict, file, default=myconverter)
