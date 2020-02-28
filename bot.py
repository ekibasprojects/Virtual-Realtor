#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import scale

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

class RealtorBot():
    def __init__(self):
        self.started = False
        self.city = ''
        self.area = 0
        self.building_type = ''
        self.condition = ''
        self.level_count = 0
        self.year = 0
        self.district = ''
        
        self.cities = {
            'алматы': 'Алматы',
            '1': 'Алматы',
            'астана': 'Астана',
            '2': 'Астана',
            'актау': 'Актау',
            '3': 'Актау',
            'караганда': 'Караганда',
            '4': 'Караганда',
            'актобе': 'Актобе',
            '5': 'Актобе',
            'шымкент': 'Шымкент',
            '6': 'Шымкент',
            'усть-каменогорск': 'Усть-Каменогорск',
            '7': 'Усть-Каменогорск',
            'павлодар': 'Павлодар',
            '8': 'Павлодар',
            'тараз': 'Тараз',
            '9': 'Тараз',
            'семей': 'Семей',
            '10': 'Семей',
            'уральск': 'Уральск',
            '11': 'Уральск',
            'атырау': 'Атырау',
            '12': 'Атырау',
            'костанай': 'Костанай',
            '13': 'Костанай',
            'кокшетау': 'Кокшетау',
            '14': 'Кокшетау',
            'талдыкорган': 'Талдыкорган',
            '15': 'Талдыкорган',
            'кызылорда': 'Кызылорда',
            '16': 'Кызылорда',
            'петропавловск': 'Петропавловск',
            '17': 'Петропавловск',
            'темиртау': 'Темиртау',
            '18': 'Темиртау',
            'экибастуз': 'Экибастуз',
            '19': 'Экибастуз',
            'жезказган': 'Жезказган',
            '20': 'Жезказган',
            'рудный': 'Рудный',
            '21': 'Рудный',
            'жанаозен': 'Жанаозен',
            '22': 'Жанаозен',
            'капчагай': 'Капчагай',
            '23': 'Капчагай',
            'косшы': 'Косшы',
            '24': 'Косшы',
            'щучинск': 'Щучинск',
            '25': 'Щучинск',
            'талгар': 'Талгар',
            '26': 'Талгар',
            'боралдай': 'Боралдай',
            '27': 'Боралдай',
            'аксай': 'Аксай',
            '28': 'Аксай',
            'аксу': 'Аксу',
            '29': 'Аксу',
            'отеген батыр': 'Отеген Батыр',
            '30': 'Отеген Батыр',
            'сатпаев': 'Сатпаев',
            '31': 'Сатпаев',
            'каскелен': 'Каскелен',
            '32': 'Каскелен',
            'балхаш': 'Балхаш',
            '33': 'Балхаш'
        }

    def make_prediction(self, city, user_data):
        
        # reading in the data
        data = pd.read_csv("clean.csv")
        
        # cities with districts defined
        cities_d = data[data['district'].notnull()]['city'].value_counts().index
        if city in cities_d:
            cols_to_drop = ['price', 'city', 'room_count', 'building_year', 'mortgage', 'level', 'title', 'street']
        else:
            cols_to_drop = ['price', 'city', 'room_count', 'building_year', 'mortgage', 'level', 'district', 'title', 'street']
    
        # preparing data        
        X = data[data['city'] == city].drop(columns=cols_to_drop)
        y_train = data[data['city'] == city]['price']
            
        X = X.append(user_data)
        X = X.reset_index(drop=True)
            
        X = self.get_processed_data(X)
            
        # building model
        X_train = X.iloc[:-1]
        X_test = X.iloc[-1:]
        
        lm = LinearRegression()
        lm.fit(X_train, y_train)
        
        # getting prediction
        predicted_price = np.exp(lm.predict(X_test))[0]
        predicted_price = int(round(predicted_price, -3))
        
        return predicted_price
                
        
    def get_processed_data(self, df):
        # getting categorical columns
        categorical = df.select_dtypes(include=['object'])
        
        # transforming into dummies
        cat_dummies = pd.get_dummies(categorical, drop_first=True)
        
        # getting numerical columns
        df = df.drop(columns=categorical)
        
        # merging numerical with categorical dummies
        df_X = pd.merge(df, cat_dummies, on=df.index)
        df_X.drop(columns='key_0', inplace=True)
        df_X.index = df.index
        
        cols = df_X.columns
        X_scaled = pd.DataFrame(scale(df_X), index=df_X.index)
        X_scaled.columns = cols
        
        return X_scaled
    
    def get_df_from_user_input(self):
        if self.year >= 2000:
            building_new = 1
        else:
            building_new = 0
            
        user_data = {
            'area': [np.log(self.area)],
            'building_type': [self.building_type],
            'condition': [self.condition],
            'level_count': [self.level_count],
            'building_new': [building_new]
        }
        
        if(self.district != ''):
            user_data['district'] = self.district
        
        df = pd.DataFrame(data=user_data)
        return df
    
    def clear_input(self):
        self.started = False
        self.city = ''
        self.area = 0
        self.building_type = ''
        self.condition = ''
        self.level_count = 0
        self.year = 0
        self.district = ''
        
    def input_prompt(self):
        self.started = True
        if(self.city == ''):
            return self.city_prompt()
        elif(self.district == '' and self.city in ['Алматы', 'Астана', 'Караганда', 'Актобе', 'Шымкент']):
            return self.district_prompt()
        elif(self.area == 0):
            return self.area_prompt()
        elif(self.year == 0):
            return self.year_prompt()
        elif(self.building_type == ''):
            return self.building_type_prompt()
        elif(self.condition == ''):
            return self.condition_prompt()
        elif(self.level_count == 0):
            return self.level_count_prompt()
        else:
            user_df = self.get_df_from_user_input()
            price = self.make_prediction(self.city, user_df)
            self.clear_input()
            return "Примерная цена вашей квартиры: " + "{:,}".format(price) + " тг."
        
    def process_input(self, user_input):
        if(self.city == ''):
            return self.process_city_input(user_input)
        elif(self.district == '' and self.city in ['Алматы', 'Астана', 'Караганда', 'Актобе', 'Шымкент']):
            return self.process_district_input(user_input)
        elif(self.area == 0):
            return self.process_area_input(user_input)
        elif(self.year == 0):
            return self.process_year_input(user_input)
        elif(self.building_type == ''):
            return self.process_building_type_input(user_input)
        elif(self.condition == ''):
            return self.process_condition_input(user_input)
        elif(self.level_count == 0):
            return self.process_level_count_input(user_input)
        
    def city_prompt(self):
        return """*В каком городе ваша квартира?*
        
1. Алматы
2. Астана
3. Актау
4. Караганда
5. Актобе
6. Шымкент
7. Усть-Каменогорск
8. Павлодар
9. Тараз
10. Семей
11. Уральск
12. Атырау
13. Костанай
14. Кокшетау
15. Талдыкорган
16. Кызылорда
17. Петропавловск
18. Темиртау
19. Экибастуз
20. Жезказган
21. Рудный
22. Жанаозен
23. Капчагай
24. Косшы
25. Щучинск
26. Талгар
27. Боралдай
28. Аксай
29. Аксу
30. Отеген Батыр
31. Сатпаев
32. Каскелен
33. Балхаш

_Введите название или номер города из списка, или введите *0*, чтобы начать заново._"""
    
    def process_city_input(self, city):
        if(city in self.cities):
            self.city = self.cities[city]
            return "OK"
        else:
            return "К сожалению, я не знаю такой город.\n\n_Выберите город из списка, или введите *0*, чтобы начать заново._"
        
        
    def district_prompt(self):
        if self.city == 'Алматы':
            return """*В каком районе ваша квартира?*
        
1. Бостандыкский
2. Алмалинкский
3. Ауезовский
4. Медеуский
5. Жетысуский
6. Турксибский
7. Алатауский
8. Наурызбайский

_Введите номер района из списка, или введите *0*, чтобы начать заново._"""
        elif self.city == 'Астана':
            return """*В каком районе ваша квартира?*
        
1. Есильский
2. Алматинский
3. Сарыаркинский
4. Байконур

_Введите номер района из списка, или введите *0*, чтобы начать заново._"""
        elif self.city == 'Караганда':
            return """*В каком районе ваша квартира?*
        
1. Казыбек Би
3. Октябрьский

_Введите номер района из списка, или введите *0*, чтобы начать заново._"""
        elif self.city == 'Актобе':
            return """*В каком районе ваша квартира?*
        
1. Новый город
2. Старый город
3. Мкр. 12
4. Мкр. 8
5. Мкр. 5
6. Мкр. 11
7. Нур Актобе

_Введите номер района из списка, или введите *0*, чтобы начать заново._"""
        elif self.city == 'Шымкент':
            return """*В каком районе ваша квартира?*
        
1. Аль-Фарабийский
2. Енбекшинский
3. Абайский
4. Каратауский

_Введите номер района из списка, или введите *0*, чтобы начать заново._"""

    def process_district_input(self, district_input):
        if self.city == 'Алматы':
            districts = ['Бостандыкский', 'Алмалинкский', 'Ауезовский', 'Медеуский', 'Жетысуский', 'Турксибский', 'Алатауский', 'Наурызбайский']
        elif self.city == 'Астана':
            districts = ['Есильский', 'Алматинский', 'Сарыаркинский', 'Байконур']
        elif self.city == 'Караганда':
            districts = ['Казыбек Би', 'Октябрьский']
        elif self.city == 'Актобе':
            districts = ['Новый город', 'Старый город', 'Мкр. 12', 'Мкр. 8', 'Мкр. 5', 'Мкр. 11', 'Нур Актобе']
        elif self.city == 'Шымкент':
            districts = ['Аль-Фарабийский', 'Енбекшинский', 'Абайский', 'Каратауский']
            
        if(district_input.isnumeric()):
            district_input = int(district_input)
            if(district_input < 1 or district_input > len(districts)):
                return "Неверный ввод.\n\n_Повторите попытку, или введите *0*, чтобы начать заново._"
            else:
                self.district = districts[district_input-1]
                return "OK"
        else:
            return "Не могу понять ваш ответ. \n\n_Введите числовое значение, или введите *0*, чтобы начать заново._"
            
    def area_prompt(self):
        return "*Какая площадь вашей квартиры (кв.м.)?*\n\n_Введите числовое значение, или введите *0*, чтобы начать заново._"
    
    def process_area_input(self, area):
        if(area.isnumeric()):
            area = int(area)
            if(area < 0):
                return "Площадь не может быть отрицательной.\n\n_Повторите попытку, или введите *0*, чтобы начать заново._"
            else:
                if(area <= 10):
                    return "Подозрительно маленькая квартира.\n\n_Повторите попытку, или введите *0*, чтобы начать заново._"
                elif(area >= 500):
                    return "Подозрительно большая квартира.\n\n_Повторите попытку, или введите *0*, чтобы начать заново._"
                else:
                    self.area = area
                    return "OK"
        else:
            return "Не могу понять ваш ответ. Сколько квадратов в вашей квартире?\n\n_Введите числовое значение, или введите *0*, чтобы начать заново._"
            
    def building_type_prompt(self):
        return """*Какой у вас дом?*
    
1. кирпичный
2. панельный
3. монолитный
4. каркасно-камышитовый
5. иное

_Введите номер ответа из списка, или введите *0*, чтобы начать заново._"""

    def process_building_type_input(self, type_input):
        if(type_input.isnumeric()):
            type_input = int(type_input)
            if(type_input < 1 or type_input > 5):
                return "Неверный ввод.\n\n_Повторите попытку, или введите *0*, чтобы начать заново._"
            else:
                types = ['кирпичный', 'панельный', 'монолитный', 'каркасно-камышитовый', 'иное']
                self.building_type = types[type_input-1]
                return "OK"
        else:
            return "Не могу понять ваш ответ. \n\n_Введите числовое значение, или введите *0*, чтобы начать заново._"
    
    def condition_prompt(self):
        return """*В каком состоянии ваша квартира?*
    
1. хорошее
2. евроремонт
3. среднее
4. черновая отделка
5. требует ремонта
6. свободная планировка

_Введите номер ответа из списка, или введите *0*, чтобы начать заново._"""

    def process_condition_input(self, condition_input):
        if(condition_input.isnumeric()):
            condition_input = int(condition_input)
            if(condition_input < 1 or condition_input > 6):
                return "Неверный ввод.\n\n_Повторите попытку, или введите *0*, чтобы начать заново._"
            else:
                conditions = ['хорошее', 'евроремонт', 'среднее', 'черновая отделка', 'требует ремонта', 'свободная планировка']
                self.condition = conditions[condition_input-1]
                return "OK"
        else:
            return "Не могу понять ваш ответ. \n\n_Введите числовое значение, или введите *0*, чтобы начать заново._"
        
    def year_prompt(self):
        return """*В каком году построили/построят ваш дом?*\n\n_Введите числовое значение, или введите *0*, чтобы начать заново._"""
    
    def process_year_input(self, year):
        if(year.isnumeric()):
            year = int(year)
            if(year < 1900):
                return "Нет данных по домам старее 1900-го года.\n\n_Введите значение от 1900 до 2022, или введите *0*, чтобы начать заново._"
            elif(year > 2022):
                return "Нет данных по домам от 2023-го года и новее.\n\n_Введите значение от 1900 до 2022, или введите *0*, чтобы начать заново._"
            else:
                self.year = year
                return "OK"
        else:
            return "Не могу понять ваш ответ.\n\n_Введите числовое значение от 1900 до 2022, или введите *0*, чтобы начать заново._"
           
        
    def level_count_prompt(self):
        return """*Сколько этажей в вашем доме?*\n\n_Введите числовое значение, или введите *0*, чтобы начать заново._"""
    
    def process_level_count_input(self, level_count):
        if(level_count.isnumeric()):
            level_count = int(level_count)
            if(level_count < 0):
                return "Этажность не может быть отрицательной.\n\n_Повторите попытку, или введите *0*, чтобы начать заново._"
            else:
                if(level_count > 100):
                    return "Подозрительно высокий дом.\n\n_Повторите попытку, или введите *0*, чтобы начать заново._"
                else:
                    self.level_count = level_count
                    return "END"
        else:
            return "Не могу понять ваш ответ. Сколько этажей в вашем доме?\n\n_Введите числовое значение, или введите *0*, чтобы начать заново._"
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        