###############################################################################################################################################
############################################## Импортируем необходимые модули и данные ########################################################
###############################################################################################################################################

# Для работы с операционной сисемой 
import os

# Для работы с SQL
from sqlalchemy import create_engine, Table, MetaData
from sqlalchemy.inspection import inspect
from sqlalchemy.dialects.postgresql import insert as pg_insert

# Для работы с табличными данными 
import pandas as pd

# Для работы с регулярными выражениями
import re 

# Для работы с датой-временем
import datetime
import pytz

###############################################################################################################################################
############################################## Создаем объект класса ##########################################################################
###############################################################################################################################################

class SortMedia(): 
    def __init__(
        self
        ,pg_password
        ,pg_login
        ,pg_host
        ,type_file_tuple = ('.jpg', '.png', '.mp4')
        ,path_list = ['/camera_vvy', '/camera_angel']
        ,schema = 'sort_media'
        ,table_name = 'path'
    ):
        """
        Сортировка медиафайлов с найденным лицом искомого человека по папкам. 
        Вход: 
            pg_password(str) - пароль к DWH.
            pg_login(str) - логин к DWH. 
            pg_host(str) - хост DWH.
            type_file_tuple(tuple) - кортеж с типами медиафайлов.
            path_list(list) -  список путей к файлам.
            schema(str) - наименование схемы DWH.
            table_name - наименование таблицы DWH. 
            type_file(list) - список типов файлов.
            path_list(list) - список путей к файлам. 
        """
        self.type_file_tuple = type_file_tuple
        self.path_list = path_list
        self.engine = create_engine(f'postgres://{pg_login}:{pg_password}@{pg_host}:5432/{pg_login}')
        self.schema = schema
        self.table_name = table_name
        
    def pg_descriptions(self): 
        """
        Функция для возвращения таблицы с описанием таблиц в pg. 
        Вход: 
            schema(str) - наименование схемы.
        Выход: 
            desription_df(DataFrame) - таблица с описанием таблиц pg.
        """
        schema = f"'{self.schema}'"     
        query_table_description = f"""
        with 
             tables as (
             select distinct
                    table_name
                    ,table_schema
            from 
                    information_schema.columns 
            where 
        -- Отбираем схему
                    table_schema in ({schema})
             )
        select 
                nspname as scheme_name, 
                obj_description(n.oid) as scheme_description,
                relname as table_name, 
                attname as column_name, 
                format_type(atttypid, atttypmod), 
                obj_description(c.oid) as table_description, 
                col_description(c.oid, a.attnum) as column_description 
        from 
                pg_class as c 
        join pg_attribute as a on (a.attrelid = c.oid) 
        join pg_namespace as n on (n.oid = c.relnamespace)
        join tables on tables.table_name = c.relname and tables.table_schema = n.nspname
        where
                format_type(atttypid, atttypmod) not in ('oid', 'cid', 'xid', 'tid', '-')
        """

        desription_df = pd.read_sql(
            sql = query_table_description
            ,con = self.engine
        )
        return desription_df

    def input_files(self):
        """
        Получение путей к медиафайлам для обработки. 
        Вход: 
            path_list(list) - лист путей к папкам, в которых необходимо получить список файлов. 
        Выход: 
            result(list) - список путей к файлам. 
        """
        files_list = []
        for path in self.path_list: 
            for dirpath, subdirs, files in os.walk(path):
                files_list.extend(os.path.join(dirpath, x) for x in files if x.endswith(self.type_file_tuple))
        result = files_list if files_list else None        
        return result
    
    def metadata_file(self, path_list): 
        """
        Получение метаданных файла. 
        Вход: 
            path(str) - путь к файлу.
        Выход: 
            metadata_file(list) - метаданные файла.
        """
        metadata_file = [] 
        
        # Получаем тип файла
        type_file_list = [type_file + '$' for type_file in self.type_file_tuple]        

        # Получаем дату загрузки 
        date_load = datetime.datetime.now(pytz.timezone('Asia/Vladivostok')).strftime("%Y-%m-%d %H:%M:%S")

        # Получаем полное и короткое наименование файла 
        for path in path_list:
            type_file = re.findall('|'.join(type_file_list), path)
            type_file = type_file[0] if type_file else None
            path_split = path.split('/')
            full_name = path_split[len(path_split) - 1]
            short_name = re.sub('|'.join(type_file_list), '', full_name)        
            metadata_file.append(
                {
                    'path' : path
                    ,'file' : full_name
                    ,'file_name' : short_name
                    ,'type' : type_file
                    ,'is_processed' : False
                    ,'date_load' : date_load
                }
            )
        
        return metadata_file
    
    def write_new_path(self): 
        """
        Запись новых путей файлов в таблицу.
        Вход: 
            нет.
        Выход: 
            нет.
        """
        
        # Получаем пути файлов
        input_files_list = self.input_files()
        
        # Подготавливаем пути файлов для записи
        records = self.metadata_file(input_files_list)
        
        # Создаем объекты для записи
        metadata = MetaData(schema = self.schema)
        metadata.bind = self.engine
        table = Table(
            self.table_name
            ,metadata
            ,schema = self.schema
            ,autoload = True
        )
        primary_keys = [key.name for key in inspect(table).primary_key]
        
        # Производим запись данных
        stmt = pg_insert(table).values(records)

        update_dict = {
            c.name: c
            for c in stmt.excluded
            if not c.primary_key
        }

        if update_dict == {}:
            insert_ignore(self.table_name, records)

        update_stmt = stmt.on_conflict_do_update(
            index_elements = primary_keys,
            set_ = update_dict,
        )

        with self.engine.connect() as conn:
            result = conn.execute(update_stmt)