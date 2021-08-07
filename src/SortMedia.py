###############################################################################################################################################
############################################## Импортируем необходимые модули и данные ########################################################
###############################################################################################################################################

# Для работы с операционной сисемой 
import os
import shutil

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

# Для Deep Learning
import torch
from torch.utils.data import DataLoader
from torchvision import datasets

# Для работы с расстояниями
from scipy.spatial.distance import cosine

# Для работы с распознаванием лиц
from facenet_pytorch import MTCNN, extract_face, InceptionResnetV1

# Для сериализации и десериализации объектов Python
import pickle

# Для мониторинга выполнения циклов
from tqdm.notebook import tqdm as tqdm_notebook

# Для работы с картинками
from PIL import Image, ImageDraw, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
# Для компьютерного зрения 
import cv2 as cv, mmcv
# Для заполнения exif
from exif import Image as ImageEXIF

###############################################################################################################################################
############################################## Создаем объект класса ##########################################################################
###############################################################################################################################################

class SortMedia(): 
    def __init__(
        self
        ,pg_password
        ,pg_login
        ,pg_host
        ,device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        ,type_file_dict = {'image' : ['.jpg', '.jpeg', '.png'], 'video' : ['.mp4']}
        ,path_input_list = ['/camera_vvy', '/camera_angel']
        ,path_training_list = ['/ml_models/sort_media/veronica/']
        ,path_output_list = '/veronica'
        ,schema = 'sort_media'
        ,table_name = 'path'
        ,MTCNN = MTCNN   
        ,path = {
            'avg_embedding_human' : '/ml_models/sort_media/avg_embedding_human.bin'
            ,'model' : '/ml_models/sort_media/binary_model.bin'
        }
    ):
        """
        Сортировка медиафайлов с найденным лицом искомого человека по папкам. 
        Вход: 
            pg_password(str) - пароль к DWH.
            pg_login(str) - логин к DWH. 
            pg_host(str) - хост DWH.
            device(torch.device) - устройство для работы модели.
            type_file_dict(dict) - словарь с типами медиафайлов.
            path_input_list(list) - список путей к файлам для поиска человека.
            path_output_list(str) - путь к директории для перемещения файлов.
            path_training_list(list) - список путей к файлам искомого человека.
            schema(str) - наименование схемы DWH.
            table_name - наименование таблицы DWH. 
            MTCNN(Model.Model) - модель для распознавания лиц.
            
        """
        self.type_file_dict = type_file_dict
        self.path_input_list = path_input_list
        self.path_training_list = path_training_list
        self.path_output_list = path_output_list
        self.engine = create_engine(f'postgres://{pg_login}:{pg_password}@{pg_host}:5432/{pg_login}')
        self.schema = schema
        self.table_name = table_name
        self.device = device
        self.path = path
        self.metadata = MetaData(schema = self.schema)
        self.metadata.bind = self.engine
        self.table = Table(
            self.table_name
            ,self.metadata
            ,schema = self.schema
            ,autoload = True
        )
        self.primary_keys = [key.name for key in inspect(self.table).primary_key]
        
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

    def input_files(self, path_list = None):
        """
        Получение путей к медиафайлам для обработки. 
        Вход: 
            path_list(list) - лист путей к папкам, в которых необходимо получить список файлов. 
        Выход: 
            result(list) - список путей к файлам. 
        """
        path_list = path_list if path_list else self.path_input_list 
        files_list = []
        type_file_tuple = tuple(i for j in self.type_file_dict.values() for i in j)
        for path in path_list: 
            for dirpath, subdirs, files in os.walk(path):
                files_list.extend(os.path.join(dirpath, x) for x in files if x.lower().endswith(type_file_tuple))
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
        type_file_tuple = tuple(i for j in self.type_file_dict.values() for i in j)
        type_file_list = [type_file + '$' for type_file in type_file_tuple]        

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
        
# Производим запись данных
        stmt = pg_insert(self.table).values(records)

        update_dict = {
            c.name: c
            for c in stmt.excluded
            if not c.primary_key and c.name not in ['date_load', 'is_processed']
        }
# Обновляем поля, если строка существует 
        update_stmt = stmt.on_conflict_do_update(
            index_elements = self.primary_keys,
            set_ = update_dict
        )

        with self.engine.connect() as conn:
            result = conn.execute(update_stmt)
            
    def path_df(self): 
        """
        Новые файлы на обработку.
        Вход: 
            нет.
        Выход: 
            result_df(DataFrame) - таблица с путями к новым файлам.
        """
        
        query = """
        select
                "path" 
                ,file
                ,file_name
                ,type
                ,is_processed
                ,date_load
        from 
                sort_media."path"	
        where 
        -- Отбираем файлы, котолрые не обработаны
                is_processed  != True
        order by 
                date_load desc
        """
        
        result_df = pd.read_sql(
            sql = query
            ,con = self.engine        
        )
        return result_df
    
    def write_model_embedding(self):             
        """
        Сохранение модели для распознавания лиц в бинарный файл. 
        Вход:
            path(str) - путь для сохранения файла.
        Выход: 
            нет. 
        """        

        resnet = InceptionResnetV1(pretrained = 'vggface2')
        with open(self.path['model'], "wb") as f:
            pickle.dump(resnet, f)            
            
    def load_model_embedding(self): 
        """
        Загрузка модели из бинарного файла для описания ключевых точек.
        Вход: 
            path(str) - путь файлу модели.
        Выход: 
            model(Model.Model) - модель в режиме оценки, и перенесенная на видеокарту для вывода эмбединга лица. 
        """
        with open(f"{self.path['model']}", "rb") as f:
            model = pickle.load(f)
        return model.eval().to(self.device)
    
    def load_model_face_detect(self): 
        """
        Загрузка модели для поиска лица.
        Вход: 
            нет.
        Выход: 
            mtcnn(Model.Model) - модель, перенесенная на видеокарту. 
        """
# Создаем объект для распознавания лиц
        mtcnn = MTCNN(
            image_size = 160
            ,margin = 0
            ,min_face_size = 20
            ,thresholds = [0.6, 0.7, 0.7]
            ,factor = 0.709
            ,post_process = True
            ,device = self.device
            ,keep_all = True
        )
        return mtcnn
    
    def avg_embedding_human(self):
        """
        Возвращаем эбединг для искомого человека. 
        Вход:
            нет.
        Выход: 
            (torch.Tensor) - усредненный тензор лица искомого человека. 
        """
# Загружаем модель для получения эмбединга 
        resnet = self.load_model_embedding()
# Получаем пути к тренировочному набору 
        training_path_list = self.input_files(path_list = self.path_training_list)
    
# Получаем вектора лиц всех
        all_aligned = []
        for path in tqdm_notebook(training_path_list): 
            img = Image.open(path)
# Переводим в RGB
            if img.mode != 'RGB':
                img = img.convert('RGB')
            aligned = self.mtcnn(img)
            boxes, probs = self.mtcnn.detect(img)
            if boxes is not None:
                for x_aligned, prob, box in zip(aligned, probs, boxes):
                    if prob > 0.99:
        #                 print(f'Вероятность определения лица: {prob}')
                        all_aligned.append(x_aligned)
        #                 display(img.crop(box.tolist()))

# Получаем эмбединг лиц всех
        all_aligned = torch.stack(all_aligned).to(self.device)
        embeddings_all = resnet(all_aligned).detach().cpu()
        return embeddings_all.mean(axis = 0)
    
    def write_avg_embedding_human(self):             
        """
        Сохранение усредненного эмбединга лица искомого человека. 
        Вход:
            path(str) - путь для сохранения файла.
        Выход: 
            нет. 
        """        
        avg_embedding_human = self.avg_embedding_human()
        with open(self.path['avg_embedding_human'], "wb") as f:
            pickle.dump(avg_embedding_human, f)      
    
    def load_avg_embedding_human(self): 
        """
        Загрузка усредненного вектора лица искомого человека.
        Вход: 
            path(str) - путь файлу модели.
        Выход: 
            avg_embedding_human(torch.Tensor) - усредненный вектор лица искомого человека. 
        """
        with open(f"{self.path['avg_embedding_human']}", "rb") as f:
            avg_embedding_human = pickle.load(f)
        return avg_embedding_human
        
    def open_image(self, path, coefficient = 3): 
        """
        Определение лица искомого человека.
        Вход: 
            path(str) - путь к файлу
            coefficient(int) - коэффициент уменьшение фото. 
        Выход: 
            img_resize(bool) - булевого значение наличия лица искомого человека.
        """

    # Открываем фото и изменяем размер       
        img = Image.open(path)
        size = [int(i / coefficient) for i in img.size]
        img_resize = img.resize(tuple(size))
    # Переводим в RGB
        if img_resize.mode != 'RGB':
            img_resize = img_resize.convert('RGB')
        print(
            '\nПуть:', path, 
            '\nРазмер:', size
        )
        return img_resize
    
    def is_human_image_detected(self, path, mtcnn, resnet, avg_embedding_human, coef_decrease = 3, distance = 0.4, probability = 0.8, frame = None): 
        """
        Определение лица искомого человека.
        Вход: 
            path(str) - путь к файлу.
            mtcnn(Model.Model) - модель для распознавания лиц. 
            resnet(Model.Model) - модель в режиме оценки, и перенесенная на видеокарту для вывода эмбединга лица. 
            avg_embedding_human(torch.Tensor) - усредненный вектор лица искомого человека. 
            coef_decrease(int) - коэффициент уменьшение фото. 
            distance(float) - максимальное расстояние объектов.
            probability(float) - вероятность обнаружения лица.
            frame(PIL.Image.Image) - фото. 
        Выход: 
            (bool) - булевого значение наличия лица искомого человека.
        """

# Открываем фото и изменяем размер 
        if path: 
            img = Image.open(path)
        else: 
            img = frame
        size = [int(i / coef_decrease) for i in img.size]
# Поворачиваем картинку несколько ращз, если длина больше высоты
        if size[0] > size[1]: 
            img_resize_list = [img.resize(tuple(size)).rotate(270), img.resize(tuple(size)).rotate(90)]
        else: 
            img_resize_list = [img.resize(tuple(size))]

# Ищем лицо
        face_list = []
        for image in img_resize_list: 
# Переводим в RGB
            if image.mode != 'RGB':
                image = image.convert('RGB')

            aligned = mtcnn(image)
            boxes, probs = mtcnn.detect(image)
            if boxes is not None:
                for x_aligned, prob, box in zip(aligned, probs, boxes):
                    if prob >= probability:
                        current_embeding = resnet(x_aligned.unsqueeze(0).to(self.device)).detach().cpu()
                        dictance = cosine(avg_embedding_human, current_embeding)
                        if dictance <= distance: 
                            face_list.append(1)
                        else: 
                            face_list.append(0)
                    else: 
                        face_list.append(0)
            else: 
                face_list.append(0)
        if sum(face_list) > 0: 
            return True
        else: 
            return False

    def fill_exif(self, path, image_description = 'Volokzhanina Veronica'):
        """
        Заполнение exif фото. 
        Вход: 
            path - путь к фото. 
            image_description - значение tag exif для image_description.
        Выход: 
            нет.
        """
# Получаем картинку     
        with open(path, "rb") as palm_1_file:
            image_exif = ImageEXIF(palm_1_file)

# Записываем тег
        image_exif.image_description = image_description

# Записываем картинку с exif 
        with open(path, 'wb') as updated_file:
            updated_file.write(image_exif.get_file())
            
    def is_human_video_detected(self, path, mtcnn, resnet, avg_embedding_human, probability = 0.8):     
        """
        Определение лица искомого человека.
        Вход: 
            path(str) - путь к файлу.
            mtcnn(Model.Model) - модель для распознавания лиц. 
            resnet(Model.Model) - модель в режиме оценки, и перенесенная на видеокарту для вывода эмбединга лица. 
            avg_embedding_human(torch.Tensor) - усредненный вектор лица искомого человека. 
            coef_decrease(int) - коэффициент уменьшение фото. 
            probability(float) - вероятность обнаружения лица.
        Выход: 
            (bool) - булевого значение наличия лица искомого человека.
        """

# Получаем отдельные картинки видео
        video = mmcv.VideoReader(path)
# Переводим видео в картинки (генератор)
        frames = (Image.fromarray(cv.cvtColor(frame, cv.COLOR_BGR2RGB)) for frame in video)

# Распознаем лицо
        for i, frame in enumerate(frames):
            print('\rTracking frame: {}'.format(i + 1), end = '')    
# Распознаем лицо
            aligned = mtcnn(frame)
            boxes, probs = mtcnn.detect(frame)
            if boxes is not None: 
                for x_aligned, prob, box in zip(aligned, probs, boxes):
                    if prob >= probability and self.is_human_image_detected(
                        path = None
                        ,mtcnn = mtcnn
                        ,resnet = resnet
                        ,avg_embedding_human = avg_embedding_human
                        ,frame = frame
                    ): 
                        return True
        return False
    
    def media_sort(self):
        """
        Сортировка медиафайлов с найденным лицом искомого человека по папкам. 
        Вход: 
            нет.
        Выход: 
            нет.
        """

# Получаем не обработанные файлы 
        files_df = self.path_df()

# Загружаем модель для распознавания лиц 
        mtcnn = self.load_model_face_detect()

# Загружаем модель для получения эмбединга 
        resnet = self.load_model_embedding()

# Загружаем усредненный вектор для лица искомого человека
        avg_embedding_human = self.load_avg_embedding_human()

        for index, row in files_df.iterrows():
            if row['type'] in self.type_file_dict['image'] and self.is_human_image_detected(            
                path = row['path']
                ,mtcnn = mtcnn
                ,resnet = resnet
                ,avg_embedding_human = avg_embedding_human
            ): 
# Перемещаем файлы
                shutil.copyfile(row['path'], self.path_output_list + '/' + row['file'])
# Обновляем exif 
                self.fill_exif(path = self.path_output_list + '/' + row['file'])            
            elif row['type'] in self.type_file_dict['video'] and self.is_human_video_detected(
                path = row['path']
                ,mtcnn = mtcnn
                ,resnet = resnet
                ,avg_embedding_human = avg_embedding_human
            ): 
# Перемещаем файлы
                shutil.copyfile(row['path'], self.path_output_list + '/' + row['file'])
# Отмечаем в бд данные 
            self.write_file_db(
                path = row['path']
                ,file = row['file']
                ,file_name = row['file_name']
                ,file_type = row['type']
            )
    
    def write_file_db(self, path, file, file_name, file_type): 
        """
        Запись в бд информации о проверке файла.
        Вход: 
             path(str) - путь к файлу.
             file(str) - полное имя файла.
             file_name(str) - наименование файла.
             file_type(str) - тип файла.             
        Выход:
            нет.
        """
# Производим запись данных в pg
        records = {
                    'path' : path
                    ,'file' : file
                    ,'file_name' : file_name
                    ,'type' : file_type
                    ,'is_processed' : True
                }
        stmt = pg_insert(self.table).values(records)
        update_dict = {
            c.name: c
            for c in stmt.excluded
            if not c.primary_key and c.name != 'date_load'
        }
# Обновляем поля, если строка существует 
        update_stmt = stmt.on_conflict_do_update(
            index_elements = self.primary_keys,
            set_ = update_dict
        )

        with self.engine.connect() as conn:
            result = conn.execute(update_stmt)