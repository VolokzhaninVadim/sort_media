###############################################################################################################################################
############################################## Импортируем необходимые модули и данные ########################################################
###############################################################################################################################################
# Для работы с Airflow
from airflow import DAG
from airflow.operators.python_operator import PythonOperator 
from airflow.operators.bash_operator import BashOperator
import datetime

# Для получения балансов Телеком
from sort_media.src.SortMedia import SortMedia

# Для работы с операционной сисемой 
import os

# Получаем переменные окружения
PG_PASSWORD=os.environ['PG_PASSWORD']
LOGIN_NAME=os.environ['LOGIN_NAME']
PG_HOST = os.environ['PG_HOST']


# Создаем объект класса 
sort_media = SortMedia(
    pg_password = PG_PASSWORD
    ,pg_login = LOGIN_NAME
    ,pg_host = PG_HOST
)


# Вводим по умолчанию аргументы dag
default_args = {
    'owner': 'Volokzhanin Vadim',
    'start_date': datetime.datetime(2021, 7, 18),
    'depends_on_past': False,
    'retries': 2,
    'retry_delay': datetime.timedelta(minutes = 2)
}

##############################################################################################################################################
############################################### Создадим DAG и поток данных ##################################################################
############################################################################################################################################## 
with DAG(
    "sort_media", 
    description = "Поиск и перемещение медиафайлов с искомым человеком"
    ,default_args = default_args
    ,catchup = False
    ,schedule_interval = "@hourly"
    ,tags=['sort_media']) as dag:

# Получаем баланс Альфа-Банк
    write_new_path = PythonOperator(
        task_id = "write_new_path", 
        python_callable = sort_media.write_new_path, 
        dag = dag
        ) 
    
    media_sort = PythonOperator(
        task_id = "media_sort", 
        python_callable = sort_media.media_sort, 
        dag = dag
        ) 
    
# Порядок выполнения задач
    write_new_path >> media_sort
