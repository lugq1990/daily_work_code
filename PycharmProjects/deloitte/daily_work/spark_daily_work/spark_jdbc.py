"""Read MYSQL data with JDBC connection!

MYSQL is created by docker: 
`docker run --name mysql -e MYSQL_ROOT_PASSWORD=1234 mysql`

so mysql: username: root; password: 1234

get inside of docker mysql:
`docker exec -it mysql bash`
`mysql -u root -p`

create table sql:
`create table test_jdbc(name varchar(255), ds varchar(255), id int(32));`

insert value:
insert into test_jdbc values ("lu", "2022", 1);
insert into test_jdbc values ("lu_2020", "2022", 10);
insert into test_jdbc values ("lu_test", "2021", 100);
insert into test_jdbc values ("lu", "2022", 1000);

"""
from pyspark.sql import SparkSession

spark = SparkSession.builder\
    .config("spark.driver.extraClassPath", "/Users/guangqianglu/Downloads/mysql-connector-java-8.0.28.jar").getOrCreate()

# this is workable!
sql = "(select * from test_jdbc where name = 'lu') as t"
df = spark.read.format('jdbc').options(url="jdbc:mysql://localhost:3306/spark", 
                                       driver='com.mysql.cj.jdbc.Driver', 
                                       dbtable=sql,
                                       user='root', 
                                       password='1234',
                                       lowerBound="0",
                                       upperBound="100",
                                       numPartitions="10",
                                       partitionColumn="id").load()

df.show()
