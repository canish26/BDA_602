import sys

from pyspark import StorageLevel

# from pyspark.ml import Transformer
from pyspark.sql import SparkSession

appName = "baseball"
master = "local"
database = "baseball"
user = "root"  # pragma: allowlist secret
password = "Csridhar2601@"  # pragma: allowlist secret
server = "localhost"
port = 3306
jdbc_url = f"jdbc:mysql://{server}:{port}/{database}?permitMysqlScheme"
jdbc_driver = "org.mariadb.jdbc.Driver"


# Create a data frame by reading data from MySQL via JDBC
def load_data(sparksession, query):
    df = (
        sparksession.read.format("jdbc")
        .option("url", jdbc_url)
        .option("query", query)
        .option("user", user)
        .option("password", password)
        .option("driver", jdbc_driver)
        .load()
    )

    return df


def main():
    sparksession = SparkSession.builder.master("local[*]").getOrCreate()
    query_sql = "SELECT * FROM feature_data"
    features_data = load_data(sparksession, query_sql)
    features_data.createOrReplaceTempView("game")
    features_data.persist(StorageLevel.MEMORY_ONLY)
    # df_temp_table = sparksession.sql(""" """)
    sparksession.sql("""SELECT * FROM feature_data""").show()


if __name__ == "__main__":
    sys.exit(main())
