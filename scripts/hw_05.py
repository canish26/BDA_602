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
    game_sql = "SELECT * FROM game"
    battercounts_sql = "SELECT * FROM batter_counts"
    game = load_data(sparksession, game_sql)
    game.createOrReplaceTempView("game")
    game.persist(StorageLevel.MEMORY_ONLY)
    batter_count = load_data(sparksession, battercounts_sql)
    batter_count.createOrReplaceTempView("batter_counts")
    batter_count.persist(StorageLevel.MEMORY_ONLY)
    # df_temp_table = sparksession.sql(""" """)
    sparksession.sql("""SELECT * FROM temp_roll_avg""").show()


if __name__ == "__main__":
    sys.exit(main())
