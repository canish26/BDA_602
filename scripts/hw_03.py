import sys

from pyspark import StorageLevel
from pyspark.ml import Transformer
from pyspark.sql import SparkSession

database = "baseball"
user = "root"
password = input("password")
server = "localhost"
port = 3306
jdbc_url = f"jdbc:mysql://{server}:{port}/{database}?permitMysqlScheme"
jdbc_driver = "org.mariadb.jdbc.Driver"


class RollingTransformer(Transformer):
    def setParams(self):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def _transform(self, sparksession, df):
        temp_roll_avg = sparksession.sql(
            " SELECT a.batter, a.local_date, (SUM(b.hit) / NULLIF(SUM(b.atBat), 0)) AS rolling_avg"
            " FROM temp_roll_avg_intermediate AS a"
            " JOIN temp_roll_avg_intermediate AS b"
            " ON a.batter = b.batter AND a.local_date > b.local_date AND b.local_date "
            " BETWEEN a.local_date - INTERVAL 100 DAY AND a.local_date"
            " GROUP BY a.batter, a.local_date ORDER BY a.batter, a.local_date"
        )

        temp_roll_avg.createOrReplaceTempView("temp_roll_avg")
        df.persist(StorageLevel.MEMORY_ONLY)
        temp_roll_avg.show(10)


# Create a data frame by reading data from MySQL via JDBC
def data(sparksession, query):
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
    #    gamesql = "SELECT * FROM game"
    battersql = "SELECT * FROM batter_counts"
    game = data(sparksession, battersql)
    game.createOrReplaceTempView("game")
    game.persist(StorageLevel.MEMORY_ONLY)
    batter_count = data(sparksession, battersql)
    batter_count.createOrReplaceTempView("batter_count")
    batter_count.persist(StorageLevel.MEMORY_ONLY)

    table_join_query = batter_count.join(game, on="game_id")
    table_join_query.createOrReplaceTempView("table_join_query")
    table_join_query.persist(StorageLevel.MEMORY_ONLY)


#    roll_avg_t = RollingTransformer()


#    roll_avg = roll_avg_t._transform(sparksession, table_join_query)


if __name__ == "__main__":
    sys.exit(main())
