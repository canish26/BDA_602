import sys

from pyspark import StorageLevel
from pyspark.ml import Transformer
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


class RollingAverageTransform(Transformer):
    def __init__(self, sparksession):
        self.sparksession = sparksession

    def _transform(self, df):
        df.createOrReplaceTempView("temp_roll_avg_intermediate")
        df.persist(StorageLevel.MEMORY_ONLY)

        rolling_avg = self.sparksession.sql(
            """
            SELECT a.batter, a.local_date, (SUM(b.hit) / NULLIF(SUM(b.atBat), 0)) AS rolling_avg
  FROM temp_roll_avg_intermediate AS a
  JOIN temp_roll_avg_intermediate AS b
  ON a.batter = b.batter AND a.local_date > b.local_date AND b.local_date
  BETWEEN a.local_date - INTERVAL 100 DAY AND a.local_date
  GROUP BY a.batter, a.local_date
         """
        )
        rolling_avg.createOrReplaceTempView("rolling_avg")
        rolling_avg.persist(StorageLevel.MEMORY_ONLY)
        return rolling_avg


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
    df_temp_table = sparksession.sql(
        """
        SELECT bc.batter, g.local_date, bc.hit, bc.atBat
  FROM batter_counts AS bc
  LEFT JOIN game AS g
  ON bc.game_id = g.game_id"""
    )
    roll_t = RollingAverageTransform(sparksession)
    result = roll_t.transform(df_temp_table)
    result.createOrReplaceTempView("temp_roll_avg")
    result.persist(StorageLevel.MEMORY_ONLY)
    sparksession.sql("""SELECT * FROM temp_roll_avg""").show()


if __name__ == "__main__":
    sys.exit(main())
