import sys
from pyspark import StorageLevel
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
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
    sparksession.sql("""SELECT * FROM feature_data""").show()

    # separate features and target variable
    X = features_data.drop(['WinningTeam', 'HomeTeamWins'], axis=1)
    y = features_data['HomeTeamWins']

    # split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # define the first model - Logistic Regression
    lr_model = LogisticRegression()
    lr_model.fit(X_train, y_train)

    # make predictions on test set using logistic regression model
    lr_preds = lr_model.predict(X_test)

    # evaluate accuracy of logistic regression model
    lr_acc = accuracy_score(y_test, lr_preds)
    print("Accuracy of Logistic Regression Model:", lr_acc)

    # define the second model - Random Forest Classifier
    rf_model = RandomForestClassifier()
    rf_model.fit(X_train, y_train)

    # make predictions on test set using random forest classifier model
    rf_preds = rf_model.predict(X_test)

    # evaluate accuracy of random forest classifier model
    rf_acc = accuracy_score(y_test, rf_preds)
    print("Accuracy of Random Forest Classifier Model:", rf_acc)


if __name__ == "__main__":
    sys.exit(main())
