import mysql.connector
from flask import Flask, render_template, request

app = Flask(__name__)

# Database connection configuration
db_config = {
    "host": "localhost",
    "user": "root",
    "password": "Csridhar2601@",
    "database": "baseball",
}

# Define routes
@app.route("/")
def home():
    # Connect to the database
    conn = mysql.connector.connect(**db_config)
    cursor = conn.cursor()

    # Execute a query to get all the players
    cursor.execute("SELECT * FROM players")
    players = cursor.fetchall()

    # Close the database connection
    cursor.close()
    conn.close()

    # Render the home template with the players
    return render_template("home.html", players=players)


@app.route("/search", methods=["GET", "POST"])
def search():
    # Connect to the database
    conn = mysql.connector.connect(**db_config)
    cursor = conn.cursor()

    if request.method == "POST":
        # Get the search term from the form
        term = request.form.get("search_term")

        # Execute a query to search for players by name or position
        cursor.execute(
            f"SELECT * FROM players WHERE name LIKE '%{term}%' OR position LIKE '%{term}%'"
        )
        players = cursor.fetchall()

        # Close the database connection
        cursor.close()
        conn.close()

        # Render the search template with the search results
        return render_template("search.html", players=players)

    # Close the database connection
    cursor.close()
    conn.close()

    # Render the search template
    return render_template("search.html")


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")
