
import pandas as pd
from question_1 import attrition_customer_data
from question_4 import test_scores
import plotly.graph_objects as go
from jinja2 import Template
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64

# Read data from Excel file
file_path = 'credit_card_customers.xlsx'
df = pd.read_excel(file_path)

# Get the distributions data
distributions = attrition_customer_data(df)

# Create pie charts for each column
pie_chart_htmls = []

for column, value_counts in distributions.items():
    fig = go.Figure(data=[go.Pie(labels=value_counts.index, values=value_counts.values)])
    pie_chart_html = fig.to_html(full_html=False, include_plotlyjs='cdn')
    pie_chart_htmls.append((column, pie_chart_html))

# Simulate the output from test_scores 
dictionary = test_scores()

# Extracting values from the dictionary
matrix = dictionary["conf_matrix"]
score = dictionary["accuracy"]
f1_score_val = dictionary["f1"]
params = dictionary["params"]
selected_features = dictionary["selected_features"]
fpr, tpr = dictionary["roc_curve"]
auc_roc = dictionary["auc_roc"]

# Plotting confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
confusion_matrix_stream = io.BytesIO()
plt.savefig(confusion_matrix_stream, format='png')
plt.close()
confusion_matrix_base64 = base64.b64encode(confusion_matrix_stream.getvalue()).decode()

# Creating a Plotly pie chart for selected features
fig = go.Figure(data=[go.Pie(labels=selected_features['Feature'], values=selected_features['Importance'])])
pie_chart_html = fig.to_html(full_html=False, default_height=500, default_width=1000)

#HTML Template
template = Template("""
<!DOCTYPE html>
<html>
<head>
    <title>Bank Analysis</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        .pie-chart-container {
            display: flex;
            flex-wrap: wrap;
        }
        .pie-chart {
            width: 45%;
            margin: 2.5%;
        }
    </style>
</head>
<body>
    <div>
        <h1>Results</h1>
        <p>F1 Score: {{ f1 }}</p>
        <p>Accuracy: {{ score }}</p>
        <p>Confusion Matrix:</p>
        <img src="data:image/png;base64,{{ confusion_matrix_base64 }}" />
    </div>
    <div>
        <h2>Feature Importance</h2>
        {{ pie_chart_html | safe }}
    </div>
    <div>
        <h2>Key Facts</h2>
        <div class="pie-chart-container">
            {% for title, pie_chart_html in pie_chart_htmls %}
            <div class="pie-chart">
                <h3>{{ title }}</h3>
                {{ pie_chart_html | safe }}
            </div>
            {% endfor %}
        </div>
    </div>
</body>
</html>
""")

# Filling the template with the extracted values
html_code = template.render(
    f1=f1_score_val,
    score=score,
    confusion_matrix_base64=confusion_matrix_base64,
    pie_chart_html=pie_chart_html,
    pie_chart_htmls=pie_chart_htmls
)

# Writing the HTML to a file
with open("index.html", "w") as file:
    file.write(html_code)

print("HTML file with interactive pie charts created successfully.")

