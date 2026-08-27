from flask import Flask, request, render_template
import pandas as pd
from matplotlib.figure import Figure
import os
import io
import base64
from groq import Groq
from dotenv import load_dotenv

load_dotenv()


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
df = None
maximum = 5

ai_client = Groq()
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
TOTAL_REQUESTS = 0
MAX_ALL_REQUESTS = 500


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/uploads', methods=['POST'])
def upload_file():
    global df
    file = request.files['file']
    maximum = request.form.get("quantity")
    if maximum:
        maximum = int(maximum)
    else:
        maximum = 5
        
    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        df = pd.read_csv(filepath)
        columns = list(df.columns)[:maximum]
        #df = df[[columns[0],columns[1]]]
        #columns = list(df.columns)
        return render_template('columns.html', columns=columns)
    
    return 'No file uploaded'

@app.route('/plot', methods=['POST'])
def plot():
    global df
    x_column_name = request.form['x_columns']
    #print(x_column_name)
    y_column_name = request.form['y_columns']
    graph_type = request.form['graph_type']

    fig = Figure()
    ax = fig.subplots()

    x_column = df[x_column_name].dropna()
    y_column = df[y_column_name].dropna()
    

    if graph_type == 'line':
        ax.plot(x_column.dropna(), y_column)
    elif graph_type == 'bar':
        ax.bar(x_column, y_column)
    elif graph_type == 'scatter':
        ax.scatter(x_column, y_column)
    elif graph_type == 'histogram': 
        ax.hist(x_column, bins=20)

    ax.set_xlabel(x_column_name)
    ax.set_ylabel(y_column_name)
    ax.set_title(f"{graph_type.title()} plot of {y_column_name} vs {x_column_name}")

    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    plot_data = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()

    use_ai = request.form.get("ai_description")


    columns = df.columns.tolist()[:maximum]
    descriptions = {}

    if use_ai:
        global TOTAL_REQUESTS
        for colm in columns:
            
            if TOTAL_REQUESTS > MAX_ALL_REQUESTS:
                print("YOU HAVE REACHED YOURR LIMIT")
                continue

            sample_data = df[colm].dropna().head(10).tolist()

            prompt = f"""
            You are a data analysis assistant. Analyze the column name and its sample data to define what it represents.
            column name: {colm}
            sample data: {sample_data}

            Provide ONE sentence only, starting exactly with the phrase "refers to".
            """

            try:
                ai_response = ai_client.chat.completions.create(
                    model= "qwen/qwen3.8-27b",
                    messages = [{"role": "user", "content": prompt}],
                    max_tokens = 30,
                    temperature = 0.2
                )
                ai_text = ai_response.choices[0].message.content.strip()
                TOTAL_REQUESTS += 1
            except Exception as e:
                print("Couldnt be processed", e)
                ai_text = "Couldnt be generated"


            descriptions[colm] = {
                "description": ai_text,
                "sample_data": sample_data
            }



    


    return render_template('plot.html', plot_data=plot_data, descriptions=descriptions)




if __name__ == '__main__':
    app.run(debug=True)



