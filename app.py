from flask import Flask, request, render_template
import pandas as pd
from matplotlib.figure import Figure
import os
import io
import base64
from transformers import pipeline

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
df = None
maximum = 5

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/uploads', methods=['POST'])
def upload_file():
    global df
    file = request.files['file']
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
    x_column = request.form['x_columns']
    y_column = request.form['y_columns']
    graph_type = request.form['graph_type']

    fig = Figure()
    ax = fig.subplots()

    if graph_type == 'line':
        ax.plot(df[x_column], df[y_column])
    elif graph_type == 'bar':
        ax.bar(df[x_column], df[y_column])
    elif graph_type == 'scatter':
        ax.scatter(df[x_column], df[y_column])
    elif graph_type == 'histogram': 
        ax.hist(df[x_column], bins=20)

    ax.set_xlabel(x_column)
    ax.set_ylabel(y_column)
    ax.set_title(f"{graph_type.title()} plot of {y_column} vs {x_column}")

    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    plot_data = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()

    use_ai = request.form.get("ai_description")

    pipe = pipeline(
        "text-generation",
        model="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    )
    columns = df.columns.tolist()[:maximum]
    descriptions = {}
    if use_ai:
        for colm in columns:
            sample_data = df[colm].dropna().head(5).tolist()
            prompt = f"""
            <|system|>
            You describe what the column name refers to.

            <|user|>
            Column name: {colm}


            ONE sentence only start with refers to.

            <|assistant|>
            """

            result = pipe(
                prompt,
                max_new_tokens=20,
                temperature=0.2,
                do_sample=True,
                return_full_text=False
            )

            descriptions[colm] = {
                "description": result[0]["generated_text"],
                "sample_data": sample_data
            }



    


    return render_template('plot.html', plot_data=plot_data, descriptions=descriptions)




if __name__ == '__main__':
    app.run(debug=True)



