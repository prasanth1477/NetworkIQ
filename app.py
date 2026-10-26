
from flask import Flask, render_template, request, make_response
import pandas as pd
import networkx as nx
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import os
import io
from werkzeug.utils import secure_filename

app = Flask(__name__)

app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024

DATA_FILE = "social_network.csv"
GRAPH_FILE = os.path.join("static", "network_graph.png")
UPLOAD_FOLDER = "uploads"

os.makedirs("static", exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

CENTRALITY_OPTIONS = {
    "Degree": "Degree Centrality",
    "Betweenness": "Betweenness Centrality",
    "Closeness": "Closeness Centrality",
    "Eigenvector": "Eigenvector Centrality"
}


def get_valid_data_file(data_file):
    """Allow only the default dataset or an existing uploaded CSV."""
    if not data_file:
        return DATA_FILE

    normalized_path = os.path.normpath(data_file)
    normalized_default = os.path.normpath(DATA_FILE)
    normalized_upload_folder = os.path.normpath(UPLOAD_FOLDER)

    if normalized_path == normalized_default:
        return DATA_FILE

    expected_prefix = normalized_upload_folder + os.sep

    if (
        normalized_path.startswith(expected_prefix)
        and normalized_path.lower().endswith(".csv")
        and os.path.isfile(normalized_path)
    ):
        return normalized_path

    return DATA_FILE


def analyze_network(data_file):
    """Read the dataset, calculate centralities, and prepare graph data."""

    try:
        df = pd.read_csv(data_file)
    except pd.errors.EmptyDataError:
        raise ValueError("The CSV file is empty.")
    except UnicodeDecodeError:
        raise ValueError("The CSV must be a readable text file.")
    except pd.errors.ParserError:
        raise ValueError("The CSV file format is invalid.")

    # Validate the CSV columns
    if not {"Source", "Target"}.issubset(df.columns):
        raise ValueError(
            "CSV must contain columns named Source and Target."
        )

    df = df.dropna(subset=["Source", "Target"]).copy()

    df["Source"] = df["Source"].astype(str).str.strip()
    df["Target"] = df["Target"].astype(str).str.strip()

    df = df[
        (df["Source"] != "") &
        (df["Target"] != "")
    ]

    if df.empty:
        raise ValueError("The CSV contains no valid connections.")

    # Create an undirected network
    G = nx.from_pandas_edgelist(
        df,
        source="Source",
        target="Target"
    )

    if G.number_of_nodes() < 2:
        raise ValueError(
            "The network must contain at least two nodes."
        )

    # Calculate all four centrality measures
    degree = nx.degree_centrality(G)
    betweenness = nx.betweenness_centrality(G)
    closeness = nx.closeness_centrality(G)

    try:
        eigenvector = nx.eigenvector_centrality(
            G,
            max_iter=1000
        )
    except nx.PowerIterationFailedConvergence:
        eigenvector = {node: 0.0 for node in G.nodes()}

    # Build the results table
    results = pd.DataFrame({
        "Node": list(G.nodes()),
        "Degree": [degree[n] for n in G.nodes()],
        "Betweenness": [betweenness[n] for n in G.nodes()],
        "Closeness": [closeness[n] for n in G.nodes()],
        "Eigenvector": [eigenvector[n] for n in G.nodes()]
    })

    for column in CENTRALITY_OPTIONS:
        results[column] = results[column].round(4)

    # Prepare nodes for the interactive graph
    graph_nodes = []

    for node in G.nodes():
        graph_nodes.append({
            "id": str(node),
            "label": str(node),
            "value": 15 + degree[node] * 25,
            "title": (
                f"Node: {node}\n"
                f"Degree Centrality: {degree[node]:.4f}\n"
                f"Betweenness Centrality: {betweenness[node]:.4f}\n"
                f"Closeness Centrality: {closeness[node]:.4f}\n"
                f"Eigenvector Centrality: {eigenvector[node]:.4f}"
            )
        })

    # Prepare connections for the interactive graph
    graph_edges = []

    for source, target in G.edges():
        graph_edges.append({
            "from": str(source),
            "to": str(target)
        })

    graph_data = {
        "nodes": graph_nodes,
        "edges": graph_edges
    }

    # Generate the existing static graph as a fallback
    plt.figure(figsize=(12, 8))

    positions = nx.spring_layout(G, seed=42)

    nx.draw_networkx_edges(
        G,
        positions,
        edge_color="#94a3b8",
        width=1.5,
        alpha=0.7
    )

    plotted_nodes = nx.draw_networkx_nodes(
        G,
        positions,
        node_color=[degree[n] for n in G.nodes()],
        cmap=plt.cm.Blues,
        node_size=[
            700 + degree[n] * 1200 for n in G.nodes()
        ],
        edgecolors="white",
        linewidths=1.5
    )

    nx.draw_networkx_labels(
        G,
        positions,
        font_size=10,
        font_weight="bold"
    )

    plt.colorbar(plotted_nodes, label="Degree Centrality")
    plt.title("Social Network Graph", fontsize=16)
    plt.axis("off")
    plt.tight_layout()

    plt.savefig(
        GRAPH_FILE,
        dpi=160,
        bbox_inches="tight",
        facecolor="white"
    )

    plt.close()

    return (
        results,
        G.number_of_nodes(),
        G.number_of_edges(),
        graph_data
    )


@app.route("/", methods=["GET", "POST"])
def home():
    error = None
    uploaded_filename = None

    # Preserve the selected dataset when changing the ranking
    data_file = get_valid_data_file(
        request.values.get("uploaded_file", DATA_FILE)
    )

    selected_measure = request.values.get("measure", "Degree")

    if selected_measure not in CENTRALITY_OPTIONS:
        selected_measure = "Degree"

    if data_file != DATA_FILE:
        uploaded_filename = os.path.basename(data_file)

    if request.method == "POST":
        uploaded_file = request.files.get("dataset")

        if uploaded_file and uploaded_file.filename:
            if not uploaded_file.filename.lower().endswith(".csv"):
                error = "Invalid file type. Please upload a CSV file."
            else:
                filename = secure_filename(uploaded_file.filename)

                if not filename:
                    error = "Invalid filename."
                else:
                    upload_path = os.path.join(
                        UPLOAD_FOLDER,
                        filename
                    )

                    try:
                        uploaded_file.save(upload_path)
                        data_file = upload_path
                        uploaded_filename = filename
                    except Exception:
                        error = "Unable to save the uploaded file."

    try:
        if error:
            raise ValueError(error)

        (
            results,
            node_count,
            edge_count,
            graph_data
        ) = analyze_network(data_file)

        # Sort by the selected measure
        results = results.sort_values(
            by=selected_measure,
            ascending=False
        ).reset_index(drop=True)

        results.insert(
            0,
            "Rank",
            range(1, len(results) + 1)
        )

        top_node = results.iloc[0]["Node"]
        top_score = results.iloc[0][selected_measure]

        return render_template(
            "index.html",
            results=results.to_dict(orient="records"),
            node_count=node_count,
            edge_count=edge_count,
            graph_file="network_graph.png",
            graph_data=graph_data,
            uploaded_filename=uploaded_filename,
            uploaded_file=data_file,
            error=None,
            centrality_options=CENTRALITY_OPTIONS,
            selected_measure=selected_measure,
            selected_measure_label=CENTRALITY_OPTIONS[
                selected_measure
            ],
            top_node=top_node,
            top_score=top_score
        )

    except (ValueError, OSError, pd.errors.ParserError) as e:
        app.logger.warning("Network analysis failed: %s", e)

        return render_template(
            "index.html",
            results=[],
            node_count=0,
            edge_count=0,
            graph_file=None,
            graph_data=None,
            uploaded_filename=None,
            uploaded_file=data_file,
            error=str(e),
            centrality_options=CENTRALITY_OPTIONS,
            selected_measure=selected_measure,
            selected_measure_label=CENTRALITY_OPTIONS[
                selected_measure
            ],
            top_node=None,
            top_score=None
        )


@app.route("/download")
def download_results():
    data_file = get_valid_data_file(
        request.args.get("uploaded_file", DATA_FILE)
    )

    measure = request.args.get("measure", "Degree")

    if measure not in CENTRALITY_OPTIONS:
        measure = "Degree"

    results, _, _, _ = analyze_network(data_file)

    results = results.sort_values(
        by=measure,
        ascending=False
    ).reset_index(drop=True)

    results.insert(
        0,
        "Rank",
        range(1, len(results) + 1)
    )

    output = io.StringIO()
    results.to_csv(output, index=False)

    response = make_response(output.getvalue())

    response.headers["Content-Disposition"] = (
        "attachment; filename=centrality_results.csv"
    )
    response.headers["Content-Type"] = (
        "text/csv; charset=utf-8"
    )

    return response


@app.errorhandler(413)
def file_too_large(error):
    return (
        "File too large. Please upload a CSV file smaller than 5 MB.",
        413
    )


if __name__ == "__main__":
    app.run(debug=True)
