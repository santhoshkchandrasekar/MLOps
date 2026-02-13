import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from kneed import KneeLocator
import pickle
import os
import base64

def load_data():
    """
    Loads data from a CSV file, serializes it, and returns the serialized data.
    Returns:
        str: Base64-encoded serialized data (JSON-safe).
    """
    print("Loading Iris dataset for clustering")
    # CUSTOMIZATION: Changed from file.csv to iris.csv
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), "../data/iris.csv"))
    serialized_data = pickle.dumps(df)
    return base64.b64encode(serialized_data).decode("ascii")

def data_preprocessing(data_b64: str):
    """
    Deserializes base64-encoded pickled data, performs preprocessing,
    and returns base64-encoded pickled clustered data.
    """
    data_bytes = base64.b64decode(data_b64)
    df = pickle.loads(data_bytes)

    df = df.dropna()
    # CUSTOMIZATION: Changed to Iris dataset features (sepal_length, sepal_width, petal_length, petal_width)
    clustering_data = df[["sepal_length", "sepal_width", "petal_length", "petal_width"]]

    min_max_scaler = MinMaxScaler()
    clustering_data_minmax = min_max_scaler.fit_transform(clustering_data)

    clustering_serialized_data = pickle.dumps(clustering_data_minmax)
    return base64.b64encode(clustering_serialized_data).decode("ascii")


def build_save_model(data_b64: str, filename: str):
    """
    Builds a KMeans model on the preprocessed data and saves it.
    Returns the SSE list (JSON-serializable).
    """
    data_bytes = base64.b64decode(data_b64)
    df = pickle.loads(data_bytes)

    kmeans_kwargs = {"init": "random", "n_init": 10, "max_iter": 300, "random_state": 42}
    sse = []
    # CUSTOMIZATION: Changed range from 1-50 to 1-10 (more appropriate for Iris dataset)
    for k in range(1, 10):
        kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
        kmeans.fit(df)
        sse.append(kmeans.inertia_)

    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)
    with open(output_path, "wb") as f:
        pickle.dump(kmeans, f)

    return sse


def load_model_elbow(filename: str, sse: list):
    """
    Loads the saved model and uses the elbow method to report k.
    Returns summary of clustering results.
    """
    output_path = os.path.join(os.path.dirname(__file__), "../model", filename)
    loaded_model = pickle.load(open(output_path, "rb"))

    # CUSTOMIZATION: Updated range to match build_save_model
    kl = KneeLocator(range(1, 10), sse, curve="convex", direction="decreasing")
    print(f"Optimal no. of clusters for Iris dataset: {kl.elbow}")

    # CUSTOMIZATION: Predict on the same iris dataset instead of test.csv
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), "../data/iris.csv"))
    df_features = df[["sepal_length", "sepal_width", "petal_length", "petal_width"]].dropna()
    
    min_max_scaler = MinMaxScaler()
    df_scaled = min_max_scaler.fit_transform(df_features)
    
    predictions = loaded_model.predict(df_scaled)
    pred = predictions[0]

    try:
        return int(pred)
    except Exception:
        return pred.item() if hasattr(pred, "item") else pred