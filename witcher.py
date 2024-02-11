import pandas as pd
import numpy as np
import os
import spacy
from spacy import displacy
import networkx as nx
import re
import matplotlib.pyplot as plt
from pyvis.network import Network

def load_nlp_model():
    """Load the Spacy NLP model."""
    return spacy.load("en_core_web_sm")

def process_text_with_spacy(nlp, file_path):
    """Process text from a file using Spacy."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            book_text = file.read()
            return nlp(book_text)
    except FileNotFoundError:
        print(f"File {file_path} not found.")
        return None

def extract_entities_per_sentence(spacy_doc):
    """Extract entities per sentence and store in a dataframe."""
    sent_entity_list = [{
        "sentence": sent.text,
        "entities": [ent.text for ent in sent.ents]
    } for sent in spacy_doc.sents]
    return pd.DataFrame(sent_entity_list)

def filter_entities(entities, character_df):
    """Filter entities to only include characters."""
    return [entity for entity in entities if entity in character_df['character'].values or entity in character_df['character_firstname'].values]

def create_relationships(df, character_df, window_size=5):
    """Create relationships based on proximity within a window of sentences."""
    relationships = []
    for i in range(len(df) - window_size + 1):
        window_entities = sum(df.iloc[i:i+window_size]['character_entities'].tolist(), [])
        for a, b in zip(window_entities, window_entities[1:]):
            if a != b:
                relationships.append((a, b))

    relationship_df = pd.DataFrame(relationships, columns=["source", "target"])
    relationship_df = relationship_df.groupby(["source", "target"]).size().reset_index(name='value')
    return relationship_df


def main():
    books_graph = []
    data_path = './data/'
    
    # Load resources
    nlp = load_nlp_model()
    character_df = pd.read_csv("characters.csv")
    character_df['character_firstname'] = character_df['character'].apply(lambda x: x.split()[0])

    # Process each book
    for book_path in sorted([b.path for b in os.scandir(data_path) if b.name.endswith('.txt')]):
        print(f"Processing {book_path}...")
        book_doc = process_text_with_spacy(nlp, book_path)
        if book_doc is None:
            continue
        
        sent_entity_df = extract_entities_per_sentence(book_doc)
        sent_entity_df['character_entities'] = sent_entity_df['entities'].apply(lambda entities: filter_entities(entities, character_df))
        relationship_df = create_relationships(sent_entity_df, character_df)
        G = nx.from_pandas_edgelist(relationship_df, 
                                source = "source", 
                                target = "target", 
                                edge_attr = "value", 
                                create_using = nx.Graph())
        books_graph.append(G)
        
    # Visualization
    visualize_graph(books_graph)

def visualize_graph(books_graph):
    """Visualize the graph."""
    G = nx.compose_all(books_graph)
    net = Network(notebook=True, height="750px", width="100%", bgcolor="#222222", font_color="white")
    net.from_nx(G)
    net.show("network.html")

if __name__ == "__main__":
    main()
