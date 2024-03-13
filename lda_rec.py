import os
import shutil
import sys
import fitz  # PyMuPDF
import numpy as np
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from gensim.models.coherencemodel import CoherenceModel
from gensim import corpora
from gensim.matutils import Sparse2Corpus
from nltk.corpus import stopwords
from sklearn.datasets import fetch_20newsgroups


class TopicNode:
    """
    Class models topic hierarchy
    """
    def __init__(self, name, top_words):
        self.name = name
        self.top_words = top_words
        self.children = []
        self.docs=0

    def add_child(self,child_node):
        self.children.append(child_node)
    def add_docs(self,num):
        self.docs=num

def print_topic_hierarchy(node, indent=0):
    # Initialize an indentation string based on the current depth
    indentation = '  ' * indent  # Adjust the number of spaces as needed for visibility
    if indent == 0:  # For the root node, you might want to print a starting point or skip
        print("Topic Hierarchy:")
    else:
        # Print the current node's name and its top words
        print(f"{indentation}--{node.name}: {', '.join(node.top_words)} docs: {node.docs}")
    # Recursively call this function for all children, increasing the indentation level
    for child in node.children:
        print_topic_hierarchy(child, indent + 1)


# Example usage within the LDA processing:
root_node = TopicNode("Root", "")

def extract_text_from_pdf(pdf_path):
    with fitz.open(pdf_path) as doc:
        text = ""
        for page in doc:
            text += page.get_text()
    return text


def process_pdfs_in_directory(directory_path):
    documents = []
    document_paths = []
    pdf_files = [f for f in os.listdir(directory_path) if f.endswith('.pdf')]
    
    for pdf_file in pdf_files:
        pdf_path = os.path.join(directory_path, pdf_file)
        text = extract_text_from_pdf(pdf_path)
        documents.append(text)
        document_paths.append(pdf_path)
    return documents, document_paths

def sanitize_topic_name(topic_name):
    """
    Sanitize the topic name to make it suitable for use as a directory name.
    This involves removing special characters and limiting the length.
    """
    sanitized_name = ''.join(char for char in topic_name if char.isalnum() or char in [' ', '_', '-'])
    sanitized_name = '_'.join(sanitized_name.split()[:2])  # Use first 5 words for brevity
    return sanitized_name

def display_topics(model, feature_names, no_top_words):
    """
    Generate and print topics with their most significant words.
    Returns a list of sanitized topic names suitable for directory naming.
    """
    topic_names = []
    for topic_idx, topic in enumerate(model.components_):
        topic_features = " ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]])
        sanitized_name = sanitize_topic_name(topic_features)
        print(f"Topic {topic_idx} ({sanitized_name}): {topic_features}")
        topic_names.append(sanitized_name)
    return topic_names,feature_names


def extract_top_words_per_topic(lda_model, feature_names, n_top_words):
    topics = []
    for topic_idx, topic in enumerate(lda_model.components_):
        top_features_ind = topic.argsort()[:-n_top_words - 1:-1]
        top_features = [feature_names[i] for i in top_features_ind]
        topics.append(top_features)
    return topics

def calculate_coherence_score(documents, topics, dictionary):
    # Gensim's CoherenceModel expects a list of lists of words (topics)
    coherence_model_lda = CoherenceModel(topics=topics, texts=documents, dictionary=dictionary, coherence='c_v')
    coherence_score = coherence_model_lda.get_coherence()
    return coherence_score


def run_lda_recursive(dirname, destdir, stop_words, parent_node=None, depth=0, max_depth=5,top_words =[]):
    dir_topic_name = os.path.basename(dirname)  # Use the directory name as the topic name
    if parent_node is None:
        current_node = TopicNode("Root", top_words)
    else:
        current_node=TopicNode(dir_topic_name,top_words)
        parent_node.add_child(current_node)

    if depth > max_depth:
        print("Maximum depth reached, stopping recursion.")
        return

    documents, document_paths = process_pdfs_in_directory(dirname)
    if not documents or len(document_paths) < 2:
        print("No PDF documents found in the directory, stopping processing.")
        return
    # Assuming vectorizer, LDA setup, and document processing as before
    vectorizer = CountVectorizer(stop_words=stop_words, max_df=0.8, min_df=0.2)
    dtm = vectorizer.fit_transform(documents)
    feature_names = vectorizer.get_feature_names_out()

    lda = LatentDirichletAllocation(n_components=5, random_state=0)
    lda.fit(dtm)

    feature_names = vectorizer.get_feature_names_out()
    n_top_words = 10
    topics = extract_top_words_per_topic(lda, feature_names, n_top_words)
    # Assume the directory name itself represents a topic


    texts = [[word for word in document.lower().split() if word not in stop_words] for document in documents]
     # TODO check for scientific/other corpora possibility
    dictionary = corpora.Dictionary(texts)
    # Document processing and directory creation as before
    coherence_score = calculate_coherence_score(texts, topics, dictionary)
    print(f"Coherence Score: {coherence_score}")

    min_coherence = 0.41
    max_coherence =1;

    if coherence_score < min_coherence:
        print("Coherence score below threshold, stopping recursion.")
        return
    if coherence_score == max_coherence:
        print("")
        print("All documents under this directory the same")
        print("")
        return

    topic_names,feature_names= display_topics(lda, feature_names, 10)
    doc_topic_dist = lda.transform(dtm)

    for doc_idx, topic_dist in enumerate(doc_topic_dist):
        dominant_topic = np.argmax(topic_dist)
        topic_directory = os.path.join(destdir, topic_names[dominant_topic])
        
        if not os.path.exists(topic_directory):
            os.makedirs(topic_directory)
        
        shutil.copy(document_paths[doc_idx], topic_directory)

    print(f"Finished processing level {depth}.")
    new_dirs = [os.path.join(destdir, d) for d in os.listdir(destdir) if os.path.isdir(os.path.join(destdir, d))]
    for idx,new_dir in enumerate(new_dirs):
        print(f"Processing new directory: {new_dir}")

        dominant_topic = np.argmax(doc_topic_dist[idx])  # Assuming this matches the document to new_dir mapping
        specific_top_words = topics[dominant_topic]
        run_lda_recursive(new_dir, new_dir, stop_words, current_node, depth + 1, max_depth,specific_top_words)

    if depth == 0:
        print_topic_hierarchy(current_node)

if __name__ == '__main__':
    nltk.download('stopwords')
    stop_words = list(stopwords.words('english'))
    stop_words += ['et', 'al', 'figure', 'table', 'true', 'false', 'introduction', 'bool', 'references', 'conclusion', 'abstract', 'method', 'doi']
    dirname = sys.argv[1]  # Assumes the first argument is the directory to process
    destdir = dirname
    parent_node = None
    run_lda_recursive(dirname, destdir, stop_words, parent_node)
