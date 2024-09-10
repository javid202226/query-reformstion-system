#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <string>
#include <algorithm>
#include <cmath>
#include <map>
#include <set>

using namespace std;

// Function to convert a string to lowercase
string to_lowercase(const string& str) {
    string result = str;
    transform(result.begin(), result.end(), result.begin(), ::tolower);
    return result;
}

// Function to split a string into words
vector<string> tokenize(const string& str) {
    istringstream iss(str);
    vector<string> tokens;
    string token;
    while (iss >> token) {
        // Remove punctuation
        token.erase(remove_if(token.begin(), token.end(), ::ispunct), token.end());
        tokens.push_back(token);
    }
    return tokens;
}

vector<int> partial_match_documents(const string& query, const unordered_map<int, string>& documents) {
    vector<int> results;

    // Tokenize the query
    vector<string> query_tokens = tokenize(query);

    // Iterate over each document and check for partial matches
    for (const auto& [doc_id, doc_text] : documents) {
        // Tokenize the document text
        vector<string> doc_tokens = tokenize(doc_text);

        // Use a set for fast lookup
        unordered_set<string> doc_token_set(doc_tokens.begin(), doc_tokens.end());

        // Check for any matching tokens
        bool match_found = false;
        for (const string& query_token : query_tokens) {
            if (doc_token_set.find(query_token) != doc_token_set.end()) {
                match_found = true;
                break;
            }
        }

        // If any match found, add the document ID to results
        if (match_found) {
            results.push_back(doc_id);
        }
    }

    return results;
}



// Load stop words from a file or define a set manually
unordered_set<string> load_stopwords() {
    unordered_set<string> stopwords = {"the", "is", "at", "of", "on", "and", "a", "to", "in"};
    // You can also load from a file if needed
    return stopwords;
}

unordered_set<string> stopwords = load_stopwords();

// Structure to represent word embeddings
struct WordEmbedding {
    vector<float> embedding_vector;
    unordered_map<string, float> similar_words;
};

// Function to load word embeddings from a file
unordered_map<string, WordEmbedding> load_embeddings(const string& filepath) {
    ifstream file(filepath);
    string line;
    unordered_map<string, WordEmbedding> embeddings;
    
    while (getline(file, line)) {
        istringstream iss(line);
        string word;
        iss >> word;

        vector<float> embedding_vector;
        float value;
        while (iss >> value) {
            embedding_vector.push_back(value);
        }
        embeddings[word] = {embedding_vector, {}};
    }

    return embeddings;
}

// Function to calculate cosine similarity between two vectors
float cosine_similarity(const vector<float>& vec1, const vector<float>& vec2) {
    float dot_product = 0.0, norm_a = 0.0, norm_b = 0.0;
    for (size_t i = 0; i < vec1.size(); ++i) {
        dot_product += vec1[i] * vec2[i];
        norm_a += vec1[i] * vec1[i];
        norm_b += vec2[i] * vec2[i];
    }
    return dot_product / (sqrt(norm_a) * sqrt(norm_b));
}

// Function to build a semantic text graph
unordered_map<string, WordEmbedding> build_text_graph(const unordered_map<string, WordEmbedding>& embeddings, int top_n = 10) {
    unordered_map<string, WordEmbedding> graph = embeddings;

    for (auto& [word, embedding] : graph) {
        map<float, string, greater<float>> top_similar;
        for (const auto& [other_word, other_embedding] : embeddings) {
            if (word != other_word) {
                float similarity = cosine_similarity(embedding.embedding_vector, other_embedding.embedding_vector);
                top_similar[similarity] = other_word;
            }
        }

        int count = 0;
        for (const auto& [similarity, similar_word] : top_similar) {
            if (count++ < top_n) {
                embedding.similar_words[similar_word] = similarity;
            } else {
                break;
            }
        }
    }

    return graph;
}

// Function to save the graph to a file
void save_graph(const unordered_map<string, WordEmbedding>& graph, const string& filepath) {
    ofstream file(filepath);
    if (file.is_open()) {
        for (const auto& [word, embedding] : graph) {
            file << word << " ";
            for (float value : embedding.embedding_vector) {
                file << value << " ";
            }
            file << "\n";
            for (const auto& [similar_word, similarity] : embedding.similar_words) {
                file << similar_word << " " << similarity << " ";
            }
            file << "\n";
        }
        file.close();
        cout << "Graph saved to " << filepath << endl;
    } else {
        cerr << "Error: Unable to open file " << filepath << " for writing." << endl;
    }
}

// Function to load the graph from a file
unordered_map<string, WordEmbedding> load_graph(const string& filepath) {
    unordered_map<string, WordEmbedding> graph;
    ifstream file(filepath);
    if (file.is_open()) {
        string line;
        while (getline(file, line)) {
            istringstream iss(line);
            string word;
            iss >> word;
            
            vector<float> embedding_vector;
            float value;
            while (iss >> value) {
                embedding_vector.push_back(value);
            }
            
            getline(file, line);
            istringstream iss2(line);
            unordered_map<string, float> similar_words;
            string similar_word;
            while (iss2 >> similar_word) {
                float similarity;
                iss2 >> similarity;
                similar_words[similar_word] = similarity;
            }
            
            graph[word] = {embedding_vector, similar_words};
        }
        file.close();
        cout << "Graph loaded from " << filepath << endl;
    } else {
        cerr << "Error: Unable to open file " << filepath << " for reading." << endl;
    }
    return graph;
}

// Function to reformulate query
// Function to reformulate query using semantic graph
string reformulate_query(const string& query, const unordered_map<string, WordEmbedding>& graph) {
    vector<string> tokens = tokenize(to_lowercase(query));
    set<string> expanded_query(tokens.begin(), tokens.end());

    // Track words already considered in the query
    unordered_set<string> seen_words(tokens.begin(), tokens.end());

    for (const string& token : tokens) {
        if (graph.find(token) != graph.end()) {
            // Get similar words sorted by similarity score
            vector<pair<string, float>> similar_words(graph.at(token).similar_words.begin(), graph.at(token).similar_words.end());
            sort(similar_words.begin(), similar_words.end(), [](const pair<string, float>& a, const pair<string, float>& b) {
                return b.second < a.second; // Sort in descending order of similarity score
            });

            // Add top similar words to expanded query
            int count = 0;
            for (const auto& [similar_word, similarity] : similar_words) {
                if (seen_words.find(similar_word) == seen_words.end()) { // Only add if not already in query
                    expanded_query.insert(similar_word);
                    seen_words.insert(similar_word);
                    if (++count >= 3) { // Limit the number of similar words added per token
                        break;
                    }
                }
            }
        }
    }

    ostringstream oss;
    for (const string& word : expanded_query) {
        oss << word << " ";
    }
    return oss.str();
}


// Structure to represent the inverted index
unordered_map<string, set<int>> inverted_index;

// Function to add a document to the index
void add_to_index(int doc_id, const string& text) {
    vector<string> tokens = tokenize(to_lowercase(text));
    for (const string& token : tokens) {
        inverted_index[token].insert(doc_id);
    }
}

// Function to search documents based on a query
vector<int> search_documents(const string& query) {
    vector<string> query_tokens = tokenize(to_lowercase(query));
    unordered_map<int, int> doc_scores;

    for (const string& token : query_tokens) {
        if (inverted_index.find(token) != inverted_index.end()) {
            for (int doc_id : inverted_index[token]) {
                doc_scores[doc_id]++;
            }
        }
    }

    // Convert scores to sorted document IDs
    vector<pair<int, int>> sorted_docs(doc_scores.begin(), doc_scores.end());
    sort(sorted_docs.begin(), sorted_docs.end(), [](const pair<int, int>& a, const pair<int, int>& b) {
        return b.second < a.second; // Sort in descending order of score
    });

    vector<int> ranked_doc_ids;
    for (const auto& [doc_id, score] : sorted_docs) {
        ranked_doc_ids.push_back(doc_id);
    }

    return ranked_doc_ids;
}

int main() {
    // Example documents
    unordered_map<int, string> documents = {
        {1, "The best restaurants for dining in NYC"},
        {2, "Places to visit in New York"},
        {3, "Top spots to eat in New York City"},
        {4, "Must-see attractions in Manhattan"},
        {5, "Best neighborhoods to live in NYC"},
        {6, "Cultural events happening in New York"},
        {7, "Family-friendly activities in Brooklyn"},
        {8, "History of the Statue of Liberty"},
        {9, "Financial district guide in NYC"},
        {10, "Broadway shows and performances"},
        {11, "Guide to Central Park activities"},
        {12, "Museums to visit in New York City"},
        {13, "Shopping districts in NYC"},
        {14, "Public transportation in NYC"},
        {15, "Guide to Times Square nightlife"},
        {16, "New York City skyline views"},
        {17, "Summer festivals in NYC"},
        {18, "Winter activities in New York"},
        {19, "Springtime in Central Park"},
        {20, "Fall foliage in New York State"},
        {21, "Best rooftop bars in NYC"},
        {22, "Art galleries in Chelsea"},
        {23, "Guide to Brooklyn Bridge Park"},
        {24, "Statue of Liberty ferry schedule"},
        {25, "Historic landmarks in NYC"},
        {26, "Exploring Little Italy"},
        {27, "Chinatown attractions in NYC"},
        {28, "Restaurants with a view in NYC"},
        {29, "Luxury hotels in Manhattan"},
        {30, "Affordable accommodations in NYC"},
        {31, "Day trips from New York City"},
        {32, "Visiting the Empire State Building"},
        {33, "Guide to Wall Street"},
        {34, "Artists and art studios in NYC"},
        {35, "Music venues in Brooklyn"},
        {36, "Theater performances in NYC"},
        {37, "Parks and recreation in Queens"},
        {38, "Guide to the Bronx Zoo"},
        {39, "New York Botanical Garden highlights"},
        {40, "Historic churches in Harlem"},
        {41, "Dining guide to Greenwich Village"},
        {42, "Guide to the High Line park"},
        {43, "Brooklyn Museum exhibits"},
        {44, "Outdoor activities in Staten Island"},
        {45, "Events calendar for NYC"},
        {46, "Food trucks in New York"},
        {47, "Diverse neighborhoods in NYC"},
        {48, "Guide to SoHo shopping"},
        {49, "Best brunch spots in NYC"},
        {50, "Healthy dining options in NYC"},
        {51, "Nightlife guide to Lower East Side"},
        {52, "Bookstores and libraries in NYC"},
        {53, "Guide to street art in Bushwick"},
        {54, "Family attractions in NYC"},
        {55, "Educational activities for kids in NYC"},
        {56, "Guide to Jewish heritage sites in NYC"},
        {57, "LGBTQ+ friendly places in NYC"},
        {58, "Haunted places in New York City"},
        {59, "Unique experiences in NYC"},
        {60, "Hidden gems in New York"},
        {61, "Guide to subway art"},
        {62, "Comedy clubs in NYC"},
        {63, "Sports events in New York"},
        {64, "Famous film locations in NYC"},
        {65, "Guide to street markets in NYC"},
        {66, "Fashion districts in New York"},
        {67, "Guide to urban parks in NYC"},
        {68, "Best pizza places in New York"},
        {69, "Healthy eating in the city"},
        {70, "Guide to food festivals in NYC"},
        {71, "Art festivals in New York"},
        {72, "Fitness centers in NYC"},
        {73, "Ice cream shops in New York"},
        {74, "Coffee culture in NYC"},
        {75, "Guide to vintage shops"},
        {76, "Local breweries in New York"},
        {77, "Cultural celebrations in NYC"},
        {78, "Guide to seafood restaurants"},
        {79, "Dessert spots in NYC"},
        {80, "Best bagel shops in New York"},
        {81, "Guide to food trucks"},
        {82, "Theater festivals in NYC"},
        {83, "Guide to holiday markets"},
        {84, "Christmas celebrations in NYC"},
        {85, "Guide to Halloween events"},
        {86, "Summer concerts in NYC"},
        {87, "Art exhibitions in New York"},
        {88, "Guide to wine bars"},
        {89, "Vegetarian dining options in NYC"},
        {90, "Guide to Italian cuisine"},
        {91, "Ethnic restaurants in NYC"},
        {92, "Guide to jazz clubs"},
        {93, "Best burgers in New York"},
        {94, "Guide to speakeasy bars"},
        {95, "Guide to luxury dining"},
        {96, "Guide to dive bars"},
        {97, "Guide to beer gardens"},
        {98, "Guide to food markets"},
        {99, "Guide to brunch places"},
        {100, "Guide to late-night dining in NYC"}
    };

    // Add documents to the inverted index
    for (const auto& [doc_id, text] : documents) {
        add_to_index(doc_id, text);
    }

    // Load embeddings
    // Update the path to your embeddings file
    string embeddings_path = "/glove.6B.50d.txt";  // Replace with your file path
    cout<<"loading embeddings."<<endl;
    unordered_map<string, WordEmbedding> embeddings = load_embeddings(embeddings_path);
    cout<<"building graph."<<endl;

    // Build text graph
    unordered_map<string, WordEmbedding> text_graph = build_text_graph(embeddings);
    cout<<"graph built successfully."<<endl;

    // Save initial graph
    string graph_filepath = "semantic_graph.txt";
    save_graph(text_graph, graph_filepath);

    // Example query
    string query;

    // Reformulate query
    while(true){
        cout<<"what do you want to serach (enter x to exit): ";
        // cin>>query;
        getline(cin,query);
        if(query == "x")break;
        string reformulated_query = reformulate_query(query, text_graph);
        cout << "Original Query: " << query << endl;
        cout << "Reformulated Query: " << reformulated_query << endl;

        // Search documents
        vector<int> search_results = search_documents(reformulated_query);

        // Display search results
        if(search_results.empty())search_results = partial_match_documents(query, documents);
        cout << "Search Results:" << endl;
        for (int doc_id : search_results) {
            cout << doc_id << ": " << documents[doc_id] << endl;
        }
    }

    // Optionally, update the graph and save it again
    text_graph = build_text_graph(embeddings);  // Update the graph with new data

    // Save updated graph
    save_graph(text_graph, graph_filepath);

    return 0;
}
